import torch
import torch.nn as nn
import torch.nn.functional as F


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
        br = torch.min((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou**2
        elif self.loss_type == "giou":
            c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
            c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class CustomLoss(object):
    def __init__(self, num_classes, device, fp16=False):
        self.num_classes = num_classes
        self.device = device
        self.fp16 = fp16

        self.strides = torch.tensor([8, 16, 32], device=device)

        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")

    def __call__(self, inputs, labels):
        outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for stride, output in zip(self.strides, inputs):
            output, grid = self.get_output_and_grid(output, stride)
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)
            outputs.append(output)

        concat_outputs = torch.cat(outputs, dim=1)
        loss = self.get_losses(x_shifts, y_shifts, expanded_strides, labels, concat_outputs)

        return loss

    def get_output_and_grid(self, output, stride):
        height, width = output.shape[-2:]
        grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        grid_y, grid_x = grid_y.to(self.device), grid_x.to(self.device)
        grid = torch.stack((grid_x, grid_y), dim=2).view(1, height, width, 2).view(1, -1, 2)
        output = output.flatten(start_dim=2).permute(0, 2, 1).contiguous()
        output[:, :, :2] = (output[:, :, :2] + grid.to(output.dtype)) * stride
        output[:, :, 2:4] = torch.exp(output[:, :, 2:4]) * stride

        return output, grid

    def get_losses(self, x_shifts, y_shifts, expanded_strides, labels, outputs):
        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4:5]
        cls_preds = outputs[:, :, 5:]
        total_num_anchors = outputs.shape[1]

        x_shifts = torch.cat(x_shifts, dim=1).to(outputs.dtype)
        y_shifts = torch.cat(y_shifts, dim=1).to(outputs.dtype)
        expanded_strides = torch.cat(expanded_strides, dim=1).to(outputs.dtype)

        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        for batch_idx in range(outputs.shape[0]):
            num_gt = len(labels[batch_idx])

            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx][:, :4].to(outputs.dtype)
                gt_classes = labels[batch_idx][:, 4].to(outputs.dtype)
                bboxes_preds_per_image = bbox_preds[batch_idx]
                cls_preds_per_image = cls_preds[batch_idx]
                obj_preds_per_image = obj_preds[batch_idx]
                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.get_assignments(
                    num_gt,
                    total_num_anchors,
                    gt_bboxes_per_image,
                    gt_classes,
                    bboxes_preds_per_image,
                    cls_preds_per_image,
                    obj_preds_per_image,
                    expanded_strides,
                    x_shifts,
                    y_shifts,
                )
                num_fg += num_fg_img
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(cls_target.dtype))
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() / num_fg

        reg_weight = 5.0

        return reg_weight * loss_iou, loss_obj, loss_cls

    @torch.no_grad()
    def get_assignments(
        self,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        cls_preds_per_image,
        obj_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
    ):
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds = cls_preds_per_image[fg_mask]
        obj_preds = obj_preds_per_image[fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        pair_wise_ious = self.calc_bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        with torch.amp.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu", enabled=False):
            cls_preds = (
                cls_preds.unsqueeze(dim=0).repeat(num_gt, 1, 1).sigmoid()
                * obj_preds.unsqueeze(dim=0).repeat(num_gt, 1, 1).sigmoid()
            )
            gt_cls_per_image = (
                F.one_hot(gt_classes.to(torch.int64), self.num_classes)
                .to(cls_preds.dtype)
                .unsqueeze(dim=1)
                .repeat(1, num_in_boxes_anchor, 1)
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                torch.clamp(cls_preds.sqrt(), 1e-6, 1.0 - 1e-6), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds

        cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center).float()

        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = self.simota_matching(
            cost, pair_wise_ious, gt_classes, num_gt, fg_mask
        )
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg

    def calc_bboxes_iou(self, bboxes_a, bboxes_b):
        lt_bboxes_a = bboxes_a[:, None, :2] - (bboxes_a[:, None, 2:] / 2)
        lt_bboxes_b = bboxes_b[None, :, :2] - (bboxes_b[None, :, 2:] / 2)
        rb_bboxes_a = bboxes_a[:, None, :2] + (bboxes_a[:, None, 2:] / 2)
        rb_bboxes_b = bboxes_b[None, :, :2] + (bboxes_b[None, :, 2:] / 2)

        lt = torch.max(lt_bboxes_a, lt_bboxes_b)
        rb = torch.min(rb_bboxes_a, rb_bboxes_b)

        area_a = torch.prod(bboxes_a[:, 2:], dim=1)
        area_b = torch.prod(bboxes_b[:, 2:], dim=1)

        is_valid_overlap = (lt < rb).to(lt.dtype).prod(dim=2)

        area_intersection = torch.prod(rb - lt, dim=2) * is_valid_overlap

        iou = area_intersection / (area_a[:, None] + area_b[None, :] - area_intersection + 1e-8)

        return iou

    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt):
        center_radius = 1.5
        expanded_strides_per_image = expanded_strides[0]

        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(dim=0).repeat(num_gt, 1)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(dim=0).repeat(num_gt, 1)

        gt_bboxes_per_image_ltx = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(dim=1).repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_rbx = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(dim=1).repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_lty = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(dim=1).repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_rby = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(dim=1).repeat(1, total_num_anchors)
        )

        b_ltx = x_centers_per_image - gt_bboxes_per_image_ltx
        b_rbx = gt_bboxes_per_image_rbx - x_centers_per_image
        b_lty = y_centers_per_image - gt_bboxes_per_image_lty
        b_rby = gt_bboxes_per_image_rby - y_centers_per_image
        bbox_deltas = torch.stack([b_ltx, b_lty, b_rbx, b_rby], dim=2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        gt_bboxes_per_image_ltx = (gt_bboxes_per_image[:, 0]).unsqueeze(dim=1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(dim=0).repeat(num_gt, 1)
        gt_bboxes_per_image_rbx = (gt_bboxes_per_image[:, 0]).unsqueeze(dim=1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(dim=0).repeat(num_gt, 1)
        gt_bboxes_per_image_lty = (gt_bboxes_per_image[:, 1]).unsqueeze(dim=1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(dim=0).repeat(num_gt, 1)
        gt_bboxes_per_image_rby = (gt_bboxes_per_image[:, 1]).unsqueeze(dim=1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(dim=0).repeat(num_gt, 1)

        c_ltx = x_centers_per_image - gt_bboxes_per_image_ltx
        c_rbx = gt_bboxes_per_image_rbx - x_centers_per_image
        c_lty = y_centers_per_image - gt_bboxes_per_image_lty
        c_rby = gt_bboxes_per_image_rby - y_centers_per_image
        center_deltas = torch.stack([c_ltx, c_lty, c_rbx, c_rby], dim=2)

        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        # is_in_boxes_anchor = is_in_boxes_all | (is_in_centers_all & is_in_boxes_all)

        is_in_boxes_and_center = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]

        return is_in_boxes_anchor, is_in_boxes_and_center

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
