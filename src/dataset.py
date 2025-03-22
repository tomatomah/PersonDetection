from random import sample, shuffle

from PIL import Image, ImageEnhance
import data_utils
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, config, training):
        self.image_paths = image_paths
        self.labels = labels
        self.config = config
        self.training = training

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = data_utils.imread(image_path)

        # label.shape -> (n, 5:[cx, cy, width, height, class_id])
        label = self.labels[idx]
        image, label = self._preprocess_data(image, label, idx)

        return torch.from_numpy(image), torch.from_numpy(label)

    @staticmethod
    def custom_collate_fn(batch):
        batch_image, batch_labels = zip(*batch)
        batch_image = torch.stack(batch_image)
        return batch_image, batch_labels

    def _preprocess_data(self, image, label, idx):
        """
        Preprocess input image and label data with optional augmentation.
        """
        # Skip augmentation for validation/testing
        if not self.training:
            image, label = self._scale_image_with_aspect_ratio(image, label)
            image = self._normalize_and_transpose_image(image)
            return image, label

        # Apply augmentations during training
        # Mosaic and Mixup augmentation
        if self.config["mosaic"] and self._rand() < self.config["mosaic"]:
            image, label = self._get_mosaic_data(image, label, idx)
            if self.config["mix_up"] and self._rand() < self.config["mix_up"]:
                image, label = self._get_mixup_data(image, label, idx)
        else:
            image, label = self._scale_image_with_aspect_ratio(image, label)

        # Final normalization
        image = self._normalize_and_transpose_image(image)

        return image, label

    def _normalize_and_transpose_image(self, image):
        """
        Normalize pixel values to [0-1] and transpose image from HWC to CHW format.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return image

    def _augment_hsv(self, image):
        """
        Apply random HSV augmentation to the image using Pillow.
        """
        hue_shift = np.random.uniform(-self.config["hsv_h"], self.config["hsv_h"])
        sat_shift = np.random.uniform(-self.config["hsv_s"], self.config["hsv_s"])
        val_shift = np.random.uniform(-self.config["hsv_v"], self.config["hsv_v"])

        image = image.convert("HSV")
        hue, sat, val = image.split()
        hue = hue.point(lambda x: ((x + int(hue_shift * 255)) % 256))
        sat = sat.point(lambda x: np.clip(x + int(sat_shift * 255), 0, 255))
        val = val.point(lambda x: np.clip(x + int(val_shift * 255), 0, 255))
        image = Image.merge("HSV", (hue, sat, val)).convert("RGB")

        return image

    def _get_mixup_data(self, image, label, idx):
        """
        Create a mixup image by blending current image with a randomly selected sample,
        and combine their corresponding labels.
        """
        # Get and scale random sample
        sample_image, sample_label = self._pick_up_sample_data(sample_num=1, idx=idx)
        sample_image, sample_label = self._scale_image_with_aspect_ratio(sample_image[0], sample_label[0])

        image_arr = np.array(image)
        sample_image_arr = np.array(sample_image)

        # Blend images with equal weights
        mixup_image_arr = (image_arr.astype(np.float32) * 0.5 + sample_image_arr.astype(np.float32) * 0.5).astype(
            np.uint8
        )
        mixup_image = Image.fromarray(mixup_image_arr)

        # Combine labels, handling empty label cases
        if len(label) == 0:
            mixup_boxes = sample_label
        elif len(sample_label) == 0:
            mixup_boxes = label
        else:
            mixup_boxes = np.concatenate([label, sample_label], axis=0)

        return mixup_image, mixup_boxes

    def _scale_image_with_aspect_ratio(self, image, label):
        """
        Scale image to target size while maintaining aspect ratio and adjust object labels accordingly.
        Creates a gray background of target size and places the scaled image at a random position.
        """
        # Create gray background of target size (gray_value = 114)
        dst_image = Image.new("RGB", (self.config["input_width"], self.config["input_height"]), (114, 114, 114))
        dst_label = np.zeros((label.shape[0], label.shape[1]), dtype=label.dtype)

        width, height = image.size

        resize_rate = min((self.config["input_height"] / height), (self.config["input_width"] / width))
        resize_height = int(height * resize_rate)
        resize_width = int(width * resize_rate)

        # Resize image using appropriate interpolation method
        if (self.config["input_height"] * self.config["input_width"]) < (height * width):
            # Use LANCZOS for downsampling (equivalent to cv2.INTER_LANCZOS4)
            resize_image = image.resize((resize_width, resize_height), Image.LANCZOS)
        else:
            # Use BICUBIC for upsampling (equivalent to cv2.INTER_CUBIC)
            resize_image = image.resize((resize_width, resize_height), Image.BICUBIC)

        # Scale object coordinates
        dst_label[:, :4] = label[:, :4] * resize_rate
        dst_label[:, 4] = label[:, 4]  # Preserve class IDs

        # Place scaled image on background and update object coordinates
        dst_dx, dst_dy = self._paste_image_at_random_position(resize_image, dst_image)
        dst_label[:, 0] += dst_dx
        dst_label[:, 1] += dst_dy

        return dst_image, dst_label

    def _paste_image_at_random_position(self, src_image, dst_image):
        """
        Paste source image at a random position within destination image during training,
        or at center position during inference.
        """
        # Generate random position factor during training, 0 during inference
        if self.training:
            pb = np.random.rand()
        else:
            pb = 0.0

        # Get image dimensions
        src_width, src_height = src_image.size
        dst_width, dst_height = dst_image.size

        # Calculate paste coordinates based on position factor
        dst_dx = int(pb * (dst_width - src_width))
        dst_dy = int(pb * (dst_height - src_height))

        # Paste source image at calculated position
        dst_image.paste(src_image, (dst_dx, dst_dy))

        return dst_dx, dst_dy

    def _get_mosaic_data(self, image, label, idx):
        """
        Create a mosaic image by combining 4 images (current image and 3 randomly selected samples)
        with data augmentation and bounding box adjustments.
        """
        # Pick up 3 sample data
        images, labels = self._pick_up_sample_data(sample_num=3, idx=idx)

        # Append current image data and label to the sample data
        images.append(image)
        labels.append(label)

        # Random shuffle
        sample_indices = list(range(0, len(images)))
        shuffle(sample_indices)

        # Reorder images and labels based on shuffled indices
        images = [images[idx] for idx in sample_indices]
        labels = [labels[idx] for idx in sample_indices]

        # Create the mosaic image
        mosaic_image = Image.new("RGB", (self.config["input_width"], self.config["input_height"]), (114, 114, 114))

        # Calculate the placement position for each image
        cutx = int(self.config["input_width"] * self._rand(0.3, 0.7))
        cuty = int(self.config["input_height"] * self._rand(0.3, 0.7))

        # place = [left-top, left-bottom, right-bottom, right-top]
        place_x = [0, 0, cutx, cutx]
        place_y = [0, cuty, cuty, 0]

        bboxes = []
        for i, img in enumerate(images):
            bbox = np.zeros_like(labels[i])

            image_w, image_h = img.size

            # Convert cx,cy,w,h -> xmin,ymin,xmax,ymax
            bbox = data_utils.cxcywh2xyxy(labels[i])

            # Flipping image horizontally
            if self._rand() < self.config["flip_lr"]:
                img, bbox = self._flip_horizontal(img, bbox, image_w)

            # Flipping image vertically
            if self._rand() < self.config["flip_ud"]:
                img, bbox = self._flip_vertical(img, bbox, image_h)

            # Calculate new aspect ratio by multiplying original aspect ratio
            new_aspect_ratio = (image_w / image_h) * (self._rand(0.7, 1.3))

            # Scale value for resizing the images
            scale = self._rand(0.4, 1)

            # Calculate the image size based on the new aspect ratio
            if new_aspect_ratio < 1:  # Vertical image
                nh = int(scale * self.config["input_height"])
                nw = int(nh * new_aspect_ratio)
            else:  # Horizontal image
                nw = int(scale * self.config["input_width"])
                nh = int(nw * (1 / new_aspect_ratio))

            # Choose interpolation method based on image size
            if (self.config["input_width"] * self.config["input_height"]) < (image_w * image_h):
                resize_image = img.resize((nw, nh), Image.LANCZOS)
            else:
                resize_image = img.resize((nw, nh), Image.BICUBIC)

            # Get the placement position for the image
            x1 = place_x[i]
            y1 = place_y[i]

            # Calculate the valid range for each quadrant based on cutx and cuty
            if i == 0:  # left-top
                x2, y2 = cutx, cuty
            elif i == 1:  # left-bottom
                x2, y2 = cutx, self.config["input_height"]
            elif i == 2:  # right-bottom
                x2, y2 = self.config["input_width"], self.config["input_height"]
            else:  # i == 3, right-top
                x2, y2 = self.config["input_width"], cuty

            # Calculate destination width and height for this quadrant
            dst_width = min(nw, x2 - x1)
            dst_height = min(nh, y2 - y1)

            # Place the image in the correct quadrant
            mosaic_image.paste(resize_image.crop((0, 0, dst_width, dst_height)), (x1, y1))

            # Update the bounding box coordinates
            bbox[:, [0, 2]] = bbox[:, [0, 2]] * (nw / image_w) + x1
            bbox[:, [1, 3]] = bbox[:, [1, 3]] * (nh / image_h) + y1

            # Select valid bounding boxes
            box_w = bbox[:, 2] - bbox[:, 0]
            box_h = bbox[:, 3] - bbox[:, 1]
            bbox = bbox[np.logical_and(box_w > 1, box_h > 1)]  # w, h > 1pixel

            bboxes.append(bbox)

        # Apply HSV augmentation
        mosaic_image = self._augment_hsv(mosaic_image)

        bboxes = self._merge_bboxes(bboxes, cutx, cuty)
        bboxes = np.array(bboxes, dtype=np.float32)

        # Convert xmin,ymin,xmax,ymax -> cx,cy,w,h
        if len(bboxes.shape) != 2 or bboxes.shape[1] != 5:
            mosaic_boxes = np.zeros((0, 5), dtype=bboxes.dtype)
        else:
            mosaic_boxes = data_utils.xyxy2cxcywh(bboxes)

        return mosaic_image, mosaic_boxes

    def _merge_bboxes(self, bboxes, cutx, cuty):
        """
        Merge and adjust bounding boxes at mosaic split lines by clipping or removing protruding boxes.
        """
        merge_bbox = []
        for i in range(len(bboxes)):
            for bbox in bboxes[i]:
                tmp_box = []
                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

                # Adjust the bounding box based on the mosaic image split position
                if i == 0:  # Top-left image
                    if ymin > cuty or xmin > cutx:  # Box is protruding to the bottom-left or top-right
                        continue
                    if ymax >= cuty and ymin <= cuty:
                        ymax = cuty
                    if xmax >= cutx and xmin <= cutx:
                        xmax = cutx

                if i == 1:  # Bottom-left image
                    if ymax < cuty or xmin > cutx:  # Box is protruding to the top-left or bottom-right
                        continue
                    if ymax >= cuty and ymin <= cuty:
                        ymin = cuty
                    if xmax >= cutx and xmin <= cutx:
                        xmax = cutx

                if i == 2:  # Bottom-right image
                    if ymax < cuty or xmax < cutx:  # Box is protruding to the top-right or bottom-left
                        continue
                    if ymax >= cuty and ymin <= cuty:
                        ymin = cuty
                    if xmax >= cutx and xmin <= cutx:
                        xmin = cutx

                if i == 3:  # Top-right image
                    if ymin > cuty or xmax < cutx:  # Box is protruding to the bottom-right or top-left
                        continue
                    if ymax >= cuty and ymin <= cuty:
                        ymax = cuty
                    if xmax >= cutx and xmin <= cutx:
                        xmin = cutx

                # Add the adjusted bounding box to the temporary list
                tmp_box.append(xmin)
                tmp_box.append(ymin)
                tmp_box.append(xmax)
                tmp_box.append(ymax)
                tmp_box.append(bbox[4])

                # Add the temporary list to the merge list
                merge_bbox.append(tmp_box)

        return merge_bbox

    def _pick_up_sample_data(self, sample_num, idx):
        """
        Randomly selects sample images and their labels from the dataset, excluding the specified index.
        """
        data_indices = list(range((len(self.image_paths))))
        data_indices.pop(idx)
        sample_indices = sample(data_indices, sample_num)

        images = []
        labels = []
        for data_idx in sample_indices:
            image_path = self.image_paths[data_idx]
            image = data_utils.imread(image_path)
            label = self.labels[data_idx]
            images.append(image)
            labels.append(label)
        return images, labels

    def _rand(self, min_value=0, max_value=1):
        """
        Generates a random float number within the specified range [min_value, max_value].
        """
        return np.random.rand() * (max_value - min_value) + min_value

    def _flip_horizontal(self, image, bbox, image_w):
        """
        Flip image horizontally and adjust bounding box coordinates accordingly.
        """
        # Flip horizontally using Pillow's method
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Adjust bounding box coordinates
        bbox[:, [0, 2]] = image_w - bbox[:, [2, 0]]

        return image, bbox

    def _flip_vertical(self, image, bbox, image_h):
        """
        Flip image vertically and adjust bounding box coordinates accordingly.
        """
        # Flip vertically using Pillow's method
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

        # Adjust bounding box coordinates
        bbox[:, [1, 3]] = image_h - bbox[:, [3, 1]]

        return image, bbox
