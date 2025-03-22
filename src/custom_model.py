import torch
import torch.nn as nn


class CB(nn.Module):
    """
    Base convolution block with batch normalization.
    """

    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)

    def forward(self, x):
        return self.bn(self.conv(x))


class CBS(nn.Module):
    """
    Convolution + BatchNorm + SiLU activation.
    """

    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1):
        super().__init__()
        self.cb = CB(in_ch, out_ch, k, s, p, g)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.cb(x))


class ELAN(nn.Module):
    """
    Efficient Layer Aggregation Network.
    """

    def __init__(self, c1, c2, c3, n, ids=None):
        super().__init__()
        if ids is None:
            ids = list(range(-min(n + 2, 4), 0))

        self.ids = ids
        self.n = n
        self.cv1 = CBS(c1, c2, k=1)
        self.cv2 = CBS(c1, c2, k=1)
        self.cv3 = nn.ModuleList([CBS(c2, c2, k=3, p=1) for _ in range(n)])

        cat_channels = c2 * len(self.ids)
        self.cv4 = CBS(cat_channels, c3, k=1)

    def forward(self, x):
        out1 = self.cv1(x)
        out2 = self.cv2(x)
        outputs = [out1, out2]

        for i in range(self.n):
            out2 = self.cv3[i](out2)
            outputs.append(out2)

        valid_outputs = []
        for idx in self.ids:
            actual_idx = idx if idx >= 0 else len(outputs) + idx

            if 0 <= actual_idx < len(outputs):
                valid_outputs.append(outputs[actual_idx])
            else:
                valid_outputs.append(outputs[-1])

        return self.cv4(torch.cat(valid_outputs, dim=1))


class MP(nn.Module):
    """
    Max Pooling layer.
    """

    def __init__(self, k=2, s=2):
        super().__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s)

    def forward(self, x):
        return self.m(x)


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer for YOLO.
    """

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # Hidden channels
        self.cv1 = CBS(c1, c_, k=1)
        self.cv2 = CBS(c_ * 4, c2, k=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class YOLOBackbone(nn.Module):
    """
    YOLO backbone network.
    """

    def __init__(self, channels, depth):
        super().__init__()
        self.stage1 = CBS(channels[0], channels[1], k=3, s=2, p=1)
        self.stage2 = nn.Sequential(
            CBS(channels[1], channels[2], k=3, s=2, p=1),
            ELAN(channels[2], channels[2] // 2, channels[2], n=depth[0]),
        )
        self.stage3 = nn.Sequential(
            MP(),
            ELAN(channels[2], channels[3] // 2, channels[3], n=depth[1]),
        )
        self.stage4 = nn.Sequential(
            MP(),
            ELAN(channels[3], channels[4] // 2, channels[4], n=depth[2]),
        )
        self.stage5 = nn.Sequential(
            MP(),
            ELAN(channels[4], channels[5] // 2, channels[5], n=depth[3]),
            SPPF(channels[5], channels[5]),
        )

    def forward(self, x):
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5


class YOLONeck(nn.Module):
    """
    YOLO Feature Pyramid Network (FPN) neck.
    """

    def __init__(self, channels, depth):
        super().__init__()
        # Top-down path
        self.cv1 = CBS(channels[5], channels[4], k=1)  # P5 -> P4
        self.cv2 = CBS(channels[4], channels[4], k=1)  # C4 lateral
        self.elan1 = ELAN(channels[4] * 2, channels[4] // 2, channels[4], n=depth[4])

        self.cv3 = CBS(channels[4], channels[3], k=1)  # P4 -> P3
        self.cv4 = CBS(channels[3], channels[3], k=1)  # C3 lateral
        self.elan2 = ELAN(channels[3] * 2, channels[3] // 2, channels[3], n=depth[4])

        # Bottom-up path
        self.down1 = CBS(channels[3], channels[4], k=3, s=2, p=1)  # P3 -> P4
        self.elan3 = ELAN(channels[4] * 2, channels[4] // 2, channels[4], n=depth[4])

        self.down2 = CBS(channels[4], channels[5], k=3, s=2, p=1)  # P4 -> P5
        self.elan4 = ELAN(channels[5] * 2, channels[5] // 2, channels[5], n=depth[4])

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, inputs):
        c3, c4, c5 = inputs

        # Top-down pathway (fine to coarse)
        p5 = c5
        p5_up = self.upsample(self.cv1(p5))

        p4 = torch.cat([self.cv2(c4), p5_up], dim=1)
        p4 = self.elan1(p4)
        p4_up = self.upsample(self.cv3(p4))

        p3 = torch.cat([self.cv4(c3), p4_up], dim=1)
        p3 = self.elan2(p3)

        # Bottom-up pathway (coarse to fine)
        p3_down = self.down1(p3)
        p4 = torch.cat([p3_down, p4], dim=1)
        p4 = self.elan3(p4)

        p4_down = self.down2(p4)
        p5 = torch.cat([p4_down, p5], dim=1)
        p5 = self.elan4(p5)

        return p3, p4, p5


class YOLOHead(nn.Module):
    """
    YOLO detection head.
    """

    def __init__(self, nc, channels):
        super().__init__()
        # Detection head for each scale
        self.heads = nn.ModuleList([self._build_head(nc, ch) for ch in channels])

    def _build_head(self, nc, ch):
        return nn.ModuleDict(
            {
                "stem": CBS(ch, ch, k=1),
                "cls_conv": nn.Sequential(
                    CBS(ch, ch, k=3, p=1),
                    CBS(ch, ch, k=3, p=1),
                ),
                "reg_conv": nn.Sequential(
                    CBS(ch, ch, k=3, p=1),
                    CBS(ch, ch, k=3, p=1),
                ),
                "cls_pred": nn.Conv2d(ch, nc, kernel_size=1),
                "reg_pred": nn.Conv2d(ch, 4, kernel_size=1),
                "obj_pred": nn.Conv2d(ch, 1, kernel_size=1),
            }
        )

    def forward(self, x):
        p3, p4, p5 = x
        features = [p3, p4, p5]
        outputs = []

        for i, feat in enumerate(features):
            head = self.heads[i]
            stem = head["stem"](feat)

            # Classification branch
            cls_feat = head["cls_conv"](stem)
            cls_output = head["cls_pred"](cls_feat)

            # Regression branch
            reg_feat = head["reg_conv"](stem)
            reg_output = head["reg_pred"](reg_feat)
            obj_output = head["obj_pred"](reg_feat)

            # Concatenate outputs: [x, y, w, h, obj, cls1, cls2, ...]
            output = torch.cat([reg_output, obj_output, cls_output], dim=1)
            outputs.append(output)

        return outputs


class YOLOModel(nn.Module):
    """
    YOLO object detection model with configurable size.
    """

    def __init__(self, num_classes=80, size="s"):
        super().__init__()
        # Configure model based on size
        config = self._get_config(size)

        self.backbone = YOLOBackbone(config["channels"], config["depth"])
        self.neck = YOLONeck(config["channels"], config["depth"])
        self.head = YOLOHead(num_classes, [config["channels"][3], config["channels"][4], config["channels"][5]])
        self.size = size
        self.num_classes = num_classes

    def _get_config(self, size):
        """
        Returns channel and depth configurations based on model size.

        Supported sizes:
        - 'n': nano
        - 't': tiny
        - 's': small
        - 'm': medium
        - 'l': large
        - 'x': xlarge
        """
        configs = {
            "n": {  # nano
                "channels": [3, 16, 32, 64, 128, 256],
                "depth": [1, 1, 1, 1, 1],
            },
            "t": {  # tiny
                "channels": [3, 24, 48, 96, 192, 384],
                "depth": [1, 1, 1, 1, 1],
            },
            "s": {  # small
                "channels": [3, 32, 64, 128, 256, 512],
                "depth": [1, 1, 1, 1, 1],
            },
            "m": {  # medium
                "channels": [3, 48, 96, 192, 384, 768],
                "depth": [2, 2, 2, 2, 2],
            },
            "l": {  # large
                "channels": [3, 64, 128, 256, 512, 1024],
                "depth": [3, 3, 3, 3, 3],
            },
            "x": {  # xlarge
                "channels": [3, 80, 160, 320, 640, 1280],
                "depth": [3, 3, 3, 3, 3],
            },
        }

        if size not in configs:
            supported = ", ".join([f"'{k}'" for k in configs.keys()])
            raise ValueError(f"Unsupported model size: '{size}'. Supported sizes: {supported}")

        return configs[size]

    def forward(self, x):
        return self.head(self.neck(self.backbone(x)))


def yolo_n(num_classes=80):
    """
    Creates a nano YOLO model.
    """
    return YOLOModel(num_classes, size="n")


def yolo_t(num_classes=80):
    """
    Creates a tiny YOLO model.
    """
    return YOLOModel(num_classes, size="t")


def yolo_s(num_classes=80):
    """
    Creates a small YOLO model.
    """
    return YOLOModel(num_classes, size="s")


def yolo_m(num_classes=80):
    """
    Creates a medium YOLO model.
    """
    return YOLOModel(num_classes, size="m")


def yolo_l(num_classes=80):
    """
    Creates a large YOLO model.
    """
    return YOLOModel(num_classes, size="l")


def yolo_x(num_classes=80):
    """
    Creates an xlarge YOLO model.
    """
    return YOLOModel(num_classes, size="x")


def yolo(num_classes=80):
    """
    Creates the default YOLO model.
    """
    return yolo_s(num_classes)
