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
    Convolution block with batch normalization and SiLU activation.
    """

    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1):
        super().__init__()
        self.cb = CB(in_ch, out_ch, k, s, p)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.cb(x))


class ResidualBlock(nn.Module):
    """
    Residual block with two convolution layers.
    """

    def __init__(self, ch, e=0.5):
        super().__init__()
        self.cbs1 = CBS(ch, int(ch * e), k=3, p=1)
        self.cbs2 = CBS(int(ch * e), ch, k=3, p=1)

    def forward(self, x):
        return x + self.cbs2(self.cbs1(x))


class CSPBlock(nn.Module):
    """
    Cross Stage Partial block.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.cbs1 = CBS(in_ch, out_ch // 2)
        self.cbs2 = CBS(in_ch, out_ch // 2)
        self.cbs3 = CBS(2 * (out_ch // 2), out_ch)
        self.res_block = nn.Sequential(
            ResidualBlock(out_ch // 2, e=1.0),
            ResidualBlock(out_ch // 2, e=1.0),
        )

    def forward(self, x):
        res_block_out = self.res_block(self.cbs1(x))
        concat_out = torch.cat((res_block_out, self.cbs2(x)), dim=1)
        return self.cbs3(concat_out)


class CSPNet(nn.Module):
    """
    Cross Stage Partial Network.
    """

    def __init__(self, in_ch, out_ch, n, csp, r):
        super().__init__()
        self.cbs1 = CBS(in_ch, 2 * (out_ch // r))
        self.cbs2 = CBS((2 + n) * (out_ch // r), out_ch)

        if not csp:
            self.mod_list = nn.ModuleList(ResidualBlock(out_ch // r) for _ in range(n))
        else:
            self.mod_list = nn.ModuleList(CSPBlock(out_ch // r, out_ch // r) for _ in range(n))

    def forward(self, x):
        features = list(self.cbs1(x).chunk(chunks=2, dim=1))
        for mod in self.mod_list:
            features.append(mod(features[-1]))

        concat_out = torch.cat(features, dim=1)
        return self.cbs2(concat_out)


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast.
    """

    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.cbs1 = CBS(in_ch, in_ch // 2)
        self.cbs2 = CBS(in_ch * 2, out_ch)
        self.mp = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        y1 = self.cbs1(x)
        y2 = self.mp(y1)
        y3 = self.mp(y2)
        y4 = self.mp(y3)
        concat_out = torch.cat(tensors=[y1, y2, y3, y4], dim=1)
        return self.cbs2(concat_out)


class Attention(nn.Module):
    """
    Multi-head self-attention block.
    """

    def __init__(self, ch, num_head):
        super().__init__()
        self.num_head = num_head
        self.dim_head = ch // num_head
        self.dim_key = self.dim_head // 2
        self.scale = self.dim_key**-0.5

        self.qkv = CB(ch, ch * 2)

        self.cb1 = CB(ch, ch, k=3, p=1, g=ch)
        self.cb2 = CB(ch, ch)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(b, self.num_head, self.dim_key * 2 + self.dim_head, h * w)

        q, k, v = qkv.split([self.dim_key, self.dim_key, self.dim_head], dim=2)

        attn = (q.transpose(dim0=-2, dim1=-1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        y = (v @ attn.transpose(dim0=-2, dim1=-1)).view(b, c, h, w) + self.cb1(v.reshape(b, c, h, w))
        return self.cb2(y)


class PSABlock(nn.Module):
    """
    Parallel Self-Attention Block.
    """

    def __init__(self, ch, num_head):
        super().__init__()
        self.attn = Attention(ch, num_head)
        self.conv = nn.Sequential(
            CBS(ch, ch * 2),
            CB(ch * 2, ch),
        )

    def forward(self, x):
        y = x + self.attn(x)
        return y + self.conv(y)


class C2PSA(nn.Module):
    """
    C2 block with Parallel Self-Attention.
    """

    def __init__(self, ch, n):
        super().__init__()
        self.cbs1 = CBS(ch, 2 * (ch // 2))
        self.cbs2 = CBS(2 * (ch // 2), ch)
        self.psa_mod = nn.Sequential(*(PSABlock(ch // 2, ch // 128) for _ in range(n)))

    def forward(self, x):
        y1, y2 = self.cbs1(x).chunk(chunks=2, dim=1)
        psa_mod_out = self.psa_mod(y2)
        concat_out = torch.cat(tensors=(y1, psa_mod_out), dim=1)
        return self.cbs2(concat_out)


class YOLOBackBone(nn.Module):
    """
    YOLO backbone network.
    """

    def __init__(self, ch, depth, csp):
        super().__init__()
        self.p1 = CBS(ch[0], ch[1], k=3, s=2, p=1)
        self.p2 = nn.Sequential(
            CBS(ch[1], ch[2], k=3, s=2, p=1),
            CSPNet(ch[2], ch[3], depth[0], csp[0], r=4),
        )
        self.p3 = nn.Sequential(
            CBS(ch[3], ch[3], k=3, s=2, p=1),
            CSPNet(ch[3], ch[4], depth[1], csp[0], r=4),
        )
        self.p4 = nn.Sequential(
            CBS(ch[4], ch[4], k=3, s=2, p=1),
            CSPNet(ch[4], ch[4], depth[2], csp[1], r=2),
        )
        self.p5 = nn.Sequential(
            CBS(ch[4], ch[5], k=3, s=2, p=1),
            CSPNet(ch[5], ch[5], depth[3], csp[1], r=2),
            SPPF(ch[5], ch[5]),
            C2PSA(ch[5], depth[4]),
        )

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class YOLONeck(nn.Module):
    """
    YOLO neck network for feature pyramid.
    """

    def __init__(self, ch, depth, csp):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.csp1 = CSPNet(ch[4] + ch[5], ch[4], depth[5], csp[0], r=2)
        self.csp2 = CSPNet(ch[4] + ch[4], ch[3], depth[5], csp[0], r=2)
        self.cbs1 = CBS(ch[3], ch[3], k=3, s=2, p=1)
        self.csp3 = CSPNet(ch[3] + ch[4], ch[4], depth[5], csp[0], r=2)
        self.cbs2 = CBS(ch[4], ch[4], k=3, s=2, p=1)
        self.csp4 = CSPNet(ch[4] + ch[5], ch[5], depth[5], csp[1], r=2)

    def forward(self, x):
        p3, p4, p5 = x
        p4 = self.csp1(torch.cat(tensors=[self.upsample(p5), p4], dim=1))
        p3 = self.csp2(torch.cat(tensors=[self.upsample(p4), p3], dim=1))
        p4 = self.csp3(torch.cat(tensors=[self.cbs1(p3), p4], dim=1))
        p5 = self.csp4(torch.cat(tensors=[self.cbs2(p4), p5], dim=1))
        return p3, p4, p5


class YOLOHead(nn.Module):
    """
    YOLO detection head.
    """

    def __init__(self, nc, fpn_ch):
        super().__init__()
        self.nc = nc
        self.obj = nn.ModuleList(
            nn.Sequential(
                CBS(x, x, k=3, p=1),
                CBS(x, x, k=3, p=1),
                nn.Conv2d(x, out_channels=1, kernel_size=1),
            )
            for x in fpn_ch
        )
        self.stem = nn.ModuleList(
            nn.Sequential(
                CBS(x, x, k=3, p=1, g=x),
                CBS(x, x),
                CBS(x, x, k=3, p=1, g=x),
                CBS(x, x),
            )
            for x in fpn_ch
        )
        self.box = nn.ModuleList([nn.Conv2d(x, out_channels=4, kernel_size=1) for x in fpn_ch])
        self.cls = nn.ModuleList([nn.Conv2d(x, out_channels=self.nc, kernel_size=1) for x in fpn_ch])

    def forward(self, x):
        outputs = []
        for i, (obj, stem, box, cls) in enumerate(zip(self.obj, self.stem, self.box, self.cls)):
            outputs.append(torch.cat(tensors=(box(stem(x[i])), obj(x[i]), cls(stem(x[i]))), dim=1))
        return outputs


class CustomYOLO(nn.Module):
    """
    Custom YOLO model.
    """

    def __init__(self, ch, depth, csp, num_classes):
        super().__init__()
        self.yolo_backbone = YOLOBackBone(ch, depth, csp)
        self.yolo_neck = YOLONeck(ch, depth, csp)
        self.yolo_head = YOLOHead(num_classes, (ch[3], ch[4], ch[5]))

    def forward(self, x):
        return self.yolo_head(self.yolo_neck(self.yolo_backbone(x)))


def yolo_n(num_classes: int) -> nn.Module:
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    ch = [3, 16, 32, 64, 128, 256]
    return CustomYOLO(ch, depth, csp, num_classes)


def yolo_t(num_classes: int) -> nn.Module:
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    ch = [3, 24, 48, 96, 192, 384]
    return CustomYOLO(ch, depth, csp, num_classes)


def yolo_s(num_classes: int) -> nn.Module:
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    ch = [3, 32, 64, 128, 256, 512]
    return CustomYOLO(ch, depth, csp, num_classes)


def yolo_m(num_classes: int) -> nn.Module:
    csp = [True, True]
    depth = [1, 1, 1, 1, 1, 1]
    ch = [3, 64, 128, 256, 512, 512]
    return CustomYOLO(ch, depth, csp, num_classes)


def yolo_l(num_classes: int) -> nn.Module:
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    ch = [3, 64, 128, 256, 512, 512]
    return CustomYOLO(ch, depth, csp, num_classes)


def yolo_x(num_classes: int) -> nn.Module:
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    ch = [3, 96, 192, 384, 768, 768]
    return CustomYOLO(ch, depth, csp, num_classes)
