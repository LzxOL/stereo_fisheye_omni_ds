"""Isolated mask-aware HEALPix GEV using frozen OmniDS convolution weights."""
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def packed(x):
    return x.flatten(0, 1)


def restore(x, batch, faces):
    return x.unflatten(0, (batch, faces))


def clear(x, support):
    return x.masked_fill(~support, 0)


class FaceOps:
    def __init__(self, earth, cross_face=True):
        self.earth = earth
        self.cross_face = cross_face

    def halo(self, x, support, width):
        x = clear(x, support)
        if not width:
            return x, support
        if not self.cross_face:
            edges = (width,) * 4
            return F.pad(x, edges), F.pad(support, edges)
        if x.shape[1] != 12 or x.shape[-2] != x.shape[-1]:
            raise ValueError("Earth2Grid expects 12 square faces")
        # One identical topology operator for data and masks. At an ambiguous
        # corner both real contributors must be valid (0.5 support is invalid).
        # "cuda" may bind the default device, so run it under the input device.
        backend = self.earth.PaddingBackends.cuda if x.is_cuda else self.earth.PaddingBackends.indexing
        device = torch.cuda.device(x.device) if x.is_cuda else torch.device("cpu")
        with device, self.earth.pad_backend(backend):
            flat, mask = (tensor.reshape(*tensor.shape[:2], -1, *tensor.shape[-2:])
                          for tensor in (x, support))
            halo = self.earth.pad(flat.contiguous(), width)
            halo_valid = self.earth.pad(mask.float().contiguous(), width) == 1
        new_hw = halo.shape[-2:]
        halo = halo.reshape(*x.shape[:-2], *new_hw)
        halo_valid = halo_valid.reshape(*support.shape[:-2], *new_hw)
        return clear(halo, halo_valid), halo_valid

    def conv2d(self, x, support, layer, destination=None):
        p, q = layer.padding
        if p != q:
            raise ValueError("Spatial padding must be symmetric")
        x, _ = self.halo(x, support, p)
        out = F.conv2d(packed(x), layer.weight, layer.bias, layer.stride, 0, layer.dilation, layer.groups)
        return clear(restore(out, *x.shape[:2]), support if destination is None else destination)

    def conv3d(self, x, support, layer, destination=None):
        depth, p, q = layer.padding
        if p != q:
            raise ValueError("Spatial padding must be symmetric")
        x, _ = self.halo(x, support, p)
        out = F.conv3d(packed(x), layer.weight, layer.bias, layer.stride,
                       (depth, 0, 0), layer.dilation, layer.groups)
        return clear(restore(out, *x.shape[:2]), support if destination is None else destination)

    def transpose3d(self, x, support, layer, destination):
        # A low-resolution neighboring cell can contribute via the existing
        # transposed kernel. The two high-resolution halo columns are cropped.
        halo = 1 if self.cross_face else 0
        x, _ = self.halo(x, support, halo)
        out = F.conv_transpose3d(packed(x), layer.weight, layer.bias, layer.stride, layer.padding,
                                 layer.output_padding, layer.groups, layer.dilation)
        if halo:
            h, w = (s * halo for s in layer.stride[-2:])
            out = out[..., h:-h, w:-w]
        return clear(restore(out, *x.shape[:2]), destination)

    def upsample_spatial(self, x, support, destination):
        halo = 1 if self.cross_face else 0
        x, support = self.halo(x, support, halo)
        out = F.interpolate(packed(x), scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        valid = F.interpolate(packed(support.float()), scale_factor=(1, 2, 2),
                              mode="trilinear", align_corners=False) >= 1 - 1e-6
        if halo:
            out, valid = out[..., 2:-2, 2:-2], valid[..., 2:-2, 2:-2]
        valid = restore(valid, *x.shape[:2]) & destination
        return clear(restore(out, *x.shape[:2]), valid), valid


def pool(x, kernel, mode="average"):
    reducer = F.max_pool3d if mode == "any" else F.avg_pool3d
    return restore(reducer(packed(x), kernel, kernel), *x.shape[:2])


def parent(mask, kernel):
    return pool(mask.float(), kernel, "any").bool()


class FaceGEV(nn.Module):
    """Reuse the stock stem, down/up blocks, guide projections and output head."""
    def __init__(self, stock, ops):
        super().__init__()
        self.stock = stock
        self.ops = ops

    def layer(self, x, support, layer, destination):
        x = self.ops.conv3d(x, support, layer.conv, destination)
        result = layer.relu(layer.bn(packed(x)))
        return clear(restore(result, *x.shape[:2]), destination)

    def layers(self, x, mask, layers: Sequence):
        for layer in layers:
            x = self.layer(x, mask, layer, mask)
        return x

    def down(self, x, support, block):
        mask = parent(support, (2, 2, 2))
        x = self.layer(x, support, block.down, mask)
        return self.layers(x, mask, (block.conv1, block.conv2)), mask

    def up(self, x, support, skip, skip_support, block):
        raised = self.ops.transpose3d(x, support, block.up, skip_support)
        raised = clear(restore(block.relu(block.bn(packed(raised))), *x.shape[:2]), skip_support)
        return self.layers(torch.cat((raised, skip), dim=2), skip_support, (block.conv1, block.conv2))

    def guides(self, guide, original_support):
        # Keep the exact stock averaging denominator. Invalid values have zero
        # contribution, and the original valid samples are never expanded.
        support = original_support.any(dim=3)
        guide = clear(guide, support)
        if self.stock.use_spatial_downsample:
            guide = restore(F.avg_pool2d(packed(guide), 2), *guide.shape[:2])
            support = restore(F.max_pool2d(packed(support.float()), 2).bool(), *guide.shape[:2])
        fields = []
        for level, projection in enumerate(self.stock.guide_projs):
            x = self.ops.conv2d(guide, support, projection)
            if level:
                x = restore(F.avg_pool2d(packed(x), 2**level), *x.shape[:2])
            fields.append(x.unsqueeze(3))
        return fields

    def forward(self, costs, guide, valid, audit=False):
        reg = self.stock.regularization
        base, base_valid = clear(costs, valid), valid
        if self.stock.use_spatial_downsample:
            base, base_valid = pool(base, (1, 2, 2)), parent(base_valid, (1, 2, 2))
        gates = self.guides(guide, valid)
        x0 = self.layers(base, base_valid, reg.stem)
        x0 = clear(x0 * gates[0].sigmoid(), base_valid)
        x1, mask1 = self.down(x0, base_valid, reg.down1)
        x1 = clear(x1 * gates[1].sigmoid(), mask1)
        x2, mask2 = self.down(x1, mask1, reg.down2)
        x2 = clear(x2 * gates[2].sigmoid(), mask2)
        y1 = self.up(x2, mask2, x1, mask1, reg.up2)
        y0 = self.up(y1, mask1, x0, base_valid, reg.up1)
        gev = self.ops.conv3d(y0, base_valid, reg.out_conv)
        gev_valid = base_valid
        if self.stock.use_spatial_downsample:
            gev, gev_valid = self.ops.upsample_spatial(gev, gev_valid, valid)
        pyramid = [(gev, gev_valid)]
        for _ in range(1, self.stock.num_pyramid_levels):
            prev, prev_valid = pyramid[-1]
            pyramid.append((pool(prev, (2, 1, 1)), parent(prev_valid, (2, 1, 1))))
        shapes = None
        if audit:
            stages = [(costs,None), (x0,base_valid), (x1,mask1), (x2,mask2),
                      (y1,mask1), (y0,base_valid), *pyramid]
            names = ["input", "stem", "down1", "down2", "up2", "up1"] + [
                f"lookup{i}" for i in range(len(pyramid))]
            shapes = {}
            for name, (value, mask) in zip(names, stages):
                item = {"shape": list(value.shape)}
                if mask is not None:
                    assert torch.isfinite(value).all()
                    assert (value.masked_fill(mask, 0) == 0).all()
                    item["valid"] = int(mask.sum())
                    item["mask_shape"] = list(mask.shape)
                shapes[name] = item
            assert not (pyramid[0][1] & ~valid).any()
        return pyramid, shapes
