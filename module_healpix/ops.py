"""Apply trained operators to PAD-XY faces with mask-aware neighborhoods."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.healpix_probe.stage2.face_gev import FaceGEV, FaceOps, clear, packed, parent, restore, pool


class LevelOps(FaceOps):
    def transpose3d(self, x, support, layer, destination):
        if not self.cross_face:
            return super().transpose3d(x, support, layer, destination)
        halo, _ = self.halo(x, support, 1)
        out = F.conv_transpose3d(packed(halo), layer.weight, layer.bias, layer.stride,
                                 layer.padding, layer.output_padding, layer.groups, layer.dilation)
        # The stock kernel is anchored at the even-index child. Move its
        # spatial phase by half a fine cell around the center of the 2×2 group.
        out = (out[..., 2:-2, 2:-2] + out[..., 1:-3, 2:-2] +
               out[..., 2:-2, 1:-3] + out[..., 1:-3, 1:-3]) * .25
        return clear(restore(out, *x.shape[:2]), destination)


class ParentAlignedGEV(FaceGEV):
    def down(self, x, support, block):
        conv = block.down.conv
        halo, _ = self.ops.halo(x, support, conv.padding[-1])
        sampled = F.conv3d(packed(halo), conv.weight, conv.bias, (conv.stride[0], 1, 1),
                           (conv.padding[0], 0, 0), conv.dilation, conv.groups)
        sampled = clear(restore(sampled, *x.shape[:2]), parent(support, (2, 1, 1)))
        value = pool(sampled, (1, 2, 2))
        mask = parent(support, (2, 2, 2))
        value = clear(restore(block.down.relu(block.down.bn(packed(value))), *x.shape[:2]), mask)
        return self.layers(value, mask, (block.conv1, block.conv2)), mask


def sequential_spatial(layers, ops, value, mask):
    """Preserve stock pointwise activations and normalization on each face."""
    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            value = ops.conv2d(value, mask, layer)
        elif isinstance(layer, (nn.ReLU, nn.GroupNorm, nn.BatchNorm2d)):
            value = clear(restore(layer(packed(value)), *value.shape[:2]), mask)
        else:
            raise TypeError(f"Unsupported face layer: {type(layer).__name__}")
    return value


def aggregate_profile(aggregator, ops, profile, valid):
    if aggregator is None:
        return clear(profile, valid)
    pixels = valid.any(dim=2, keepdim=True)
    x, mask = profile, valid
    final = len(aggregator.conv) - 1
    for i, layer in enumerate(aggregator.conv):
        if isinstance(layer, nn.Conv2d):
            destination = valid if i == final else pixels
            x = ops.conv2d(x, mask, layer, destination=destination)
            mask = destination
        elif isinstance(layer, (nn.GroupNorm, nn.ReLU)):
            x = clear(restore(layer(packed(x)), *x.shape[:2]), mask)
        else:
            raise TypeError("HEALPix only supports the stock standard Similarity Conv2D aggregator")
    if aggregator.use_residual:
        x = profile + aggregator.residual_weight.sigmoid() * x
    return clear(x, valid)


def update_state(block, ops, state, sim, context, gev, depth, mask):
    motion = block.encoder

    def pair(first, second, value):
        value = ops.conv2d(value, mask, first).relu()
        return ops.conv2d(value, mask, second).relu()

    merged = torch.cat((
        pair(motion.conv_sim1, motion.conv_sim2, sim),
        pair(motion.conv_dino1, motion.conv_dino2, context),
        pair(motion.conv_gev1, motion.conv_gev2, gev),
        pair(motion.conv_depth1, motion.conv_depth2, depth),
    ), dim=2)
    codes = ops.conv2d(merged, mask, motion.conv_out).relu()
    x = torch.cat((context, codes, clear(depth, mask)), dim=2)
    gru = block.gru
    hx = torch.cat((state, x), dim=2)
    z = ops.conv2d(hx, mask, gru.convz).sigmoid()
    r = ops.conv2d(hx, mask, gru.convr).sigmoid()
    q = ops.conv2d(torch.cat((r * state, x), dim=2), mask, gru.convq).tanh()
    state = clear((1 - z) * state + z * q, mask)
    head = block.depth_head
    delta = ops.conv2d(ops.conv2d(state, mask, head.conv1).relu(), mask, head.conv2)
    return state, delta


def convex_children(ops, depth, logits, allowed):
    b, faces, _, h, w = depth.shape
    halo, support = ops.halo(depth, allowed, 1)
    patch = F.unfold(packed(halo), (3, 3)).reshape(b, faces, 9, 1, 1, h, w)
    valid = (F.unfold(packed(support.float()), (3, 3)) == 1).reshape(b, faces, 9, 1, 1, h, w)
    scores = logits.reshape(b, faces, 9, 2, 2, h, w)
    weights = scores.masked_fill(~valid, torch.finfo(scores.dtype).min).softmax(dim=2) * valid
    out = (patch * weights).sum(dim=2)
    out = out.permute(0, 1, 4, 2, 5, 3).reshape(b, faces, 1, 2*h, 2*w)
    child_support = allowed.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    return clear(out, child_support), child_support
