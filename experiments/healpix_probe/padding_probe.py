"""Test upstream face padding against the independent healpy neighbour table."""
from importlib.util import module_from_spec, spec_from_file_location

import healpy as hp
import numpy as np
import torch
from torch.nn import functional as F

from geometry_probe import stats


def checked_orders(nside, earth):
    face, row, col = np.indices((12, nside, nside), dtype=np.int64)
    # Standard XY uses (x,y)=South-origin coordinates. Earth PAD XY reverses
    # each axis and swaps them before writing row-major (face,row,col).
    xy = hp.xyf2pix(nside, col, row, face, nest=True)
    pad = hp.xyf2pix(nside, nside - 1 - row, nside - 1 - col, face, nest=True)
    ring = hp.nest2ring(nside, pad)
    ids = np.arange(pad.size)
    np.testing.assert_array_equal(np.sort(pad, axis=None), ids)
    np.testing.assert_array_equal(hp.ring2nest(nside, ring), pad)
    np.testing.assert_array_equal(hp.vec2pix(nside, *hp.pix2vec(nside, pad, nest=True), nest=True), pad)
    np.testing.assert_array_equal(hp.vec2pix(nside, *hp.pix2vec(nside, ring)), ring)
    xyz = hp.pix2vec(nside, pad, nest=True)
    np.testing.assert_allclose(xyz, hp.pix2vec(nside, ring), rtol=0, atol=1e-13)
    if earth is not None:
        t = torch.arange(pad.size)
        reordered = earth.reorder(t, earth.NEST, earth.HEALPIX_PAD_XY)
        np.testing.assert_array_equal(reordered.numpy(), pad.ravel())
        np.testing.assert_array_equal(earth.reorder(t, earth.NEST, earth.XY()).numpy(), xy.ravel())
        np.testing.assert_array_equal(earth.reorder(t, earth.RING, earth.HEALPIX_PAD_XY).numpy(), ring.ravel())
        torch.testing.assert_close(earth.reorder(reordered, earth.HEALPIX_PAD_XY, earth.NEST), t)
    return pad


def backends(earth_source):
    try:
        from earth2grid import healpix as earth
    except (ImportError, OSError) as error:
        if earth_source is None:
            raise RuntimeError("Install Earth2Grid or supply an official source tree with --earth-source") from error
        # Earth2Grid includes this reference with its upstream license. Import
        # the standalone PyTorch file directly if the compiled package fails.
        path = earth_source / "earth2grid/third_party/zephyr/healpix.py"
        spec = spec_from_file_location("upstream_zephyr_healpix", path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return None, module.healpix_pad, repr(error)

    def indexing(tensor, width):
        with earth.pad_backend(earth.PaddingBackends.indexing):
            return earth.pad(tensor, width)

    return earth, indexing, None


# healpy order is SW,W,NW,N,NE,E,SE,S in HEALPix conventions. The
# standalone lookup below follows the independently computed PAD XY mapping.
STENCIL = np.array([[3, 4, 5], [2, 8, 6], [1, 0, 7]])


def convolution(padded, nside):
    field = padded.reshape(12, 1, nside + 2, nside + 2)
    return F.conv2d(field, field.new_full((1, 1, 3, 3), 1 / 9)).reshape(12, nside, nside)


def check_padding(nest, earth, pad, device):
    nside = nest.shape[-1]
    labels = torch.from_numpy(nest.astype(np.float64))
    id_halo = pad(labels[None], 1)[0]
    torch.testing.assert_close(id_halo[:, 1:-1, 1:-1], labels, atol=0, rtol=0)

    # Extract stencil labelled with NEST IDs. Regular positions must match
    # the directional oracle exactly, including along every face boundary.
    actual = id_halo.unfold(-2, 3, 1).unfold(-2, 3, 1).numpy()
    neighbours = hp.get_all_neighbours(nside, nest, nest=True).transpose(1, 2, 3, 0)
    expected = np.concatenate((neighbours, nest[..., None]), axis=-1)[..., STENCIL]
    regular = expected >= 0
    np.testing.assert_array_equal(actual[regular], expected[regular])
    f, rows, cols, dy, dx = np.where(~regular)
    assert len(f) == 24, len(f)
    assert ((dy != 1) & (dx != 1)).all()
    corners = []
    for fi, row, col, ir, ic in zip(f, rows, cols, dy, dx):
        along_row, along_col = actual[fi, row, col, ir, 1], actual[fi, row, col, 1, ic]
        value = actual[fi, row, col, ir, ic]
        if 4 <= fi < 8:
            assert value == (along_row + along_col) / 2
            policy = "mean"
        else:
            assert value in (along_row, along_col)
            policy = "repeat"
        corners.append(dict(face=int(fi), row=int(row), col=int(col), dy=int(ir - 1), dx=int(ic - 1),
                            two_edge_nest=[int(along_row), int(along_col)],
                            policy=policy, label_value=float(value)))

    constant = labels.new_ones((1, *nest.shape))
    torch.testing.assert_close(pad(constant, 2), torch.ones_like(pad(constant, 2)), rtol=0, atol=0)
    torch.testing.assert_close(convolution(pad(constant, 1), nside), constant[0], atol=1e-15, rtol=0)

    # Put the impulse in a cell across face 0's top side, then verify the
    # response against independent neighbor membership over all regular cells.
    source_nest = int(expected[0, 0, nside // 2, 0, 1])
    assert source_nest // nside**2 != 0
    impulse = (labels == source_nest).double()
    response = convolution(pad(impulse[None], 1), nside).numpy()
    normal_cell = regular.all(axis=(-1, -2))
    response_oracle = (expected == source_nest).sum(axis=(-1, -2)) / 9
    np.testing.assert_allclose(response[normal_cell], response_oracle[normal_cell], rtol=0, atol=1e-15)
    np.testing.assert_allclose(response[0, 0, nside // 2], 1 / 9, atol=1e-15)

    # A continuous spherical scalar with no zero seam or polar artifact by
    # construction. Compare identical 3x3 mean filters with two halo policies.
    x, y, z = hp.pix2vec(nside, nest, nest=True)
    field = torch.from_numpy(1.5 + .4 * x -.3 * y + .2 * z + .15 * x * z)
    padded = pad(field[None], 1)
    smooth = convolution(padded, nside).numpy()
    zero_smooth = convolution(F.pad(field, (1, 1, 1, 1))[None], nside).numpy()
    by_nest = np.empty(nest.size)
    by_nest[nest] = field.numpy()
    oracle = by_nest[expected.clip(min=0)].mean(axis=(-1, -2))
    np.testing.assert_allclose(smooth[normal_cell], oracle[normal_cell], rtol=0, atol=2e-14)
    edge = np.ones(nest.shape, dtype=bool)
    edge[:, 1:-1, 1:-1] = False
    cross = regular & ((expected // nside**2) != np.arange(12)[:, None, None, None, None])
    edge_pairs = cross & ((STENCIL == 4) | (STENCIL == 2) | (STENCIL == 6) | (STENCIL == 0))
    cross_edges = []
    for fi, row, col, sr, sc in zip(*np.where(edge_pairs)):
        if (sr, sc) == (0, 1):
            side = "top"
        elif (sr, sc) == (1, 0):
            side = "left"
        elif (sr, sc) == (1, 2):
            side = "right"
        else:
            side = "bottom"
        to_nest = int(expected[fi, row, col, sr, sc])
        cross_edges.append((int(fi), side, to_nest // nside**2))
    adjacency = {str(fi): {side: sorted({target for face, edge_side, target in cross_edges
                                      if (fi, side) == (face, edge_side)})
                           for side in ("top", "left", "right", "bottom")} for fi in range(12)}
    assert all(len(targets) == 1 for sides in adjacency.values() for targets in sides.values())

    backend_results = {}
    if earth is not None:
        for backend, target_device in ((earth.PaddingBackends.zephyr, "cpu"),
                                       (earth.PaddingBackends.cuda, device)):
            if backend == earth.PaddingBackends.cuda and device.type != "cuda":
                continue
            try:
                with earth.pad_backend(backend):
                    for width in (1, 2):
                        sample = field[None].float()
                        other = earth.pad(sample.to(target_device), width).cpu()
                        torch.testing.assert_close(other, pad(sample, width), rtol=0, atol=1e-7)
                backend_results[backend.name] = {"status": "match", "widths": [1, 2]}
            except (RuntimeError, ImportError, OSError, AttributeError, AssertionError,
                    NotImplementedError, ValueError) as exc:
                backend_results[backend.name] = {"status": "failed", "error": str(exc)}

    return {
        "nside": nside,
        "regular_stencil_entries": int(regular.sum()),
        "cross_face_directional_entries": int(cross.sum()),
        "cross_face_edge_entries": int(edge_pairs.sum()),
        "face_adjacency": adjacency,
        "degenerate_corner_cases": corners,
        "constant_convolution_max_error": float((convolution(pad(constant, 1), nside) - 1).abs().max()),
        "impulse_source_nest": source_nest,
        "impulse_source_face": source_nest // nside**2,
        "impulse_on_face_0": float(response[0, 0, nside // 2]),
        "continuous_edge_abs_delta": stats(np.abs(smooth - field.numpy())[edge]),
        "continuous_inner_abs_delta": stats(np.abs(smooth - field.numpy())[~edge]),
        "zero_pad_edge_abs_delta": stats(np.abs(zero_smooth - field.numpy())[edge]),
        "backend_checks": backend_results,
    }, (field.numpy(), smooth, zero_smooth)
