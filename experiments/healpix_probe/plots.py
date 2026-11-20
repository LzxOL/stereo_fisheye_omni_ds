"""Readability-checked visualizations of measured HEALPix projection and seams."""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.lines import Line2D
import numpy as np

BLUE = "#0072B2"
ORANGE = "#E69F00"
GRAY = "#D3D6DB"


def save(fig, path):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=r"Glyph .* missing")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        w, h = fig.canvas.get_width_height()
        labels = list(fig.texts)
        for legend in fig.legends:
            labels.extend(legend.get_texts())
        for ax in fig.axes:
            labels.extend([ax.title, *ax.texts])
            if ax.get_legend() is not None:
                labels.extend(ax.get_legend().get_texts())
            # A hidden face axis may retain tick objects that are never drawn.
            if not ax.axison:
                continue
            for axis in (ax.xaxis, ax.yaxis):
                if not axis.get_visible():
                    continue
                labels.extend([axis.label, axis.get_offset_text()])
                low, high = sorted(axis.get_view_interval())
                for tick in (*axis.get_major_ticks(), *axis.get_minor_ticks()):
                    if low <= tick.get_loc() <= high:
                        labels.extend([tick.label1, tick.label2])
        for label in labels:
            if label.get_visible() and label.get_text():
                bbox = label.get_window_extent(renderer)
                assert bbox.x0 >= -3 and bbox.y0 >= -3 and bbox.x1 <= w + 3 and bbox.y1 <= h + 3, label
        fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_projections(output, sweeps, images):
    fig, axes = plt.subplots(len(sweeps), 2, figsize=(8.3, 3.2 * len(sweeps)),
                             layout="constrained", squeeze=False)
    for row, sweep in enumerate(sweeps):
        k = len(sweep.selected) // 2
        depth = sweep.selected[k]
        pair = sweep.joint[depth]
        for ax, image, uv, valid, side in zip(axes[row], images, (sweep.left_uv, sweep.right_uv_selected[k]),
                                             (sweep.left_valid, sweep.right_valid[depth]), ("cam4", "cam3")):
            h, w = image.shape[:2]
            ax.imshow(image, alpha=.54)
            native_only = uv[valid & ~pair]
            common = uv[pair]
            size = {8: 11, 16: 5, 64: 1.0}.get(sweep.nside, 4)
            ax.scatter(*native_only.T, s=size * 1.4, c=ORANGE, marker="x", linewidth=.6, alpha=.8)
            ax.scatter(*common.T, s=size, c=BLUE, marker="o", linewidth=0, alpha=.75)
            ax.set(xlim=(0, w-1), ylim=(h-1, 0),
                   title=f"{side} | Nside {sweep.nside} | {pair.sum():,} common cells")
            ax.set_xticks([0, 400, 799])
            ax.set_yticks([0, 384, 767])
            if row == len(sweeps) - 1:
                ax.set_xlabel("Image column (px)")
            if side == "cam4":
                ax.set_ylabel("Image row (px)")
    legend = [Line2D([], [], marker="o", color=BLUE, linestyle="none", label="Both cameras valid"),
              Line2D([], [], marker="x", color=ORANGE, linestyle="none", label="This camera only")]
    fig.legend(handles=legend, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(f"OmniThings camera projections at inverse-depth index {sweeps[0].selected[1]}")
    save(fig, output / "camera_projections.png")


def face_sheet(cells):
    # A display sheet for the twelve labeled faces, without claiming that
    # neighboring panels are neighboring faces.
    return np.stack([np.concatenate(row, axis=1) for row in cells.reshape(3, 4, *cells.shape[1:])]).reshape(
        3 * cells.shape[-1], 4 * cells.shape[-1])


def decorate_faces(ax, nside):
    for c in range(1, 4):
        ax.axvline(c*nside-.5, c="white", lw=1)
    for r in range(1, 3):
        ax.axhline(r*nside-.5, c="white", lw=1)
    for face in range(12):
        y, x = divmod(face, 4)
        ax.text(x*nside + .07*nside, y*nside + .12*nside, str(face), fontsize=8,
                color="#24272A", bbox=dict(facecolor="white", alpha=.82, edgecolor="none", pad=1))
    ax.set_axis_off()


def plot_masks(output, sweep, axis, all_sweeps):
    nside = sweep.nside
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.2), layout="constrained")
    color = ListedColormap([GRAY, ORANGE, BLUE])
    ax = axes[0, 0]
    ax.imshow(face_sheet(sweep.left_valid.astype(np.uint8)), cmap=color, vmin=0, vmax=2,
              interpolation="nearest")
    ax.set_title("cam4 feature support")
    decorate_faces(ax, nside)
    for ax, d in zip((axes[0, 1], axes[1, 0]), (sweep.selected[0], sweep.selected[-1])):
        values = sweep.left_valid.astype(np.uint8) + sweep.joint[d]
        ax.imshow(face_sheet(values), cmap=color, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(f"Stereo support | inverse-depth index {d}")
        decorate_faces(ax, nside)
    ax = axes[1, 1]
    for item, line in zip(all_sweeps, (":", "--", "-")):
        counts = np.asarray(item.report["depth_joint_counts"])
        ax.plot(axis, 100 * counts / item.report["cells"], label=f"Nside {item.nside}", ls=line)
    ax.set(xlabel=r"Inverse distance (m$^{-1}$)", ylabel="Jointly valid (% of sphere)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.17)
    labels = ("Outside cam4 support", "cam4 valid", "Both cameras valid")
    fig.legend(handles=[Line2D([], [], marker="s", color=c, linestyle="none", label=label)
                        for c, label in zip((GRAY, ORANGE, BLUE), labels)],
               loc="outside lower center", frameon=False, ncol=3)
    fig.suptitle(f"Candidate masks and coverage | face numbers in PAD XY order (Nside {nside})")
    save(fig, output / "candidate_masks.png")


def plot_seam(output, fields):
    field, smooth, zero = fields
    nside = field.shape[-1]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1), layout="constrained")
    for ax, other, title in zip(axes, (smooth, zero), ("Cross-face padding", "Independent zero padding")):
        delta = np.maximum(np.abs(other - field), 1e-5)
        heat = ax.imshow(face_sheet(delta), cmap="magma", norm=LogNorm(vmin=1e-4, vmax=1.5),
                         interpolation="nearest")
        decorate_faces(ax, nside)
        ax.set_title(title)
    colorbar = fig.colorbar(heat, ax=list(axes), shrink=.82, label="Absolute change",
                            ticks=[1e-4, 1e-3, 1e-2, 1e-1, 1])
    colorbar.minorticks_off()
    fig.suptitle(f"3×3 mean filter on a continuous spherical field | Nside {nside}")
    save(fig, output / "padding_seams.png")
