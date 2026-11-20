"""Read-only OmniDS HEALPix feasibility probe. Outputs go to --out."""
import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import json
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import healpy
import numpy as np
import torch

from experiments.local_spherical_grid.camera_geometry import load_pair
from geometry_probe import build_sweep, candidate_axis, hash_file, save_grids
from padding_probe import backends, checked_orders, check_padding
from plots import plot_masks, plot_projections, plot_seam
from volumes import frozen_features, run_volumes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="OmniThings dataset folder containing config.yaml")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trusted OmniDS Yin–Yang training checkpoint")
    parser.add_argument("--selection", type=Path,
                        default=ROOT / "experiments/local_spherical_grid/outputs_v1_yin50/grid_v1_selected.json")
    parser.add_argument("--earth-source", type=Path, help="Earth2Grid source checkout for the standalone padding fallback")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--nsides", type=int, nargs="+", default=[8, 16, 64])
    parser.add_argument("--frame", type=int, default=4887)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--depth-chunk", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--save-grids", action="store_true")
    args = parser.parse_args()
    if min(args.nsides) < 4 or any(n & (n - 1) for n in args.nsides):
        parser.error("Each Nside must be a power of two and at least 4")
    if min(args.depth_chunk, args.repeats) < 1:
        parser.error("--depth-chunk and --repeats must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    earth, pad, backend_issue = backends(args.earth_source)
    try:
        earth_version = version("earth2grid")
    except PackageNotFoundError:
        earth_version = None
    print(f"padding: {'Earth2Grid indexing' if earth is not None else 'upstream Zephyr'}", flush=True)
    if backend_issue:
        print("Earth2Grid import issue:", backend_issue, flush=True)
    cameras, images, config, provenance = load_pair(args.data, frame=args.frame)
    axis = candidate_axis(args.selection, config)
    output = {
        "platform": platform.platform(), "python": sys.version.split()[0], "torch": torch.__version__,
        "torch_compiled_cuda": torch.version.cuda, "numpy": np.__version__, "healpy": healpy.__version__,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "earth2grid": earth_version, "padding_backend": "indexing" if earth is not None else "zephyr",
        "earth2grid_import_issue": backend_issue,
        "data": str(args.data), "config_sha256": hash_file(args.data / "config.yaml"),
        "frame": args.frame, "cameras": provenance,
        "raw_images_sha256": [hash_file(Path(camera["image"])) for camera in provenance],
        "raw_masks_sha256": [hash_file(Path(camera["mask"])) for camera in provenance],
        "selection": str(args.selection), "selection_sha256": hash_file(args.selection),
        "inverse_depth_candidates": axis.tolist(), "depth_chunk": args.depth_chunk,
        "timed_repeats": args.repeats, "resolutions": {},
    }

    sweeps = []
    with torch.inference_mode():
        features, output["encoder"] = frozen_features(args.checkpoint, cameras, images, device)
        for nside in args.nsides:
            tick = perf_counter()
            ids = checked_orders(nside, earth)
            padding_stats, fields = check_padding(ids, earth, pad, device)
            print(f"Nside {nside}: orientation and padding passed", flush=True)
            sweep = build_sweep(ids, cameras, axis)
            sweep.report["cpu_setup_seconds"] = perf_counter() - tick
            print(f"Nside {nside}: masks and camera geometry passed", flush=True)
            sweeps.append(sweep)
            if nside == (16 if 16 in args.nsides else args.nsides[0]):
                seam_fields = fields
            if args.save_grids:
                save_grids(args.out / f"grids_n{nside}.npz", sweep, axis)
            dtypes = (torch.float32, torch.float16) if device.type == "cuda" else (torch.float32,)
            volumes = [run_volumes(features, sweep, device, args.depth_chunk, dtype, args.repeats) for dtype in dtypes]
            print(f"Nside {nside}: sampled feature and GWC volumes checked", flush=True)
            output["resolutions"][str(nside)] = {"geometry": sweep.report, "padding": padding_stats, "volumes": volumes}
            (args.out / "metrics.json").write_text(json.dumps(output, indent=2) + "\n")
    plot_seam(args.out, seam_fields)
    plot_projections(args.out, sweeps, images)
    plot_masks(args.out, sweeps[-1], axis, sweeps)


if __name__ == "__main__":
    main()
