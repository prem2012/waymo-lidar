"""`wl index` / `wl serve`: analytics + viewer.  `wl train` / `wl eval` / `wl predict` / `wl export`: the detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="wl", description="Waymo Open Dataset v2 LiDAR tooling")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="build/update the analytics cache (incremental)")
    pi.add_argument("root", type=Path, help="folder holding training/ and validation/")
    pi.add_argument("--cache", type=Path, help="cache dir (default: <root>/../cache/<root name>)")
    pi.add_argument("--stride", type=int, default=10, help="decode every Nth frame for point stats (default 10)")
    pi.add_argument("--workers", type=int, help="parallel processes (default: half the CPUs)")
    pi.add_argument("--force", action="store_true", help="re-index segments already in the cache")

    ps = sub.add_parser("serve", help="run the web tool (indexes new segments first)")
    ps.add_argument("root", type=Path)
    ps.add_argument("--cache", type=Path)
    ps.add_argument("--port", type=int, default=8000, help="web tool port (default 8000)")
    ps.add_argument("--viewer-port", type=int, default=9090, help="Rerun web viewer port (default 9090)")
    ps.add_argument("--memory-limit", default="2GiB", help="Rerun gRPC server buffer (default 2GiB)")
    ps.add_argument("--no-index", action="store_true", help="skip indexing new segments on startup")

    pt = sub.add_parser("train", help="train the detector (TensorBoard logs under runs/<name>/tb)")
    pt.add_argument("root", type=Path)
    pt.add_argument("--name", help="run name (default: a timestamp); runs/<name> must not exist unless --resume")
    pt.add_argument("--config", type=Path, default=Path("configs/bev_centerpoint.yaml"))
    pt.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    pt.add_argument("--resume", action="store_true", help="continue runs/<name> from checkpoints/last.pt")
    pt.add_argument("--overfit", metavar="SEGMENT", help="smoke test: train and validate on one training segment")
    pt.add_argument("--max-minutes", type=float, help="stop cleanly after this long (validates and saves first)")

    pe = sub.add_parser("eval", help="evaluate a checkpoint")
    pe.add_argument("root", type=Path)
    pe.add_argument("--ckpt", type=Path, required=True)
    pe.add_argument("--split", default="validation")
    pe.add_argument("--max-frames", type=int, default=0)

    # `wl predict` / `wl export` options live in their modules and are parsed there (so other commands don't import
    # torch / rerun just to build the parser); it's listed here only for `wl -h`.
    sub.add_parser("predict", add_help=False, help="view a checkpoint's predictions vs GT in Rerun (wl predict -h)")
    sub.add_parser("export", add_help=False, help="export the network to ONNX / TensorRT (wl export -h)")
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["predict"]:
        from .det.predict import main as predict_main
        predict_main(args[1:])
        return
    if args[:1] == ["export"]:
        from .det.export import main as export_main
        export_main(args[1:])
        return

    a = p.parse_args(argv)
    if a.cmd in ("train", "eval"):
        root = a.root.expanduser().resolve()
        if a.cmd == "train":
            import time

            from .det.config import load_config
            from .det.train import train
            train(root, load_config(a.config, a.set), a.name or time.strftime("%Y%m%d-%H%M%S"),
                  resume=a.resume, overfit=a.overfit, max_minutes=a.max_minutes)
        else:
            from .det.train import evaluate_checkpoint
            evaluate_checkpoint(root, a.ckpt, a.split, a.max_frames)
        return

    from .analytics.index import build_index, default_cache_dir

    cache = a.cache or default_cache_dir(a.root)
    if a.cmd == "index":
        build_index(a.root, cache, stride=a.stride, workers=a.workers, force=a.force)
    elif a.cmd == "serve":
        if not a.no_index:
            build_index(a.root, cache)
        import uvicorn

        from .web.app import create_app

        app = create_app(a.root.expanduser().resolve(), cache, web_port=a.viewer_port, memory_limit=a.memory_limit)
        print(f"\n  Analytics: http://localhost:{a.port}/analytics\n  Viewer:    http://localhost:{a.port}/viewer\n")
        uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
