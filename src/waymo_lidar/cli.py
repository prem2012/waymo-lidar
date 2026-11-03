"""`wl index <root>` builds the analytics cache; `wl serve <root>` runs the web tool."""

from __future__ import annotations

import argparse
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

    a = p.parse_args(argv)
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
