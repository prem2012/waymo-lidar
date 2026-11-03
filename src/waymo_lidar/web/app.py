"""Local web tool: /analytics (charts from the index cache) and /viewer (Rerun, streamed from Parquet)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse

from ..analytics.stats import load_stats
from ..data.dataset import WaymoDataset
from ..viz.stream import Streamer

STATIC = Path(__file__).parent / "static"
SECTIONS = ("overview", "labels", "spatial", "pointcloud", "scenes")


def create_app(root: Path, cache: Path, web_port: int = 9090, memory_limit: str = "2GiB") -> FastAPI:
    ds = WaymoDataset(root)
    streamer = Streamer(ds, web_port=web_port, memory_limit=memory_limit)
    app = FastAPI(title="Waymo LiDAR tool")

    def stats():
        try:
            return load_stats(cache)
        except FileNotFoundError as e:
            raise HTTPException(503, str(e)) from e

    @app.get("/")
    def index():
        return RedirectResponse("/analytics")

    @app.get("/analytics")
    def analytics_page():
        return FileResponse(STATIC / "analytics.html")

    @app.get("/viewer")
    def viewer_page():
        return FileResponse(STATIC / "viewer.html")

    @app.get("/static/{name}")
    def static(name: str):
        path = (STATIC / name).resolve()
        if path.parent != STATIC.resolve() or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/api/splits")
    def splits():
        return stats().splits()

    @app.get("/api/stats/{section}")
    def section(section: str, split: str = Query("all")):
        if section not in SECTIONS:
            raise HTTPException(404, f"unknown section {section}")
        s = stats()
        if split != "all" and split not in s.splits():
            raise HTTPException(400, f"unknown split {split}")
        return getattr(s, section)(split)

    @app.get("/api/segments")
    def segments():
        # Every segment on disk, enriched with index info where available.
        try:
            info = {(r["split"], r["segment"]): r for r in load_stats(cache).segment_list()}
        except FileNotFoundError:
            info = {}
        return [info.get((s, n), {"split": s, "segment": n}) for s, n in ds.all_segments()]

    @app.post("/api/view")
    def view(split: str, segment: str, returns: str = Query("1", pattern="^(1|1,2)$")):
        try:
            return streamer.open(split, segment, tuple(int(r) for r in returns.split(",")))
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    @app.get("/api/view/status")
    def view_status():
        return streamer.status.public()

    @app.on_event("shutdown")
    def shutdown():
        streamer.close()

    return app
