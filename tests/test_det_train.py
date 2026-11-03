"""Training-loop tests: tiny grid, 4 cached frames of one pinned segment, a few steps each."""

import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from waymo_lidar.data.dataset import WaymoDataset
from waymo_lidar.det import train as T
from waymo_lidar.det.config import load_config
from waymo_lidar.det.data import load_sample
from waymo_lidar.det.targets import TargetBuilder

ROOT = Path(os.environ.get("WAYMO_ROOT", "~/dev/data/waymo_v2")).expanduser()
SEG = "10023947602400723454_1120_000_1140_000"
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not (ROOT / "training" / "lidar_box" / f"{SEG}.parquet").is_file(), reason="stage-1 data missing"),
]
SMALL = ["grid.x_range=[-25.6, 25.6]", "grid.y_range=[-25.6, 25.6]", "train.batch_size=2", "train.log_every=1",
         "train.eval_every=1", "optim.lr=1e-3", "train.lr_schedule=constant"]


def cfg_for(tmp_path, *extra):
    return load_config(overrides=[*SMALL, f"train.runs_dir={tmp_path}", *extra])


@pytest.fixture(scope="module")
def cached():
    cfg = load_config(overrides=SMALL)
    tb = TargetBuilder(cfg)
    ts = WaymoDataset(ROOT).segment("training", SEG).timestamps
    return [tb(load_sample(ROOT, "training", SEG, int(t), cfg)) for t in ts[[0, 50, 100, 150]]]


@pytest.fixture
def four_frames(monkeypatch, cached):
    monkeypatch.setattr(T, "overfit_samples", lambda root, seg, cfg: cached)


def tags(run: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(run / "tb"), size_guidance={"scalars": 0, "images": 0, "tensors": 0})
    acc.Reload()
    return acc


def test_overfit_samples_one_pass_is_fast():
    import time
    cfg = load_config()
    t = time.perf_counter()
    samples = T.overfit_samples(ROOT, SEG, cfg)
    dt = time.perf_counter() - t
    assert len(samples) == len(WaymoDataset(ROOT).segment("training", SEG).timestamps)
    assert {"points", "heatmap", "target_reg"} <= samples[0].keys()
    assert dt < 60  # one pass (~9 s); per-frame load_sample would take ~5 min


def test_lr_schedules():
    f = lambda sch, s: T.lr_factor(sch, s, 1000, 0.4)  # noqa: E731
    assert f("onecycle", 0) == pytest.approx(0.1) and f("onecycle", 400) == pytest.approx(1.0)
    assert f("onecycle", 1000) == pytest.approx(1e-4) and f("onecycle", 5000) == pytest.approx(1e-4)  # clamped
    vals = [f("onecycle", s) for s in range(0, 1001, 10)]
    peak = int(np.argmax(vals))
    assert all(a <= b for a, b in zip(vals[:peak], vals[1:peak + 1])) and all(a >= b for a, b in zip(vals[peak:], vals[peak + 1:]))
    assert f("cosine", 0) == pytest.approx(1.0) and f("cosine", 1000) == pytest.approx(1e-4)
    assert f("constant", 0) == f("constant", 999) == 1.0


def test_short_run_writes_everything(tmp_path, four_frames):
    cfg = cfg_for(tmp_path, "train.epochs=2")
    run = T.train(ROOT, cfg, "short", overfit=SEG, log=lambda *a: None)
    assert (run / "config.yaml").is_file() and (run / "checkpoints" / "last.pt").is_file()
    assert (run / "checkpoints" / "best.pt").is_file() and (run / "eval_0.md").is_file() and (run / "eval_1.md").is_file()
    acc = tags(run)
    scalars = set(acc.Tags()["scalars"])
    for t in ("loss/total", "loss/heatmap/vehicle", "loss/reg/dx", "train/lr", "train/frames_per_s",
              "train/data_wait_frac", "val/loss/total", "eval/bev_ap@0.5/mean", "eval/dist_ap@1m/vehicle"):
        assert t in scalars, t
    assert "val/heatmap/vehicle" in acc.Tags()["images"]
    losses = [e.value for e in acc.Scalars("loss/total")]
    assert len(losses) == 4 and all(math.isfinite(v) for v in losses)  # 2 epochs x 2 steps


def test_existing_run_refused(tmp_path, four_frames):
    cfg = cfg_for(tmp_path, "train.epochs=1")
    T.train(ROOT, cfg, "dup", overfit=SEG, log=lambda *a: None)
    with pytest.raises(FileExistsError):
        T.train(ROOT, cfg, "dup", overfit=SEG, log=lambda *a: None)


def test_resume_continues_step_and_schedule(tmp_path, four_frames):
    sched = ["train.lr_schedule=cosine"]
    T.train(ROOT, cfg_for(tmp_path, "train.epochs=4", *sched), "res", overfit=SEG, max_minutes=1e-9, log=lambda *a: None)
    ck = torch.load(tmp_path / "res" / "checkpoints" / "last.pt", weights_only=False)
    assert ck["epoch"] == 0 and ck["step"] == 1  # stopped right after the first step
    lr_before = ck["scheduler"]["_last_lr"][0]
    logs = []
    T.train(ROOT, cfg_for(tmp_path, "train.epochs=4", *sched), "res", resume=True, overfit=SEG, log=logs.append)
    assert any("resumed" in m and "epoch 1, step 1" in m for m in logs)
    ck2 = torch.load(tmp_path / "res" / "checkpoints" / "last.pt", weights_only=False)
    assert ck2["epoch"] == 3 and ck2["step"] == 1 + 3 * 2  # epochs 1..3, 2 steps each
    total = 4 * 2  # steps_per_epoch = ceil(4 frames / batch 2)
    assert lr_before == pytest.approx(1e-3 * T.lr_factor("cosine", 1, total, 0.4))
    acc = tags(tmp_path / "res")
    steps = [e.step for e in acc.Scalars("train/lr")]
    assert steps == sorted(steps) and steps[-1] == 7  # no restart of the step counter


def test_nan_loss_is_skipped(tmp_path, four_frames, monkeypatch):
    orig = T.CenterPointLoss.forward
    calls = {"n": 0}

    def flaky(self, out, batch):
        res = orig(self, out, batch)
        calls["n"] += 1
        if calls["n"] == 2 and self.training:
            res.total = res.total * float("nan")
        return res

    monkeypatch.setattr(T.CenterPointLoss, "forward", flaky)
    run = T.train(ROOT, cfg_for(tmp_path, "train.epochs=2"), "nan", overfit=SEG, log=lambda *a: None)
    ck = torch.load(run / "checkpoints" / "last.pt", weights_only=False)
    assert ck["step"] == 3  # 4 batches, 1 skipped
    assert all(torch.isfinite(v).all() for v in ck["model"].values() if v.is_floating_point())
    assert tags(run).Scalars("train/skipped_steps")[-1].value == 1


def test_checkpoint_reloads_to_same_metrics(tmp_path, four_frames, cached):
    run = T.train(ROOT, cfg_for(tmp_path, "train.epochs=1"), "ck", overfit=SEG, log=lambda *a: None)
    model, cfg, meta = T.load_model(run / "checkpoints" / "last.pt")
    loader = torch.utils.data.DataLoader(T.InMemoryFrames(cached), batch_size=2, collate_fn=T.collate)
    from waymo_lidar.det.voxelize import Voxelizer
    res, _, _, n = T.validate(model, Voxelizer(cfg.grid).cuda(), T.CenterPointLoss(cfg).cuda(), loader, cfg,
                              torch.device("cuda"), T.AMP[cfg.train.precision])
    assert n == 4
    assert res.table() in (run / "eval_0.md").read_text()
