# Detector loss

Part of the [Phase 2 detector](README.md). Code: `src/waymo_lidar/det/loss.py`; tests:
`tests/test_det_loss.py`; settings: the `loss` section of the [config](config.md).

CenterPoint's loss: a **Gaussian focal loss** on the class heatmaps, plus a masked **L1 or smooth L1**
regression at object centres. The [model's outputs](model.md#output-contract) and the
[targets](targets.md) share a layout, so the loss is a direct comparison with no reshuffling.

## Usage

```python
from waymo_lidar.det.loss import CenterPointLoss

loss_fn = CenterPointLoss(cfg).cuda()
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(bev)
res = loss_fn(out, batch)          # batch from FrameStream(transform=TargetBuilder(cfg)) + collate
res.total.backward()
res.scalars()                      # {"loss/total": 10.68, "loss/heatmap": 9.40, ...}: syncs, so only when logging
```

The loss always computes in **float32**: bf16 logits from autocast are cast before any log or power.

## Heatmap: Gaussian focal loss

With p = sigmoid(logit), gt the target heatmap, and N the number of cells in the **whole batch** where
gt = 1 (one per object, since the Gaussians peak at exactly 1.0):

```
L_pos = −Σ_{gt=1}  (1 − p)^α · log p
L_neg = −Σ_{gt<1}  (1 − gt)^β · p^α · log(1 − p)
L_hm  = (L_pos + L_neg) / max(N, 1)             α = loss.focal_alpha = 2,  β = loss.focal_beta = 4
```

- **(1 − p)^α and p^α** down-weight cells the model already gets right. Almost all 3 × 376² cells are
  easy negatives, so without this they would swamp the loss.
- **(1 − gt)^β** softens the penalty near object centres: a cell just next to a centre (gt ≈ 0.8)
  costs only 0.2⁴ = 0.0016 of a far-away false positive.
- **Normalised by objects, not cells.** A crowded frame and an empty one give comparable magnitudes,
  and a batch without any objects divides by 1.
- **Per-class breakdown:** each class's part of the same sum, divided by the same N. The logged
  `loss/heatmap/<class>` values add up exactly to `loss/heatmap`.

**One difference from CenterPoint's code:** CenterPoint clamps p to [1e-4, 1 − 1e-4] before taking
logs. Here, log p and log(1 − p) come from `logsigmoid(±logit)`, which is exact and stable, so no
clamp is needed. Wherever the clamp doesn't kick in, the value is identical; the tests check it
against a re-implementation of CenterPoint's formula (rtol 1e-5). The difference only shows when the
model is extremely confident, where CenterPoint's clamp cuts the gradient off.

## Regression: masked L1 or smooth L1

```
pred = reg gathered at target_ind               (B, 8, 376²) → (B, K, 8)
e    = pred − target_reg
ℓ(e) = |e|                                      loss.reg_loss: l1          (default, as in CenterPoint)
ℓ(e) = 0.5 · e² / β   if |e| < β                loss.reg_loss: smooth_l1
       |e| − 0.5 · β  otherwise                 β = loss.smooth_l1_beta (default 1/9)
L_reg[f] = Σ mask · ℓ(e)[f] / (Σ mask + 1e-4)   per field f
L_reg    = Σ_f code_weights[f] · L_reg[f]
```

- **Fields** are `REG_FIELDS`: dx, dy (cell offset), z, log l, log w, log h, sin, cos.
- **Padding:** slots with `target_mask` = 0 contribute nothing, even though their `target_ind` points
  at cell 0.
- **Reduction:** the same as CenterPoint's `RegLoss`: summed over objects and divided by the number
  of objects, per field.
- **Smooth L1** is quadratic for small errors, so its gradient shrinks as a prediction gets close
  (L1's stays at ±1), and linear for large ones. β = 1/9 is the classic Faster R-CNN value. Our
  regression values are of order 0–1 (cell offsets, sin/cos, log-size errors), so β = 1/9 covers the
  fine-tuning regime.

## Total

```
L = L_hm + reg_weight · L_reg          reg_weight = loss.reg_weight = 0.25
```

## Logged values

`res.logs` holds detached scalars still on the GPU. `res.scalars()` turns them into floats with a
single transfer, so only call it on steps that log. These names go to Weights & Biases unchanged:

| Name | Value |
|---|---|
| `loss/total` | L |
| `loss/heatmap` | L_hm |
| `loss/heatmap/vehicle`, `…/pedestrian`, `…/cyclist` | per-class parts of L_hm (sum to it) |
| `loss/reg` | L_reg (code-weighted, before `reg_weight`) |
| `loss/reg/dx` … `loss/reg/cos` | per-field L_reg[f], before `code_weights` |
| `targets/num_objects` | objects in the batch |

## Sanity checks

**Initial loss.** At initialisation every logit is about `heatmap_bias` (p ≈ 0.1), so the expected
heatmap loss follows from the targets alone:

```
L_hm ≈ [ Σ_{gt<1} (1 − gt)⁴ · 0.1² · −log 0.9  +  N · 0.9² · −log 0.1 ] / N
```

On real frames this is ~9–10: the many negatives add ~7–8, and each object adds 1.87. A test checks
`focal_loss` on constant logits against this formula exactly (rel 1e-4). The real initialised model
measured 9.40, so the head init works as intended.

**Overfit.** One fixed batch of 2 real frames (129 objects), `simple` backbone, full size, bf16,
AdamW lr 1e-3, 150 steps (~6 s on the 4090):

| Step | L1: total | heatmap | reg | Smooth L1: total | heatmap | reg |
|---|---|---|---|---|---|---|
| 0 | 10.68 | 9.40 | 5.11 | 10.57 | 9.40 | 4.69 |
| 10 | 4.84 | 4.41 | 1.74 | 4.76 | 4.40 | 1.47 |
| 25 | 1.52 | 1.35 | 0.68 | 1.38 | 1.32 | 0.24 |
| 50 | 0.42 | 0.35 | 0.29 | 0.35 | 0.34 | 0.04 |
| 100 | 0.16 | 0.12 | 0.19 | 0.12 | 0.11 | 0.02 |
| 150 | **0.10** | 0.07 | 0.14 | **0.07** | 0.06 | 0.005 |

Both drive the loss down by more than 99%, with the heatmap and regression terms falling together.
This is the first time the model, targets and loss have trained end to end. Smooth L1's regression
values are smaller partly by construction (−β/2 per element, and quadratic near 0), so the two `reg`
columns aren't directly comparable. Evaluation will compare them properly.

## Tests

`uv run pytest tests/test_det_loss.py` (15 tests, ~28 s):

- **Focal loss:** matches CenterPoint's formula; perfect predictions give 0; gradient signs are
  right; a batch without objects is finite; bf16 input equals fp32 on the rounded logits.
- **Regression:**
  - Hand-placed predictions at known indices give the hand-computed per-field L1.
  - Garbage at padded slots changes nothing.
  - Smooth L1 matches `F.smooth_l1_loss`, with the L1 − β/2 and 0.5 e²/β limits; the default β is
    1/9.
  - Exact predictions give 0.
- **CenterPointLoss:** log names and breakdown sums; `code_weights` and `reg_weight` scale the right
  terms; `reg_loss: smooth_l1` is picked up from the config.
- **Real data:** the initial loss equals the analytic estimate; both regression losses overfit one
  batch by more than 90% (GPU).
