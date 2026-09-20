# Solar CE historical execution-recipe reconstruction and numerical reproduction audit

## 1. Executive summary

**Verdict: C — NO CANDIDATE MATCHED FROZEN IDENTITY TOLERANCE.**

The archived Solar CE A readout is recoverably associated with a V100 execution
path and an evaluation extraction batch size of 128.  A bounded seed-2,
64-row reproduction using that path passed the pre-existing frozen gate.  The
required full seed-2 readout validation, however, preserved IDs, labels, mode,
and exact-L1 decisions but had a maximum probability difference of
`2.3392936e-06`, above the frozen `2e-06` tolerance.  No feature archive was
promoted, no N head was fit, and CE Phase B remains blocked.

This audit establishes an evidence-backed historical recipe, but it does not
establish a full numerical reproduction of the archived CE arrays.

## 2. Current numerical issue

Prior CE exports agreed in data identity and decision outputs but differed by
small floating-point amounts.  The seed-2 V100 localization spike found that
the first observable divergence was at the penultimate 512-D feature: normalized
inputs agreed exactly across batch sizes while penultimate features and logits
varied with batch structure.  This motivated recovery of the historical
evaluation recipe rather than tolerance relaxation.

## 3. Historical evidence sources

- Historical CE A/C task: SLURM array `4399479`, task `1` / seed 2; its saved
  stdout records `Tesla V100-SXM2-32GB`, node `acidsgcn011`,
  `torch 2.6.0+cu124`, and CUDA 12.4.
- Historical launcher: `scripts/slurm_solar_ac_mechanism_replication.sbatch`.
  It records `qGPU24`, `--gres=gpu:1`, eight CPU cores, 64 GiB, the `ocqr`
  environment, and the command arguments `--extract-batch 128 --workers 4
  --device cuda:0`.
- Historical runner: `scripts/run_ac_mechanism_replication.py`; it constructs
  sequential Solar evaluation loaders and captures features under `eval()` and
  `torch.no_grad()`.
- Current seed-2 candidate and full-readout job records: `4437393` and
  `4437794`, respectively.  Both were allocated a typed V100 on `qGPU24`.

The source revision `1d80e095d7a552f0a89369fa708d67c1463511f7` is strongly
supported by the saved launcher/log provenance, but was recorded after the
historical run; it is therefore not treated as a verified historical checkout.

## 4. Recovered environment

| Field | Historical | Current full seed-2 | Evidence | Relevant? |
| --- | --- | --- | --- | --- |
| GPU | Tesla V100-SXM2-32GB | Tesla V100-SXM2-32GB | verified runtime records | yes |
| node | `acidsgcn011` | `acidsgcn007` | verified scheduler/runtime records | yes |
| partition/account | `qGPU24` / `csc344r253` | `qGPU24` / `csc344r253` | verified accounting | possibly |
| PyTorch / CUDA | 2.6.0+cu124 / 12.4 | 2.6.0+cu124 / 12.4 | verified historical log; runtime query | yes |
| cuDNN | unknown | 90100 | historical metadata unavailable | yes |
| driver | unknown | 580.159.04 | historical metadata unavailable | possibly |
| torchvision / Python | unknown | 0.21.0+cu124 / 3.11.15 | historical metadata unavailable | possibly |
| evaluation batch / workers | 128 / 4 | 128 / 4 | verified launcher command | yes |
| pin memory / sequential loader | true / no shuffle | true / no shuffle | verified runner | yes |
| persistent workers / prefetch / drop-last | false / 2 / false | false / 2 / false | runner defaults, strongly supported | possibly |
| `eval()` / execution context | `eval()` / `no_grad()` | same | verified runner/current diagnostic | yes |
| cuDNN benchmark/deterministic | unknown | false / false | no historical backend record | yes |
| deterministic algorithms | unknown | false | no historical backend record | yes |
| TF32 / matmul precision | unknown | matmul false; cuDNN true; highest | no historical backend record | yes |
| autocast | no explicit use | false | runner/current diagnostic | yes |
| script revision | runner strongly supported; exact checkout unknown | current compatible runner | log/source archaeology | yes |

## 5. Recovered execution parameters

The recovered seed-2 candidate intentionally uses the original evaluation
shape: `seed_everything(2)`, batch 128, four workers, pin memory, sequential
sampling, no drop-last, `model.eval()`, and `torch.no_grad()`.  It uses the
verified seed-2 checkpoint SHA256
`c746ee847afd2305001195f0da47b77638f5549a11893390735bc8dcdf65fd01` and the
verified normalization artifact SHA256
`1178b05f9fafad22eec58608236ee5ba035a2f4629bfc17c10281797da8ed02b`.

## 6. Historical/current diff table

The material recovered difference from the preceding 64-row diagnostic is
batch structure: the original archived evaluation used a 128-row extraction
batch, not the earlier 32/64 diagnostic batches.  The exact historical CUDA,
cuDNN, driver, torchvision, Python, backend-flag, and source-checkout values
remain unavailable, so their role cannot be excluded.

## 7. Seed-2 candidate recipes

Two evidence-guided candidates were run on the same fixed 64 archived IDs in
the first 128-row historical batch on V100 job `4437393`:

1. `historical_recorded_defaults`: recovered batch/loader/no-grad recipe with
   framework defaults.
2. `historical_recorded_strict_fp32_control`: the same recipe with TF32
   explicitly disabled and highest float32 matmul precision.

No combinatorial tuning or full-dataset selection occurred.

## 8. 64-row results

Both candidates produced the same result: maximum/mean logit error
`2.3841858e-06` / `6.3087791e-07`, maximum probability error
`1.9760548e-07`, zero mode differences, and zero exact-L1 differences.  Both
therefore passed the existing subset identity gate.  Repeated execution had
zero observed feature and logit drift in this fixed candidate process.

## 9. Full seed-2 validation

The historical-recorded-default recipe was then evaluated over the complete
archived seed-2 readout under job `4437794` (`qGPU24`, typed V100,
`acidsgcn007`, completed successfully in 37:32).  It did not capture or
promote train/validation/evaluation feature archives.

| Quantity | Result |
| --- | ---: |
| sample IDs / labels | exact / exact |
| maximum / mean logit error | `1.0490417e-05` / `9.2857411e-07` |
| maximum / mean probability error | `2.3392936e-06` / `6.2059481e-08` |
| mode differences | 0 |
| exact-L1 differences | 0 |
| frozen result | **MISMATCH**: probability maximum exceeds `2e-06` |

## 10. Seeds 1/3/4 subset validation

Not run.  The protocol permits those checks only after a passing full seed-2
gate; seed 2 failed its full probability gate.

## 11. Source revision differences

No source-level discrepancy was found between the recovered runner behavior
and the candidate path for checkpoint loading, preprocessing, feature hook,
loader construction, dtype, `eval()`, or `no_grad()`.  The historical git
checkout itself is not recoverable from retained provenance, so this is not an
exact source-revision proof.

## 12. Earliest remaining divergence

For the earlier fixed 64-row V100 diagnostic, inputs matched exactly and the
first measured divergence was the penultimate feature stage.  On the recovered
128-row path the fixed subset passes, but the full readout exposes a residual
probability discrepancy.  Existing artifacts do not retain archived features,
so the earliest full-population divergence cannot be localized beyond the
model-output path without additional authorized provenance work.

## 13. Historical recipe verdict

The execution recipe is partially reconstructed and sufficient for a passing
fixed subset, but it is **not** a historically compatible full-readout recipe
under the frozen gate.  The remaining mismatch should not be attributed solely
to GPU model, TF32, or batching without direct historical environment evidence.

## 14. Full-regeneration readiness

Not ready.  A full train/validation/evaluation CE feature regeneration would
not be valid to promote because seed 2 fails the required full A-output
identity comparison.

## 15. Remaining uncertainty

Unavailable historical metadata includes the exact cuDNN, driver, torchvision,
Python, backend-flag, and source-checkout state.  A retained historical
environment/container or exact runtime metadata would be required before a
new bounded reproduction candidate can be justified.

## 16. Recommended next step

Keep Solar CE N fitting blocked.  Do not rerun CE features or change frozen
tolerances.  If further work is separately authorized, first recover an exact
historical environment/container or backend metadata; then conduct one bounded
seed-2 reproduction diagnostic before considering any full regeneration.

## Final bounded seed-2 historical replay

The single authorized full-feature replay ran as job `4441691` on `acidsgcn007` under account `csc344r253` and partition `qGPU24`. Scheduler accounting and direct runtime output both recorded a `Tesla V100-SXM2-32GB` (driver `580.159.04`), with PyTorch `2.6.0+cu124`, torchvision `0.21.0+cu124`, CUDA `12.4`, and cuDNN `90100`. It used batch size 128, four workers, a pinned sequential loader, no drop-last, `eval()`, `torch.no_grad()`, no autocast, and float32 model and inputs. Current runtime values were cuDNN benchmark/deterministic false, deterministic algorithms false, matmul TF32 false, cuDNN TF32 true, and float32 matmul precision `highest`; the historical backend-flag values remain unknown.

The original CE seed-2 checkpoint (SHA256 `c746ee847afd2305001195f0da47b77638f5549a11893390735bc8dcdf65fd01`) and normalization artifact (SHA256 `1178b05f9fafad22eec58608236ee5ba035a2f4629bfc17c10281797da8ed02b`) were used with strict state loading. The temporary archive has the expected 45,047/2,431/28,006 train/validation/evaluation rows; its SHA256 values are `7ccf6383f7a0ff88d8992226d4d42fc3de0101440d06460d31a5d4c3629ffef8`, `d2aaaf8ebca81c8dfaaa8c11aedde2d9ac7f221b0898ac2848cd3e6103480384`, and `ddb4af3817340a4d26b17f23aabb58c9d40cb6a84218fbe41b1f1379bbd34d5f`.

The full frozen A gate retained exact sample IDs, labels, mode decisions, and exact-L1 decisions. Maximum/mean logit error was `1.0490417e-05` / `9.2778343e-07`, within the `2e-05` frozen logit limit. Maximum/mean probability error was `2.8676560e-06` / `6.2734405e-08`, exceeding the frozen `2e-06` probability limit. The result is therefore **MISMATCH**. The temporary tree `outputs/mechanism_replication/features/solar/ce/seed_2_historical_replay_tmp/` is retained, was not promoted, and must not be used for CE Phase B. No further retry is authorized; Solar CE remains 3/4 validated and CE N fitting remains blocked.
