# Phase 3.18B — Solar CE Representation Robustness

## Decision

\[
\boxed{\text{A — TWO-DOMAIN CROSS-OBJECTIVE DIRECTION ROBUSTNESS CONFIRMED}}
\]

On the predeclared archived Solar readout, direction-only adaptation of the frozen CE representation moves rare X localization decisively outward. The same qualitative A→C effect is now observed for both CE- and RPS-trained frozen representations in both RetinaMNIST and Solar. This is limited to the tested frozen protocols; it is neither objective-independent nor universal.

## Frozen protocol and integrity

The existing Phase 3.9 CE cache was valid and was reused; no feature extraction or backbone training occurred.

| Item | Frozen choice |
| --- | --- |
| CE checkpoint | `outputs/solar/phase3_8_shrinkage_confirmation/ce/seed_0/selected_checkpoint.pt` |
| CE feature cache | `outputs/solar/phase3_9_mechanism_audit/features/ce/` |
| Feature | input to `model.fc`, 512 dimensions |
| B/C fitting split | aligned training features only, 45,047 examples |
| Validation role | provenance/integrity only; no selection or early stopping |
| Archived readout | 28,006 aligned examples, including 921 X |
| B/C objective | identical replacement class-balanced batches / CE |
| Optimizer/schedule | AdamW, lr `.001`, batch `64`, 100 epochs, seed `0` |
| B/C constraints | B full linear, weight decay `1e-4`; C fixed-A norms/biases, direction-only, decay `0` |

C uses \(w_k^C=\|w_k^A\|_2v_k^C/\|v_k^C\|_2\) and \(b_k^C=b_k^A\). B and C reset the same sampler generator and therefore use matching balanced batches. No controlled scale, ROP, bias-only condition, representation update, new loss, sampling rule, checkpoint, seed, or dataset was used.

The only job, `4368045`, completed on `qGPU24` / `acidsgcn011` in `27:19` (account `csc344r253`). It used `ocqr`, PyTorch `2.6.0+cu124`, CUDA `12.4`, and a Tesla V100-SXM2-32GB. CUDA was available; there was no CPU fallback. The submitted script and logs are `scripts/slurm_phase3_18b_solar_ce_direction.sbatch` and `logs/phase3_18b/solar_ce_direction_4368045.{out,err}`.

- Counts train/validation/test are `45,047/2,431/28,006`; X supports are `562/24/921`; all cached values are finite.
- IDs are unique and all pairwise split overlaps are zero.
- Cached CE test-logit replay error is `7.63e-6`; C initialization replay error is `1.91e-6`.
- C maximum row-norm error is `5.96e-8`; maximum fixed-bias error is `0`.

## Head diagnostics

| Class | A norm | B norm | B Δnorm | B Δbias | A→B cosine | A→C cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | .591 | 8.361 | +7.769 | -.067 | .108 | .701 |
| 1 | .598 | 6.567 | +5.970 | -.082 | .174 | .741 |
| 2 | .581 | 6.910 | +6.329 | -.174 | .206 | .671 |
| 3 | .605 | 9.808 | +9.203 | -.018 | .234 | .612 |
| 4 (X) | .612 | 17.942 | +17.330 | +.459 | .162 | .512 |

B changes direction, scale, and bias. C has material direction rotation while preserving the original CE scale and biases exactly.

## Primary X-class results

All decisions are exact discrete L1 Bayes decisions.

| Condition | X routing `0/1/2/3/4` | Exact / 921 | X MAE | Severe | Mean / median p(X) | Mean / median p(M) | Mean / median p(M)+p(X) | Predictive mean | Shrinkage | Mean / median L1 risk |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A-CE | `0/23/60/838/0` | 0 | 1.115 | 9.0% | .040 / .035 | .812 / .907 | .852 / .956 | 2.864 | 1.136 | .163 / .094 |
| B-CE | `1/19/363/209/329` | 329 | 1.081 | 41.6% | .362 / .211 | .228 / .116 | .590 / .668 | 2.919 | 1.081 | .296 / .250 |
| C-CE | `0/23/132/207/559` | 559 | .586 | 16.8% | .590 / .636 | .194 / .161 | .784 / .911 | 3.344 | .656 | .381 / .394 |

C moves X outward relative to A: exact recovery `0→559`, MAE `1.115→.586`, mean p(X) `.040→.590`, predictive mean `2.864→3.344`, and shrinkage `1.136→.656`. C also exceeds B on the primary X outcomes, although that was not a decision criterion.

For the predeclared adjacent-interior class M (class 3), the X margin is:

| Condition | Mean `zX-zM` | Median | Positive fraction |
| --- | ---: | ---: | ---: |
| A-CE | -3.109 | -3.114 | 0.0% |
| B-CE | -.116 | -.068 | 49.2% |
| C-CE | 1.190 | 1.144 | 76.2% |

## Representation/head decomposition

Using the saved Phase 3.9 raw train-centroid diagnostic, 724 X examples are X-like and 197 are representation-inward.

| Subgroup | Support | A routing | C routing | Exact A→C | MAE A→C |
| --- | ---: | --- | --- | ---: | ---: |
| X-like | 724 | `0/0/9/715/0` | `0/0/49/116/559` | 0→559 | 1.012→.296 |
| Representation-inward | 197 | `0/23/51/123/0` | `0/23/83/91/0` | 0→0 | 1.492→1.655 |

All exact recoveries are in the X-like subgroup. This is descriptive evidence, not causal proof of recoverability or irrecoverability.

## Collateral effects

For class 0, C changes MAE `.104→.114`, severe prevalence `2.23%→2.16%`, mean p0 `.871→.848`, lower-neighborhood mass `.961→.959`, and predictive mean `.179→.213`. C therefore carries a modest lower-endpoint MAE cost.

Global L1 accuracy/MAE/QWK/severe are `.577/.456/.814/2.95%` (A) versus `.493/.595/.774/8.43%` (C). NLL/Brier/RPS/ECE are `1.064/.606/.084/.203` versus `1.211/.689/.104/.203`; Spearman/AUROC/AUPRC/selective MAE are `.254/.689/.049/.323` versus `.126/.519/.081/.505`. Rare-end localization does not imply a global predictive or UQ gain.

## Saved RPS comparison and synthesis

| Representation | A X MAE | C X MAE | Exact A→C | Shrinkage A→C | p(X) A→C | Predictive mean A→C | Interpretation |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| RPS (Phase 3.15) | 1.197 | .675 | 0→496 | 1.228→.773 | .033→.513 | 2.772→3.227 | saved outward response |
| CE (this phase) | 1.115 | .586 | 0→559 | 1.136→.656 | .040→.590 | 2.864→3.344 | confirmed outward response |

The qualitative A→C sign agrees for Solar CE/RPS. It is not a CE-versus-RPS superiority claim.

| Dataset | Representation | Direction-only A→C rare-end effect |
| --- | --- | --- |
| RetinaMNIST | RPS | saved outward C4 response |
| RetinaMNIST | CE | confirmed in Phase 3.18A |
| Solar | RPS | saved outward X response |
| Solar | CE | confirmed in this phase |

## Questions answered and scope

1. **B head-actionability?** Yes: B recovers 329 exact X decisions, with large severe/global/probability trade-offs.
2. **C-CE versus A-CE?** Clear improvement in exact routing, MAE, p(X), predictive mean, shrinkage, and X/M margin.
3. **Solar CE/RPS sign consistency?** Yes.
4. **CE/RPS across both domains?** Yes, within the tested frozen protocols.
5. **Magnitude?** Both are large; raw magnitudes are not an objective ranking.
6. **Exact recoveries X-like?** Yes, all 559 C recoveries are X-like.
7. **Collateral effects?** C modestly worsens class-0 MAE and worsens global L1/probability/risk metrics despite lower class-0 severe prevalence.
8. **Paper limitation?** It can be narrowed: direct direction evidence covers the tested frozen CE and RPS representations in RetinaMNIST and Solar.

No method, final evaluation, tuning, additional seed, UTKFace CE, scale/bias experiment, ROP, or representation retraining is authorized by this result.

Artifacts: `outputs/solar/phase3_18b_ce_direction_robustness/`.
