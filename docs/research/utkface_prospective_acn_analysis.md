# UTKFace prospective A/C/N analysis

## Status

All eight V100-only feature-head units completed. A is the source head; C and N train only `DirectionOnlyLinear.direction` from the same A initialization with fixed row norms and biases. C uses balanced replacement batches; N uses one shuffled natural empirical pass per epoch. Both use CE, AdamW lr `.001`, zero direction weight decay, batch 64, and 100 terminal epochs. No backbone was retrained, feature archive re-exported, or post-outcome rerun performed.

All source feature archives were V100-produced and replay-validated. Every fitted C/N head preserved norms and biases within `1e-6`; only direction was trainable. Test evaluation uses exact discrete L1 decisions and class-4 support is 67.

## CE results

| Seed | A endpoint MAE | N endpoint MAE | C endpoint MAE | N-A | C-A | N-C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.447761 | 0.761194 | 0.432836 | 0.313433 | -0.014925 | 0.328358 |
| 2 | 0.537313 | 0.567164 | 0.268657 | 0.029851 | -0.268657 | 0.298507 |
| 3 | 0.641791 | 0.791045 | 0.477612 | 0.149254 | -0.164179 | 0.313433 |
| 4 | 1.328358 | 0.895522 | 0.492537 | -0.432836 | -0.835821 | 0.402985 |

C improves A in 4/4; N improves A in 1/4; C is stronger than N in 4/4.
C-A mean -0.320896, SD 0.358727, descriptive 95% CI [-0.8917099486228106, 0.24991890384669113]. N-A mean 0.014925, SD 0.320345, descriptive 95% CI [-0.494815760782853, 0.5246665070515096].
Mean endpoint A/N/C: {'A': 0.7388059701492538, 'N': 0.753731343283582, 'C': 0.417910447761194}; global A/N/C: {'A': 0.2650780261493041, 'N': 0.25516659637283845, 'C': 0.2761493040911008}; macro A/N/C: {'A': 0.42360313684992223, 'N': 0.41164122178426515, 'C': 0.33710070625150146}; severe A/N/C: {'A': 0.01560522986081822, 'N': 0.012442007591733445, 'C': 0.016975959510754954}.

## RPS results

| Seed | A endpoint MAE | N endpoint MAE | C endpoint MAE | N-A | C-A | N-C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.089552 | 0.761194 | 0.432836 | -0.328358 | -0.656716 | 0.328358 |
| 2 | 0.522388 | 0.776119 | 0.358209 | 0.253731 | -0.164179 | 0.417910 |
| 3 | 0.432836 | 0.716418 | 0.462687 | 0.283582 | 0.029851 | 0.253731 |
| 4 | 0.492537 | 0.597015 | 0.268657 | 0.104478 | -0.223881 | 0.328358 |

C improves A in 3/4; N improves A in 1/4; C is stronger than N in 4/4.
C-A mean -0.253731, SD 0.289670, descriptive 95% CI [-0.7146612429581096, 0.20719855639094548]. N-A mean 0.078358, SD 0.282237, descriptive 95% CI [-0.3707437203733252, 0.5274601382837729].
Mean endpoint A/N/C: {'A': 0.6343283582089552, 'N': 0.7126865671641791, 'C': 0.3805970149253731}; global A/N/C: {'A': 0.26992830029523407, 'N': 0.24799662589624633, 'C': 0.276043863348798}; macro A/N/C: {'A': 0.4343493732068971, 'N': 0.4010706866936231, 'C': 0.3229333175002562}; severe A/N/C: {'A': 0.01665963728384648, 'N': 0.011914803880219316, 'C': 0.015288907633909743}.

## Interpretation and limitations

CE shows a consistent C response (4/4) and an inconsistent N response (1/4). RPS shows a mixed/partial C response (3/4) and an inconsistent N response (1/4). C is stronger than N in all eight matched UTKFace units. This is supporting, setting-specific evidence based on four seeds and 67 class-4 test examples per unit; it does not establish universal causality or deployment optimality. No H2a, support-severity analysis, extra ablation, or manuscript update was performed.

Machine-readable outputs: `outputs/utkface/prospective_replication/analysis/analysis_manifest.json`, `per_seed_metrics.csv`, objective summaries, and `redistribution.csv`.
