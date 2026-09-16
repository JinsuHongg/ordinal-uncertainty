# Confirmatory Global Trade-off Analysis

**Verdict: B — CONFIRMATORY GLOBAL TRADE-OFF ANALYSIS COMPLETE; CLEAR TRADE-OFF.**

The deterministic analysis reads only saved A/C arrays for all 16 confirmatory
runs under `outputs/mechanism_replication/ac/{retina,solar}/{ce,rps}/seed_{1..4}`.
Retina uses the same 1,080-row training-only OOF population (66 endpoint rows);
Solar uses the 28,006-row archived readout (921 endpoint rows). Manifest
identity, sample-ID uniqueness, counts, endpoint support, and finite arrays are
asserted. Frozen exact-L1 endpoint deltas reproduce H1.

| Setting | Δ endpoint MAE | Δ global MAE | Δ macro MAE |
| --- | ---: | ---: | ---: |
| Retina CE | -.6553 | +.0153 | -.1442 |
| Retina RPS | -.4280 | +.0558 | -.0520 |
| Solar CE | -.9229 | +.0889 | -.1270 |
| Solar RPS | -1.0309 | +.0618 | -.1699 |

Endpoint MAE improves in every frozen seed, but global MAE worsens on average
in all settings while macro MAE improves in all settings. Severe error decreases
slightly for Retina and increases for Solar. Endpoint mass increases for every
true class, most strongly for the adjacent and endpoint classes. C therefore
has a measurable redistribution cost and is not a cost-free global improvement;
the class-gradient and macro improvement do not reduce the effect to a uniform
upward-shift artifact. The analysis does not change H1/H2a or select a head.
Per-seed, per-class, routing, and mass tables are under
`outputs/mechanism_replication/analysis/global_tradeoff/`.
