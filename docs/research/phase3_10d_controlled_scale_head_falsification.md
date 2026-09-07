# Phase 3.10D — Controlled-Scale Head Adaptation Falsification

## Question and fixed protocol

Phase 3.10C showed that direction-only adaptation retains partial rare-class
benefit but loses most exact recovery. This causal test asks whether a
predeclared intermediate fixed scale can preserve substantial balanced-head
recovery while reducing its class-0/global cost.

For each fold/class, D uses
\(s_k(\alpha)=(1-\alpha)\|w_k^A\|+\alpha\|w_k^B\|\), fixed original bias,
and trainable direction only. The grid was frozen as `.25/.50/.75`; no other
alpha, ROP, logit adjustment, class weights, loss, backbone, validation/test
data, seed, or dataset was used. It reused the Phase 3.10A training-only OOF
folds, frozen `(1080,512)` RPS features, canonical RPS head, and restored B
heads. AdamW used LR `.001`, batch 64, 100 epochs, and zero direction weight
decay because normalization removes radial decay from the effective classifier.
All final norm errors were within floating-point tolerance (`<=1e-6`).

## Pooled OOF comparison

| Condition | L1 acc / MAE / QWK / severe % | C4 MAE / exact / routing 0/1/2/3/4 | C0 MAE / severe % |
|---|---|---|---|
| A original | .518 / .696 / .604 / 19.1 | 1.697 / 0 / 1/1/41/23/0 | .570 / 25.1 |
| B balanced | .507 / .744 / .612 / 19.4 | 1.121 / 11 / 0/2/15/38/11 | .733 / 24.3 |
| C direction-only | .503 / .721 / .621 / 18.0 | 1.348 / 2 / 0/3/19/42/2 | .691 / 22.0 |
| D α=.25 | .504 / .725 / .621 / 18.4 | 1.318 / 4 / 0/3/19/40/4 | .698 / 22.4 |
| D α=.50 | .503 / .731 / .619 / 18.9 | 1.258 / 8 / 0/3/19/36/8 | .698 / 22.4 |
| D α=.75 | .502 / .740 / .608 / 19.2 | 1.288 / 7 / 0/3/20/36/7 | .710 / 22.4 |

The α=.50 setting retains 8 exact C4 decisions versus B's 11 while improving
C0 MAE by `.035` and global MAE by `.013` over B. It improves exact recovery
by 6 cases versus C at only `.006` C0-MAE cost. α=.75 retains 7 exact C4
decisions and still improves C0 MAE over B. Thus the useful region is not an
isolated point.

Probability quality at α=.25/.50/.75 is NLL `1.221/1.221/1.219`, Brier
`.599/.599/.597`, RPS `.133/.133/.133`, ECE `.061/.060/.059`; all improve B's
`1.236/.603/.136/.072`. Risk metrics are mixed but not catastrophically worse:
Spearman `.449/.460/.459` versus B `.471`, AUROC `.710/.704/.701` versus B
`.700`, AUPRC `.342/.342/.346` versus B `.354`, and selective MAE
`.426/.427/.430` versus B `.427`.

## Margins and retention

At α=.50, mean margins are `z4-z3=-.075`, `z4-z2=.223`, and `z0-z1=1.150`.
They lie between the direction-only and balanced-head mechanisms, while exact
C4 recovery is disproportionately closer to B than C. Retention relative to
B is: α=.25 `4/11` exact and `23/37` inward-improved; α=.50 `6/11` and
`24/37`; α=.75 `6/11` and `23/37`. Of B's 84 damaged C0 cases, D restores
29/28/23 at α=.25/.50/.75.

The result is not merely outcome interpolation: α=.50 adds six exact recoveries
over C while holding C0 MAE nearly unchanged, and α=.50/.75 preserve a
substantial shared B-recovery subset while remaining safer than B on C0/global
MAE.

## Decision

\[
\boxed{\text{GO — CONTROLLED SCALE SUPPORTED}}
\]

The causal mechanism is supported on training-only OOF evidence. The
best-supported region is α `.50–.75`, with α `.50` the stronger recovery/safety
point. This is not a final method, does not authorize test evaluation or any
adaptive/class-specific scale variant, and does not establish cross-dataset
validity. Any next design or method-freeze decision requires separate
authorization.

Artifacts: `outputs/retinamnist/phase3_10d_controlled_scale_head/`.
