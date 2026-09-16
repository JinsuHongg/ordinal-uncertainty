# Manuscript Methods / Experimental Setup Audit

**Status:** first-draft audit complete (2026-09-15)  
**Scope:** Methods and Experimental Setup only; no results, figures, experiments, citation resolution, commit, or push.

## Outcome

The two manuscript Methods inputs now provide the main-text first draft. The draft preserves the frozen paper identity: an empirical mechanism analysis of rare upper-end inward localization, not a new classifier method or a claim that imbalance has been independently isolated as a cause.

## Required-concept completeness matrix

Status is assessed after this draft. ``PRESENT AND SUFFICIENT'' means sufficient for the main text; it does not imply that every executable hyperparameter is repeated there.

| # | Required concept | Status | Main-text location / note |
| ---: | --- | --- | --- |
| 1 | Ordinal prediction notation | PRESENT AND SUFFICIENT | Problem setup, paragraph 1 |
| 2 | Predictive probabilities \(p_k\) | PRESENT AND SUFFICIENT | Problem setup, paragraph 1 |
| 3 | Mode decision | PRESENT AND SUFFICIENT | Problem setup, paragraph 2 |
| 4 | Exact discrete L1 decision / predictive median | PRESENT AND SUFFICIENT | Problem setup, paragraph 2 |
| 5 | Predictive mean | PRESENT AND SUFFICIENT | Problem setup, paragraph 3 |
| 6 | Inward shrinkage | PRESENT AND SUFFICIENT | Problem setup, paragraph 3 |
| 7 | Rare-end class definition | PRESENT AND SUFFICIENT | Problem setup, paragraph 1 |
| 8 | Severe-error definition | PRESENT AND SUFFICIENT | Problem setup, paragraph 2 |
| 9 | Classifier-head parameterization | PRESENT AND SUFFICIENT | Problem setup, paragraphs 1 and 7 |
| 10 | Frozen representation | PRESENT AND SUFFICIENT | Problem setup, paragraph 7; experimental setup, paragraph 2 |
| 11 | A original head | PRESENT AND SUFFICIENT | Problem setup, paragraph 7 |
| 12 | B balanced full head | SUPPLEMENT-ONLY | Problem setup, paragraph 7; historical comparison only |
| 13 | C direction-only head | PRESENT AND SUFFICIENT | Problem setup, paragraph 7; experimental setup, paragraph 2 |
| 14 | D controlled-scale head | SUPPLEMENT-ONLY | Problem setup, paragraph 7; not primary A/C contrast |
| 15 | Balanced sampling | PRESENT AND SUFFICIENT | Problem setup, paragraph 7; experimental setup, paragraph 2 |
| 16 | CE objective | PRESENT AND SUFFICIENT | Problem setup, paragraph 5 |
| 17 | RPS objective | PRESENT AND SUFFICIENT | Problem setup, paragraph 5 |
| 18 | Historical seed-0 role | PRESENT AND SUFFICIENT | Experimental setup, paragraph 1 |
| 19 | Confirmatory seeds 1--4 role | PRESENT AND SUFFICIENT | Experimental setup, paragraphs 1 and 3 |
| 20 | H1 statistic and gate | PRESENT AND SUFFICIENT | Experimental setup, paragraph 3 |
| 21 | H2a variables | PRESENT AND SUFFICIENT | Problem setup, paragraph 6; experimental setup, paragraph 4 |
| 22 | M0/M1 models | PRESENT AND SUFFICIENT | Experimental setup, paragraph 4 |
| 23 | \(\beta_{\mathrm{ord}}\) interpretation | PRESENT AND SUFFICIENT | Experimental setup, paragraph 4 |
| 24 | Generic centroid separation \(g\) | PRESENT AND SUFFICIENT | Problem setup, paragraph 6 |
| 25 | Endpoint-vs-adjacent margin \(m_{\mathrm{adj}}\) | PRESENT AND SUFFICIENT | Problem setup, paragraph 6 |
| 26 | LOOCV comparison | PRESENT AND SUFFICIENT | Experimental setup, paragraph 4 |
| 27 | H2a gate | PRESENT AND SUFFICIENT | Experimental setup, paragraph 4 |
| 28 | Retina OOF evaluation design | PRESENT AND SUFFICIENT | Experimental setup, paragraph 2 |
| 29 | Solar fixed archived readout design | PRESENT AND SUFFICIENT | Experimental setup, paragraphs 1--2 |
| 30 | UTKFace supporting-only role | PRESENT AND SUFFICIENT | Experimental setup, paragraph 1 |
| 31 | Phase 3.19 factor study | PRESENT AND SUFFICIENT | Experimental setup, paragraph 5 |
| 32 | Phase 3.20A support-severity study | PRESENT AND SUFFICIENT | Experimental setup, paragraph 5 |

## Training-detail classification

| Detail | Classification | Treatment |
| --- | --- | --- |
| RetinaMNIST resolution, RGB input, backbone family/stem, fixed train counts | MAIN TEXT | Needed to identify the canonical setup and OOF population. |
| Retina input normalization | MAIN TEXT | Short reproducibility statement included. |
| Solar channels, resolution, architecture, train-only normalization | MAIN TEXT | Included at high level because they identify the archived setup. |
| Solar temporal alignment and full manifest details | SUPPLEMENT-ONLY | Cite artifact manifest/provenance table; do not overload main protocol. |
| Backbone optimizer schedules, augmentations, early stopping, selected epochs, full checkpoint inventory | SUPPLEMENT-ONLY | Existing execution artifacts/logs are the appropriate source. |
| C sampler, loss, optimizer, learning rate, batch size, epochs, direction weight decay, fixed norm/bias constraint | MAIN TEXT | Required to interpret C as a controlled direction probe. |
| Fold assignments and fitting-fold-only centroid computation | MAIN TEXT | Necessary leakage/provenance control. |
| Historical B bias-only and D controlled-scale mechanics | SUPPLEMENT-ONLY | Not part of the primary A/C confirmatory comparison. |
| H2b response-magnitude analysis | SUPPLEMENT-ONLY | Secondary, no decision gate. |
| Phase 3.19 exact cell-level outputs | SUPPLEMENT-ONLY | Main text only needs the bounded factorial-design distinction. |
| Phase 3.20A complete per-seed grid outputs | SUPPLEMENT-ONLY | Main text states the controlled design and bounded conclusion. |

No core main-text detail is currently classified as missing. The supplement still needs an organized manifest/configuration table and an explicit archived Solar readout provenance table before submission; these are organization tasks, not unresolved scientific claims.

## Additions made

- Defined the probability, decision, localization, severity, and geometry quantities with consistent orientation for the upper endpoint.
- Specified CE and the implemented cumulative-probability RPS.
- Made the A/C fixed-representation and fixed-norm/fixed-bias interpretation explicit.
- Added the seed-role, replication-unit, OOF, centroid-provenance, H1, and H2a rules that govern the confirmatory claims.
- Distinguished the frozen A/C study from the Phase 3.19 training-only factorial and the Phase 3.20A full-model severity grid.

## Cross-section consistency

The new text matches the abstract, Introduction, Results, Discussion, and Limitations on the following boundaries: the primary unit is a backbone seed; only seeds 1--4 are confirmatory; UTKFace and seed 0 are supporting or historical; H1 is the robust direction-only finding; H2a is setting-dependent; and neither balanced sampling nor the severity grid is presented as a new method or a universal causal result. No factual inconsistency requiring a Results or Discussion edit was found.

The protocol's ``not replicated'' H2a gate corresponds to the Results table's reporting label ``Not supported''; both denote at most two beneficial coefficients and do not change the statistical rule.

## Next manuscript step

Organize the supplement and verified bibliography, then perform a full cross-section consistency and venue-format pass. Do not change the frozen claim hierarchy or run additional experiments as part of that writing work.
