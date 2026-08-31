# Phase 1.75 Decision Rule Audit

Frozen native-28x28 probability CSVs for seeds 0--4 were reused; no model was
retrained. Decisions minimize exact expected loss over discrete actions 0--4, with
the smallest minimizer selected on ties. Mode is argmax, L1 is the predictive median,
and L2 is discrete expected-squared-loss minimization.

Mode/L1/L2 mean metrics are accuracy 0.5320/0.5295/0.4975, MAE
0.7430/0.6950/0.7080, QWK 0.5526/0.5771/0.5600, and severe-error prevalence
20.3%/18.3%/17.8%. L1 changes 6.75%--36.0% of decisions per seed; L2 changes
15.0%--51.25%. Thus L1 primarily converts ordinal mistakes to less severe ones with
a small accuracy trade-off, while L2 has a larger hit-rate trade-off.

`prediction_distance_l1` is exactly mode-centered expected L1 risk. `bayes_risk_l2`
equals the exact discrete L2 Bayes risk. Matched decision/risk alignment, detection,
and coverage outputs are saved per seed in the audit directory.

**Decision: DECISION RULE INTRODUCES A TRADE-OFF.** Decision-rule mismatch is an
important contributor to MAE and severe errors, but ordinal Bayes decisions should
not be presented as uniformly better because accuracy declines. Any later Phase 2
must separate learned distributions, decision rules, and risk scores.
