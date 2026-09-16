# ICLR 2027 Submission Audit

**Verdict:** **B — ICLR 2027 FORMAT READY; MINOR SUBMISSION ITEMS REMAIN** (2026-09-15)

## Constraints and template

The official [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) require at most nine main-text pages, anonymous main and supplementary text, and an AI-use statement; references and appendices do not count. The official `iclr-2027-style-files.zip` was retrieved from `media.iclr.cc/Conferences/ICLR2027/` on 2026-09-15. The dedicated source tree is `manuscript/iclr2027/`; the generic manuscript is preserved.

## Page budget

| Version | Main text | References begin | Appendix | Total PDF |
| --- | ---: | ---: | ---: | ---: |
| Faithful baseline | 13 | 14 | 15--19 | 19 |
| ICLR submission version | 4 | 5 | 7--11 | 11 |
| Controlled main-text expansion | 6 | 6 | 8--12 | 12 |

The final main text is below the nine-page limit without altering margins, fonts, spacing, or the official style. References occupy pages 5--6; the appendix is integrated after references.

## Compression and content disposition

The compact ICLR body retains the phenomenon, A/C probe, H1 gate/results, H2a setting dependence, controlled-factor and severity boundaries, all four canonical figures, and two primary tables. Detailed historical seed-0 analyses, seed-level results, hyperparameters, provenance, UTKFace support, and secondary diagnostics are in the appendix. Related work and discussion were condensed rather than removed.

## Statements and ethics

The submission includes anonymous AI-use, reproducibility, and ethics statements. The AI-use wording is a draft for author confirmation before upload. A short ethics statement is appropriate because the analysis uses medical and face/age datasets but makes no deployment or IRB claim.

## Anonymity audit

The author block is anonymous; PDF metadata has empty Title/Author fields. The copied appendix had one identifying cluster path (`/scratch/users/jhong36`), which was replaced with an anonymous description. No author, institution, personal repository, home path, or acknowledgement is intentionally present in the submission sources. Do not include the local baseline directory or original generic sources in the upload.

## Verification and remaining items

`latexmk -pdf main.tex` succeeds. The remaining warnings are non-material underfull bibliography/appendix boxes and float underfull-vbox notices; no undefined citations or references remain after the final rerun. `git diff --check` is required before submission. Before OpenReview upload, authors must confirm the AI-use disclosure, re-run the anonymization search on the exact upload bundle, and decide whether to provide anonymous code.
