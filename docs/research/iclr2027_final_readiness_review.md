# ICLR 2027 Final Submission Readiness Review

**Verdict: A — CONTENT AND ANONYMITY READY; AUTHOR CONFIRMATION ONLY.**

The original four-page version was too compressed: A/C constraints, seed roles,
H2a variables and LOO criterion, and the difference between replicated H1 and
setting-dependent H2a were underexplained. The final six-page main text now
states the ordinal phenomenon, prior-work boundary, fixed norm/bias
direction-only probe, CE/RPS settings, seed-1--4 replication unit, Retina OOF
and Solar readout roles, H1 sign gate, \(m_{\rm adj}\), generic gap,
\(\Delta_{\rm LOO}\), controlled-study boundaries, and limitations. A reviewer
can identify all core claims without consulting the appendix.

Long-tail reviewers can distinguish the work from balanced classifier fitting;
ordinal/UQ reviewers can identify the L1 decision and location diagnostics; and
general ML reviewers can follow the replication unit and claim boundaries.
Likely questions about external generalization, Solar's archived-readout role,
and setting-specific H2a are addressed and bounded in the main text.

The official ICLR 2027 template compiles successfully. Main text occupies pages
1--6, references begin on page 6, and the integrated appendix occupies pages
8--12. Figures 1--4 and compact H1/H2a tables remain in main; seed-level,
historical, provenance, UTKFace, and secondary material remains in appendix.

The AI-use statement accurately describes assistance with drafting/editing,
code/documentation, citation organization, analysis scripting, and workflow
organization while assigning responsibility and empirical verification to the
authors. Its status is **NEEDS AUTHOR CONFIRMATION**. The reproducibility
statement is accurate and does not promise a public repository. The ethics
statement makes no IRB, consent, clinical, or demographic claim.

Source-tree searches found no author name, institution, personal URL, username,
home/cluster path, acknowledgement, grant, ORCID, or email in submission files.
The earlier `/scratch/users/jhong36` appendix path was removed. The PDF has the
anonymous author block and review header; PDF Title, Author, Subject, and
Keywords are empty.

Upload `manuscript/iclr2027/main.pdf` and, if sources are required, only the
minimal anonymous source set. Exclude `baseline/`, build logs, audit documents,
generic manuscript sources, and artifacts. Do not include a code URL unless an
anonymous repository is separately prepared. The only author decisions are AI
disclosure confirmation and whether to supply an anonymous code/artifact package.
`latexmk` has no errors or undefined references/citations. Remaining warnings
are small appendix-only overfull boxes (at most 4.28pt) and bibliography
underfull boxes.
