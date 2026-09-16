# Manuscript Citation Verification

**Status:** complete (2026-09-15)  
**Scope:** replacement of the six manuscript `\cite{TODO}` placeholders only.

## TODO inventory and claim-to-source mapping

| ID | File | Claim/topic | Sources inserted | Support |
| ---: | --- | --- | --- | --- |
| 1 | `introduction.tex` | Decoupling, balanced head retraining, and logit adjustment | Kang et al. (ICLR 2020); Alshammari et al. (CVPR 2022); Menon et al. (ICLR 2021) | DIRECTLY SUPPORTED |
| 2 | `introduction.tex` | Classifier/feature geometry; ordinal costs and resampling | Wang et al. (CVPR 2022); Zhu et al. (CVPR 2022); Yi et al. (ICLR 2025); Wang et al. (AAAI 2026); Zhu et al. (KBS 2019); L\'azaro and Figueiras-Vidal (PR 2023) | DIRECTLY SUPPORTED |
| 3 | `related_work.tex` | Long-tail classifier rebalancing | Kang et al.; Alshammari et al.; Menon et al. | DIRECTLY SUPPORTED |
| 4 | `related_work.tex` | Angular/norm, representation, and feature/classifier geometry | C2AM; Balanced Contrastive Learning; Yi et al.; Space Alignment Matters | DIRECTLY SUPPORTED |
| 5 | `related_work.tex` | Imbalanced ordinal weighting/resampling | SMOR; L\'azaro and Figueiras-Vidal | DIRECTLY SUPPORTED |
| 6 | `discussion.tex` | Prior-art boundary for head rebalancing, direction/norm effects, and alignment | Kang et al.; Alshammari et al.; C2AM; Space Alignment Matters | DIRECTLY SUPPORTED |

The Introduction wording was minimally split so that the long-tail intervention
claim and the geometry claim each have sources that directly support them. No
empirical claim, gate, result, or novelty boundary was changed.

## Verified bibliography records

| Key | Verified source(s) |
| --- | --- |
| `kang2020decoupling` | [OpenReview](https://openreview.net/forum?id=r1gRTCVFvB) |
| `alshammari2022weight` | [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Alshammari_Long-Tailed_Recognition_via_Weight_Balancing_CVPR_2022_paper.html) |
| `menon2021logit` | [OpenReview](https://openreview.net/forum?id=37nvvqkCo5) |
| `c2am2022angular` | [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_C2AM_Loss_Chasing_a_Better_Decision_Boundary_for_Long-Tail_Object_CVPR_2022_paper.html) |
| `zhu2022balanced` | [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_Balanced_Contrastive_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2022_paper.html) |
| `yi2025geometry` | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/adb2075b6dd31cb18dfa727240d2887e-Abstract-Conference.html) |
| `wang2026alignment` | [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/39835) |
| `zhu2019smor` | [publisher record](https://www.sciencedirect.com/science/article/pii/S0950705118306166) |
| `lazaro2023bayesian` | [institutional publisher record](https://researchportal.uc3m.es/display/act556413) |

The project audit's shorthand ``Michael Yao'' for the ICLR 2025 geometry paper
was corrected to the verified author name **Jiachen Yao** in the BibTeX entry.

## Novelty-boundary check

The verified literature directly confirms that balanced/decoupled classifier
training, logit adjustment, classifier angular or norm effects, long-tail
representation geometry, feature/classifier alignment, and imbalanced ordinal
methods are established themes. The manuscript continues to disclaim novelty
for those components and confines its contribution to the stated ordinal
localization mechanism analysis.

## Remaining gaps

There are no unresolved citation placeholders. The bibliography is intentionally
limited to sources needed by the six existing claims; it is not a supplement or
a comprehensive literature survey.
