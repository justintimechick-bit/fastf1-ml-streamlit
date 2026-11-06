# Project requirements and this implementation

The supplied **CIS8695 MM2 AY2026/27 Machine Learning Project** rubric is the source
for the requirements below. It informs the project; its upload and submission
instructions are not instructions for this application to submit work.

| Rubric requirement | Evidence in this project | Remaining student deliverable |
| --- | --- | --- |
| Public dataset; explain acquisition and cleaning | Public FIA final classifications and independent entry lists, document URLs/publication bounds/SHA-256, real snapshot coverage, FastF1/Jolpica source audit | Describe source limitations and why FIA final labels are used |
| At least two different ML methods | Three families: linear/logistic, random forest, histogram gradient boosting | Explain both strengths and limitations; avoid describing three tasks as three model families |
| Preprocessing, visual exploration, model building | Shifted driver/team history; fold-local imputation/scaling/encoding; unavailable columns excluded; interface coverage and prediction charts | Add figures that support the research question |
| Performance evaluation and validation | Chronological development events, baseline comparison, separate selected historical-target metrics, event/known-rank counts | Discuss results that lose to a baseline as well as improvements |
| Performance and interpretability | Simple linear reference, tree ensembles, visible input groups and held-out results | Explain feature effects cautiously; predictive association does not establish causation |
| Reproducible Python notebooks with displayed output | `tools/export_real_notebook.py` executes analysis cells against an existing sealed real run | Review exported `.ipynb`, include any additional exploratory/model analysis required for the report |
| Dataset or public link | Parsed real factual observations and official source links; local source cache retained | Include the dataset/source link with the final submission |

The proposal is due **November 6**: at most one page, 12-point font, single-spaced.
It needs the problem and importance, data acquisition/cleaning, proposed methods,
comparisons and metrics. The project presentation is **December 2**, **13–17
minutes**. The final report is due **December 6** in the attached project rubric:
**10–15 pages**, 12-point font, 1.5 spacing, no cover page. Its requested sections
are executive summary, background, data, methods/parameters, evaluation,
results/discussion/limitations, conclusions and appendix. These dates come from
the project document; the earlier high-level syllabus image uses a different
report milestone.

The linear family connects directly to Sessions 1–2 (linear, ridge, logistic
regression). Random forest and gradient boosting connect to Session 3 (CART,
forests, boosting and ensembles). Chronological features and validation connect
to time-dependent prediction, but this implementation does not claim to fit a
classical univariate forecasting model such as ARIMA. PCA, clustering and every
other syllabus topic need not be added simply to satisfy the requirement of at
least two methods.

Use [the actual-run guide](real-model-lab.md) for commands and the interpretation
of incomplete rank labels, missing timing features and exploratory historical
replays. A clean interface is supporting evidence, not a replacement for the
report's discussion of evaluation and limitations.
