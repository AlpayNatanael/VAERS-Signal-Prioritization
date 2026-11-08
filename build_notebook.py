"""Create the explanatory notebook. Run run_presentation.py to execute it."""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))

def code(text):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))

md(r"""
    # Can machine learning help prioritize vaccine–event pairs for review?

    **Natanael Alpay · University of California, Irvine**  
    Presentation notebook · 8 November 2026

    We use historical VAERS reports to ask a practical question: **which vaccine–event pairs should we examine first if we want to find pairs that will cross a statistical screening threshold next quarter?**

    This notebook explains the completed study, refits the models, and rebuilds its graphs. The data cover 2022Q1–2026Q2. These are retrospective results from periods we have already examined.

    **How to use it:** read from top to bottom, or select **Run All**. All calculations and plots are in `analysis/`, so the code cells stay short. The complete ZIP includes the frozen processed inputs. Install the supplied `requirements.txt` before running.
""")

md(r"""
    ## 1. What problem are we trying to solve?

    VAERS contains reports of events after vaccination. There are many possible vaccine–event pairs, and reviewing them takes effort. We want to put promising candidates closer to the top of a review list.

    Here, a **statistical alert** means that a pair passes a fixed rule based on its report count and reporting proportions. A threshold crossing is a prompt for further investigation. It does not by itself establish that a vaccine caused an event.

    Our task has three steps:

    1. Find pairs that appear this quarter but are below the screening threshold.
    2. Use their current information and reporting history to rank them.
    3. Check whether they cross the threshold in the following quarter.

    A pair that was above the threshold in an earlier quarter can return to the list after falling below it. We therefore distinguish **first observed crossings** from **returning crossings**.
""")

md(r"""
    ## 2. Run the study

    The next cell trains logistic regression and gradient boosting from scratch in each of nine chronological windows. It also runs the independent checks of labels, rankings, subgroup counts, and tied scores.

    This run starts from the frozen, previously audited quarterly panel. It does not download or rebuild all raw VAERS reports. The panel and corrected pipeline are included so the numerical results can be reproduced.
""")

code("""
    %matplotlib inline
    from analysis import presentation as p

    study = p.run_analysis()
""")

md(r"""
    ## 3. What are we predicting?

    A data point is a **vaccine-type × event-term pair in one quarter**. It is not an individual patient. The event terms are MedDRA preferred terms and include symptoms, administrative errors, and product issues.

    A pair crosses our screening threshold when all three conditions hold:

    - At least **3 distinct retained reports** contain the pair.
    - The **proportional reporting ratio (PRR) is at least 2**.
    - The lower 95% confidence bound for PRR is **greater than 1**.

    PRR compares how common an event is in reports containing a vaccine type with how common it is in other reports. These are reporting proportions, not event rates among vaccinated people. The implementation uses a 0.5 correction in all four contingency-table cells.

    In symbols, for an eligible pair, our target is

    $$Y_{p,t}=1\quad\text{if pair }p\text{ crosses the rule in quarter }t+1.$$

    The final test ranks **2026Q1** candidates and checks their outcomes in **2026Q2**.
""")

code("""
    p.study_overview(study)
""")

md(r"""
    Of the 294 positive outcomes in the final test, **245 were returning crossings** and **49 were first observed crossings**.

    “First observed” means no earlier threshold crossing in the available history beginning in 2022. It does not mean a medically new or previously unknown adverse reaction. The candidates in this group have already been reported; what is new within our history is the threshold crossing.
""")

md(r"""
    ## 4. What information and methods do we use?

    We give the models **31 features**, grouped as follows:

    | Information | Examples | Features |
    |---|---|---:|
    | Reporting volume | Pair counts, vaccine counts, event counts | 5 |
    | Current statistical measures | PRR and related reporting scores | 5 |
    | Report context | Seriousness, age, sex, vaccination setting | 10 |
    | Reporting history | Previous-quarter counts, changes, earlier crossings | 11 |

    We compare two learned models with simple ranking rules:

    | Method | Simple explanation |
    |---|---|
    | **Threshold proximity** | Rank a pair by how close it is to satisfying all parts of the screening rule. |
    | **History + proximity** | Put pairs with an earlier crossing first, then use threshold proximity within each group. |
    | **Logistic regression** | Learn a weighted combination of the features to produce a probability score. |
    | **Gradient boosting** | Combine many small decision trees so the score can depend on interactions and nonlinear patterns. |

    History + proximity is an important comparator: returning alerts may be relatively easy to anticipate. A useful model should be compared with a rule that already uses this information.
""")

md(r"""
    ## 5. How do we keep the evaluation in time order?

    For each test quarter, the model is fitted using earlier candidate quarters. A separate, later historical quarter is used to adjust its probability scores. The test outcomes are then used only for evaluation.

    For the final window:

    | Step | Candidate quarters | Outcomes used |
    |---|---|---|
    | Fit the model | Before 2025Q4 | The following quarter for each training row |
    | Calibrate its scores | 2025Q4 | 2026Q1 |
    | Rank the test candidates | 2026Q1 | Evaluate against 2026Q2 |

    Missing-age values are filled using the median from the fitting window. The same value is then used for calibration and testing. The original model settings and random seed 42 are preserved.

    **Timing limitation:** these splits use historical report receipt dates. They do not reconstruct the exact information publicly available on each forecast date. A live deployment claim would need a separate data-availability evaluation.
""")

md(r"""
    ## 6. How many pairs do we need to examine?

    The next graph answers the most practical question in this study. Start at the top of each ranking and examine more pairs. The vertical axis shows what fraction of the **294 eligible future crossings** have been captured.

    A higher curve is better: it finds more of those future crossings within the same review budget.
""")

code("""
    p.plot_review_efficiency(study)
""")

md(r"""
    **How to explain this graph:** to capture half of the eligible future crossings, gradient boosting needed **806 pairs**, while threshold proximity needed **1,531**. That is a **47.4% reduction in the number of pairs**.

    This percentage is relative to threshold proximity. The history-aware rule is a stronger baseline and narrows the gap, as shown below. All methods are evaluated on the same eligible candidates.
""")

code("""
    p.workload_table(study)
""")

md(r"""
    Several pairs have identical scores under the history rule. With uniform random ordering within the tied block, the expected number examined to find 147 positives is **994.24**. The possible tie-order extremes are 979–1,802; these are not confidence limits.

    The curve uses expected capture at each budget within ties. Expected capture and expected stopping time are different calculations. The reported 806 and 1,531 cutoffs have unique scores.

    These results count candidate pairs. We did not measure staff time, cost, or whether the pairs led to useful clinical findings.
""")

md(r"""
    ## 7. Does the model help with first observed crossings?

    **Average precision (AP)** summarizes how well positives appear near the top of a ranking across different cutoffs. Higher is better. AP is not classification accuracy, and its scale depends on how common positive outcomes are.

    We first compare models on all eligible pairs. We then repeat the comparison using only pairs with no earlier threshold crossing in the available history.
""")

code("""
    p.plot_final_performance(study)
    p.final_results_table(study)
""")

md(r"""
    Gradient boosting has the highest AP among these methods in the final test: **0.2047 overall** and **0.0604 among candidates with no earlier crossing**.

    The latter group contains only 49 positives among 4,581 candidates. Its low positive-outcome rate is one reason to show absolute counts alongside AP. We should compare models within each population and examine what their top-ranked lists actually contain.
""")

code("""
    p.plot_first_and_returning(study)
""")

md(r"""
    **How to explain this graph:** the combined top 100 contained **32 future crossings, all returning alerts**. If we instead rank only the candidates with no earlier crossing, the top 100 contains **4 first observed crossings**.

    These are two separate 100-pair lists from different candidate populations. They are not two uses of one shared 100-pair budget.

    The model improves the ranking for first observed crossings, but the absolute yield remains modest. Most of the positives in the combined high-ranking list are returning alerts. At the 806-pair cutoff, **142 of the 147 captured crossings were returning** and **5 were first observed**.
""")

md(r"""
    ## 8. Is the result consistent across quarters?

    We repeat the same fitting and evaluation procedure across nine candidate quarters, from 2024Q1 through 2026Q1. Every point below uses an earlier training period and predicts the following quarter.
""")

code("""
    p.plot_quarterly_performance(study)
""")

md(r"""
    Gradient boosting outperformed threshold proximity and the history rule in all nine overall windows. Logistic regression performed better than gradient boosting in **2024Q1** and **2025Q4**.

    These quarters were already examined during development and share historical information. The counts of wins describe the observed results; they are not nine independent significance tests or a new untouched validation study.
""")

md(r"""
    ## 9. What does the ranking leave out?

    The main results apply to pairs that were observed and below the rule in the candidate quarter. Some future crossings belong to pairs outside that group.

    In 2026Q2, **727 pairs** met the rule. Their status in the preceding quarter was:
""")

code("""
    p.coverage_table(study)
""")

md(r"""
    Only the **294 eligible crossings** appear in the main capture calculation. Another 294 pairs were already above the rule, and 139 were absent from the current-quarter candidate set. Therefore, “50% capture” means half of the eligible future crossings, not half of every pair meeting the rule next quarter.

    The study supports a retrospective ranking result. It does not establish vaccine causality, real-time public-data availability, or clinical usefulness. Those would require additional evidence.
""")

md(r"""
    ## 10. A short explanation to say aloud

    We studied whether machine learning can help experts decide which vaccine–event pairs to review first. We used current reporting patterns and reporting history to predict whether a pair would cross a statistical screening threshold next quarter. In the final test, gradient boosting found half of the eligible future crossings within 806 pairs, compared with 1,531 for threshold proximity. Most of the captured crossings were returning alerts, so we also examined first observed crossings separately. The results show a ranking benefit in this historical study; the clinical value still needs to be evaluated.

    **The next research steps** are stronger temporal baselines, a later evaluation fixed before its outcomes are examined, and a careful study of data availability and first-crossing performance. New ensemble models and simulations are planned extensions, not results presented here.
""")

md(r"""
    ## Reproducibility record

    The summary below comes from this notebook's execution. The original numerical scripts check the fixed labels, scores, time splits, prediction agreement, and tie handling. Additional presentation checks compare the plotted quantities with the fresh prediction files.

    The frozen inputs are the project's own pickle files; the accompanying hashes identify them. Only load pickle files from a source you trust. Readable result tables are also supplied as CSV files.
""")

code("""
    p.execution_summary(study)
""")

md(r"""
    **Files to explore:** `analysis/source/vaers_pipeline.py` contains the corrected scientific pipeline; `analysis/presentation.py` contains the plotting helpers; `analysis/results/` contains the detailed results; `figures/` contains PNG and vector PDF versions of the presentation graphs.

    **Sources:** [VAERS public data](https://vaers.hhs.gov/data.html); the archived corrected project and numerical verification package dated 13 October 2026. The supplied `analysis/README.txt` records the original verification scope. This presentation reruns those completed comparisons.
""")

nb=nbf.v4.new_notebook(cells=cells,metadata={
    "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.12.14"},
    "title":"VAERS Signal Prioritization — Presentation",
    "authors":[{"name":"Natanael Alpay"}],
})
nbf.validate(nb)
path=ROOT/"VAERS_Presentation.ipynb"
nbf.write(nb,path)
print(f"Created {path.name}: {len(cells)} cells")
