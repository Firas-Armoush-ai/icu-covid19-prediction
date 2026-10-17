# ICU Prediction for COVID-19 Patients: Project Summary

## Problem

Predict whether a confirmed COVID-19 patient will need ICU admission, using only data from the first two hours after hospital admission. Missing a patient who will need ICU is the costliest error.

## Data

* Hospital Sírio-Libanês, *COVID-19 – Clinical Data to assess diagnosis* (Kaggle, CC BY-NC 4.0): 385 patients, São Paulo and Brasília, January–April 2020, values pre-scaled to −1..1 by the provider.
* One row per patient from the 0–2 hour window; target = ICU admission in any window.
* **32 patients already in ICU during that window were excluded**, following the provider's warning that those measurements may come from after admission. Final dataset: 353 patients, 163 admitted to ICU.

## Approach

* Fixed 80/20 stratified split: 282 training patients, 71 held-out test patients.
* Logistic regression, random forest and XGBoost (library defaults), each with and without sigmoid calibration, compared by 5-fold × 10-repeat cross-validation on the training split, with all preprocessing refitted inside every fold.
* Decision threshold set from out-of-fold predictions to reach 90% recall.
* A one-feature baseline (logistic regression on age group alone) scored on the same folds and, afterwards, on the same test patients.
* Final model and threshold committed before a single evaluation on the test set, with bootstrap intervals.

## Results

* Cross-validation: models were indistinguishable on ROC-AUC (0.67–0.71); uncalibrated models were badly overconfident (calibration slopes 0.14–0.31), and sigmoid calibration fixed it.
* Final model: **calibrated random forest, threshold 0.352**.
* Test set: ROC-AUC 0.643 (95% CI 0.502–0.765); at the threshold, recall 0.818 (27 of 33), precision 0.519, specificity 0.342.
* Most-used features: age, gender, lymphocytes, C-reactive protein, platelets, urea, neutrophils.
* **Age alone does as well:** the age-only baseline reaches ROC-AUC 0.701 in cross-validation (final model 0.695) and 0.723 on the test set (difference −0.080, 95% CI −0.228 to +0.055).

## Takeaways

* Fixing the 0–2 hour leakage and evaluating properly lowered the headline numbers, and made them trustworthy.
* Early data supports a cautious screen (rule out about a third of patients who will not need ICU), not a confident predictor, and most of its signal is age.
* A trivial baseline is the first thing to check: it showed that the labs add little in the first two hours.
* Probability calibration matters as much as discrimination when a dashboard shows probabilities.

## Possible next steps

* Any new model has to beat the age-only baseline.
* Validate on data from another hospital or a later period.
* Reduce the redundant lab columns (at 0–2 hours each lab's MEAN, MEDIAN, MIN and MAX are identical) and test a regularised logistic model with nested cross-validation.
* Use later time windows for a separate, later-stage prediction task.

Details: [README](../README.md), [cross-validation report](cv_report.md), [test-set evaluation](evaluation_test.md).
