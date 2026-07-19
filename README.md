# Algorithmic Fairness Audit: Healthcare Risk Prediction

## Overview
A machine learning audit and mitigation pipeline for a Heart Failure prediction model. This project demonstrates how standard global accuracy metrics obscure severe localized biases against demographic subgroups, and implements a mathematical mitigation strategy to rectify safety disparities without expanding the dataset.

## Audit Methodology & Findings
An intersectional audit (Age × Sex) was performed on a baseline Support Vector Machine (SVM) trained on clinical diagnostic data.

* **Global Performance:** The baseline model achieved 86% global accuracy.
* **Subgroup Failure:** The model exhibited a 29.2% False Negative Rate (FNR) for young female patients.
* **Safety Disparity:** Young females experienced a 4.5x higher misdiagnosis rate (predicting "Healthy" when sick) compared to the reference group of older males (6.5% FNR).

## Mitigation Architecture
To reduce the safety gap, a multi-stage mitigation pipeline was engineered:

1. **Decoupled Classifiers:** Partitioned the training architecture to fit independent SVMs for male and female cohorts, isolating distribution shifts.
2. **Algorithmic Constraints:** Enforced `class_weight='balanced'` and mapped a non-linear RBF Kernel exclusively for the female cohort to compensate for data scarcity.
3. **Decision Boundary Thresholding:** Shifted the classification threshold (T = -0.2) for the female model, explicitly prioritizing Recall (Sensitivity) over Precision to minimize missed clinical diagnoses.

## Empirical Results

| Demographic Subgroup | Metric | Baseline Model | Mitigated Model | Absolute Delta |
| :--- | :--- | :--- | :--- | :--- |
| **Young Female** | False Negative Rate | 29.2% | 19.4% | -9.8% |
| **Old Male** | False Negative Rate | 6.5% | 6.5% | 0.0% (Reference) |

*Conclusion: The decoupled architecture successfully reduced the fatal misdiagnosis rate for the disadvantaged subgroup by approximately one-third, mathematically trading a controlled increase in False Positives (system burden) for a strict reduction in False Negatives (patient safety risk).*

## Execution

### Tech Stack
* **Python:** Core runtime.
* **Scikit-Learn:** SVM architecture, pipeline orchestration, scaling.
* **Pandas:** Demographic slicing and matrix manipulation.
* **Matplotlib:** Trade-off curve visualization.

### Run the Audit Pipeline
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute:**
   ```bash
   python src/main.py
   ```

*The pipeline will output the demographic matrix, execute cross-validation, and print the raw Safety Gap metrics to the terminal.*
