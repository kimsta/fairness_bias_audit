# Algorithmic Fairness Audit: Healthcare Risk Prediction

## 📋 Executive Summary

A machine learning audit of a Heart Failure prediction model, demonstrating how standard accuracy metrics mask dangerous biases against minority demographic groups.

I identified a **4.5x Safety Disparity** for young female patients compared to the reference group (older males). I then engineered a hybrid mitigation strategy that reduced this safety gap by **~50%** without requiring additional data collection.

## 🔍 The Problem (Diagnosis)

Using a Support Vector Machine (SVM) on clinical data, I performed an intersectional audit (Age x Sex).

* **Finding:** The model achieved **86% global accuracy**.

* **The Failure:** However, it exhibited a **29.2% False Negative Rate (FNR)** for Young Females.

* **Impact:** Nearly 1 in 3 sick young women were misdiagnosed as "Healthy," compared to only 1 in 15 (6.5%) for Old Men.

## 🛠️ The Solution (Mitigation)

I implemented a multi-stage mitigation pipeline:

1. **Decoupling:** Trained separate classifiers for Men and Women to handle distribution shifts.

2. **Constraints:** Applied `class_weight='balanced'` and an **RBF Kernel** for the female cohort to handle data scarcity and non-linearity.

3. **Threshold Tuning:** Shifted the decision boundary ($T = -0.2$) for women to prioritize Recall (Sensitivity) over Precision.

## 📊 Results

| Subgroup | Metric | Baseline Model | Mitigated Model | Impact | 
 | ----- | ----- | ----- | ----- | ----- | 
| **Young Female** | **Miss Rate (FNR)** | **29.2%** | **19.4%** | **📉 Risk Reduced (~10pts)** | 
| **Old Male** | Miss Rate (FNR) | 6.5% | 6.5% | (Reference) | 

*The mitigation explicitly traded a slight increase in False Alarms (Burden) for a significant decrease in Missed Diagnoses (Safety).*

## 🔧 Tech Stack

* **Scikit-Learn:** SVM, Pipeline, Scaling.

* **Pandas:** Data manipulation and demographic slicing.

* **Matplotlib:** Visualization of trade-off curves.

## 🚀 How to Run this Experiment

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Audit**

    ```bash
    python src/main.py
    ```
3. **Output** The script will generate the demographic breakdown, run the cross-validation audit, and print the **Safety Gap** results to your terminal.
    