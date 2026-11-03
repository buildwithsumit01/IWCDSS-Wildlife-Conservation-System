# IWCDSS — Intelligent Wildlife Conservation Decision Support System

An integrated platform combining Artificial Intelligence, Knowledge Representation, and field ecology to assist wildlife biologists, forest officers, and patrol rangers in tiger presence/absence prediction, threat management, and patrol route optimization.

---

## 🌟 Architecture Overview

The system covers all 9 Computational Intelligence tasks organized into 5 primary operational sections:

1. **Overview & Foundation (Tasks 1, 2, 4)**
   - **Task 1:** Domain Ontology & Conceptual ER model (12 entities, 17 semantic relationships).
   - **Task 2:** Production Rule-Based Expert System (Rules R1–R10, forward & backward chaining, interactive threat evaluator).
   - **Task 4:** Data Preparation & Quality Inspection (Distinguishes the 7,989 WEKA-cleaned dataset with IQR outlier filters from the 8,085 baseline survey dataset; includes spatial scatter maps and multivariate plot matrix).

2. **Tiger Presence Predictor (Tasks 5, 6, 7, 9)**
   - **Manual Survey Assessment:** Real-time multi-module consensus (Expert Rules + XGBoost + Dense DNN).
   - **Dynamic Local SHAP:** Feature attribution bar chart explaining why a specific point is predicted Present or Absent.
   - **Actionable Conservation Recommendations:** Priority levels (`High`, `Moderate`, `Low`, `Very low`) with customized field protocols.
   - **Official Report Export:** One-click generation of field summary report in Microsoft Word format (`.docx`).
   - **Batch Assessment:** Upload multi-row survey CSVs and download prioritized findings.

3. **Model Benchmarks & Evaluation (Tasks 5, 6, 7)**
   - **Feature Engineering (Task 5):** Correlation matrix heatmap and Random Forest feature importance rankings.
   - **ML Model Leaderboard (Task 6):** Preserved official metrics for XGBoost, Decision Tree, Random Forest, SVM, and Logistic Regression.
   - **Deep Neural Network Evaluation (Task 7):** Preserved metrics and training history over 42 epochs; analysis of 88.99% high recall in minimizing false negatives.
   - **Authentic Visualizations:** ROC Curves, Confusion Matrices, and K-Means Clustering scatter plot.

4. **Explainable AI (Task 8)**
   - Global SHAP feature ranking across reserves.
   - LIME and SHAP comparison for True Positives, False Positives, and False Negatives.
   - Counterfactual Perturbation Analysis (determining minimal prey/survey shifts needed to flip false negative predictions).
   - Confidence Trust Metric (83.2%) and Subgroup Ecological Fairness across Terai, Chure Hills, and Streambed habitats.

5. **Patrol Route Optimizer (Task 3)**
   - State space search network across 8 sanctuary zones (Entry Gate to Poaching Hotspot).
   - Heuristic distance table $h(n)$ and algorithm complexity analysis.
   - Interactive route solver comparing BFS, DFS, and A* Search (identifying the optimal 8 km patrol path).

---

## 📊 Preserved Model Metrics vs. Local Regeneration

| Model | Official Notebook Metrics | Locally Regenerated Metrics | Notes |
| :--- | :---: | :---: | :--- |
| **XGBoost (Best Model)** | Acc: 0.8745, F1: 0.6862, ROC-AUC: 0.9292 | Acc: 0.8745, F1: 0.6862, ROC-AUC: 0.9292 | **Exact match** down to 4 decimal places. |
| **Decision Tree** | Acc: 0.8856, F1: 0.6805, ROC-AUC: 0.8036 | Acc: 0.8856, F1: 0.6805, ROC-AUC: 0.8036 | **Exact match**. |
| **Random Forest** | Acc: 0.8794, F1: 0.6382, ROC-AUC: 0.9193 | Acc: 0.8794, F1: 0.6382, ROC-AUC: 0.9193 | **Exact match**. |
| **SVM** | Acc: 0.7842, F1: 0.5749, ROC-AUC: 0.8810 | Acc: 0.7842, F1: 0.5749, ROC-AUC: 0.8810 | **Exact match**. |
| **Logistic Regression** | Acc: 0.7687, F1: 0.5302, ROC-AUC: 0.8305 | Acc: 0.7687, F1: 0.5302, ROC-AUC: 0.8305 | **Exact match**. |
| **Dense DNN** | Acc: 0.7749, Recall: 0.8899, ROC-AUC: 0.9062 | Acc: 0.7527, Recall: 0.8578, ROC-AUC: 0.8969 | Minor variation due to oneDNN CPU instruction differences on Windows vs Linux Colab. Preserved official metrics are highlighted in the UI. |

---

## 🚀 How to Launch the Application

Run the following command in PowerShell / Command Prompt:

```bash
streamlit run app/app.py
```
Or double-click `run_app.bat`.

The application will be accessible at: `http://localhost:8501`.
