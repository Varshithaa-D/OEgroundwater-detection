# Groundwater Over-Exploitation Detection - Unified Benchmark Results

| Category | Model Name | ROC_AUC | Accuracy | Balanced_Accuracy |
| --- | --- | --- | --- | --- |
| Spatial Control | Geographic k-NN (Coordinates Only) | 45.0 | 75.93 | 46.59 |
| Classical ML | Logistic Regression (L2) | 84.32 | 79.63 | 72.05 |
| Classical ML | Linear Support Vector Classifier | 82.27 | 77.78 | 51.59 |
| Classical ML | Random Forest (100 Trees) | 68.64 | 79.63 | 64.32 |
| Classical ML | Extra-Trees (Extremely Randomized) | 66.36 | 77.78 | 59.32 |
| Classical ML | Non-linear RBF Kernel SVM | 59.32 | 81.48 | 53.86 |
| Classical ML | Histogram Gradient Boosting | 42.05 | 59.26 | 44.09 |
| Bio-Inspired | Genetic Algorithm (GA) + SVM | 79.32 | 81.48 | 53.86 |
| Bio-Inspired | Genetic Algorithm (GA) + LogReg | 72.27 | 66.67 | 56.36 |
| Bio-Inspired | Particle Swarm Optimization (PSO) + SVM | 57.05 | 77.78 | 51.59 |
| Deep Learning | Temporal Bidirectional GRU | 75.91 | 72.22 | 63.64 |
| Deep Learning | Temporal 1D-CNN (Kernel=3) | 75.91 | 68.52 | 65.23 |
| Deep Learning | Small Temporal Transformer | 77.73 | 74.07 | 68.64 |
