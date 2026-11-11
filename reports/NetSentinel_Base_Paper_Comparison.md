# NetSentinel Base Paper Comparison

**Reference paper:** M. Ali et al., “Hybrid Machine Learning Model for Efficient Botnet Attack Detection in IoT Environment,” IEEE Access, vol. 12, pp. 40682–40699 (2024), [DOI 10.1109/ACCESS.2024.3376400](https://doi.org/10.1109/ACCESS.2024.3376400). The [article PDF](https://cloudstechnologies.in/cloudtech-admin/basepaperfiles/1739747659hybrid_machine_learning_model_for_efficient_botnet_attack_detection_in_iot_environment.pdf) reports the 82,332-row UNSW-NB15 training file, its 70/30 training/test division, and headline accuracies in the methodology/results sections and Tables 8–13.

## Headline accuracy comparison

The project column comes from the completed run’s 24,700-row held-out test set. Delta is project minus paper in percentage points (pp). Positive values mean the project score is higher; they do not establish statistical significance.

| Model | Paper accuracy | Project accuracy | Delta (pp) |
|---|---:|---:|---:|
| ANN | 75.68% | 95.7611% | +20.0811 |
| CNN | 94.40% | 95.6680% | +1.2680 |
| LSTM | 96.51% | 95.8745% | −0.6355 |
| RNN | 95.22% | 95.2429% | +0.0229 |
| ACLR | 96.98% | 95.8785% | −1.1015 |

The paper’s prose and tables report ANN 0.7568, CNN 0.9440, LSTM 0.9651, RNN 0.9522, and ACLR 0.9698 accuracy. The paper reports ACLR precision 0.9691, recall 0.9693, F1 0.9692, ROC-AUC 0.9934, and PR-AUC 0.9950. It reports five-fold ACLR accuracy 0.9749. These are paper-reported values, not results from this project’s run.

## Metric detail

| Model | Paper P / R / F1 | Project P / R / F1 |
|---|---|---|
| ANN | 0.7817 / 0.7568 / 0.7559 | 0.958507 / 0.964779 / 0.961633 |
| CNN | 0.9444 / 0.9440 / 0.9439 | 0.950655 / 0.971765 / 0.961094 |
| LSTM | 0.9651 / 0.9651 / 0.9651 | 0.962571 / 0.962500 / 0.962535 |
| RNN | 0.9523 / 0.9522 / 0.9521 | 0.958115 / 0.955368 / 0.956739 |
| ACLR | 0.9691 / 0.9693 / 0.9692 | 0.955408 / 0.970441 / 0.962866 |

For ACLR, project ROC-AUC is 0.993818 versus paper 0.9934; project PR-AUC is 0.995363 versus paper 0.9950. Other project ROC-AUC and PR-AUC values are in [metrics_test.json](../runs/NetSentinel_20261103_113830/reports/metrics_test.json); the paper’s individual-model comparison centers on accuracy, precision, recall and F1.

## Method differences and interpretation

1. **Train/validation/test protocol.** The paper describes splitting the 82,332-row training file 70/30. This run preserves that outer test partition but further splits the 70% outer training portion into 48,987 fit rows and 8,645 validation rows. This extra validation partition supports selecting best checkpoints without looking at test results and reduces the rows available for model fitting.
2. **Checkpoint selection.** Project base and meta checkpoints are selected using validation loss. The held-out test partition is evaluated after fitting. The paper describes a 70/30 split and epoch-based results but does not document this exact validation-checkpoint protocol.
3. **Leakage-safe stacking.** Project ACLR meta-training uses five-fold out-of-fold predictions on the fit partition. Each training row receives base-model predictions from a model that excluded it. The paper describes combining the four base models, but does not specify this OOF procedure.
4. **Reproducibility choices.** This project used seed 42. A matching paper seed is not stated in the implementation materials available here. Metrics were calculated at a 0.5 probability threshold.
5. **External cross-validation.** The paper’s 0.9749 figure is an external ACLR cross-validation accuracy. This project’s five OOF folds generate training features for ACLR; they are not a five-fold held-out score. An optional external CV run was not included in the completed primary experiment.

The results are therefore a transparent comparison, not an exact reproduction under identical training and checkpoint-selection procedures. The small ACLR accuracy gap versus LSTM within this run is only about 0.004 percentage points; no significance claim is made.

## Project run artifacts

The project values are from [runs/NetSentinel_20261103_113830/reports/metrics_test.json](../runs/NetSentinel_20261103_113830/reports/metrics_test.json), with labels and predictions under [runs/NetSentinel_20261103_113830/outputs/](../runs/NetSentinel_20261103_113830/outputs/). The full training procedure and plots are documented in [NetSentinel_Final_Training_Report.md](./NetSentinel_Final_Training_Report.md).

