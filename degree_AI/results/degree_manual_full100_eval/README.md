# Degree Manual Full 100 Evaluation

This folder contains the merged 100 manual labels and the degree_AI predictions joined from `degree_aihub_summary.csv`.

## Main result
- Samples: 100
- Accuracy: 0.5300
- Macro F1: 0.2702
- Weighted F1: 0.4347
- Correct/Wrong: 53/47

## Files
- `combined_manual_degree_100_with_predictions.csv`: merged manual labels + degree_AI prediction columns.
- `eval_rows_full100.csv`: valid rows used in evaluation.
- `manual_degree_confusion_matrix_full100.csv`: confusion matrix.
- `manual_degree_label_metrics_full100.csv`: precision/recall/f1 by label.
- `manual_degree_mismatches_full100.csv`: wrong cases.
- `manual_degree_eval_metrics_full100.json`: machine-readable metrics.
- `manual_degree_full100_report.txt`: human-readable report.
