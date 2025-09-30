# Reproduction

To compute the MCQ acc:

```bash
cd .
python compute_human_mcq_acc.py --input processed_mcq_freerecall_scores_p1_to_p32.csv --output _mcq_acc/human_mcq_acc_metrics.json
```

To compute the free recall score based on the USE frameworks proposed by paper *Studying memory narratives with natural language processing*
```bash
cd .
python compute_human_free_recall_score.py
```