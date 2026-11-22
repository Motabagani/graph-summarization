# Alignment with published results (ours minus published; ROUGE F1 x100, no stemming)

| Baseline | Dataset | Source | Ours (R-1 / R-2 / R-L) | Published | Difference | Mean abs. diff |
|---|---|---|---|---|---|---|
| PacSum (unnormalized) | cnn_dailymail | PacSum, Table 2 | 38.67 / 16.91 / 35.20 | 39.20 / 16.30 / 35.30 | -0.53 / +0.61 / -0.10 | 0.41 |
| TextRank (unnormalized) | cnn_dailymail | PacSum, Table 2 | 33.14 / 13.09 / 29.89 | 33.20 / 11.80 / 29.60 | -0.06 / +1.29 / +0.29 | 0.55 |
| DegreeCentrality (unnormalized) | cnn_dailymail | PacSum, Table 2 | 32.96 / 13.07 / 29.73 | 33.00 / 11.70 / 29.50 | -0.04 / +1.37 / +0.23 | 0.55 |
| Lead | cnn_dailymail | PacSum, Table 2 | 38.65 / 17.00 / 35.26 | 40.50 / 17.70 / 36.70 | -1.85 / -0.70 / -1.44 | 0.67 |
| Lead | pubmed | HipoRank, Table 3 | 35.22 / 12.06 / n/c | 35.63 / 12.28 / n/c | -0.41 / -0.22 / n/c | 0.67 |
| Lead | arxiv | HipoRank, Table 4 | 33.73 / 8.91 / n/c | 33.66 / 8.94 / n/c | +0.07 / -0.03 / n/c | 0.67 |
| LexPageRank (corpus idf) | pubmed | HipoRank (LexRank), Table 3 | 40.24 / 14.87 / n/c | 39.19 / 13.89 / n/c | +1.05 / +0.98 / n/c | 1.89 |
| LexPageRank (corpus idf) | arxiv | HipoRank (LexRank), Table 4 | 37.93 / 12.21 / n/c | 33.85 / 10.73 / n/c | +4.08 / +1.48 / n/c | 1.89 |
| Oracle | cnn_dailymail | PacSum, Table 2 | 56.06 / 31.21 / 51.66 | 54.70 / 30.40 / 50.80 | +1.36 / +0.81 / +0.86 | 2.85 |
| Oracle | pubmed | HipoRank, Table 3 | 55.93 / 34.02 / n/c | 55.05 / 27.48 / n/c | +0.88 / +6.54 / n/c | 2.85 |
| Oracle | arxiv | HipoRank, Table 4 | 55.27 / 31.18 / n/c | 53.88 / 23.05 / n/c | +1.39 / +8.13 / n/c | 2.85 |
