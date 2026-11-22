# Robustness variants vs main-table baselines (ROUGE F1 x100, no stemming; R-L = Lsum)


## cnn_dailymail

| Variant | R-1 | R-2 | R-L | change vs main (R-1 / R-2 / R-L) |
|---|---|---|---|---|
| TextRank (tf-idf) [main table] | 33.14 | 13.09 | 29.89 |  |
| TextRank (cosine) | 34.27 | 13.79 | 31.00 | +1.13 / +0.70 / +1.12 |
| TextRank (Mihalcea 2004) | 31.97 | 11.79 | 28.65 | -1.18 / -1.31 / -1.24 |
| DegreeCentrality (tf-idf) [main table] | 32.96 | 13.07 | 29.73 |  |
| DegreeCentrality (cosine) | 33.56 | 13.49 | 30.37 | +0.60 / +0.42 / +0.63 |
| LexPageRank [main table] | 33.58 | 12.82 | 30.48 |  |
| LexPageRank (document idf) | 37.39 | 16.17 | 34.07 | +3.82 / +3.35 / +3.59 |
| PacSum (tf-idf) [main table] | 38.67 | 16.91 | 35.20 |  |
| PacSum (cosine) | 38.54 | 16.85 | 35.14 | -0.13 / -0.06 / -0.06 |

## pubmed

| Variant | R-1 | R-2 | R-L | change vs main (R-1 / R-2 / R-L) |
|---|---|---|---|---|
| TextRank (tf-idf) [main table] | 34.20 | 10.51 | 29.06 |  |
| TextRank (cosine) | 38.48 | 14.03 | 33.54 | +4.27 / +3.51 / +4.48 |
| TextRank (Mihalcea 2004) | 37.64 | 12.41 | 32.12 | +3.44 / +1.89 / +3.05 |
| DegreeCentrality (tf-idf) [main table] | 33.86 | 10.39 | 28.79 |  |
| DegreeCentrality (cosine) | 37.91 | 13.77 | 33.10 | +4.05 / +3.38 / +4.31 |
| LexPageRank [main table] | 40.24 | 14.87 | 35.56 |  |
| LexPageRank (document idf) | 39.59 | 14.18 | 34.52 | -0.65 / -0.69 / -1.04 |
| PacSum (tf-idf) [main table] | 36.40 | 12.84 | 32.10 |  |
| PacSum (cosine) | 39.63 | 15.26 | 35.36 | +3.23 / +2.42 / +3.26 |

## arxiv

| Variant | R-1 | R-2 | R-L | change vs main (R-1 / R-2 / R-L) |
|---|---|---|---|---|
| TextRank (tf-idf) [main table] | 29.47 | 7.44 | 24.66 |  |
| TextRank (cosine) | 34.39 | 10.04 | 29.56 | +4.92 / +2.60 / +4.90 |
| TextRank (Mihalcea 2004) | 35.36 | 9.73 | 29.61 | +5.89 / +2.29 / +4.95 |
| DegreeCentrality (tf-idf) [main table] | 29.10 | 7.33 | 24.38 |  |
| DegreeCentrality (cosine) | 33.37 | 9.68 | 28.77 | +4.27 / +2.35 / +4.39 |
| LexPageRank [main table] | 37.93 | 12.21 | 32.85 |  |
| LexPageRank (document idf) | 37.65 | 11.89 | 32.34 | -0.28 / -0.32 / -0.51 |
| PacSum (tf-idf) [main table] | 34.31 | 9.41 | 30.05 |  |
| PacSum (cosine) | 37.23 | 11.73 | 32.49 | +2.92 / +2.31 / +2.45 |

## billsum

| Variant | R-1 | R-2 | R-L | change vs main (R-1 / R-2 / R-L) |
|---|---|---|---|---|
| TextRank (tf-idf) [main table] | 38.63 | 16.39 | 28.66 |  |
| TextRank (cosine) | 39.88 | 18.21 | 30.35 | +1.25 / +1.82 / +1.69 |
| TextRank (Mihalcea 2004) | 39.28 | 17.00 | 29.27 | +0.65 / +0.61 / +0.61 |
| DegreeCentrality (tf-idf) [main table] | 38.48 | 16.26 | 28.53 |  |
| DegreeCentrality (cosine) | 39.00 | 17.67 | 29.62 | +0.52 / +1.40 / +1.09 |
| LexPageRank [main table] | 40.57 | 18.44 | 31.23 |  |
| LexPageRank (document idf) | 40.19 | 17.90 | 30.52 | -0.38 / -0.54 / -0.72 |
| PacSum (tf-idf) [main table] | 39.35 | 17.30 | 29.80 |  |
| PacSum (cosine) | 40.16 | 18.40 | 31.29 | +0.81 / +1.10 / +1.49 |

## wikihow

| Variant | R-1 | R-2 | R-L | change vs main (R-1 / R-2 / R-L) |
|---|---|---|---|---|
| TextRank (tf-idf) [main table] | 23.69 | 5.51 | 21.59 |  |
| TextRank (cosine) | 24.63 | 6.04 | 22.71 | +0.93 / +0.53 / +1.12 |
| TextRank (Mihalcea 2004) | 24.23 | 5.67 | 21.97 | +0.54 / +0.16 / +0.38 |
| DegreeCentrality (tf-idf) [main table] | 23.56 | 5.48 | 21.48 |  |
| DegreeCentrality (cosine) | 24.39 | 5.98 | 22.52 | +0.82 / +0.49 / +1.03 |
| LexPageRank [main table] | 24.91 | 6.13 | 22.95 |  |
| LexPageRank (document idf) | 24.66 | 5.94 | 22.63 | -0.25 / -0.20 / -0.32 |
| PacSum (tf-idf) [main table] | 23.53 | 5.34 | 21.67 |  |
| PacSum (cosine) | 24.37 | 5.85 | 22.53 | +0.84 / +0.51 / +0.86 |

## tldr17

| Variant | R-1 | R-2 | R-L | change vs main (R-1 / R-2 / R-L) |
|---|---|---|---|---|
| TextRank (tf-idf) [main table] | 15.41 | 2.65 | 11.49 |  |
| TextRank (cosine) | 14.89 | 2.73 | 11.44 | -0.52 / +0.07 / -0.05 |
| TextRank (Mihalcea 2004) | 16.43 | 2.65 | 11.91 | +1.03 / -0.00 / +0.42 |
| DegreeCentrality (tf-idf) [main table] | 15.38 | 2.66 | 11.48 |  |
| DegreeCentrality (cosine) | 14.59 | 2.70 | 11.29 | -0.79 / +0.05 / -0.19 |
| LexPageRank [main table] | 14.66 | 2.69 | 11.32 |  |
| LexPageRank (document idf) | 14.98 | 2.66 | 11.41 | +0.31 / -0.03 / +0.09 |
| PacSum (tf-idf) [main table] | 14.75 | 2.52 | 11.23 |  |
| PacSum (cosine) | 13.96 | 2.55 | 10.95 | -0.79 / +0.03 / -0.29 |

## Effect of PacSum's directionality on the cosine graph (ROUGE-1)

| Dataset | DegreeCentrality (cosine, no direction) | PacSum (cosine, directed) | direction effect | LexPageRank | PacSum (cosine) vs LexPageRank | tuned (beta, lambda1) |
|---|---|---|---|---|---|---|
| cnn_dailymail | 33.56 | 38.54 | +4.99 | 33.58 | +4.97 | (0.0, 0.6) |
| pubmed | 37.91 | 39.63 | +1.72 | 40.24 | -0.61 | (0.0, 0.4) |
| arxiv | 33.37 | 37.23 | +3.85 | 37.93 | -0.70 | (0.0, 0.6) |
| billsum | 39.00 | 40.16 | +1.16 | 40.57 | -0.41 | (0.0, 0.2) |
| wikihow | 24.39 | 24.37 | -0.02 | 24.91 | -0.54 | (0.0, 0.1) |
| tldr17 | 14.59 | 13.96 | -0.64 | 14.66 | -0.71 | (0.0, 0.0) |
