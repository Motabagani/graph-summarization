# Baseline results (ROUGE F1 x100, no stemming)


## cnn_dailymail

| Method | R-1 | R-2 | R-L | R-Lsum | docs | length | tuned |
|---|---|---|---|---|---|---|---|
| Lead | 38.65 | 17.00 | 24.44 | 35.26 | 11490 | 3 sent |  |
| TextRank (tf-idf) | 33.14 | 13.09 | 20.41 | 29.89 | 11490 | 3 sent |  |
| DegreeCentrality (tf-idf) | 32.96 | 13.07 | 20.45 | 29.73 | 11490 | 3 sent |  |
| LexPageRank | 33.58 | 12.82 | 21.13 | 30.48 | 11490 | 3 sent | {'threshold': 0.3} |
| PacSum (tf-idf) | 38.67 | 16.91 | 23.80 | 35.20 | 11490 | 3 sent | {'beta': 0.0, 'lambda1': 0.7, 'lambda2': 0.30000000000000004} |
| Oracle | 56.06 | 31.21 | 38.46 | 51.66 | 11490 | 3 sent |  |

## pubmed

| Method | R-1 | R-2 | R-L | R-Lsum | docs | length | tuned |
|---|---|---|---|---|---|---|---|
| Lead | 35.22 | 12.06 | 19.52 | 31.77 | 6658 | 203 words |  |
| TextRank (tf-idf) | 34.20 | 10.51 | 17.63 | 29.06 | 6658 | 203 words |  |
| DegreeCentrality (tf-idf) | 33.86 | 10.39 | 17.56 | 28.79 | 6658 | 203 words |  |
| LexPageRank | 40.24 | 14.87 | 20.68 | 35.56 | 6658 | 203 words | {'threshold': 0.2} |
| PacSum (tf-idf) | 36.40 | 12.84 | 18.75 | 32.10 | 6658 | 203 words | {'beta': 0.0, 'lambda1': 0.7, 'lambda2': 0.30000000000000004} |
| Oracle | 55.93 | 34.02 | 36.65 | 51.79 | 6658 | 203 words |  |

## arxiv

| Method | R-1 | R-2 | R-L | R-Lsum | docs | length | tuned |
|---|---|---|---|---|---|---|---|
| Lead | 33.73 | 8.91 | 17.26 | 29.82 | 6440 | 220 words |  |
| TextRank (tf-idf) | 29.47 | 7.44 | 15.23 | 24.66 | 6440 | 220 words |  |
| DegreeCentrality (tf-idf) | 29.10 | 7.33 | 15.08 | 24.38 | 6440 | 220 words |  |
| LexPageRank | 37.93 | 12.21 | 19.40 | 32.85 | 6440 | 220 words | {'threshold': 0.1} |
| PacSum (tf-idf) | 34.31 | 9.41 | 17.10 | 30.05 | 6440 | 220 words | {'beta': 0.0, 'lambda1': 0.9, 'lambda2': 0.09999999999999998} |
| Oracle | 55.27 | 31.18 | 31.05 | 50.38 | 6440 | 220 words |  |

## billsum

| Method | R-1 | R-2 | R-L | R-Lsum | docs | length | tuned |
|---|---|---|---|---|---|---|---|
| Lead | 38.94 | 17.86 | 24.55 | 31.14 | 3269 | 184 words |  |
| TextRank (tf-idf) | 38.63 | 16.39 | 22.97 | 28.66 | 3269 | 184 words |  |
| DegreeCentrality (tf-idf) | 38.48 | 16.26 | 22.89 | 28.53 | 3269 | 184 words |  |
| LexPageRank | 40.57 | 18.44 | 23.72 | 31.23 | 3269 | 184 words | {'threshold': 0.2} |
| PacSum (tf-idf) | 39.35 | 17.30 | 23.80 | 29.80 | 3269 | 184 words | {'beta': 0.0, 'lambda1': 0.0, 'lambda2': 1.0} |
| Oracle | 58.96 | 38.44 | 43.30 | 49.56 | 3269 | 184 words |  |

## wikihow

| Method | R-1 | R-2 | R-L | R-Lsum | docs | length | tuned |
|---|---|---|---|---|---|---|---|
| Lead | 22.57 | 4.99 | 14.56 | 20.94 | 213294 | 55 words |  |
| TextRank (tf-idf) | 23.69 | 5.51 | 15.02 | 21.59 | 213294 | 55 words |  |
| DegreeCentrality (tf-idf) | 23.56 | 5.48 | 14.99 | 21.48 | 213294 | 55 words |  |
| LexPageRank | 24.91 | 6.13 | 15.77 | 22.95 | 213294 | 55 words | {'threshold': 0.1} |
| PacSum (tf-idf) | 23.53 | 5.34 | 14.83 | 21.67 | 213294 | 55 words | {'beta': 0.1, 'lambda1': 0.7, 'lambda2': 0.30000000000000004} |
| Oracle | 35.47 | 12.40 | 23.02 | 32.76 | 213294 | 55 words |  |

## tldr17

| Method | R-1 | R-2 | R-L | R-Lsum | docs | length | tuned |
|---|---|---|---|---|---|---|---|
| Lead | 12.30 | 2.29 | 9.74 | 9.92 | 3847330 | 1 sent |  |
| TextRank (tf-idf) | 15.41 | 2.65 | 11.20 | 11.49 | 3847330 | 1 sent |  |
| DegreeCentrality (tf-idf) | 15.38 | 2.66 | 11.19 | 11.48 | 3847330 | 1 sent |  |
| LexPageRank | 14.66 | 2.69 | 11.09 | 11.32 | 3847330 | 1 sent | {'threshold': 0.1} |
| PacSum (tf-idf) | 14.75 | 2.52 | 10.98 | 11.23 | 3847330 | 1 sent | {'beta': 0.0, 'lambda1': 0.1, 'lambda2': 0.9} |
| Oracle | 27.03 | 9.14 | 20.84 | 21.24 | 3847330 | 1 sent |  |
