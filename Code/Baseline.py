from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
from lightfm.evaluation import recall_at_k
from lightfm.evaluation import reciprocal_rank

from lightfm.datasets import fetch_movielens
from lightfm.data import Dataset

from math import log
import numpy as np
from scipy import sparse
from scipy.sparse import identity
from operator import itemgetter
import pandas as pd


def ndcg_at_k(model, user_ids, item_ids, actual_scores, item_features=None, user_features=None, k=10):
    pred_scores = []
    ndcg_k = []
    for user in user_ids:
        pred_score = model.predict(user_ids=user_ids,
                                   item_ids=item_ids)
        pred_score = (pred_score / sum(pred_score)) * 10000
        pred_scores.append(pred_score.tolist())

    matrix = np.array(pred_scores)
    pred_scores = sparse.csr_matrix(matrix)
    scores = []
    real_scores = []
    actual_scores = actual_scores.tocsr()
    for i in range(0, 943):
        for j in range(0, 1682):
            scores.append((j, pred_scores[i, j]))
            try:
                real_scores.append((j, actual_scores[i, j]))
            except:
                real_scores.append((j, 0))

        scores.sort(key=itemgetter(1), reverse=True)
        real_scores.sort(key=itemgetter(1), reverse=True)
        total = 0
        ideal_total = 0
        for n in range(0, k):
            total += real_scores[scores[n][0]][1] / log(n + 2, 2)
            ideal_total += real_scores[n][1] / log(n + 2, 2)

        if ideal_total != 0:
            ndcg_k.append(total / ideal_total)

    return ndcg_k


movielens = fetch_movielens(indicator_features=True, genre_features=True)

train = movielens['train']
test = movielens['test']

row = []
col = []
val = []
file = open('data/ml-100k/u.data', encoding="ISO-8859-1")
for line in file:
    interaction = []
    data = line.split('\t')
    row.append(int(data[0]))
    col.append(int(data[1]))
    val.append(int(data[2]))

actual_data = sparse.coo_matrix((val, (row, col)), dtype=np.float32)

model = LightFM(learning_rate=0.1, loss='warp')
model.fit(train, epochs=10)

train_precision = precision_at_k(model, train, k=10).mean()
test_precision = precision_at_k(model, test, k=10).mean()

train_recall = recall_at_k(model, train, k=10).mean()
test_recall = recall_at_k(model, test, k=10).mean()

train_auc = auc_score(model, train).mean()
test_auc = auc_score(model, test).mean()

print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('Recall: train %.2f, test %.2f.' % (train_recall, test_recall))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

ndcg = ndcg_at_k(model,
                 user_ids=np.array([x for x in range(0, 943)]),
                 item_ids=np.array([x for x in range(0, 1682)]),
                 actual_scores=actual_data,
                 k=10)

print(ndcg)

# print("-----------------------")