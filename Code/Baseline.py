from lightfm import LightFM
from Dataset import Dataset
from math import log
import numpy as np
import scipy.sparse as sp
import pandas as pd
from time import time
from evaluate import evaluate_model
import argparse

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='so',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    return parser.parse_args()


def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in xrange(num_negatives):
            j = np.random.randint(num_items)
            while train.has_key((u, j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


if __name__ == '__main__':
    args = parse_args()
    num_factors = args.num_factors
    num_negatives = args.num_neg
    learning_rate = args.lr
    epochs = args.epochs
    verbose = args.verbose
    
    
    topK = 10
    evaluation_threads = 1
    
    print("MF arguments: %s" %(args))


    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    # Build model
    model = LightFM(learning_rate=learning_rate, loss='warp')
    
    # Train model
    best_hr, best_ndcg, best_iter = 0, 0, -1
    for epoch in xrange(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        
        # Training
        hist = model.fit_partial(train, epochs=1)
        t2 = time()
        
        # Evaluation
        if epoch %verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    

    
    
    
    
    
    
    
    