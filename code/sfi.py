"""
Filename: sfi.py
Author: Quang Minh Nguyen
Description: the implementation of the Segmented Frequent Itemsets (SFI) 
algorithm. To run the code, run the following command:

python sfi.py dataset/user_itemset_training.csv dataset/user_itemset_test_query.csv

"""


# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt 
# sparse matrix processing
from scipy.sparse import coo_matrix, csr_matrix, bmat
from scipy.sparse.linalg import svds
# clustering
from sklearn.cluster import KMeans
# general
import sys
from collections import defaultdict
from time import time


def get_SVD_embeddings(matrix, k):
    """
    Get the SVD embeddings for users and items.

    Parameters:
        matrix: np.array, the matrix to be factorized
        k: int, number of latent factors
    
    Returns:
        U: np.array, user embeddings
        V: np.array, itemset embeddings
    """
    U, _, V = svds(matrix, k=k)
    return U, V


def get_frequent_itemsets(user_itemset):
    """
    Get a list of itemsets sorted by their number
    of appearances in user_itemset.

    Parameters:
        user_itemset: np.array, each row is (user, itemset)

    Returns:
        frequent_itemsets: list[int], a list of itemsets
        frequent_itemsets_count: list[int], a list of counts
    """
    user_itemset = np.array(user_itemset)
    unique_itemsets, counts = np.unique(user_itemset[:, 1], return_counts=True)
    itemset_count = np.array(sorted(list(zip(unique_itemsets, counts)), key=lambda x: x[1], reverse=True))
    frequent_itemsets = itemset_count[:, 0]
    frequent_itemsets_count = itemset_count[:, 1]
    return frequent_itemsets, frequent_itemsets_count


def get_clusters(n_clusters, U):
    """
    Perform k-means clustering given user embeddings U.

    Parameters:
        n_clusters: int, number of clusters
        U: np.array, user embeddings

    Returns:
        cluster_user: dict[int, list[int]], cluster_user[i] 
            is a list of user ids in cluster i
        user_cluster: dict[int, int], user_cluster[i] is the 
            cluster id of user i
    """
    # cluster the users
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(U)
    # get the cluster labels
    user_cluster = kmeans.labels_
    # get inverse mapping
    cluster_user = {}
    for i in range(n_clusters):
        cluster_user[i] = []
    for i in range(len(user_cluster)):
        cluster_user[user_cluster[i]].append(i)
    return cluster_user, user_cluster


if __name__=='__main__':

    # hyperparameters
    N_EMBED = 8
    N_CLUSTER = 15
    N_FI = 6000

    # get data from command line
    user_itemset_training_path = sys.argv[1]
    user_itemset_test_query_path = sys.argv[2]

    # import the data
    user_itemset_training = np.loadtxt(user_itemset_training_path, delimiter=',', dtype=int)
    user_itemset_test_query = np.loadtxt(user_itemset_test_query_path, delimiter=',', dtype=int)

    # the sparse utility matrix
    sparse_user_itemset = coo_matrix(
        ([1]*len(user_itemset_training), 
        (user_itemset_training[:, 0], user_itemset_training[:, 1])), 
        dtype=float)
    sparse_user_itemset_csr = sparse_user_itemset.tocsr()  # in csr format
    sparse_user_itemset_csc = sparse_user_itemset.tocsc()  # in csc format

    # get the SVD embeddings
    U, V = get_SVD_embeddings(sparse_user_itemset_csr, N_EMBED)
    # get the clusters
    cluster_user, user_cluster = get_clusters(N_CLUSTER, U)
    # get the itemsets relevant to each cluster
    cluster_user_itemset = defaultdict(list)
    for user, itemset in user_itemset_training:
        cluster_user_itemset[user_cluster[user]].append((user, itemset))
    # get the top itemsets for each cluster
    cluster_frequent_itemsets = defaultdict(list)
    for i in range(N_CLUSTER):
        cluster_frequent_itemsets[i], _ = get_frequent_itemsets(cluster_user_itemset[i])
    user_itemset_test_prediction = np.zeros(len(user_itemset_test_query))
    for j in range(len(user_itemset_test_query)):
        user, itemset = user_itemset_test_query[j]
        if itemset in cluster_frequent_itemsets[user_cluster[user]][:N_FI]:
            user_itemset_test_prediction[j] = 1
    
    # export the prediction to csv
    np.savetxt('result/user_itemset_test_prediction.csv', user_itemset_test_prediction, delimiter=',', fmt='%d')
