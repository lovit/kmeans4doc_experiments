data_directory = '/mnt/sdc2/kmeans4doc/dataset/'
experiment_directory = '/mnt/sdc2/kmeans4doc/experiments_v1_20180123/'
package_path = '/mnt/lovit/git/soyclustering/'
beta_array = [0.1, 0.05, 0.02, 0.01]
k_array = [10, 20, 50, 100]

import sys
sys.path.append(package_path)

import pickle
import numpy as np
from glob import glob
from scipy.io import mmread
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from soyclustering import SphericalKMeans
from sklearn.feature_extraction.text import TfidfTransformer

begin_dataset = 5
end_dataset = 7

for data_index in range(begin_dataset, end_dataset+1):

    train_count, total_count = 0, len(beta_array) * len(k_array)

    for beta in beta_array:
        for k in k_array:

            # load dense center
            center_path = '{}/sparsity_test/d{}/beta_{}__k_{}/cluster_centers_dense.pkl'.format(
                experiment_directory, data_index, beta, k)
            with open(center_path, 'rb') as f:
                dense_centers = pickle.load(f)
            
            # load sparse center
            center_path = '{}/sparsity_test/d{}/beta_{}__k_{}/cluster_centers_sparse.pkl'.format(
                experiment_directory, data_index, beta, k)
            with open(center_path, 'rb') as f:
                sparse_centers = pickle.load(f)
            
            dist = np.zeros(k)
            for i in range(k):
                dist[i] = pairwise_distances(
                    dense_centers[i].reshape(1,-1),
                    sparse_centers[i].reshape(1,-1),
                    metric = 'cosine'
                )
            
            # save dist
            dist_path = '{}/sparsity_test/d{}/beta_{}__k_{}/dense_sparse_center_difference.txt'.format(
                experiment_directory, data_index, beta, k)
            np.savetxt(dist_path, dist)
                
            train_count += 1
            print('dataset {}, {} / {} done'.format(data_index, train_count, total_count))

    print('all experiments of dataset {} was done\n\n\n'.format(data_index))
