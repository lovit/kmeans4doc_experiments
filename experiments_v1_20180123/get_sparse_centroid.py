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
from soyclustering import SphericalKMeans
from soyclustering._kmeans import _minimum_df_projections
from sklearn.feature_extraction.text import TfidfTransformer

begin_dataset = 5
end_dataset = 7

for data_index in range(begin_dataset, end_dataset+1):

    print('loading dataset {} ... '.format(data_index), end='', flush=True)

    x_path = '{}/d{}.mtx'.format(data_directory, data_index)
    x = mmread(x_path).tocsr()
    x = TfidfTransformer().fit_transform(x)

    print('done', flush=True)

    train_count, total_count = 0, len(beta_array) * len(k_array)

    for beta in beta_array:
        for k in k_array:
            
            # load labels
            label_path = glob('{}/sparsity_test/d{}/beta_{}__k_{}/*iter10.txt'.format(
                experiment_directory, data_index, beta, k))[0]
            label = np.asarray(np.loadtxt(label_path), dtype=np.int64)
            
            # create dense center
            centers = np.zeros((k, x.shape[1]))
            for i in range(k):
                row_idx = np.where(label == i)[0]
                row = normalize(x[row_idx,:].sum(axis=0))
                if type(row) is not np.ndarray:
                    row = row.todense()
                centers[i] = row
            
            # save dense center
            center_path = '{}/sparsity_test/d{}/beta_{}__k_{}/cluster_centers_dense.pkl'.format(
                experiment_directory, data_index, beta, k)
            with open(center_path, 'wb') as f:
                pickle.dump(centers, f)
            
            # projection
            centers = _minimum_df_projections(x, centers, label, beta)
            
            # save sparse center
            center_path = '{}/sparsity_test/d{}/beta_{}__k_{}/cluster_centers_sparse.pkl'.format(
                experiment_directory, data_index, beta, k)
            with open(center_path, 'wb') as f:
                pickle.dump(centers, f)
                
            train_count += 1
            print('dataset {}, {} / {} done'.format(data_index, train_count, total_count))

    print('all experiments of dataset {} was done\n\n\n'.format(data_index))
