data_directory = '/mnt/sdc2/kmeans4doc/dataset/'
package_path = '/mnt/lovit/git/soyclustering/'
beta_array = [0.1, 0.05, 0.02, 0.01]
k_array = [10, 20, 50, 100]

import sys
sys.path.append(package_path)

from scipy.io import mmread
from soyclustering import SphericalKMeans
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
            debug_directory = './sparsity_test/d{}/beta_{}__k_{}/'.format(data_index, beta, k)
            kmeans = SphericalKMeans(n_clusters = k,
                                     init = 'similar_cut',
                                     max_similar = 0.4,
                                     sample_factor = 3,
                                     debug_directory = debug_directory,
                                     debug_centroid_on = False,
                                     max_iter = 10,
                                     sparsity='minimum_df',
                                     minimum_df_factor = beta
                                    )
            kmeans.fit(x)
            train_count += 1
            print('dataset {}, {} / {} done'.format(data_index, train_count, total_count))

    print('all experiments of dataset {} was done\n\n\n'.format(data_index))
