data_directory = '/mnt/sdc2/kmeans4doc/dataset/'
package_path = '/mnt/lovit/git/soyclustering/'
alpha_array = [1.5, 3, 5, 10]
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
    
    train_count, total_count = 0, len(alpha_array) * (len(k_array) + 1)
    
    for alpha in alpha_array:
        for k in k_array:
            debug_directory = './initial_test/d{}/alpha_{}__k_{}/'.format(data_index, alpha, k)
            kmeans = SphericalKMeans(n_clusters = k,
                                     init = 'similar_cut',
                                     max_similar = 0.4,
                                     sample_factor = alpha,
                                     debug_directory = debug_directory,
                                     debug_centroid_on = False,
                                     max_iter = 10
                                    )
            kmeans.fit(x)
            train_count += 1
            print('dataset {}, {} / {} done'.format(data_index, train_count, total_count))
    
    for k in k_array:
        debug_directory = './initial_test/d{}/kmeanspp_{}__k_{}/'.format(data_index, alpha, k)
        kmeans = SphericalKMeans(n_clusters = k,
                                 init = 'kmeans++',
                                 debug_directory = debug_directory,
                                 debug_centroid_on = False,
                                 max_iter = 10
                                )
        kmeans.fit(x)
        train_count += 1
        print('dataset {}, {} / {} done'.format(data_index, train_count, total_count))

