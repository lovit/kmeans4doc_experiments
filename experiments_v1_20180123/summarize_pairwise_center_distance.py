data_directory = '/mnt/sdc2/kmeans4doc/dataset/'
experiment_directory = '/mnt/sdc2/kmeans4doc/experiments_v1_20180123/'
beta_array = [0.1, 0.05, 0.02, 0.01]
k_array = [10, 20, 50, 100]


begin_dataset = 1
end_dataset = 7

output_path = 'pairwise_distance_of_centers.txt'

import pickle
from sklearn.metrics import pairwise_distances

with open(output_path, 'w', encoding='utf-8') as fo:
    for data_index in range(begin_dataset, end_dataset+1):
        for beta in beta_array:
            for k in k_array:
                center_path = '{}/sparsity_test/d{}/beta_{}__k_{}/cluster_centers_dense.pkl'.format(experiment_directory, data_index, beta, k)
                spec = 'beta_{}__k_{}'.format(beta, k)
                with open(center_path, 'rb') as f:
                    center = pickle.load(f)
                average_dist = pairwise_distances(center, metric='cosine').mean()
                
                # dataset, spec, average distance
                fo.write('{}\t{}\t{}\n'.format(data_index, spec, average_dist))

        print('all experiments of dataset {} was done'.format(data_index))