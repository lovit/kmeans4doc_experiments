data_directory = '/mnt/sdc2/kmeans4doc/dataset/'
package_path = '/mnt/lovit/git/soyclustering/'
experiment_directory = '/mnt/sdc2/kmeans4doc/experiments_v1_20180123/'
beta_array = [0.1, 0.05, 0.02, 0.01]
k_array = [10, 20, 50, 100]

begin_dataset = 1
end_dataset = 7

output_path = 'sparsity_test_result.txt'


from glob import glob
import numpy as np
import pickle

with open(output_path, 'w', encoding='utf-8') as fo:

    for data_index in range(begin_dataset, end_dataset+1):

        log_paths = glob('{}/sparsity_test/d{}/*/*logs.txt'.format(experiment_directory, data_index))

        for log_path in log_paths:

            spec = log_path.split('/')[-2]

            with open(log_path, encoding='utf-8') as f:
                for _ in range(10):
                    next(f)
                sparsity = next(f).strip().split('sparsity=')[1]

            log_path = '/'.join(log_path.split('/')[:-1])+'/dense_sparse_center_difference.txt'
            diff = np.loadtxt(log_path)
            average_difference = diff.mean()
            
            center_path = '/'.join(log_path.split('/')[:-1])+'/cluster_centers_dense.pkl'
            with open(center_path, 'rb') as f:
                centers = pickle.load(f)

            p_nnz_of_dense = centers.nonzero()[0].shape[0] / (centers.shape[0] * centers.shape[1])

            # percentage of nonzero elements, dense-sparse centroid difference, p_nnz_of_dense
            fo.write('{}\t{}\t{}\t{}\t{}\n'.format(data_index, spec, sparsity, average_difference, p_nnz_of_dense))

        print('done with dataset = {}'.format(data_index))