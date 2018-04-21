experiment_directory = '/mnt/sdc2/kmeans4doc/experiments_v1_20180123/'

import numpy as np
from glob import glob

begin_dataset = 5
end_dataset = 7

for data_index in range(begin_dataset, end_dataset+1):
    
    diff_paths = glob('{}/sparsity_test/d{}/*/dense_sparse_center_difference.txt'.format(
        experiment_directory, data_index))
    
    summary_path = '{}/sparsity_diff/d{}.txt'.format(experiment_directory, data_index)
    with open(summary_path, 'w', encoding='utf-8') as f:    
        for path in diff_paths:
            diffs = np.loadtxt(path)
            max_diff = diffs.max()
            settings = path.split('/')[-2]
            f.write('{}\t{}\n'.format(settings, max_diff))
    
    print('data_index = {} done'.format(data_index, max_diff))
