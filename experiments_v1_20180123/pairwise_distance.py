root_directory = '/mnt/sdc2/kmeans4doc/'
data_directory = '{}/dataset/'.format(root_directory)
n_samples = 1000

from scipy.io import mmread
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

begin_dataset = 1
end_dataset = 7

for i in range(begin_dataset, end_dataset+1):
    # exception
    if i == 7:
        n_samples = 80
    else:
        n_samples = 1000

    mtx_path = '{}/d{}.mtx'.format(data_directory, i)
    x = mmread(mtx_path).tocsr()
    x = TfidfTransformer().fit_transform(x).tocsr()
    n_data = x.shape[0]
    idx = np.random.permutation(n_data)[:n_samples]
    
    dist = pairwise_distances(x, x[idx,:], metric='cosine')    
    hist, bin_edges = np.histogram(dist, bins=20)
    hist_path = './pairwise_distance/d{}.txt'.format(i)
    with open(hist_path, 'w', encoding='utf-8') as f:
        for h, b, e in zip(hist, bin_edges, bin_edges[1:]):
            f.write('{} - {}\t{}\n'.format(b, e, h))
    print('done dataset {}'.format(i))
print('everything was done')
