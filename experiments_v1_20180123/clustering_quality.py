from glob import glob
import pickle
import numpy as np
from scipy.io import mmread
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin_min



def load_labels(path):
    with open(path, encoding='utf-8') as f:
        labels = np.asarray([int(doc.strip()) for doc in f])
    return labels

def create_centroids(x, labels):
    n_clusters = np.unique(labels).shape[0]
    centroids = np.zeros((n_clusters, x.shape[1]))
    for idx in range(n_clusters):
        xsub = x[np.where(labels == idx)[0]]
        centroids[idx] = np.asarray(xsub.sum(axis=0))
    return centroids

def load_centroids(path):
    with open(path, 'rb') as f:
        centroids = pickle.load(f)
    return centroids

def get_setting(path):
    return path.split('/')[-2]

def write(path, message):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(message)
    print(message)

def intra_inter_cluster_distance(x, labels, centroids):
    n_clusters = np.unique(labels).shape[0]
    intra = 0
    inter = 0
    for idx in range(n_clusters):
        xsub = x[np.where(labels == idx)[0]]
        intra += pairwise_distances(xsub,
            centroids[idx].reshape(1,-1), metric='cosine').sum()
        for oidx in range(n_clusters):
            if oidx == idx:
                continue
            inter += pairwise_distances(xsub,
                centroids[oidx].reshape(1,-1), metric='cosine').sum()
        print('\rintra, inter distance {}/{} ...'.format(idx, n_clusters), end='')
    print('\rintra, inter distance was done {0}/{0} ...'.format(n_clusters))
    intra /= x.shape[0]
    inter /= (x.shape[0] * (n_clusters - 1))
    return intra, inter

def silhouette_index(x, labels, centroids):
    n_clusters = np.unique(labels).shape[0]
    silhouette = 0
    for idx in range(n_clusters):
        xsub = x[np.where(labels == idx)[0]]
        c = centroids[idx].reshape(1,-1)
        idx_ = np.asarray([i for i in range(n_clusters) if i != idx])
        c_ = centroids[idx_]
        a = pairwise_distances(xsub, c, metric='cosine').reshape(-1)
        _, b = pairwise_distances_argmin_min(xsub, c_, metric='cosine')
        abmax = np.hstack([a.reshape(-1,1), b.reshape(-1,1)]).max(axis=1)
        silhouette += ((b - a) / abmax).sum()
        print('\rsilhouette {}/{} ...'.format(idx, n_clusters), end='')
    print('\rsilhouette was done {0}/{0} ...'.format(n_clusters))
    return silhouette / x.shape[0]

def initializer_test():
    expname = 'initial_test'
    table_path = 'clustering_quality_initializer.txt'
    header = '\t'.join(['dataname', 'setting', 'intra', 'inter', 'silhouette']) + '\n'
    write(table_path, header)

    for didx in range(1, 8):
        dataname = 'd%d' % didx
        datapath = '../dataset/{}.mtx'.format(dataname)
        x = mmread(datapath).tocsr()
        label_paths = glob('{}/{}/*/*iter10.txt'.format(expname, dataname))

        for i, label_path in enumerate(label_paths):
            print('Initializer setting {} / {}'.format(i+1, len(label_paths)))
            setting = get_setting(label_path)

            labels = load_labels(label_path)
            centroids = create_centroids(x, labels)

            intra, inter = intra_inter_cluster_distance(x, labels, centroids)
            silhouette = silhouette_index(x, labels, centroids)

            message = '{}\t{}\t{}\t{}\t{}\n'.format(dataname, setting, intra, inter, silhouette)
            write(table_path, message)
        #break # break for dataset

def sparsity_test():
    expname = 'sparsity_test'
    table_path = 'clustering_quality_sparsity.txt'
    header = '\t'.join(['dataname', 'setting', 'intra', 'inter', 'silhouette', 'centroid_silhouette']) + '\n'
    write(table_path, header)

    for didx in range(1, 8):
        dataname = 'd%d' % didx
        datapath = '../dataset/{}.mtx'.format(dataname)
        x = mmread(datapath).tocsr()
        label_paths = glob('{}/{}/*/*iter10.txt'.format(expname, dataname))

        for i, label_path in enumerate(label_paths):
            print('Sparsity setting {} / {}'.format(i+1, len(label_paths)))
            setting = get_setting(label_path)

            labels = load_labels(label_path)
            centroids = create_centroids(x, labels)

            intra, inter = intra_inter_cluster_distance(x, labels, centroids)
            silhouette = silhouette_index(x, labels, centroids)

            centroid_path = label_path.rsplit('/', 1)[0] + '/cluster_centers_sparse.pkl'
            centroids = load_centroids(centroid_path)
            centroid_silhouette = silhouette_index(x, labels, centroids)

            message = '{}\t{}\t{}\t{}\t{}\t{}\n'.format(dataname, setting, intra, inter, silhouette, centroid_silhouette)
            write(table_path, message)
        #break # break for dataset

def main():
    initializer_test()
    sparsity_test()

if __name__ == '__main__':
    main()
