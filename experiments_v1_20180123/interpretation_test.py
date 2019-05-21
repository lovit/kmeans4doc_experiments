def main():
    ##########
    # import #
    import pickle
    import sys
    sys.path.append('/mnt/lovit/git/clustering4docs/')

    try:
        from soyclustering import SphericalKMeans
        from scipy.io import mmread
        from sklearn.feature_extraction.text import TfidfTransformer
    except:
        print('Import failed. Terminated')


    ####################
    # training k-means #

    # (data, k, beta)
    configurations = [
        # ('d1', 100, 0.1), # a6
        # ('d2', 100, 0.1), # tuscon
        # ('d3', 500, 0.1), # sonata
        # ('d4', 1000, 0.1) # imdb
        # ('d5', 1000, 0.1), # reuter
        # d6 ignore, movielens
        # ('d7', 1000, 0.05) # yelp
    ]

    data_dir = '/mnt/sdc2/kmeans4doc/dataset/'
    result_dir = '/mnt/sdc2/kmeans4doc/experiments_v1_20180123/interpretation_test/'

    print('Begin kmeans', flush=True)

    for data, k, beta in configurations:

        spherical_kmeans = SphericalKMeans(
            n_clusters = k,
            max_iter = 10, #max_iter = 10,
            verbose = 1,
            init = 'similar_cut',
            sparsity = 'minimum_df', 
            minimum_df_factor = beta
        )

        x_path = '{}{}.mtx'.format(data_dir, data)
        x = mmread(x_path).tocsr()
        x = TfidfTransformer().fit_transform(x)

        labels = spherical_kmeans.fit_predict(x)
        labels_path = '{}{}_k{}_label.txt'.format(result_dir, data, k)
        with open(labels_path, 'w', encoding='utf-8') as f:
            for label in labels:
                f.write('{}\n'.format(label))

        centers = spherical_kmeans.cluster_centers_
        centers_path = '{}{}_k{}_center.pkl'.format(result_dir, data, k)
        with open(centers_path, 'wb') as f:
            pickle.dump(centers, f)

        print('K-MEANS DONE with {}'.format(data), end='\n\n', flush=True)


    #######################
    # clustering labeling #

    print('begin interpretation')

    from soyclustering import proportion_keywords

    configurations = [
        ('d1', 100),
        ('d2', 100),
        ('d3', 500),
        ('d4', 1000),
        # ('d5', 1000), # RCV1 released as vectorized version
        ('d7', 1000),
        ('d7', 500)
    ]

    for data, k in configurations:

        labels_path = '{}{}_k{}_label.txt'.format(result_dir, data, k)
        with open(labels_path, encoding='utf-8') as f:
            labels = [doc.strip() for doc in f]

        centers_path = '{}{}_k{}_center.pkl'.format(result_dir, data, k)
        with open(centers_path, 'rb') as f:
            centers = pickle.load(f)

        vocabs_path = '{}{}.vocab'.format(data_dir, data)
        with open(vocabs_path, encoding='utf-8') as f:
            index2word = [word.strip() for word in f]

        keywords = proportion_keywords(centers, labels, index2word=index2word, topk=40, candidates_topk=100)    
        keywords_path = '{}{}_k{}_keywords.txt'.format(result_dir, data, k)
        with open(keywords_path, 'w', encoding='utf-8') as f:
            for keyword in keywords:
                keyword = [word for word, _ in keyword]
                keyword_strf = ' '.join(keyword)
                f.write('{}\n'.format(keyword_strf))

        print('DONE LABELING {}'.format(data))
    print('everything was done')

if __name__ == '__main__':
    main()
