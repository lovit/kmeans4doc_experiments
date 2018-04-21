data_directory = '/mnt/sdc2/kmeans4doc/dataset/'
package_path = '/mnt/lovit/git/soyclustering/'
experiment_directory = '/mnt/sdc2/kmeans4doc/experiments_v1_20180123/'
alpha_array = [1.5, 3, 5, 10]
k_array = [10, 20, 50, 100]

begin_dataset = 1
end_dataset = 7

output_path = 'initialize_test_result.txt'


from glob import glob

with open(output_path, 'w', encoding='utf-8') as fo:

    for data_index in range(begin_dataset, end_dataset+1):

        log_paths = glob('{}/initial_test/d{}/*/*logs.txt'.format(experiment_directory, data_index))

        for log_path in log_paths:
            
            spec = log_path.split('/')[-2]
            with open(log_path, encoding='utf-8') as f:
                initial_time = next(f).split('=')[1].split()[0]
                for _ in range(9):
                    next(f)
                n_changed_at_10 = next(f).split('changed=')[1].split(',')[0]

            # dataset, spec, initial time, n changes at iter 10
            fo.write('{}\t{}\t{}\t{}\n'.format(data_index, spec, initial_time, n_changed_at_10))

        print('done with dataset = {}'.format(data_index))