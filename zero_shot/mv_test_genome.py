import datasets
from datasets import load_from_disk
import pandas as pd
import os

GTDB_PATH = '/data5/zhanghaohong/download/GTDB/gtdb_genomes_reps_r220/'

test_set = load_from_disk('/home/zhanghaohong/data2/projects/nt_gtdb/datasets/GTDB/toy_dataset_1000g/test')

file_list = pd.read_csv(f'{GTDB_PATH}/genome_paths.tsv', sep=' ', header=None)
file_list.index = file_list[0].apply(lambda x: x.split('_genomic.fna')[0])
file_list['path'] = GTDB_PATH + file_list[1] + file_list[0]

test_files = set()
for genome_id in test_set['GenomeID']:
    if genome_id in file_list.index:
        test_files.add(file_list.loc[genome_id, 'path'])
        
# cpy test genome files to a separate folder
os.makedirs('test_genomes_1000g', exist_ok=True)
for file_path in test_files:
    os.system(f'cp {file_path} test_genomes_1000g/')
    # gunzip the file
    os.system(f'gunzip test_genomes_1000g/{os.path.basename(file_path)}')
