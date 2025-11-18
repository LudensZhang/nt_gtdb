import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from Bio import SeqIO
# open gzipped fasta file
import os
import gzip
from tqdm import tqdm
# argparse
import argparse

WINDOW_SIZE = 6000 # num of nucleotides in each sample

def create_samples_from_genome(genome_file, window_size=WINDOW_SIZE):
    # return a datasets samples object
    # each sample has 'GenomeID', 'ContigID', 'Start', 'End', 'Sequence'
    samples = []
    nts = 0
    with gzip.open(genome_file, 'rt') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            contig_id = record.id
            sequence = str(record.seq)
            seq_length = len(sequence)
            for start in range(0, seq_length, window_size):
                end = min(start + window_size, seq_length)
                sample_seq = sequence[start:end]
                sample = {
                    'GenomeID': genome_file.split('/')[-1].split('_genomic')[0],
                    'ContigID': contig_id,
                    'Start': start,
                    'End': end,
                    'Sequence': sample_seq
                }
                samples.append(sample)
                nts += len(sample_seq)
    return datasets.Dataset.from_list(samples), nts

def create_datasets_from_genomes(genome_files, window_size=WINDOW_SIZE):
    all_samples = []
    nts = 0
    for genome_file in tqdm(genome_files):
        samples, genome_nts = create_samples_from_genome(genome_file, window_size)
        nts += genome_nts
        all_samples.extend(samples)
    return datasets.Dataset.from_list(all_samples), nts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create datasets from genome files.')
    parser.add_argument('--gtdb_path', type=str, required=True, help='Path to GTDB genome files directory.',
                        default='/data5/zhanghaohong/download/GTDB/gtdb_genomes_reps_r220/')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output dataset.',
                        default='datasets/GTDB/full_dataset')
    parser.add_argument('--n_genomes', type=int, default=None, help='Number of genomes to process. If None, process all genomes.')
    parser.add_argument('--window_size', type=int, default=WINDOW_SIZE, help='Window size for each sample.')
    parser.add_argument('--split_ratio', type=str, default='0.8,0.1,0.1', help='Train, validation, test split ratio.')
    args = parser.parse_args()

    genome_dir = args.gtdb_path
    path_file = os.path.join(genome_dir, 'genome_paths.tsv')
    genome_path_df = pd.read_csv(path_file, sep=' ', header=None)
    genome_paths = genome_path_df[1] + genome_path_df[0]
    genome_paths = [os.path.join(genome_dir, path) for path in genome_paths]

    if args.n_genomes is not None:
        genome_paths = genome_paths[:args.n_genomes]
    
    # split genome paths
    train_ratio, val_ratio, test_ratio = map(float, args.split_ratio.split(','))
    train_paths, temp_paths = train_test_split(genome_paths, train_size=train_ratio, random_state=42)
    val_paths, test_paths = train_test_split(temp_paths, test_size=test_ratio/(test_ratio + val_ratio), random_state=42)
    
    datasets_dict = {}
    nts = 0
    for split_name, paths in zip(['train', 'validation', 'test'], [train_paths, val_paths, test_paths]):
        print(f'Creating {split_name} dataset from {len(paths)} genomes...')
        dataset, split_nts = create_datasets_from_genomes(paths, window_size=args.window_size)
        nts += split_nts
        datasets_dict[split_name] = dataset
    
    dataset = datasets.DatasetDict(datasets_dict)
    
    # save dataset to disk
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    dataset.save_to_disk(args.output_path)
    print(f'Dataset saved to {args.output_path} with {args.n_genomes if args.n_genomes else "all"} genomes. \n \
          Total basepairs: {nts/1e9:.2f} billion. \n \
          Estimated B tokens: {nts/6/1e9:.2f} billion.')
    
# genome_path = pd.read_csv('/data5/zhanghaohong/download/GTDB/gtdb_genomes_reps_r220/genome_paths.tsv', sep=' ', header=None)

# genome_path = genome_path[1] + genome_path[0]
# # test with one genome
# # test_path = f'/data5/zhanghaohong/download/GTDB/gtdb_genomes_reps_r220/{genome_path[0]}'
# # samples = create_samples_from_genome(test_path)
# # random 100 samples as toy dataset
# toy_genome_paths = genome_path.sample(1000, random_state=42).tolist()
# toy_genome_paths = [f'/data5/zhanghaohong/download/GTDB/gtdb_genomes_reps_r220/{path}' for path in toy_genome_paths]
# toy_dataset = create_datasets_from_genomes(toy_genome_paths)
# os.makedirs('datasets/GTDB', exist_ok=True)
# toy_dataset.save_to_disk('datasets/GTDB/toy_dataset_1000')

# print how many basepairs in the dataset (XXXB nucleotides, roughly XXX/6 B tokens)
# total_basepairs = sum(len(sample['Sequence']) for sample in toy_dataset)
# print(f'Toy dataset has {total_basepairs/1e9:.2f} billion basepairs, roughly {total_basepairs/6/1e9:.2f} billion B tokens.')