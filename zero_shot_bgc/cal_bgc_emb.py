import datasets
# read gbk
from Bio import SeqIO
import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

def parse_gbk(gbk_file):
    # return sequence and domain locations
    sequences = []
    domains = []
    for record in SeqIO.parse(gbk_file, 'genbank'):
        seq = str(record.seq)
        sequences.append(seq)
        for feature in record.features:
            if feature.type in ['aSDomain']:
                start = int(feature.location.start)
                end = int(feature.location.end)
                domains.append((start, end, feature.qualifiers['aSDomain'][0]))
    return ''.join(sequences), domains

def seq2datasets(seq, max_length):
    # split seq into chunks of max_length
    samples = []
    seq_length = len(seq)
    for start in range(0, seq_length, max_length):
        end = min(start + max_length, seq_length)
        sample_seq = seq[start:end]
        samples.append({'Sequence': sample_seq, 'Start': start, 'End': end})
    return datasets.Dataset.from_list(samples)

# map labels to each sample
def get_label(start, end):
    # return a list of labels for the given start and end nts
    nts_labels = []
    for nt in range(start, end):
        for loc_start, loc_end, label in locations:
            if loc_start <= nt <= loc_end:
                nts_labels.append(label)
                break
        else:
            nts_labels.append('No domain')
    return nts_labels

MODEL_PATH = '/data5/zhanghaohong/transformers_model/nucleotide-transformer-v2-50m-multi-species'
MAX_LEN = 1000

# zero_shot_bgc/BGC0000001.gbk
seqs, locations = parse_gbk('BGC0000001.gbk')
dataset = seq2datasets(seqs, MAX_LEN*6)


dataset = dataset.map(
    lambda x: {'Labels': get_label(x['Start'], x['End'])},
    num_proc=64,
)
# filter out samples without domain
dataset = dataset.filter(lambda x: any(label != 'No domain' for label in x['Labels']))

tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=True,
    )
model = AutoModelForMaskedLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

# tokenize function
def tokenize_fn(batch):
        return tokenizer(
            batch["Sequence"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN + 1,
            return_tensors="pt",
        )
tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=64,
    )
# extract embeddings for each nts in 1, 5, 9 13 layers
# set model return hidden states
model.config.output_hidden_states = True

def extract_embeddings(batch, model=model):
    inputs = {k: torch.tensor(v) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states  # tuple of (batch_size, seq_len, hidden_size)
    selected_layers = [1, 5, 9, 13]
    embeddings = []
    for layer in selected_layers:
        layer_embeddings = hidden_states[layer - 1][:, 1:, :].cpu().numpy()  # exclude special tokens
        embeddings.append(layer_embeddings)
    # return a dict of embeddings for each layer
    return {f'layer_{layer}_embeddings': embeddings[i] for i, layer in enumerate(selected_layers)}
tokenized = tokenized.map(
    extract_embeddings,
    batched=True,
    num_proc=64,
)

# narrow labels
df = pd.DataFrame(tokenized)
df['Narrowed_Labels'] = df['Labels'].apply(lambda x: [x[i] for i in range(0, len(x), 6)])
# save embeddings to disk
os.makedirs('bgc_embeddings_dfs', exist_ok=True)
df.to_pickle('bgc_embeddings_dfs/bgc0000001_embeddings.pkl')

# finetuned model
finetuned_model_path = '/data5/zhanghaohong/projects/nt_gtdb/finetuned_models/nt_transformer_gtdb_1k/checkpoint-41058'
finetuned_model = model.from_pretrained(finetuned_model_path)
finetuned_model.config.output_hidden_states = True
tokenized_ft = tokenized.map(
    lambda batch: extract_embeddings(batch, model=finetuned_model),
    batched=True,
    num_proc=64,
)
df_ft = pd.DataFrame(tokenized_ft)
df_ft['Narrowed_Labels'] = df_ft['Labels'].apply(lambda x: [x[i] for i in range(0, len(x), 6)])
df_ft.to_pickle('bgc_embeddings_dfs/bgc0000001_embeddings_finetuned.pkl')
