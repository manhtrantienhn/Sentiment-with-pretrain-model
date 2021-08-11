import json
import numpy as np
from tqdm import tqdm
import random
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader
import nltk
nltk.download('punkt')

def read_data(train_dataset, valid_dataset):
    with open(train_dataset, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(valid_dataset, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    return train_data, val_data

def create_vocab(train_dataset):
    origins = ' '.join(i.strip() for i in train_dataset['origin'])
    masks = ' '.join(i.strip() for i in train_dataset['mask'])

    origin_words = set(nltk.word_tokenize(origins))
    origin_vocab = ['<sos>', '<eos>', '<pad>', '<unk>'] + list(origin_words)
    origin_w2idx = {w:i for i, w in enumerate(origin_vocab)}
    origin_idx2w = {i:w for i, w in enumerate(origin_vocab)}

    mask_words = set(nltk.word_tokenize(masks))
    mask_vocab = ['<sos>', '<eos>', '<pad>', '<unk>'] + list(mask_words)
    mask_w2idx = {w:i for i, w in enumerate(mask_vocab)}
    mask_idx2w = {i:w for i, w in enumerate(origin_vocab)}

    sos_idx = origin_w2idx['<sos>']
    eos_idx = origin_w2idx['<eos>']
    pad_idx = origin_w2idx['<pad>']
    unk_idx = origin_w2idx['<unk>']

    return origin_w2idx, origin_idx2w, mask_w2idx, mask_idx2w, sos_idx, eos_idx, pad_idx, unk_idx

def text_to_seq(dataset, word2idx, unk_idx):
    seqs = []
    for sentence in tqdm(dataset):
        seqs.append([word2idx[word] if word in word2idx else unk_idx for word in nltk.word_tokenize(sentence)])
    return seqs

def pad_and_truncate_seqs(seqs, max_seq_len, pad_idx, sos_idx, eos_idx):
    seq_pads = np.zeros((len(seqs), max_seq_len))
    for i, seq in tqdm(enumerate(seqs)):
        pad_len = max_seq_len - len(seq) - 2
        if pad_len > 0:
            seq_pads[i] = np.pad([sos_idx] + seq + [eos_idx], (0, pad_len), 'constant', constant_values=(pad_idx))
        else:
            seq_pads[i] =  [sos_idx] + seq[:max_seq_len - 2] + [eos_idx]
    return seq_pads

def create_data_loader(train_dataset, valid_dataset, batch_size, origin_max_len, mask_max_len):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    print('reading datasets...')
    train_data, val_data = read_data(train_dataset, valid_dataset)

    print('creating vocab...')
    origin_w2idx, _, mask_w2idx, _, sos_idx, eos_idx, pad_idx, unk_idx = create_vocab(train_data)

    print('convert text to sequence...')
    origin_train = text_to_seq(train_data['origin'], origin_w2idx, unk_idx)
    restr_train = text_to_seq(train_data['resconstruct'], origin_w2idx, unk_idx)
    mask_train = text_to_seq(train_data['mask'], mask_w2idx, unk_idx)
    origin_val = text_to_seq(val_data['origin'], origin_w2idx, unk_idx)
    restr_val = text_to_seq(val_data['resconstruct'], origin_w2idx, unk_idx)
    mask_val = text_to_seq(val_data['mask'], mask_w2idx, unk_idx)

    print('padding and truncating seq...')
    origin_train_pad = pad_and_truncate_seqs(origin_train, origin_max_len, pad_idx, sos_idx, eos_idx)
    restr_train_pad = pad_and_truncate_seqs(restr_train, origin_max_len, pad_idx, sos_idx, eos_idx)
    mask_train_pad = pad_and_truncate_seqs(mask_train, mask_max_len, pad_idx, sos_idx, eos_idx)
    origin_val_pad = pad_and_truncate_seqs(origin_val, origin_max_len, pad_idx, sos_idx, eos_idx)
    restr_val_pad = pad_and_truncate_seqs(restr_val, origin_max_len, pad_idx, sos_idx, eos_idx)
    mask_val_pad = pad_and_truncate_seqs(mask_val, mask_max_len, pad_idx, sos_idx, eos_idx)
    
    origin_bert_token_train = tokenizer.batch_encode_plus(train_data['origin'], padding=True, truncation=True, max_length=origin_max_len, return_tensors='pt')
    restr_bert_token_train = tokenizer.batch_encode_plus(train_data['resconstruct'], padding=True, truncation=True, max_length=origin_max_len, return_tensors='pt')
    mask_bert_token_train = tokenizer.batch_encode_plus(train_data['mask'], padding=True, truncation=True, max_length=mask_max_len, return_tensors='pt')
    
    origin_bert_token_val = tokenizer.batch_encode_plus(val_data['origin'], padding=True, truncation=True, max_length=origin_max_len, return_tensors='pt')
    restr_bert_token_val = tokenizer.batch_encode_plus(val_data['resconstruct'], padding=True, truncation=True, max_length=origin_max_len, return_tensors='pt')
    mask_bert_token_val = tokenizer.batch_encode_plus(val_data['mask'], padding=True, truncation=True, max_length=mask_max_len, return_tensors='pt')

    train_tensor = TensorDataset(torch.tensor(origin_train_pad, dtype=torch.long), origin_bert_token_train['input_ids'], origin_bert_token_train['attention_mask'],
                                 torch.tensor(restr_train_pad, dtype=torch.long), restr_bert_token_train['input_ids'], restr_bert_token_train['attention_mask'],
                                 torch.tensor(mask_train_pad, dtype=torch.long), mask_bert_token_train['input_ids'], mask_bert_token_train['attention_mask'])
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

    val_tensor = TensorDataset(torch.tensor(origin_val_pad, dtype=torch.long), origin_bert_token_val['input_ids'], origin_bert_token_val['attention_mask'],
                               torch.tensor(restr_val_pad, dtype=torch.long), restr_bert_token_val['input_ids'], restr_bert_token_val['attention_mask'],
                               torch.tensor(mask_val_pad, dtype=torch.long), mask_bert_token_val['input_ids'], mask_bert_token_val['attention_mask'])
    val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, origin_w2idx, mask_w2idx