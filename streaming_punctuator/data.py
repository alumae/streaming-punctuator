import torch
import tqdm
import random
import string
from torch.nn.utils.rnn import pad_sequence
import logging


PUNCTUATIONS = ["", ",", ".", "?", "!"]
PUNCTUATION2ID = {s:i for i, s in enumerate(PUNCTUATIONS)}


def line2seqs(l, tokenizer, label_delay):
    words = l.split()
    token_seq = []
    label_seq = []
    token_seq.append(tokenizer.token_to_id("<s>"))
    label_seq.extend([-1] * (label_delay + 1) )

    label = ""
    j = 0

    while j < len(words):
        word = words[j]
        label = ""
        if (j + 1 < len(words)) and len(words[j+1]) == 1 and words[j+1][0] in string.punctuation:            
            if words[j+1] in PUNCTUATION2ID:
                label = words[j+1]

        token_ids = tokenizer.encode(word).ids
        if len(token_ids) > 0:
            token_seq.extend(token_ids)
            if len(token_ids) > 1:
                label_seq.extend([-1] * (len(token_ids) - 1))
            label_seq.append(PUNCTUATION2ID[label])

        if label != "":
            j += 1    
        j += 1

    token_seq.extend([tokenizer.token_to_id("</s>")] * (label_delay + 1))
    label_seq.append(-1)
    #label_seq.append(-1)
    
    return token_seq, label_seq


class PunctuationDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, filename, label_delay=1, **kwargs):
        self.tokenizer = tokenizer
        self.token_sequences = []
        self.label_sequences = []
        self.lengths = []
        logging.debug(f"Reading text from {filename}")
        with open(filename) as f:
            for l in f:   
                token_seq, label_seq = line2seqs(l, self.tokenizer, label_delay)   
                if len(token_seq) != len(label_seq):
                    breakpoint()
                assert len(token_seq) == len(label_seq)

                if (len(token_seq)) > 0:
                    self.token_sequences.append(torch.tensor(token_seq))
                    self.label_sequences.append(torch.tensor(label_seq))
                    self.lengths.append(len(token_seq))

    def __len__(self):
        return len(self.label_sequences)

    def __getitem__(self, idx):
        return {"tokens" : self.token_sequences[idx], "labels": self.label_sequences[idx], "length": self.lengths[idx]}


    def collate_batch(self, batch):
        label_list, token_list, length_list = [], [], []
        for item in batch:
            label_list.append(item["labels"])
            token_list.append(item["tokens"])
            length_list.append(item["length"])
        #breakpoint()
        return {"tokens": pad_sequence(token_list, batch_first=True, padding_value=self.tokenizer.token_to_id("</s>")), 
                "labels": pad_sequence(label_list, batch_first=True, padding_value=-1),
                "lengths": torch.tensor(length_list)}

            
