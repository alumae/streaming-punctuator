import unittest
from streaming_punctuator.data import line2seqs, PunctuationDataset
import torch
from tokenizers import Tokenizer
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torchnlp.samplers import BucketBatchSampler

class TestLine2Seqs(unittest.TestCase):

    def test_line2seqs(self):
        tokenizer = Tokenizer.from_file("test/tokenizer.json") 
        tokenizer.add_special_tokens(["<s>", "</s>"])
        token_seq, label_seq = line2seqs("tere ! minu nimi on Baabuu .", tokenizer, label_delay=1)
        print([tokenizer.id_to_token(i) for i in token_seq])
        print(label_seq)
        self.assertEqual(len(token_seq), len(label_seq))
        self.assertEqual([tokenizer.id_to_token(i) for i in token_seq], ['<s>', 'tere</w>', 'minu</w>', 'nimi</w>', 'on</w>', 'Baa', 'bu', 'u</w>', '</s>', '</s>'])
        self.assertEqual(label_seq,                                     [-1,    -1,          4,          0,          0,        0,     -1,    -1,      2,      -1])


    def test_line2seqs_label_delay(self):
        tokenizer = Tokenizer.from_file("test/tokenizer.json") 
        tokenizer.add_special_tokens(["<s>", "</s>"])
        token_seq, label_seq = line2seqs("tere ! minu nimi on Baabuu .", tokenizer, label_delay=2)
        print([tokenizer.id_to_token(i) for i in token_seq])
        print(label_seq)
        self.assertEqual(len(token_seq), len(label_seq))
        self.assertEqual([tokenizer.id_to_token(i) for i in token_seq], ['<s>', 'tere</w>', 'minu</w>', 'nimi</w>', 'on</w>', 'Baa', 'bu', 'u</w>', '</s>', '</s>', '</s>'])
        self.assertEqual(label_seq, [-1, -1, -1, 4, 0, 0, 0, -1, -1, 2, -1])


        token_seq, label_seq = line2seqs("tere ! minu nimi on Baabuu .", tokenizer, label_delay=0)
        print([tokenizer.id_to_token(i) for i in token_seq])
        print(label_seq)
        self.assertEqual(len(token_seq), len(label_seq))
        self.assertEqual([tokenizer.id_to_token(i) for i in token_seq], ['<s>', 'tere</w>', 'minu</w>', 'nimi</w>', 'on</w>', 'Baa', 'bu', 'u</w>', "</s>"])
        self.assertEqual(label_seq, [-1, 4, 0, 0, 0, -1, -1, 2, -1])


class TestDataLoader(unittest.TestCase):

    def test_dataloader(self):
        tokenizer = Tokenizer.from_file("test/tokenizer.json") 
        tokenizer.add_special_tokens(["<s>", "</s>"])

        dataset = PunctuationDataset(tokenizer, "test/dev.txt")
        
        batch_size = 8

        random_sampler = RandomSampler(dataset)
        
        batch_iterator = BucketBatchSampler(random_sampler, batch_size=batch_size, drop_last=False, sort_key=lambda x: dataset[x]["length"], bucket_size_multiplier=100)
        dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_iterator, collate_fn=dataset.collate_batch)

        for i in range(2):
            print(f"Testing epoch {i}")
            for j, batch in enumerate(dataloader):
                if j == 0:
                    # make sure that the length difference inside a batch is not > 20%
                    self.assertTrue((batch["lengths"].max() - batch["lengths"].min()) / batch["lengths"].max() < 0.2 )
                

