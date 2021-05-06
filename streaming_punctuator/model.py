import os
import logging
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch import optim

import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1
from tokenizers import Tokenizer

import streaming_punctuator.data

# From: https://github.com/pytorch/pytorch/issues/1333#issuecomment-453702879
class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class StreamingPunctuatorModel(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = Tokenizer.from_file(self.hparams.tokenizer_file)
        self.tokenizer.add_special_tokens(["<s>", "</s>"])
        vocab_size = self.tokenizer.get_vocab_size()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.hparams.embedding_dim)
        current_input_dim = self.hparams.embedding_dim

        if (self.hparams.use_transformer):
            self.pos_encoder = PositionalEncoding(self.hparams.embedding_dim, 0.2)            
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.hparams.embedding_dim, nhead=8)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
            self.dropout = nn.Dropout(p=0.2)
            self.output = nn.Linear(self.hparams.embedding_dim, out_features=len(streaming_punctuator.data.PUNCTUATIONS))


        else:
            if len(self.hparams.conv_dims) > 0:
                self.conv_dims = [int(s) for s in self.hparams.conv_dims.split(",")]
                self.conv_sizes = [int(s) for s in self.hparams.conv_sizes.split(",")]
                self.conv_dilations = [int(s) for s in self.hparams.conv_dilations.split(",")]
                assert(len(self.conv_dims) == len(self.conv_sizes))
                assert(len(self.conv_dims) == len(self.conv_dilations))

                conv_layers = []
                current_input_dim = self.hparams.embedding_dim
                for i in range(len(self.conv_dims)):
                    conv_layer = CausalConv1d(current_input_dim, self.conv_dims[i], self.conv_sizes[i], 
                                        dilation=self.conv_dilations[i], 
                                        bias=False)                                
                    conv_layers.append(conv_layer)            
                    conv_layers.append(nn.BatchNorm1d(self.conv_dims[i], affine=False))
                    conv_layers.append(nn.ReLU(inplace=True))
                    current_input_dim = self.conv_dims[i]
                self.conv_layers = nn.Sequential(*conv_layers)
            else:
                self.conv_layers = None

            self.lstm = nn.LSTM(input_size=current_input_dim, 
                    hidden_size=self.hparams.lstm_hidden_size,
                    bidirectional=self.hparams.use_bidirectional,
                    num_layers=self.hparams.lstm_num_layers,
                    batch_first=True)
            self.dropout = nn.Dropout(p=0.2)
            lstm_output_size = self.hparams.lstm_hidden_size + self.hparams.lstm_hidden_size * self.hparams.use_bidirectional
            self.output = nn.Linear(lstm_output_size, out_features=len(streaming_punctuator.data.PUNCTUATIONS))


    def forward(self, tokens):
        #logging.debug(f"Processing batch with shape {tokens.shape}")
        x = self.embedding(tokens)
        if (self.hparams.use_transformer):
            x *=  math.sqrt(self.hparams.embedding_dim)
            x = x.permute(1, 0, 2)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            x = x.permute(1, 0, 2)
            x = F.log_softmax(self.dropout(self.output(x)), dim=-1)
            return x

        else:
            if (self.conv_layers is not None):
                x = x.permute(0, 2, 1)
                x = self.conv_layers(x)
                x = x.permute(0, 2, 1)

            x, _ = self.lstm(x)
            x = F.log_softmax(self.dropout(self.output(x)), dim=-1)
            return x

    def training_step(self, batch, batch_idx):
        tokens = batch["tokens"]
        labels = batch["labels"]
        y_hat = self.forward(tokens)
        
        flattened_labels = labels[labels != -1]
        flattened_y_hat = y_hat[labels != -1]
        loss = F.nll_loss(flattened_y_hat, flattened_labels)
        #self.log('train_loss', loss)
        self.log('lr', torch.tensor(self.trainer.optimizers[0].param_groups[0]['lr'], device=loss.device), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens = batch["tokens"]
        labels = batch["labels"]
        y_hat = self.forward(tokens)
        #breakpoint()
        flattened_labels = labels[labels != -1]
        flattened_y_hat = y_hat[labels != -1]
        loss = F.nll_loss(flattened_y_hat, flattened_labels)
        acc = accuracy(flattened_y_hat.exp(), flattened_labels)
        f1_value = f1(flattened_y_hat.exp(), flattened_labels, 
            average='micro',
            ignore_index=streaming_punctuator.data.PUNCTUATION2ID[""], 
            num_classes=len(streaming_punctuator.data.PUNCTUATIONS))

        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)
        self.log('val_f1', f1_value, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        pass


    def process_line(self, text):
        toks, _ = streaming_punctuator.data.line2seqs(text, self.tokenizer, self.hparams.label_delay)
        y_hat = self.forward(torch.tensor(toks).unsqueeze(0)).squeeze()
        punctuation_ids = y_hat.argmax(dim=1)
        
        #print(toks)
        #print(punctuation_ids)
        
        result = ""
        
        for i, tok_id in enumerate(toks):
            tok = self.tokenizer.id_to_token(tok_id)
            if tok not in ["<s>", "</s>"]:                
                if tok.endswith("</w>"):
                    result += tok[:-4] + " "
                else:
                    result += tok

            if self.tokenizer.id_to_token(toks[i]).endswith("</w>"):

                if streaming_punctuator.data.PUNCTUATIONS[punctuation_ids[i + self.hparams.label_delay]] != "":

                    result += streaming_punctuator.data.PUNCTUATIONS[punctuation_ids[i+ self.hparams.label_delay]] + " "
            
        return result.strip()



    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )


        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)        
        return [optimizer], [scheduler]

    # learning rate warm-up
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.learning_rate

        # update params
        optimizer.step(closure=closure)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("StreamingPunctuatorModel")
        parser.add_argument('--tokenizer-file', required=False, type=str)
        parser.add_argument('--embedding-dim', type=int, default=256)
        parser.add_argument('--conv-dims', type=str, default="")
        parser.add_argument('--conv-sizes', type=str, default="")
        parser.add_argument('--conv-dilations', type=str, default="")
        parser.add_argument('--lstm-hidden-size', type=int, default=512)
        parser.add_argument('--lstm-num-layers', type=int, default=2)
        parser.add_argument('--label-delay', type=int, default=1)
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--learning-rate', type=float, default=1e-3)
        parser.add_argument('--use-transformer', action='store_true')
        parser.add_argument('--warmup-steps', type=int, default=5000)
        parser.add_argument('--use-bidirectional', action='store_true')
        return parent_parser        