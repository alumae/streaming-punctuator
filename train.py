import os
import sys
from argparse import ArgumentParser
import multiprocessing as mp

import logging
import numpy as np
import torch
import torch.utils.data 
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torchnlp.samplers import BucketBatchSampler

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


from streaming_punctuator.model import StreamingPunctuatorModel
from streaming_punctuator.data import PunctuationDataset



seed_everything(234)

def main(args):
    """
    Main training routine specific for this project
    :param hparams:
    """
    if args.train_file is not None and args.dev_file is not None:
        if (args.load_checkpoint):
            model = StreamingPunctuatorModel.load_from_checkpoint(args.load_checkpoint, **vars(args))
        else:
            model = StreamingPunctuatorModel(**vars(args))

        batch_size = args.batch_size

        train_dataset = PunctuationDataset(tokenizer=model.tokenizer, filename=args.train_file, label_delay=args.label_delay)
        dev_dataset = PunctuationDataset(tokenizer=model.tokenizer, filename=args.dev_file, label_delay=args.label_delay)

        random_sampler = RandomSampler(train_dataset)
        
        batch_iterator = BucketBatchSampler(random_sampler, batch_size=batch_size, drop_last=False, sort_key=lambda i: train_dataset[i]["length"], bucket_size_multiplier=100)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_iterator, collate_fn=train_dataset.collate_batch, num_workers=8)


        dev_loader = torch.utils.data.DataLoader(
                dataset=dev_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=dev_dataset.collate_batch,
                num_workers=2)



        checkpoint_callback = ModelCheckpoint(
                save_top_k=4,
                save_last=True,
                verbose=True,
                monitor='val_f1',
                mode='max',
                prefix=''
        )        
        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             callbacks=[
                                             ])
        trainer.fit(model, train_dataloader=train_loader,
                    val_dataloaders=dev_loader)
    elif args.process_stdin and args.load_checkpoint is not None:
        model = StreamingPunctuatorModel.load_from_checkpoint(args.load_checkpoint)
        while True:
            l = sys.stdin.readline()
            if not l: break
            print(model.process_line(l.strip()))
            sys.stdout.flush()
        
    else:
        raise Exception("Either --train-file and --dev-file or --process-stdin and --load-checkpoint should be specified")

        


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    #mp.set_start_method('fork')

    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)
    

    # each LightningModule defines arguments relevant to it
    parser = StreamingPunctuatorModel.add_model_specific_args(parent_parser)
    #parser = PunctuationDataset.add_data_specific_args(parser, root_dir)
    parser = Trainer.add_argparse_args(parser)

    # data
    parser.add_argument('--train-file', required=False, type=str)       
    parser.add_argument('--dev-file', required=False, type=str)
    
    parser.add_argument('--load-checkpoint', required=False, type=str)        
    parser.add_argument('--process-stdin', action='store_true')        


    args = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)