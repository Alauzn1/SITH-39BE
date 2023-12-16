from argparse import ArgumentParser
import pytorch_lightning as pl
from models.model.transformer import Transformer_Model
from datasets import *
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import *


def cli_main():
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--d_embed', type=int, default=512)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument('--h', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dr_rate', type=float, default=0.1)
    parser.add_argument('--norm_eps', type=float, default=1e-5)
    parser.add_argument('--dataset', type=str, default='multi30k')

    parser = pl.Trainer.add_argparse_args(parser)
    parser = Transformer_Model.add_model_specific_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    ds = ds_dict[args.dataset](args)
    ds.prepare_data()
    ds.setup()

    early_stop_callback = EarlyStopping(
        monitor='valid_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )
    checkpoint_callback = ModelCheckpoint(monitor='valid_acc', mode='max')
    logger = TensorBoardLogger('logs/multi30k', name='bert')

    model = Transformer_Model(len(ds.vocab_src), len(ds.vocab_tgt), args=args)

    trainer = pl.Trainer.from_argparse_args(args,
                                            gpus=[1],
                                            deterministic=True,
                                            callbacks=[early_stop_callback, checkpoint_callback],
                                            # precision=16,
                                            # fast_dev_run=True,
                                            log_every_n_steps=10,
                                            logger=logger,
                                            max_epochs=1000)
    trainer.fit(model, datamodule=ds)
    test_eval = trainer.test(model, datamodule=ds)

    print('=============== RESULT ===================')
    print(test_eval)
    print('=============== RESULT ===================')

if __name__ == '__main__':
    cli_main()

