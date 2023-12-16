from argparse import ArgumentParser
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.attn import *
from datasets import *


class LitClassifier(pl.LightningModule):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(input_dim, args.embedding_dim)
        if args.rnn_type == 'RNN':
            self.rnn = nn.RNN(args.embedding_dim, args.hidden_dim, bidirectional=args.bi, batch_first=True)
        elif args.rnn_type == 'GRU':
            self.rnn = nn.GRU(args.embedding_dim, args.hidden_dim, bidirectional=args.bi, batch_first=True)
        elif args.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(args.embedding_dim, args.hidden_dim, bidirectional=args.bi, batch_first=True)

        self.fc = nn.Linear(args.hidden_dim, output_dim)

        if args.attn_type == 'tanh_attn':
            self.attn = TanhAttention(args.hidden_dim)
        elif args.attn_type == 'dot_attn':
            self.attn = DotAttention()
        elif args.attn_type == '':
            self.attn = None
        else:
            raise Exception('attn_type error')

        self.learning_rate = args.learning_rate
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def get_middle(self, input):
        """
        验证过程获取中间变量的部分
        :param input: (x, x_len)
        :return:
        """
        x, x_len = input
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, x_len, batch_first=True)
        output, hidden = self.rnn(packed)
        unpacked_output, length = pad_packed_sequence(output, batch_first=True)

        if self.hparams['args'].bi:
            b, s, _ = unpacked_output.shape
            unpacked_output = unpacked_output.view(b, s, 2, -1)
            unpacked_output = unpacked_output.sum(dim=2)

        if self.attn:
            attn_output, attn_dist = self.attn(unpacked_output)
            wait_output = attn_output.sum(dim=1)
        else:
            # wait_output = unpacked_output.sum(dim=1)
            attn_output = None
            wait_output = torch.einsum('ble, b->be', unpacked_output, 1 / length.to(x.device))

        return unpacked_output, attn_output, wait_output

    def forward(self, text):
        x, x_len = text
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, x_len, batch_first=True)
        output, hidden = self.rnn(packed)
        unpacked_output, length = pad_packed_sequence(output, batch_first=True)

        if self.hparams['args'].bi:
            b, s, _ = unpacked_output.shape
            unpacked_output = unpacked_output.view(b, s, 2, -1)
            unpacked_output = unpacked_output.sum(dim=2)

        if self.attn:
            wait_output, attn_dist = self.attn(unpacked_output)
            wait_output = wait_output.sum(dim=1)
        else:
            # wait_output = unpacked_output.sum(dim=1)
            wait_output = torch.einsum('ble, b->be', unpacked_output, 1 / length.to(x.device))

        # _ = self.fc(hidden.squeeze(dim=0))
        _ = self.fc(wait_output)
        return _

    def training_step(self, batch, batch_idx):
        x, x_len, y = batch
        y_hat = self((x, x_len))
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_len, y = batch
        y_hat = self((x, x_len))
        loss = F.cross_entropy(y_hat, y)

        self.log('valid_loss', loss)
        self.valid_acc(torch.max(y_hat, 1)[1], y)

    def test_step(self, batch, batch_idx):
        x, x_len, y = batch
        y_hat = self((x, x_len))
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        self.test_acc(torch.max(y_hat, 1)[1], y)

    def validation_epoch_end(self, outs):
        self.log('valid_acc', self.valid_acc.compute())

    def test_epoch_end(self, outs):
        self.log('test_acc', self.test_acc.compute())

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        # return torch.optim.Adam(self.parameters(), lr=self.hparams['args'].learning_rate)

        return torch.optim.Adam(params, lr=self.hparams['args'].learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=3e-3)
        return parser


def cli_main():

    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden_dim', type=int, default=50)
    parser.add_argument('--embedding_dim', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--rnn_type', type=str, default='GRU')  # RNN, LSTM, GRU
    parser.add_argument('--attn_type', type=str, default='tanh_attn')  # dot_attn, tanh_attn
    parser.add_argument('--bi', action='store_true', default=False)  # 是否是双向的
    parser.add_argument('--dataset', type=str, default='ag_news')  # sst/imdb/ag_news/yelp_polarity
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    # ------------
    # data
    # ------------
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

    # model = LitClassifier(ds.vocab_size, ds.label_size, args)
    # trainer = pl.Trainer.from_argparse_args(args,
    #                                         gpus=[0],
    #                                         deterministic=True,
    #                                         callbacks=[early_stop_callback, checkpoint_callback],
    #                                         log_every_n_steps=10,
    #                                         # precision=16,
    #                                         auto_lr_find=True)
    #
    # lr_finder = trainer.tuner.lr_find(model, datamodule=ds)
    # args.learning_rate = lr_finder.suggestion()
    # print('============Learning Rate=============')
    # print(lr_finder.suggestion())
    # print('============Learning Rate=============')

    model = LitClassifier(ds.vocab_size, ds.label_size, args)
    trainer = pl.Trainer.from_argparse_args(args,
                                            # gpus=[0],
                                            deterministic=True,
                                            callbacks=[early_stop_callback, checkpoint_callback],
                                            # precision=16,
                                            # fast_dev_run=True,
                                            log_every_n_steps=10)
    trainer.fit(model, datamodule=ds)
    test_eval = trainer.test(model, datamodule=ds)

    print('=============== RESULT ===================')
    print(test_eval)
    print('=============== RESULT ===================')


if __name__ == '__main__':
    cli_main()