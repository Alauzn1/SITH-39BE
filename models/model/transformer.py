import logging
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from data import Multi30k
from utils_t import get_bleu_score, greedy_decode
from argparse import ArgumentParser
from models.build_model import build_model


device = torch.device('cuda:1')

class Transformer_Model(pl.LightningModule):

    def __init__(self, src_vocab_size, tgt_vocab_size, args):
        super(Transformer_Model, self).__init__()
        self.save_hyperparameters()
        tran_model = build_model(src_vocab_size, tgt_vocab_size, args)
        self.src_embed = tran_model.src_embed
        self.tgt_embed = tran_model.tgt_embed
        self.encoder = tran_model.encoder
        self.decoder = tran_model.decoder
        self.generator = tran_model.generator

        self.DATASET = Multi30k()
        #self.test_acc = Get_Bleu_Score()
        #self.valid_acc = Get_Bleu_Score()
        #self.test_acc = BLEUScore(n_gram=1)
        #self.valid_acc = BLEUScore(n_gram=1)

        self.cross_loss = nn.CrossEntropyLoss(ignore_index=self.DATASET.pad_idx)



    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)


    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)


    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)
        encoder_out, en_layer_out = self.encode(src, src_mask)
        decoder_out, de_layer_out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
        out = self.generator(decoder_out)
        out = F.log_softmax(out, dim=-1)
        return out, decoder_out


    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask


    def make_tgt_mask(self, tgt):
        pad_mask = self.make_pad_mask(tgt, tgt)
        seq_mask = self.make_subsequent_mask(tgt, tgt)
        mask = pad_mask & seq_mask
        return pad_mask & seq_mask


    def make_src_tgt_mask(self, src, tgt):
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask


    def make_pad_mask(self, query, key, pad_idx=1):
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)    # (n_batch, 1, query_seq_len, key_seq_len)

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # (n_batch, 1, query_seq_len, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False
        return mask


    def make_subsequent_mask(self, query, key):
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8') # lower triangle without diagonal
        mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)
        return mask

    def itos(self, x):
        x = list(x.cpu().numpy())
        tokens = self.DATASET.vocab_tgt.lookup_tokens(x)
        text = ''
        tokens = list(filter(lambda x: x not in {"", " ", "."} and x not in list(self.DATASET.specials.keys()), tokens))
        for t in tokens:
            text += t
            text += " "
        return text

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_x = tgt[:, :-1]
        tgt_y = tgt[:, 1:]
        output, _ =self(src, tgt_x)
        y_hat = output.contiguous().view(-1, output.shape[-1])
        y_gt = tgt_y.contiguous().view(-1)
        loss = self.cross_loss(y_hat, y_gt)
        params = filter(lambda p: p.requires_grad, self.parameters())
        nn.utils.clip_grad_norm_(params, 1.0)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_x = tgt[:, :-1]
        tgt_y = tgt[:, 1:]
        output, _ = self(src, tgt_x)
        y_hat = output.contiguous().view(-1, output.shape[-1])
        y_gt = tgt_y.contiguous().view(-1)
        loss = self.cross_loss(y_hat, y_gt)
        # pred = [out.max(dim=1)[1] for out in output]
        # pred_str = list(map(self.itos, pred))
        # gt_str = list(map(lambda x: [self.itos(x)], tgt_y))
        self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('valid_loss', loss)
        score = get_bleu_score(output, tgt_y, self.DATASET.vocab_tgt, self.DATASET.specials)
        # print(f"_score:{score:.5f}")
        # self.valid_acc = score
        # self.valid_acc(output, tgt_y, self.DATASET.vocab_tgt, self.DATASET.specials)
        # self.valid_acc(pred_str, gt_str)
        return {'val_loss': loss, 'valid_acc': score}

    def validation_epoch_end(self, outs):
        val_loss = torch.stack([x['val_loss'] for x in outs]).mean()
        self.log('val_loss', val_loss, prog_bar=True)
        for output in outs:
            print(output["valid_acc"])
            print('\n')
        bleu_score = sum([output["valid_acc"] for output in outs]) / len([output["valid_acc"] for output in outs])
        print(f"bleu_score:{bleu_score:.5f}")
        self.log('bleu_score', bleu_score, prog_bar=True)

    def training_epoch_end(self, training_ouputs):
        avg_loss = torch.tensor([x["loss"] for x in training_ouputs]).mean() 
        self.log('training_loss_epoch_end', avg_loss, prog_bar=True)
        print(f"training_loss_epoch_end:{avg_loss:.5f}")

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        # return torch.optim.Adam(self.parameters(), lr=self.hparams['args'].learning_rate)
        optimizer = optim.Adam(params, lr=self.hparams['args'].learning_rate, weight_decay=self.hparams['args'].weight_decay,
                         eps=self.hparams['args'].adam_eps)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True, factor=self.hparams['args'].scheduler_factor,
                                                         patience=self.hparams['args'].scheduler_patience),
            'monitor': 'val_loss'
        }

        return [optimizer], [scheduler]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-5)
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        parser.add_argument('--adam_eps', type=float, default=5e-9)
        parser.add_argument('--scheduler_factor', type=float, default=0.9)
        parser.add_argument('--scheduler_patience', type=float, default=10)
        return parser
