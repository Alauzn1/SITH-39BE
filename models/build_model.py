import torch
import torch.nn as nn
from models.model.encoder import Encoder
from models.model.decoder import Decoder
from models.block.encoder_block import EncoderBlock
from models.block.decoder_block import DecoderBlock
from models.layer.multi_head_attention_layer import MultiHeadAttentionLayer
from models.layer.position_wise_feed_forward_layer import PositionWiseFeedForwardLayer
from models.embedding.transformer_embedding import TransformerEmbedding
from models.embedding.token_embedding import TokenEmbedding
from models.embedding.positional_encoding import PositionalEncoding
import pytorch_lightning as pl


class build_model(pl.LightningModule):
    def __init__(self, src_vocab_size, tgt_vocab_size, args):
        super(build_model, self).__init__()
        self.save_hyperparameters()
        import copy
        copy = copy.deepcopy
        src_token_embed = TokenEmbedding(
                                         d_embed = args.d_embed,
                                         vocab_size = src_vocab_size)
        tgt_token_embed = TokenEmbedding(
                                         d_embed = args.d_embed,
                                         vocab_size = tgt_vocab_size)
        pos_embed = PositionalEncoding(
                                       d_embed = args.d_embed,
                                       max_len = args.max_len,
                                       dr_rate = args.dr_rate)

        self.src_embed = TransformerEmbedding(
                                         token_embed = src_token_embed,
                                         pos_embed = copy(pos_embed),
                                         dr_rate = args.dr_rate)
        self.tgt_embed = TransformerEmbedding(
                                         token_embed = tgt_token_embed,
                                         pos_embed = copy(pos_embed),
                                         dr_rate = args.dr_rate)

        attention = MultiHeadAttentionLayer(
                                            d_model = args.d_model,
                                            h = args.h,
                                            qkv_fc = nn.Linear(args.d_embed, args.d_model),
                                            out_fc = nn.Linear(args.d_model, args.d_embed),
                                            dr_rate = args.dr_rate)
        position_ff = PositionWiseFeedForwardLayer(
                                                   fc1 = nn.Linear(args.d_embed, args.d_ff),
                                                   fc2 = nn.Linear(args.d_ff, args.d_embed),
                                                   dr_rate = args.dr_rate)
        norm = nn.LayerNorm(args.d_embed, eps = args.norm_eps)

        encoder_block = EncoderBlock(
                                     self_attention = copy(attention),
                                     position_ff = copy(position_ff),
                                     norm = copy(norm),
                                     dr_rate = args.dr_rate)
        decoder_block = DecoderBlock(
                                     self_attention = copy(attention),
                                     cross_attention = copy(attention),
                                     position_ff = copy(position_ff),
                                     norm = copy(norm),
                                     dr_rate = args.dr_rate)

        self.encoder = Encoder(
                          encoder_block = encoder_block,
                          n_layer = args.n_layer,
                          norm = copy(norm))
        self.decoder = Decoder(
                          decoder_block = decoder_block,
                          layer_de = args.layer_de,
                          norm = copy(norm))
        self.generator = nn.Linear(args.d_model, tgt_vocab_size)


