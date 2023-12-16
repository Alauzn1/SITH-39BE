"""
@author : Hansu Kim(@cpm0722)
@when : 2022-08-21
@github : https://github.com/cpm0722
@homepage : https://cpm0722.github.io
"""

import pickle
import torch
from torchtext.data.metrics import bleu_score



# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from typing import Any, Callable, Optional
#
# import torch
#
# from pytorch_lightning.metrics.metric import Metric
# from pytorch_lightning.metrics.utils import _input_format_classification


# class Get_bleu_Score(Metric):
#
#     def __init__(
#         self,
#         threshold: float = 0.5,
#         compute_on_step: bool = True,
#         dist_sync_on_step: bool = False,
#         process_group: Optional[Any] = None,
#         dist_sync_fn: Callable = None,
#     ):
#         super().__init__(
#             compute_on_step=compute_on_step,
#             dist_sync_on_step=dist_sync_on_step,
#             process_group=process_group,
#             dist_sync_fn=dist_sync_fn,
#         )
#
#         self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
#
#         self.threshold = threshold
#
#     def update(self, preds: torch.Tensor, target: torch.Tensor):
#         """
#         Update state with predictions and targets.
#
#         Args:
#             preds: Predictions from model
#             target: Ground truth values
#         """
#         preds, target = _input_format_classification(preds, target, self.threshold)
#         assert preds.shape == target.shape
#
#         self.correct += torch.sum(preds == target)
#         self.total += target.numel()
#
#     def compute(self):
#         """
#         Computes accuracy over state.
#         """
#         return self.correct.float() / self.total





def save_pkl(data, fname):
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def load_pkl(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


def get_bleu_score(output=[], gt=[], vocab=None, specials=None, max_n=4):

    def itos(x):
        x = list(x.cpu().numpy())
        tokens = vocab.lookup_tokens(x)
        tokens = list(filter(lambda x: x not in {"", " ", "."} and x not in list(specials.keys()), tokens))
        return tokens

    pred = [out.max(dim=1)[1] for out in output]
    pred_str = list(map(itos, pred))
    gt_str = list(map(lambda x: [itos(x)], gt))

    score = bleu_score(pred_str, gt_str, max_n=max_n)
    score = score * 100
    return score


# def greedy_decode(model, src, max_len, start_symbol, end_symbol):
#     src = src.to(model.device)
#     src_mask = model.make_src_mask(src).to(model.device)
#     memory = model.encode(src, src_mask)
#
#     ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(model.device)
#     for i in range(max_len-1):
#         memory = memory.to(model.device)
#         tgt_mask = model.make_tgt_mask(ys).to(model.device)
#         src_tgt_mask = model.make_src_tgt_mask(src, ys).to(model.device)
#         out, de_layer_out = model.decode(ys, memory, tgt_mask, src_tgt_mask)
#         prob = model.generator(out[:, -1])
#         _, next_word = torch.max(prob, dim=1)
#         next_word = next_word.item()
#
#         ys = torch.cat([ys,
#                         torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
#         if next_word == end_symbol:
#             break
#     return de_layer_out


# def greedy_decode(model, src, max_len, start_symbol, end_symbol):
#     device = torch.device('cpu')
#     # src = src.to(model.device)
#     src = src.to(device)
#     # src_mask = model.make_src_mask(src).to(model.device)
#     src_mask = model.make_src_mask(src).to(device)
#     memory, memory_list = model.encode(src, src_mask)
#
#     # ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(model.device)
#     ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
#     for i in range(max_len-1):
#         # memory = memory.to(model.device)
#         memory = memory.to(device)
#         # tgt_mask = model.make_tgt_mask(ys).to(model.device)
#         tgt_mask = model.make_tgt_mask(ys).to(device)
#         # src_tgt_mask = model.make_src_tgt_mask(src, ys).to(model.device)
#         src_tgt_mask = model.make_src_tgt_mask(src, ys).to(device)
#         out, out_list = model.decode(ys, memory, tgt_mask, src_tgt_mask)
#         prob = model.generator(out[:, -1])
#         _, next_word = torch.max(prob, dim=1)
#         next_word = next_word.data[0]
#
#         ys = torch.cat([ys,
#                         torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
#         if next_word == end_symbol:
#             break
#     return memory_list, out_list


def greedy_decode(model, src, max_len, start_symbol, end_symbol):
    device = torch.device('cpu')
    # src = src.to(model.device)
    src = src.to(device)
    # src_mask = model.make_src_mask(src).to(model.device)
    src_mask = model.make_src_mask(src).to(device)
    src_embedding = model.src_embed(src).to(device)
    memory, memory_list = model.encode(src, src_mask)

    # ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(model.device)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        # memory = memory.to(model.device)
        memory = memory.to(device)
        # tgt_mask = model.make_tgt_mask(ys).to(model.device)
        tgt_mask = model.make_tgt_mask(ys).to(device)
        # src_tgt_mask = model.make_src_tgt_mask(src, ys).to(model.device)
        src_tgt_mask = model.make_src_tgt_mask(src, ys).to(device)
        out, out_list = model.decode(ys, memory, tgt_mask, src_tgt_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == end_symbol:
            break
    return src_embedding, memory_list, out_list


def gettgt(model, tgt):
    device = torch.device('cpu')
    # src = src.to(model.device)
    tgt = tgt.to(device)
    # src_mask = model.make_src_mask(src).to(model.device)
    tgt_embedding = model.tgt_embed(tgt).to(device)
    return tgt_embedding

