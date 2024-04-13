import pickle
import torch
from torchtext.data.metrics import bleu_score

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


def greedy_decode(model, src, max_len, start_symbol, end_symbol):
    device = torch.device('cpu')
    src = src.to(device)
    src_mask = model.make_src_mask(src).to(device)
    src_embedding = model.src_embed(src).to(device)
    memory, memory_list = model.encode(src, src_mask)

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = model.make_tgt_mask(ys).to(device)
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
    tgt = tgt.to(device)
    tgt_embedding = model.tgt_embed(tgt).to(device)
    return tgt_embedding

