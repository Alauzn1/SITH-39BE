#from rnns_trainer import LitClassifier
import torch.nn as nn
from models.model.transformer import Transformer_Model
import models.embedding.token_embedding
from argparse import ArgumentParser
from datasets import *
from datasets.data import Multi30k
import pytorch_lightning as pl
import numpy as np
from utils import reduce_dimension
from utils_t import greedy_decode, gettgt
import torch
import pickle
from tqdm import tqdm
import yaml
from collections import OrderedDict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_reduce_word_vec(wait_reduce, reduce_type, low_dim=2):
    '''
    数据去重, 降维
    :param wait_reduce 等待降维的
    :param low_dim: 降低到的维数,默认为2
    :return: all_vec_2_dimension, all_vec_unique
    '''
    print("开始整合坐标")
    all_vec_unique = wait_reduce
    all_vec_2_dimension = reduce_dimension.reduce_dimension(all_vec_unique, low_dim, reduce_type)
    print("坐标整合完毕")
    return all_vec_unique, all_vec_2_dimension




def collect_vectors(ckpt_file, hparams_file, vectors_file):
    """
    获取数据集中高位到低维的字典
    :return:
    """
    print('加载模型')
    # 加载要使用的模型

    model = Transformer_Model.load_from_checkpoint(ckpt_file)
    model.eval()
    print('加载模型成功')

    print('加载数据集')
    pl.seed_everything(0)
    # 加载数据集
    yaml_config = yaml.load(open(hparams_file, 'r'), Loader=yaml.FullLoader)
    ds = ds_dict[yaml_config['args'].dataset]()
    ds.prepare_data()
    ds.setup()
    print('加载数据集成功')



    # TODO: Embedding Tensor 实际上跟Hidden的维度可能是不同的
    hidden_high = None

    file_path_src = "/home/jinxin/project/LJX_1008_Convexplainer_enfr/data/test_2016_flickr.en"
    file_path_tgt = "/home/jinxin/project/LJX_1008_Convexplainer_enfr/data/test_2016_flickr.fr"

    with open(file_path_src, "r") as file1, open(file_path_tgt, 'r') as file2:
        for line1, line2 in zip(file1, file2):
            src = str(line1.strip())
            tgt = str(line2.strip())
            embedding_pos_tgt = ds.get_tgt_embed(model, tgt, gettgt)
            embedding_pos, encoder_sixall, decoder_sixall = ds.transex(model, src, greedy_decode)
            #print(len(encoder_sixall))
            #print(len(decoder_sixall))
            encoder_1 = encoder_sixall[0]
            encoder_2 = encoder_sixall[1]
            encoder_3 = encoder_sixall[2]
            encoder_4 = encoder_sixall[3]
            encoder_5 = encoder_sixall[4]
            encoder_6 = encoder_sixall[5]
            # print('encoder--------------------')
            # print(encoder_1.shape)
            # print(encoder_2.shape)
            # print(encoder_3.shape)
            # print(encoder_4.shape)
            # print(encoder_5.shape)
            # print(encoder_6.shape)
            # decoder_sixall = model.get_middle_decoder(src, tgt)
            # print('decoder--------------------')
            decoder_1 = decoder_sixall[0]
            decoder_2 = decoder_sixall[1]
            decoder_3 = decoder_sixall[2]
            decoder_4 = decoder_sixall[3]
            decoder_5 = decoder_sixall[4]
            decoder_6 = decoder_sixall[5]
            # print(decoder_1.shape)
            # print(decoder_2.shape)
            # print(decoder_3.shape)
            # print(decoder_4.shape)
            # print(decoder_5.shape)
            # print(decoder_6.shape)
            # 去掉第一维的batch
            embedding_pos_seq = embedding_pos.squeeze(0).detach().numpy()
            encoder_1_seq = encoder_1.squeeze(0).detach().numpy()
            encoder_2_seq = encoder_2.squeeze(0).detach().numpy()
            encoder_3_seq = encoder_3.squeeze(0).detach().numpy()
            encoder_4_seq = encoder_4.squeeze(0).detach().numpy()
            encoder_5_seq = encoder_5.squeeze(0).detach().numpy()
            encoder_6_seq = encoder_6.squeeze(0).detach().numpy()

            decoder_1_seq = decoder_1.squeeze(0).detach().numpy()
            decoder_2_seq = decoder_2.squeeze(0).detach().numpy()
            decoder_3_seq = decoder_3.squeeze(0).detach().numpy()
            decoder_4_seq = decoder_4.squeeze(0).detach().numpy()
            decoder_5_seq = decoder_5.squeeze(0).detach().numpy()
            decoder_6_seq = decoder_6.squeeze(0).detach().numpy()
            embedding_pos_tgt_seq = embedding_pos_tgt.squeeze(0).detach().numpy()

            _ = np.concatenate((embedding_pos_seq, encoder_1_seq, encoder_2_seq, encoder_3_seq, encoder_4_seq, encoder_5_seq, encoder_6_seq
                                , decoder_1_seq, decoder_2_seq, decoder_3_seq, decoder_4_seq, decoder_5_seq, decoder_6_seq, embedding_pos_tgt_seq))

            if hidden_high is None:
                hidden_high = _
            else:
                hidden_high = np.vstack((hidden_high, _))
        print('开始保存-------------------')
        np.savez(vectors_file, hidden_high=hidden_high)


def generate_vector_dict(ckpt_file, hparams_file, vectors_file, reduce_file, reduce_type):

    collect_vectors(ckpt_file, hparams_file, vectors_file)
    all_vectors = np.load('{}.npz'.format(vectors_file))

    all_vectors_np = np.vstack((all_vectors['hidden_high']))

    embedding_high_uni, embedding_low = get_reduce_word_vec(all_vectors_np, reduce_type)
    print(embedding_high_uni.shape)
    print(embedding_low.shape)

    np.savez_compressed(reduce_file, high=embedding_high_uni, low=embedding_low)


# if __name__ == '__main__':
#     generate_vector_dict(
#         '/data/jinxin/Transformer_multi30k_enfr_0921/model6_6/version_3/checkpoints/epoch=499-val_loss=1.305.ckpt',
#         '/data/jinxin/Transformer_multi30k_enfr_0921/model6_6/version_3/hparams.yaml',
#         f'./save/LJX_Convexplainer_enfr_20230922/vector_111', f'./save/LJX_Convexplainer_enfr_20230922/reduce_111', 'pca')
