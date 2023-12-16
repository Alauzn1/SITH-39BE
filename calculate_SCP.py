import pickle
import numpy as np
from statistics import mean
from shapely import wkt
from shapely.geometry import MultiPoint
from shapely.geometry import LineString
import pytorch_lightning as pl
from tqdm import tqdm
from utils.utils import generate_numpy_key
from datasets import *
from datasets.data import Multi30k
from utils_t import greedy_decode, gettgt
import yaml
from models.model.transformer import Transformer_Model

def calculate_each(embedding, hidden, prefix):
    # a 为词向量所对应的点集
    # b 为隐含节点所对应的点集

    # 去除重复行
    embedding_np = np.unique(np.array(embedding), axis=0)
    hidden_np = np.unique(np.array(hidden), axis=0)

    embedding_mp = MultiPoint(embedding_np)
    hidden_mp = MultiPoint(hidden_np)

    embedding_convex =embedding_mp.convex_hull
    hidden_convex = hidden_mp.convex_hull


    # 两个凸包交集面积
    area_intersection = hidden_convex.intersection(embedding_convex).area

    area_precision = area_intersection / hidden_convex.area if hidden_convex.area != 0 else 0  #SCP


    return {
        f'{prefix}-SCP': area_precision,
    }


def calculate(ckpt_file, hparams_file, dict_file, record_file):
    print('开始加载并且构造字典文件, 请等待...')
    # 加载向量文件
    print('vector_reduce = np.load dict_file', dict_file)
    vector_reduce = np.load('{}.npz'.format(dict_file))
    all_dict = {}
    for i, _ in enumerate(vector_reduce['high']):
        all_dict[generate_numpy_key(_)] = i
    print('开始加载模型文件, 请等待...')
    # 加载要使用的模型
    model = Transformer_Model.load_from_checkpoint(ckpt_file, hparams_file=hparams_file)
    model.eval()
    pl.seed_everything(0)

    # 加载数据集
    yaml_config = yaml.load(open(hparams_file, 'r'), Loader=yaml.FullLoader)

    ds = ds_dict[yaml_config['args'].dataset]()
    ds.prepare_data()
    ds.setup()


    print('开始计算...')
    all_cal = []

    file_path_src = "/home/jinxin/project/Convexplainer_enfr/data/test_2016_flickr.en"
    file_path_tgt = "/home/jinxin/project/Convexplainer_enfr/data/test_2016_flickr.fr"

    with open(file_path_src, "r") as file1, open(file_path_tgt, 'r') as file2:
        for line1, line2 in zip(file1, file2):
            src = str(line1.strip())
            tgt = str(line2.strip())
            embedding_pos_tgt = ds.get_tgt_embed(model, tgt, gettgt)
            embedding_pos, encoder_sixall, decoder_sixall = ds.transex(model, src, greedy_decode)

            encoder_1 = encoder_sixall[0]
            encoder_2 = encoder_sixall[1]
            encoder_3 = encoder_sixall[2]
            encoder_4 = encoder_sixall[3]
            encoder_5 = encoder_sixall[4]
            encoder_6 = encoder_sixall[5]

            # decoder_sixall = model.get_middle_decoder(src, tgt)
            decoder_1 = decoder_sixall[0]
            decoder_2 = decoder_sixall[1]
            decoder_3 = decoder_sixall[2]
            decoder_4 = decoder_sixall[3]
            decoder_5 = decoder_sixall[4]
            decoder_6 = decoder_sixall[5]

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

            embedding_pos_seq_low = []
            for _ in embedding_pos_seq:
                # if all_dict.get(generate_numpy_key(_)) is not None:
                embedding_pos_seq_low.append(all_dict[generate_numpy_key(_)])
            embedding_pos_seq_low = vector_reduce['low'][embedding_pos_seq_low]

            embedding_pos_tgt_seq_low = []
            for _ in embedding_pos_tgt_seq:
                # if all_dict.get(generate_numpy_key(_)) is not None:
                embedding_pos_tgt_seq_low.append(all_dict[generate_numpy_key(_)])
            embedding_pos_tgt_seq_low = vector_reduce['low'][embedding_pos_tgt_seq_low]

            encoder_1_seq_low = []
            for _ in encoder_1_seq:
                # if all_dict.get(generate_numpy_key(_)) is not None:
                encoder_1_seq_low.append(all_dict[generate_numpy_key(_)])
            encoder_1_seq_low = vector_reduce['low'][encoder_1_seq_low]

            encoder_2_seq_low = []
            for _ in encoder_2_seq:
                # if all_dict.get(generate_numpy_key(_)) is not None:
                encoder_2_seq_low.append(all_dict[generate_numpy_key(_)])
            encoder_2_seq_low = vector_reduce['low'][encoder_2_seq_low]

            encoder_3_seq_low = []
            for _ in encoder_3_seq:
                # if all_dict.get(generate_numpy_key(_)) is not None:
                encoder_3_seq_low.append(all_dict[generate_numpy_key(_)])
            encoder_3_seq_low = vector_reduce['low'][encoder_3_seq_low]

            encoder_4_seq_low = []
            for _ in encoder_4_seq:
                # if all_dict.get(generate_numpy_key(_)) is not None:
                encoder_4_seq_low.append(all_dict[generate_numpy_key(_)])
            encoder_4_seq_low = vector_reduce['low'][encoder_4_seq_low]

            encoder_5_seq_low = []
            for _ in encoder_5_seq:
                # if all_dict.get(generate_numpy_key(_)) is not None:
                encoder_5_seq_low.append(all_dict[generate_numpy_key(_)])
            encoder_5_seq_low = vector_reduce['low'][encoder_5_seq_low]

            encoder_6_seq_low = []
            for _ in encoder_6_seq:
                # if all_dict.get(generate_numpy_key(_)) is not None:
                encoder_6_seq_low.append(all_dict[generate_numpy_key(_)])
            encoder_6_seq_low = vector_reduce['low'][encoder_6_seq_low]

            decoder_1_seq_low = []
            for _ in decoder_1_seq:
                # if all_dict.get(generate_numpy_key(_)) is not None:
                decoder_1_seq_low.append(all_dict[generate_numpy_key(_)])
            decoder_1_seq_low = vector_reduce['low'][decoder_1_seq_low]

            decoder_2_seq_low = []
            for _ in decoder_2_seq:
                # if all_dict.get(generate_numpy_key(_)) is not None:
                decoder_2_seq_low.append(all_dict[generate_numpy_key(_)])
            decoder_2_seq_low = vector_reduce['low'][decoder_2_seq_low]

            decoder_3_seq_low = []
            for _ in decoder_3_seq:
                # if all_dict.get(generate_numpy_key(_)) is not None:
                decoder_3_seq_low.append(all_dict[generate_numpy_key(_)])
            decoder_3_seq_low = vector_reduce['low'][decoder_3_seq_low]

            decoder_4_seq_low = []
            for _ in decoder_4_seq:
                # if all_dict.get(generate_numpy_key(_)) is not None:
                decoder_4_seq_low.append(all_dict[generate_numpy_key(_)])
            decoder_4_seq_low = vector_reduce['low'][decoder_4_seq_low]

            decoder_5_seq_low = []
            for _ in decoder_5_seq:
                # if all_dict.get(generate_numpy_key(_)) is not None:
                decoder_5_seq_low.append(all_dict[generate_numpy_key(_)])
            decoder_5_seq_low = vector_reduce['low'][decoder_5_seq_low]

            decoder_6_seq_low = []
            for _ in decoder_6_seq:
                # if all_dict.get(generate_numpy_key(_)) is not None:
                decoder_6_seq_low.append(all_dict[generate_numpy_key(_)])
            decoder_6_seq_low = vector_reduce['low'][decoder_6_seq_low]

            feature = calculate_each(embedding_pos_seq_low, encoder_1_seq_low, 'en0_en1')

            _ = calculate_each(embedding_pos_seq_low, encoder_2_seq_low, 'en0_en2')
            feature.update(_)

            _ = calculate_each(embedding_pos_seq_low, encoder_3_seq_low, 'en0_en3')
            feature.update(_)

            _ = calculate_each(embedding_pos_seq_low, encoder_4_seq_low, 'en0_en4')
            feature.update(_)

            _ = calculate_each(embedding_pos_seq_low, encoder_5_seq_low, 'en0_en5')
            feature.update(_)

            _ = calculate_each(embedding_pos_seq_low, encoder_6_seq_low, 'en0_en6')
            feature.update(_)



            _ = calculate_each(encoder_6_seq_low, decoder_1_seq_low, 'en6_de1')
            feature.update(_)

            _ = calculate_each(encoder_6_seq_low, decoder_2_seq_low, 'en6_de2')
            feature.update(_)

            _ = calculate_each(encoder_6_seq_low, decoder_3_seq_low, 'en6_de3')
            feature.update(_)

            _ = calculate_each(encoder_6_seq_low, decoder_4_seq_low, 'en6_de4')
            feature.update(_)

            _ = calculate_each(encoder_6_seq_low, decoder_5_seq_low, 'en6_de5')
            feature.update(_)

            _ = calculate_each(encoder_6_seq_low, decoder_6_seq_low, 'en6_de6')
            feature.update(_)


            _ = calculate_each(encoder_1_seq_low, encoder_2_seq_low, 'en1_en2')
            feature.update(_)

            _ = calculate_each(encoder_2_seq_low, encoder_3_seq_low, 'en2_en3')
            feature.update(_)

            _ = calculate_each(encoder_3_seq_low, encoder_4_seq_low, 'en3_en4')
            feature.update(_)

            _ = calculate_each(encoder_4_seq_low, encoder_5_seq_low, 'en4_en5')
            feature.update(_)

            _ = calculate_each(encoder_5_seq_low, encoder_6_seq_low, 'en5_en6')
            feature.update(_)



            _ = calculate_each(decoder_1_seq_low, decoder_2_seq_low, 'de1_de2')
            feature.update(_)

            _ = calculate_each(decoder_2_seq_low, decoder_3_seq_low, 'de2_de3')
            feature.update(_)

            _ = calculate_each(decoder_3_seq_low, decoder_4_seq_low, 'de3_de4')
            feature.update(_)

            _ = calculate_each(decoder_4_seq_low, decoder_5_seq_low, 'de4_de5')
            feature.update(_)

            _ = calculate_each(decoder_5_seq_low, decoder_6_seq_low, 'de5_de6')
            feature.update(_)



            all_cal.append(feature)

        with open(record_file, 'wb') as wf:
            pickle.dump(all_cal, wf)

