import pytorch_lightning as pl
from models.model.transformer import Transformer_Model
from datasets import *
from utils import greedy_decode
from tqdm import tqdm
from torch import nn, optim
import torch
import yaml


def trans_main():
    device = torch.device('cuda:0')

    ckpt_file = '/data/model6_6/version_0/checkpoints/epoch=399.ckpt'
    hparams_file = '/data/model6_6/version_0/hparams.yaml'

    print('加载模型')
    # 加载要使用的模型

    model = Transformer_Model.load_from_checkpoint(ckpt_file)
    model.eval()
    model = model.to(device)
    print('加载模型成功')

    print('加载数据集')
    pl.seed_everything(0)
    # 加载数据集
    yaml_config = yaml.load(open(hparams_file, 'r'), Loader=yaml.FullLoader)
    ds = ds_dict[yaml_config['args'].dataset]()
    ds.prepare_data()
    ds.setup()
    print('加载数据集成功')

    file_path = "/home/project/Transformer_enfr/data/test_2016_flickr.en"
    output_file_path = "/home/project/Transformer_enfr/test_translate_result_enfr.txt"

    with open(file_path, "r") as file, open(output_file_path, "w") as output_file:
        for line in file:
            src = str(line.strip())
            sentences = ds.translate(model, src, greedy_decode)
            output_file.write(f'{sentences}\n')
   

if __name__ == '__main__':
    trans_main()

