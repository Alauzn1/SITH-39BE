import vector_dict
import calculate
import calculate_to_csv
import os
import numpy as np

BASE_NAME = '11123'
REDUCE_TYPE = 'pca'   # tsne, pca, pac_tsne

c = {
    'tanh': {
        'CKPT_FILE': 'lightning_logs/version_1/checkpoints/epoch=0-step=374.ckpt',
        'HPARAMS_FILE': 'lightning_logs/version_1/hparams.yaml',
        'VECTOR_FILE': f'./save/{BASE_NAME}/vector_tanh'.format(BASE_NAME),
        'CALCULATE_FILE': f'./save/{BASE_NAME}/cal_tanh'.format(BASE_NAME)
    },
    'dot': {
        'CKPT_FILE': 'lightning_logs/version_1/checkpoints/epoch=0-step=374.ckpt',
        'HPARAMS_FILE': 'lightning_logs/version_1/hparams.yaml',
        'VECTOR_FILE': f'./save/{BASE_NAME}/vector_dot'.format(BASE_NAME),
        'CALCULATE_FILE': f'./save/{BASE_NAME}/cal_dot'.format(BASE_NAME)
    },
    'no': {
        'CKPT_FILE': 'lightning_logs/version_1/checkpoints/epoch=0-step=374.ckpt',
        'HPARAMS_FILE': 'lightning_logs/version_1/hparams.yaml',
        'VECTOR_FILE': f'./save/{BASE_NAME}/vector_no'.format(BASE_NAME),
        'CALCULATE_FILE': f'./save/{BASE_NAME}/cal_no'.format(BASE_NAME),
    }
}

REDUCE_FILE = f'./save/{BASE_NAME}/reduce'
CALCULATE_FILE = f'./save/{BASE_NAME}/calculate.pkl'
CSV_FILE = f'./save/{BASE_NAME}/cal.csv'

if os.path.exists(f'./save/{BASE_NAME}'):
    raise Exception(f'设定的BASE_NAME: {BASE_NAME} 已经存在了')
else:
    os.mkdir(f'./save/{BASE_NAME}')

# 计算所有的可能产生的向量，并且整合到一起
all_vector_list = []
for k, v in c.items():
    print('计算向量 {}'.format(k))
    vector_dict.collect_vectors(v['CKPT_FILE'], v['HPARAMS_FILE'], v['VECTOR_FILE'])
    _ = np.load('{}.npz'.format(v['VECTOR_FILE']))
    all_vector_list.append(_['embedding_high'])
    all_vector_list.append(_['hidden_high'])

print('整合降维')
all_vectors_np = np.vstack(all_vector_list)
# 降维, 保存
embedding_high_uni, embedding_low = vector_dict.get_reduce_word_vec(all_vectors_np, REDUCE_TYPE)
np.savez_compressed(REDUCE_FILE, high=embedding_high_uni, low=embedding_low)

for k, v in c.items():
    print('计算指标 {}'.format(k))
    calculate.calculate(v['CKPT_FILE'], v['HPARAMS_FILE'], REDUCE_FILE, v['CALCULATE_FILE'])