import vector_dict
import calculate_to_csv
import calculate_CIO
import calculate_SCP
import calculate_SCC
import calculate_SCD
import os

CKPT_FILE = '/data/jinxin/Transformer_multi30k_enfr_0921/model6_6/version_3/checkpoints/epoch=499-val_loss=1.305.ckpt'
HPARAMS_FILE = '/data/jinxin/Transformer_multi30k_enfr_0921/model6_6/version_3/hparams.yaml'
REDUCE_TYPE = 'pca'   # tsne, pca, pac_tsne

# BASE_NAME = 'Convexplainer'
BASE_NAME = 'LJX_Convexplainer_enfr_202311008'

VECTOR_FILE = f'/data/jinxin/save/{BASE_NAME}/vector'
REDUCE_FILE = f'/data/jinxin/save/{BASE_NAME}/reduce'
CALCULATE_FILE_CIO = f'/data/jinxin/save/{BASE_NAME}/calculate_CIO.pkl'
CALCULATE_FILE_SCP = f'/data/jinxin/save/{BASE_NAME}/calculate_SCP.pkl'
CALCULATE_FILE_SCC = f'/data/jinxin/save/{BASE_NAME}/calculate_SCC.pkl'
CALCULATE_FILE_SCD = f'/data/jinxin/save/{BASE_NAME}/calculate_SCD.pkl'

CSV_FILE_CIO = f'/data/jinxin/save/{BASE_NAME}/cal_CIO.csv'
CSV_FILE_SCP = f'/data/jinxin/save/{BASE_NAME}/cal_SCP.csv'
CSV_FILE_SCC = f'/data/jinxin/save/{BASE_NAME}/cal_SCC.csv'
CSV_FILE_SCD = f'/data/jinxin/save/{BASE_NAME}/cal_SCD.csv'


if os.path.exists(f'/data/jinxin/save/{BASE_NAME}'):
    raise Exception(f'设定的BASE_NAME: {BASE_NAME} 已经存在了')
else:
    os.mkdir(f'/data/jinxin/save/{BASE_NAME}')

vector_dict.generate_vector_dict(CKPT_FILE, HPARAMS_FILE, VECTOR_FILE, REDUCE_FILE, REDUCE_TYPE)

calculate_CIO.calculate(CKPT_FILE, HPARAMS_FILE, REDUCE_FILE, CALCULATE_FILE_CIO)
calculate_to_csv.calculate_to_csv(CALCULATE_FILE_CIO, CSV_FILE_CIO)

calculate_SCP.calculate(CKPT_FILE, HPARAMS_FILE, REDUCE_FILE, CALCULATE_FILE_SCP)
calculate_to_csv.calculate_to_csv(CALCULATE_FILE_SCP, CSV_FILE_SCP)

calculate_SCC.calculate(CKPT_FILE, HPARAMS_FILE, REDUCE_FILE, CALCULATE_FILE_SCC)
calculate_to_csv.calculate_to_csv(CALCULATE_FILE_SCC, CSV_FILE_SCC)

calculate_SCD.calculate(CKPT_FILE, HPARAMS_FILE, REDUCE_FILE, CALCULATE_FILE_SCD)
calculate_to_csv.calculate_to_csv(CALCULATE_FILE_SCD, CSV_FILE_SCD)





