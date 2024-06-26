import vector_dict
import calculate_to_csv
import calculate_CIO
import calculate_SIA_D
import calculate_SIA_P
import calculate_SIA_I
import os

CKPT_FILE = '/data/model/version_0/checkpoints/epoch=399.ckpt'
HPARAMS_FILE = '/data/model6_6/version_0/hparams.yaml'
REDUCE_TYPE = 'pca'

BASE_NAME = 'SITH'

VECTOR_FILE = f'/data/save/{BASE_NAME}/vector'
REDUCE_FILE = f'/data/save/{BASE_NAME}/reduce'
CALCULATE_FILE_CIO = f'/data/save/{BASE_NAME}/calculate_CIO.pkl'
CALCULATE_FILE_SIA_D = f'/data/save/{BASE_NAME}/calculate_SIA_D.pkl'
CALCULATE_FILE_SIA_P = f'/data/save/{BASE_NAME}/calculate_SIA_P.pkl'
CALCULATE_FILE_SIA_I = f'/data/save/{BASE_NAME}/calculate_SIA_I.pkl'


CSV_FILE_CIO = f'/data/save/{BASE_NAME}/cal_CIO.csv'
CSV_FILE_SIA_D = f'/data/save/{BASE_NAME}/cal_SIA_D.csv'
CSV_FILE_SIA_P = f'/data/save/{BASE_NAME}/cal_SIA_P.csv'
CSV_FILE_SIA_I = f'/data/save/{BASE_NAME}/cal_SIA_I.csv'



if os.path.exists(f'/data/save/{BASE_NAME}'):
    raise Exception(f'BASE_NAME: {BASE_NAME} already exists')
else:
    os.mkdir(f'/data/save/{BASE_NAME}')

vector_dict.generate_vector_dict(CKPT_FILE, HPARAMS_FILE, VECTOR_FILE, REDUCE_FILE, REDUCE_TYPE)

calculate_CIO.calculate(CKPT_FILE, HPARAMS_FILE, REDUCE_FILE, CALCULATE_FILE_CIO)
calculate_to_csv.calculate_to_csv(CALCULATE_FILE_CIO, CSV_FILE_CIO)

calculate_SIA_D.calculate(CKPT_FILE, HPARAMS_FILE, REDUCE_FILE, CALCULATE_FILE_SIA_D)
calculate_to_csv.calculate_to_csv(CALCULATE_FILE_SIA_D, CSV_FILE_SIA_D)

calculate_SIA_P.calculate(CKPT_FILE, HPARAMS_FILE, REDUCE_FILE, CALCULATE_FILE_SIA_P)
calculate_to_csv.calculate_to_csv(CALCULATE_FILE_SIA_P, CSV_FILE_SIA_P)

calculate_SIA_I.calculate(CKPT_FILE, HPARAMS_FILE, REDUCE_FILE, CALCULATE_FILE_SIA_I)
calculate_to_csv.calculate_to_csv(CALCULATE_FILE_SIA_I, CSV_FILE_SIA_I)








