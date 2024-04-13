import pickle
import pandas as pd


def calculate_to_csv(calculate_file, csv_file):
    calculate_data = open(calculate_file, 'rb')
    calculate_data = pickle.load(calculate_data, encoding='iso-8859-1')
    data = pd.DataFrame(calculate_data)
    data.to_csv(csv_file, index=False, encoding='utf-8')

#
# if __name__ == '__main__':
#     calculate_to_csv(f'./save/new_0725_201944/calculate.pkl', f'./save/new_0725_201944/cal.csv')
