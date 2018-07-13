from utils import *

train_file = 'data/train.csv'
test_file = 'data/test.csv'

train_data = load_csv_data(train_file)
_, y, x = read_content(train_data)
