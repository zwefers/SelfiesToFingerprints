import pandas as pd

train_data = pd.read_csv("moses_train.csv")
test_data = pd.read_csv("moses_test.csv")
data = pd.concat([train_data, test_data])
data_path = data.to_csv("all_data.csv")