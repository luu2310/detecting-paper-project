import pandas as pd
from sklearn.model_selection import train_test_split
# data = []
# for i in range(1,201):
#     index = f"{i:03d}"
#     data.append({
#         "images": f"data/images/paper_{index}.jpg",
#         "labels": f"data/labels/paper_{index}.txt"
#     })

# data_df = pd.DataFrame(data)
# data_df.to_csv("data/CSVs/dataset.csv", index=False)
# print(data_df)

dataset = pd.read_csv(r"./data/CSVs/dataset.csv")

train_data, val_data = train_test_split(dataset, test_size= 0.3)
train_data.to_csv("data/CSVs/train_df.csv", index=False)
val_data.to_csv("data/CSVs/val_df.csv", index=False)

