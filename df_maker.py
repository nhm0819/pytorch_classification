from sklearn.model_selection import train_test_split
from glob import glob
import pandas as pd
import os

# image paths
electric_images = glob("E:\\work\\kesco\\raw_data\\20211008\\segmented_bad_data\\electric\\*.jpg")
flame_images = glob("E:\\work\\kesco\\raw_data\\20211008\\segmented_bad_data\\flame\\*.jpg")

# aggregate
images = electric_images + flame_images
len(images)


# data frame
dict = {'electric':0, 'flame':1}
df = pd.DataFrame(columns=['path','label'])

for image in images:
    split_path = image.split("\\")
    image_path = os.path.join(split_path[5], split_path[6], split_path[7])
    label = image.split("\\")[6]
    df = df.append({'path': image_path, 'label': dict[label]}, ignore_index=True)
len(df)

# save
df.to_csv("bad_segmented_all.csv",index=False)


# train:val:test = 8:1:1
train, val = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
# val, test = train_test_split(val, test_size=0.5, random_state=42, shuffle=True)

train.to_csv("segmented_train.csv", index=False)
val.to_csv("segmented_test.csv", index=False)
# test.to_csv("test.csv", index=False)