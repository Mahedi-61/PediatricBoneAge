# import python modules
import os, random
import pandas as pd 
import numpy as np 
import cv2

database_path = "~/laptop/present_work/pediatric_bone_age/database/rsna_bone_age"
train_label = "train_label.csv"
test_label = "test_label.csv"

train_df = pd.read_csv(os.path.join(database_path, train_label))
print(train_df.head())

pid = list(train_df["id"])
age = list(train_df["boneage"])
male = list(train_df["male"])

# getting the length of all column
print(len(pid)) # 12,611
print(len(age)) # 12,611
print(len(male))# 12,611
# foudn no missing value


train_df['gender'] = train_df['male'].map(lambda x: 'male' if x else 'female')
boneage_mean = train_df['boneage'].mean()
boneage_div = 2*train_df['boneage'].std()

# we don't want normalization for now
boneage_mean = 0
boneage_div = 1.0
train_df['boneage_zscore'] = train_df['boneage'].map(lambda x: (x-boneage_mean)/boneage_div)
train_df.dropna(inplace = True)
print(train_df.sample(3))
