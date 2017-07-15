import tensorflow as tf
import numpy as np
import pandas as pd

#1.read data
all_data = pd.read_csv("train.csv", dtype={"Age":np.float64}, )

#2.split the data into train and result column
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
train_data = all_data[predictors]
result_data = all_data["Survived"]
#print(train_data)
#print(result_data)

#3.build the computational graph
weight = tf.Variable([.1,.1,.1,.1,.1,.1,.1], dtype=tf.float32)
bias = tf.Variable([.1,.1,.1,.1,.1,.1,.1], dtype=tf.float32)

linear_model = weight * 

