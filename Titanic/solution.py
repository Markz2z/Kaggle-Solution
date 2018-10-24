import tensorflow as tf
import numpy as np
import pandas as pd

#1.read data
all_data = pd.read_csv("train.csv", dtype={"Age":np.float64}, )
sess = tf.Session()

#2.split the data into train and result column && feature engineering
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
train_data = all_data[predictors]
result_data = all_data["Survived"]

train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())

train_data.loc[train_data["Sex"]=="male", "Sex"] = 1
train_data.loc[train_data["Sex"]=="female", "Sex"] = 0

train_data["Embarked"] = train_data["Embarked"].fillna("S")
train_data.loc[train_data["Embarked"]=='S', "Embarked"] = 0
train_data.loc[train_data["Embarked"]=='C', "Embarked"] = 1
train_data.loc[train_data["Embarked"]=='Q', "Embarked"] = 2
print(train_data)
print(result_data)

#3.build the computational graph
learning_rate = 0.01
training_epoch = 25

x=tf.placeholder(tf.float32,[891,7])
y=tf.placeholder(tf.float32,[891])

W=tf.Variable(tf.zeros([7,1]))
b=tf.Variable(tf.zeros([891]))

#pred = tf.nn.softmax(tf.matmul(x,W)+b)
pred = tf.matmul(x, W) + b
#cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
cost = tf.square(pred - y)
loss = tf.reduce_sum(cost)
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

print(train_data.shape)
print(result_data.shape)
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epoch):
    	sess.run(optimizer, {x:train_data, y:result_data})
    print "Optimization Finished!"
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:train_data, y:result_data})
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
