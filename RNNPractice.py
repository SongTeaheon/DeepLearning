import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


def get_a_cell(lstm_size, keep_prob):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return drop





#기본 값 설정
time_steps = sequence_length = 7
data_dim = 6
output_dim = 1

#csv파일 가져오기
xy = pd.read_csv('/Users/songtaeheon/Documents/stockProgram/stockData/WooJinData.csv', sep='\t')

print(xy.head())

#불필요한 컬럼 제거
xy = xy.drop(columns={'Unnamed: 0', '날짜'})
xy = xy[::-1] #reverse order
print(xy.head())

#scale 0-1 - 모든 값을 0에서 1사이의 수로 바꾼다.
scaler = MinMaxScaler()
xy_scale = scaler.fit_transform(xy)

#x데이터 y데이터 설정
x = xy_scale
y= xy_scale[:,[0]] #y데이터는 종가 컬럼만 가지면 됨!

print(x[:5])
print(y[0:5])

#sequence크기만큼 자잘자잘하게 잘라서 data셋에 넣는다.

#데이터가 1 2 3 4 5 6 7 8 9이고 seq가 3이면 x가 123 234 345 ...이런 식으로
#y는 이후의 모든 결과값들이므로 456789(123의 결과) 56789(234의 결과) 6789 789 ...이런 식으로!

data_X = []
data_Y = []
for i in range(len(y) - sequence_length):
    _x = x[i:i+sequence_length]
    _y = y[i+sequence_length]
    data_X.append(_x)
    data_Y.append(_y)

#데이터 셋을 트레인용과 테스트용으로 나눈다. 7대3
train_size = int(len(data_Y)*0.7)
train_test = len(data_Y) - train_size

trainX, testX = data_X[0:train_size], data_X[train_size:len(data_Y)]
trainY, testY = data_Y[0:train_size], data_Y[train_size:len(data_Y)]

#place홀더는 변수의 타입(텐서)을 미리 정해놓는것!
#[None,sequence_length ,data_dim]는 shape임! None*sequence_length*data_dim
X = tf.placeholder(tf.float32, [None,sequence_length ,data_dim])
Y = tf.placeholder(tf.float32, [None, 1]) #sequence length 안넣는 거일 수 도

#rnn생성 및 구현
#Long short-term memory unit (LSTM) recurrent network cell.
#num_units은 노드의 수. state_is_tuple은 투플형태로 하겠냐는 거인듯...
cell = tf.nn.rnn_cell.LSTMCell(num_units=sequence_length, state_is_tuple=True)

#RNN생성 **이 함수는 deprecated됨. keras.layers.RNN(cell) 써야!
#outputs: The RNN output Tensor.
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

#outputs[:,-1]을 하는 이유는 모든 cell에서 나오는 output을 쓰는 것이 아니라 마지막에 나온 output만 쓰기 때문이다.
#activation_fn이 None인 것은 리니어한 곱이 활성함수로 사용한 것!
Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim, activation_fn=None)

#loss 함수
loss = tf.reduce_sum(tf.square(Y - Y_pred))

#optimizer : loss를 최소화!
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

#session실행
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#학습
for i in range(1000):
    _, l = sess.run([train, loss], feed_dict={X:trainX, Y:trainY})
    print(i, l)

#테스트
test_Predict = sess.run(Y_pred, feed_dict={X:testX})


import matplotlib.pyplot as plt
plt.plot(testY)
plt.plot(test_Predict)
plt.show()




