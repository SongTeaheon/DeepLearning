#데이터 분석

import tensorflow as tf
import LinearData
import matplotlib.pyplot as plt

def showGraph():
    plt.plot(LinearData.x_data, LinearData.y_data, 'ro')
    plt.plot(LinearData.x_data, pre_w * LinearData.x_data + pre_b)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-2, 2)
    plt.ylim(0.1, 0.6)
    plt.legend()
    plt.show()


#Variable은 변수 정의!
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * LinearData.x_data + b

loss = tf.reduce_mean(tf.square(y-LinearData.y_data))

#optimizer 구현(경사하강법 적용)
optimizer = tf.train.GradientDescentOptimizer(0.5) #0.5는 학습속도
train = optimizer.minimize(loss)

#변수 초기화 세션 생성, 호출
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

#training 반복(8번)

for step in range(100):
    sess.run(train)


pre_w = sess.run(W)
pre_b = sess.run(b)

print(pre_w, pre_b)

showGraph()





