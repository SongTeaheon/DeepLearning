from tensorflow.examples.tutorials.mnist import input_data
#tensorflow에서 제공하는 연습용 데이터 (mnist데이터)를 가져온다.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784], name="x") #shape : None, 784 : None은 x의 배치숫자(이미지 개수)로 뭐든지 될 수 있다는 의미, 784는 이미지 하나당 데이터 개수
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) +b)
y_ = tf.placeholder(tf.float32, [None, 10], name="y_") #y_는 결과데이터(이미지가 어떤 숫자인지에 대한 정보) 1,0,0,0,0,0... 이런 식으로 나타내고 있음.

#크로스 엔트로피를 이용해서
cross_entropy = tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    feed_dict = {x:batch_xs, y_:batch_ys}
    sess.run(train_step, feed_dict)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))