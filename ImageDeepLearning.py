import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
image = np.array([[
    [[1],[2],[3]],
    [[4], [5], [6]],
    [[7], [8], [9]]]], dtype=np.float)#1,3,3,1
print("image shape : ", image.shape)
#plt.imshow(image.reshape(3,3), cmap='Greys')
#plt.show()


#filter 2,2,1(color 수),3(filter수)

#weight = 1,1,1,1
weight = tf.constant([ [[[1, 10, -1]], [[1, 10, -1]]],
                      [[[1, 10, -1]], [[1, 10, -1]]],   ], dtype=tf.float64 )
print("weight shape : ", weight.shape)

conv2d = tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding='SAME')
conv2d_img = conv2d.eval()
print("conv2d shape : ", conv2d_img.shape)

conv2d_img = np.swapaxes(conv2d_img, 0, 3)

for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,3,i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')


#max pooling practice
image2 = np.array([[[[4]], [[3]]],[[[2]], [[1]]] ], dtype=np.float)
print("image2 shape : ", image2.shape)
pool = tf.nn.max_pool(image2, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
print("pool shape : ", pool.shape)
print("pool : ", pool.eval())



plt.show()
