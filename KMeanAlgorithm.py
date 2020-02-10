
import tensorflow as tf
import Data as d
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt



#데이터를 텐서로 옮긴다. Data에서 만든 랜덤상수들을 가지고 상수 텐서를 만든다.
vectors = tf.constant(d.vectors_set)

#k개의 중심을 랜덤으로 입력중에서 선택하도록 함.
k=2
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0,0], [k,-1]))

#해야하는 일은  각 vector와 centroides의 거리가 최소가 되는 centroides를 선택해야함
#하지만 vector는 shape가 [2000,2], centroides는 [4,2]
#뺄셈을 해야 거리를 구할 수 있으므로 이것들의 dimension을 확장시킨 후 뺀다
expanded_vectors =  tf.expand_dims(vectors, 0) #shape : [2000,2] -> [1, 2000, 2]
expanded_cetroides = tf.expand_dims(centroides, 1) #shape : [4,2] ->[4, 1, 2]
diff = tf.subtract(expanded_vectors, expanded_cetroides) #shape : [4, 2000, 2]

sqr = tf.square(diff)#차이 제곱
distances = tf.reduce_sum(sqr,2)#dimension2를 따라서 각 값들을 더해서 차원을 낮춘다. shape : [4, 2000]
assignments = tf.argmin(distances, 0)#0차원을 따라 가장 작은 값의 원소가 있는 인덱스 리턴 shape : [2000]

'''
for c in range(k):
    d = tf.equal(assignment, c)
    w = tf.where(c)
    r = tf.reshape(w, [1,-1])
    g = tf.gather(vectors, r)

r = tf.reduce_mean(g)
means = tf.concat(0, [r])
'''


means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])), axis=1) for c in range(k)], 0)

update_centroides = tf.assign(centroides, means)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(1):
    _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])

data = {"x": [], "y": [], "cluster": []}

for i in range(len(assignment_values)):
    data["x"].append(d.vectors_set[i][0])
    data["y"].append(d.vectors_set[i][1])
    data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
sb.lmplot("x", "y", data=df, fit_reg=False, height=6, hue="cluster", legend=False)
plt.show()

print(centroid_values)
