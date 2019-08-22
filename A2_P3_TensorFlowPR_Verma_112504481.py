import numpy as np
import tensorflow as tf
import sys
from pprint import pprint


filename =sys.argv[1]

fileContent = np.loadtxt(filename)
u=[]
v=[]
dict ={}
indices = []

for i in fileContent:
    fr = int(i[0])
    to = int(i[1])
    u.append(fr)
    v.append(to)
    indices.append([to,fr])
    dict[fr] = dict.get(fr ,0) + 1

maxU = max(u)
maxV = max(v)
maxValueInList= max(maxU,maxV)

Beta = 0.85

mat=tf.sparse.SparseTensor(indices=indices, values=[Beta/dict[each[1]] for each in indices], dense_shape=[maxValueInList+1, maxValueInList+1])

initialArray=np.ones((maxValueInList+1,1), dtype = float)

A_matrix = tf.placeholder(tf.float32, [maxValueInList+1, 1])
r_0 = tf.math.multiply(A_matrix, 1/maxValueInList)

r_1 = tf.placeholder(tf.float32, [maxValueInList+1, 1])

scalar=tf.math.scalar_mul(1-Beta,r_0)

init = tf.global_variables_initializer()
threshold= 0.00000001

# def error_norm(r1, r_0):
#     return tf.reduce_sum(tf.abs(r1 - r_0))

current=[]
with tf.Session() as sess:

    sess.run(init)
    previous = sess.run(r_0, feed_dict={A_matrix: initialArray})

    while(True):
        # s = sess.run(scalar,feed_dict={r_0:previous})
        current = sess.run(tf.math.add(tf.sparse.sparse_dense_matmul(mat,r_0),scalar),feed_dict={mat:mat.eval(),r_0:previous,scalar:sess.run(scalar,feed_dict={r_0:previous})})
        # err= tf.reduce_sum(tf.abs(current - previous))
        if sess.run(tf.reduce_sum(tf.abs(current - previous)))<threshold:
            break
        previous = current
final = []

for i in range(len(current)):
    final.append([i, current[i]])

list = sorted(final, key = lambda x:x[1], reverse=True)

pprint("The top 20 node ids along with their ranks : ")
pprint(list[:20])

# pprint(list[-20:][::-1])
# pprint(list[-20])
pprint("The bottom 20 node ids along with their ranks")
# pprint(list[:20])
pprint(list[-20:][::-1])



