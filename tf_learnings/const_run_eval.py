import tensorflow as tf

x = tf.constant(3.0)
y = tf.constant(2.0)
z = tf.constant(5.0)

h = x + y
g = h * z

with tf.Session() as sess:
    print(sess.run(x)) # 3
    print(sess.run(h)) # 5
    print(h.eval()) # 5
    print(sess.run(g)) # 25
    print(sess.run([g, h])) # 25, 5
