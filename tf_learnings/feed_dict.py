import tensorflow as tf

x = tf.placeholder(tf.float32, name="input1")
y = tf.placeholder(tf.float32, name="input2")

z = x * y

b = "bobo"
print({"bebe": 3})
print({b: 33}) # b evaluates to "bobo", which becomes the dictionary hash key.

with tf.Session() as sess:
    # In python land, x and y evaluate to placeholder values, which become the
    # hash keys of the feed_dict.
    print(sess.run([x, y, z], feed_dict={
        x: [3.],
        y: [.3]
    }))
    # You can use the values fed into x and y elsewhere in python code (even
    # where they are out of scope in python) because they are tracked in the
    # tensorflow session graph.  Here we repeat the above call, but refer to
    # the placeholders by the name we gave them in the tensorflow graph.
    print(sess.run([x, y, z], feed_dict={
        sess.graph.get_tensor_by_name("input1:0"): [3.],
        sess.graph.get_tensor_by_name("input2:0"): [.3]
    }))
    # It's a bit weird that we need to append ":0" to the names we gave.
    # That's necessary because we technically defined placeholder operations,
    # and we're getting the first (or zeroth) outputs of those operations.