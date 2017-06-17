import tensorflow as tf

# Define a stateful variable that we'll increment.
counter = tf.Variable(0)
# Define an intermediate variable holding the incremented counter.
new_value = counter + tf.Constant(1)
# Assign the value of the intermediate variable to the counter variable.
incremented_value = tf.assign(counter, new_value)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print(sess.run(counter)) # 0
    for _ in range(3):
        sess.run(incremented_value)
        print(sess.run(counter)) # 1, 2, 3

# What? We run an assignment operation three times, but the value of counter
# changes each time?  It looks stupid and counterintuitive, but it's a key part
# of tensorflow.  Here's that whole mess with variable names and comments that
# better reflect what's going on.

# Define a stateful variable that we'll increment.
counter = tf.Variable(0)
# Define an operation that creates an intermediate variable holding the
# incremented counter.
create_new_value = counter + tf.Constant(1)
# Define an operation that assigns the value of the intermediate variable to
# the counter variable.
assign_new_value = tf.assign(counter, create_new_value)

# With bad variable names, the above looked like we'd already done declaration,
# initialization, and mutation.  But now hopefully it's clear that all we've
# actually done is the declaration step (in the form of a computation graph).  

with tf.Session() as sess:
    # _Now_ we'll do the initialization step.
    sess.run(tf.global_variables_initializer())
    
    print(sess.run(counter)) # 0
    for _ in range(3):
        sess.run(assign_new_value)
        print(sess.run(counter)) # 1, 2, 3

# It still might look weird that we change the counter state by running the
# same assignment three times.  But we've declared a computation graph, and the
# structure of the graph is actually the structure of computation dependencies.
# The assign_new_value operation is dependent on the create_new_value
# operation, so it's run implicitly when we run assign_new_value.

# Why do we want that power?  So that we can declare a complex computation
# graph (like a neural net full of nonlinearities, convolutions, and millions
# of parameters), and then declare a single loss function and optimization
# operation that depends on the whole lot.  That way we can update variables
# across the entire computation graph implicitly, just by running the final
# optimization operation.