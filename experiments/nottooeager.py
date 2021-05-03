import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

# The scalar variable to minimize
x = tf.Variable(initial_value=0, name='x', trainable=True, dtype=tf.float32)

# Optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.01)


# Loss function
def loss_function():
    return (x - 4) ** 2


# Minimizer
def run(x_initial_guess, tol=1e-8, max_iter=10000):
    # Set initial
    x.assign(x_initial_guess)

    # Loop
    while True:

        # Save the current value to compute the error later
        x_init = x.numpy()

        # Train
        opt.minimize(loss_function, var_list=[x])

        # Calculate the error
        err = np.abs(x.numpy() - x_init)

        # Report!
        print("Iteration: %d Val: %f Error: %f" % (
            opt.iterations.numpy(),
            x.numpy(),
            err))

        # Check error
        if err < tol:
            print(f'stopping at err={err}<{tol}')
            return x.numpy()

        # Check iterations
        if opt.iterations > max_iter:
            print(f'stopping at max_iter={max_iter}')
            return x.numpy()


run(x_initial_guess=10.0)
