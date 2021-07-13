import pickle

import matplotlib.pyplot as plt
import numpy as np
from absl import flags, app
from matplotlib import animation

flags.DEFINE_string("rollout_path", None, help="Path to rollout pickle file")
flags.DEFINE_integer("step_stride", 0, help="Stride of steps to skip.")
flags.DEFINE_boolean("block_on_show", True, help="For test purposes.")

FLAGS = flags.FLAGS


def main(unused):
    if not FLAGS.rollout_path:
        raise ValueError("A `rollout_path` must be passed.")
    with open(FLAGS.rollout_path, "rb") as file:
        rollout_data = pickle.load(file)

    trajectory = np.concatenate([rollout_data["initial_velocity"], rollout_data['predicted_rollout']], axis=0)
    nodes_x = rollout_data['locations'][:, 0]
    nodes_y = rollout_data['locations'][:, 1]

    fig = plt.figure()
    plt.axes(xlim=(np.amin(nodes_x), np.amax(nodes_x)), ylim=(np.amin(nodes_x), np.amax(nodes_y)))

    def update(step_i):
        t_step = trajectory[step_i]
        t_x = t_step[:, 0]
        t_y = t_step[:, 1]
        t = np.multiply(t_x, t_y)
        cont = plt.tricontour(nodes_x, nodes_y, t)
        return cont

    anim = animation.FuncAnimation(fig, update, interval=10, repeat=True, save_count=200)
    plt.show(block=FLAGS.block_on_show)

    file = 'D:\\Users\\abdel\\Documents\\animation.gif'
    anim.save(file, writer='imagemagick', fps=30)


if __name__ == "__main__":
    app.run(main)
