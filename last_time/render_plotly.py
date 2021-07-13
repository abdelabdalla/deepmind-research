import pickle

import matplotlib.pyplot as plt
import numpy as np
from absl import flags, app

flags.DEFINE_string("rollout_path", None, help="Path to rollout pickle file")
flags.DEFINE_integer("step_stride", 0, help="Stride of steps to skip.")
flags.DEFINE_boolean("block_on_show", True, help="For test purposes.")

FLAGS = flags.FLAGS

TYPE_TO_COLOR = {
    3: "black",  # Boundary.
    5: "blue",  # Fluid.
}


def main(unused):
    if not FLAGS.rollout_path:
        raise ValueError("A `rollout_path` must be passed.")
    with open(FLAGS.rollout_path, "rb") as file:
        rollout_data = pickle.load(file)

    node_x = rollout_data['locations'][:, 0]
    node_y = rollout_data['locations'][:, 1]

    trajectory = np.concatenate([rollout_data["initial_velocity"], rollout_data['predicted_rollout']], axis=0)

    t = trajectory[0]  # .reshape((len(node_x), len(node_y)))


if __name__ == "__main__":
    app.run(main)
