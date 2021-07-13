import pickle
import numpy as np
from absl import flags, app
import meshio

flags.DEFINE_string("rollout_path", None, help="Path to rollout pickle file")

FLAGS = flags.FLAGS


def main(unused):
    if not FLAGS.rollout_path:
        raise ValueError("A `rollout_path` must be passed.")
    with open(FLAGS.rollout_path, "rb") as file:
        rollout_data = pickle.load(file)

    trajectory = np.concatenate([rollout_data["initial_velocity"], rollout_data['predicted_rollout']], axis=0)
    nodes = rollout_data['locations']
    nodes_x = rollout_data['locations'][:, 0]
    nodes_y = rollout_data['locations'][:, 1]
    edges = rollout_data['connections']

    cells = [('triangle', edges)]

    for i in range(0, len(trajectory)):
        t_step = trajectory[i]
        t_x = t_step[:, 0]
        t_y = t_step[:, 1]
        t = np.absolute(np.multiply(t_x, t_y))

        mesh = meshio.Mesh(nodes, cells, point_data={'Velocity_Magnitude': t})
        mesh.write('D:\\Users\\abdel\\Documents\\new_render\\frame' + str(i) + '.vtk')


if __name__ == "__main__":
    app.run(main)
