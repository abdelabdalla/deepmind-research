import time

from graph_nets import blocks
from graph_nets import utils_tf
from graph_nets.demos import models
from matplotlib import pyplot as plt
import numpy as np
import sonnet as snt
import tensorflow as tf

tf.enable_eager_execution()

node_noise_level = 0.04

dataset_tr = tf.data.TFRecordDataset('/Users/abdelabdalla/Downloads/data/test.tfrecord')
dataset_te = tf.data.TFRecordDataset('/Users/abdelabdalla/Downloads/data/validate.tfrecord')

feature_description = {
    'key': tf.io.FixedLenFeature([], tf.int64),
    'velocity': tf.io.FixedLenFeature([], tf.string),
    'location': tf.io.FixedLenFeature([], tf.string),
    'connections': tf.io.FixedLenFeature([], tf.string)
}


def _parse_data_func(proto):
    return tf.io.parse_single_example(proto, feature_description)


def physics(receiver_nodes, sender_nodes, current_vel):
    diff = receiver_nodes[..., 0:2] - sender_nodes[..., 0:2]
    acc = tf.subtract(current_vel[..., 0:2], diff)
    return acc


def euler_integrator(nodes, acceleration, step_size):
    is_wall = nodes[..., 2]
    # acceleration *= 1 - is_wall
    new_vel = nodes[..., 0:2] + acceleration  # * step_size
    n_v = new_vel.numpy()
    i_w = np.reshape(is_wall.numpy(), (-1, 1))
    output = np.concatenate((n_v, i_w), axis=1)
    return tf.convert_to_tensor(output)


class NSSimulator(snt.AbstractModule):

    def __init__(self, step_size, name='NSSimulator'):
        super(NSSimulator, self).__init__(name=name)
        self.step_size = step_size

        with self._enter_variable_scope():
            self._aggregator = blocks.ReceivedEdgesToNodesAggregator(
                reducer=tf.unsorted_segment_sum
            )

    def _build(self, graph):
        receiver_nodes = blocks.broadcast_receiver_nodes_to_edges(graph)
        sender_nodes = blocks.broadcast_sender_nodes_to_edges(graph)
        acceleration_per_edge = physics(receiver_nodes, sender_nodes, graph.edges)
        graph = graph.replace(edges=acceleration_per_edge)

        acceleration_per_node = self._aggregator(graph)

        updated_velocities = euler_integrator(graph.nodes, acceleration_per_node, self.step_size)

        graph = graph.replace(nodes=updated_velocities)

        return graph


def prediction_to_next_state(input_graph, predicted_graph, step_size):
    # manually integrate velocities to compute new vels
    print('1')
    new_vel = tf.add(input_graph.nodes[..., 0:2], predicted_graph.nodes[..., 0:2])  # * step_size
    print('2')
    n_v = new_vel.numpy()
    i_w = np.reshape(input_graph.nodes[..., 2].numpy(), (-1, 1))
    # new_nodes = tf.concat(
    #    [new_vel, input_graph.nodes[..., 3]], axis=1)
    comb = np.concatenate((n_v, i_w), axis=1)
    comb_tf = tf.convert_to_tensor(comb)
    print('3')
    return input_graph.replace(nodes=comb_tf)


def rollout_physics(simulator, graph, steps, step_size):
    def body(t, graph, nodes_per_step):
        predicted_graph = simulator(graph)
        if isinstance(predicted_graph, list):
            predicted_graph = predicted_graph[-1]

        graph = prediction_to_next_state(graph, predicted_graph, step_size)

        return t + 1, graph, nodes_per_step.write(t, graph.nodes)

    nodes_per_step = tf.TensorArray(
        dtype=graph.nodes.dtype, size=steps + 1, element_shape=graph.nodes.shape)
    nodes_per_step = nodes_per_step.write(0, graph.nodes)

    _, g, nodes_per_step = tf.while_loop(
        lambda t, *unused_args: t <= steps, body, loop_vars=[1, graph, nodes_per_step])

    return g, nodes_per_step.stack()


def apply_noise(graph, node_noise_level, edge_noise_level, global_noise_level):
    node_position_noise = tf.random_uniform(
        [graph.nodes.shape[0].value, 2],
        minval=-node_noise_level,
        maxval=node_noise_level,
        dtype=tf.float64)
    edge_spring_constant_noise = tf.random_uniform(
        [graph.edges.shape[0].value, 1],
        minval=-edge_noise_level,
        maxval=edge_noise_level,
        dtype=tf.float64)
    global_gravity_y_noise = tf.random_uniform(
        [graph.globals.shape[0].value, 1],
        minval=-global_noise_level,
        maxval=global_noise_level,
        dtype=tf.float64)

    return graph.replace(
        nodes=tf.concat(
            [graph.nodes[..., :2] + node_position_noise, graph.nodes[..., 2:]],
            axis=-1),
        edges=tf.concat(
            [
                graph.edges[..., :1] + edge_spring_constant_noise,
                graph.edges[..., 1:]
            ],
            axis=-1),
        globals=tf.concat(
            [
                graph.globals[..., :1],
                graph.globals[..., 1:] + global_gravity_y_noise
            ],
            axis=-1))


def generate_trajectory(simulator, graph, steps, step_size, node_noise_level, edge_noise_level, global_noise_level):
    # TODO add noise graph = apply_noise(graph, node_noise_level, edge_noise_level, global_noise_level)
    _, n = rollout_physics(simulator, graph, steps, step_size)
    return graph, n


def graph_builder_old(node_loc, node_con):
    """
    Args:
        node_loc: node locations
        node_con: node connections

    Returns:
        data_dict: dict with nodes, edges, receivers, senders, globals
        trajectory
    """
    nodes = node_loc.numpy()
    r, d = nodes.shape
    boundary = np.zeros((r, 3))
    nodes = np.concatenate((nodes, boundary), axis=1)

    for i in range(0, nodes.shape[0]):
        if nodes[i, 0] == 0 or nodes[i, 1] == 0:
            nodes[i, 4] = 1

    senders, receivers = [], []
    connections = node_con.numpy()
    for points in connections:
        senders.append(points[0])
        receivers.append(points[1])
        receivers.append(points[0])
        senders.append(points[1])

        senders.append(points[0])
        receivers.append(points[2])
        receivers.append(points[0])
        senders.append(points[2])

        senders.append(points[1])
        receivers.append(points[2])
        receivers.append(points[1])
        senders.append(points[2])

    edges = np.zeros((len(senders), 1))

    return {
        'globals': [],
        'nodes': nodes,
        'edges': edges,
        'receivers': receivers,
        'senders': senders
    }


def graph_builder(node_loc, node_con):
    node_locs = node_loc.numpy()
    r, d = node_locs.shape
    nodes = np.zeros((r, 3), dtype='float64')
    max_x = np.amax(node_locs[:, 0])
    max_y = np.amax(node_locs[:, 1])
    # check if nodes is a fluid (0), wall (1), or inlet/outlet (2)
    for i in range(0, nodes.shape[0]):
        if node_locs[i, 0] == 0 or node_locs[i, 1] == 0:
            nodes[i, 2] = 1
        elif node_locs[i, 0] == max_x or node_locs[i, 1] == max_y:
            nodes[i, 2] = 2

    edges, senders, receivers = [], [], []
    connections = node_con.numpy()
    for points in connections:
        senders.append(points[0])
        receivers.append(points[1])
        u_ij = [(node_locs[points[0], 0] - node_locs[points[1], 0]),
                (node_locs[points[0], 1] - node_locs[points[1], 1])]
        edges.append(np.concatenate((u_ij, [np.sqrt((u_ij[0] ** 2) + (u_ij[1] ** 2))])).astype('float64'))
        receivers.append(points[0])
        senders.append(points[1])
        u_ij = [(node_locs[points[1], 0] - node_locs[points[0], 0]),
                (node_locs[points[1], 1] - node_locs[points[0], 1])]
        edges.append(np.concatenate((u_ij, [np.sqrt((u_ij[0] ** 2) + (u_ij[1] ** 2))])).astype('float64'))

        senders.append(points[0])
        receivers.append(points[2])
        u_ij = [(node_locs[points[0], 0] - node_locs[points[2], 0]),
                (node_locs[points[0], 1] - node_locs[points[2], 1])]
        edges.append(np.concatenate((u_ij, [np.sqrt((u_ij[0] ** 2) + (u_ij[1] ** 2))])).astype('float64'))
        receivers.append(points[0])
        senders.append(points[2])
        u_ij = [(node_locs[points[2], 0] - node_locs[points[0], 0]),
                (node_locs[points[2], 1] - node_locs[points[0], 1])]
        edges.append(np.concatenate((u_ij, [np.sqrt((u_ij[0] ** 2) + (u_ij[1] ** 2))])).astype('float64'))

        senders.append(points[1])
        receivers.append(points[2])
        u_ij = [(node_locs[points[1], 0] - node_locs[points[2], 0]),
                (node_locs[points[1], 1] - node_locs[points[2], 1])]
        edges.append(np.concatenate((u_ij, [np.sqrt((u_ij[0] ** 2) + (u_ij[1] ** 2))])).astype('float64'))
        receivers.append(points[1])
        senders.append(points[2])
        u_ij = [(node_locs[points[2], 0] - node_locs[points[1], 0]),
                (node_locs[points[2], 1] - node_locs[points[1], 1])]
        edges.append(np.concatenate((u_ij, [np.sqrt((u_ij[0] ** 2) + (u_ij[1] ** 2))])).astype('float64'))

    return {
               'globals': np.array([-1.], dtype='float64'),
               'nodes': nodes,
               'edges': edges,
               'receivers': receivers,
               'senders': senders
           }, nodes[:, 2]


def rollout_builder(node_vel, bounds, num_time_steps):
    out = []
    i_prev = -1
    for j in range(0, len(bounds)):
        for i in range(0, num_time_steps + 1):  # len(node_vel[j])):

            print('i: ' + str(i) + ' j: ' + str(j))
            bound = np.reshape(bounds[j], (-1, 1))
            vel = node_vel[j][i].numpy()

            nodes = np.concatenate((vel, bound), axis=1)

            if i_prev != i and j == 0:
                out.append(nodes)
            else:
                out[i] = np.concatenate((out[i], nodes), axis=0)

            i_prev = i

    return out


def rollout_builder_old(node_loc, node_vel):
    out = []
    i_prev = -1
    for j in range(0, len(node_loc)):
        for i in range(0, len(node_vel[j])):

            print('i: ' + str(i) + ' j: ' + str(j))
            loc = node_loc[j].numpy()
            vel = node_vel[j][i].numpy()

            r, d = loc.shape
            boundary = np.zeros((r, 1))
            nodes = np.concatenate((loc, vel), axis=1)
            nodes = np.concatenate((nodes, boundary), axis=1)

            for k in range(0, nodes.shape[0]):
                if nodes[k, 0] == 0 or nodes[k, 1] == 0:
                    nodes[k, 4] = 1

            if i_prev != i and j == 0:
                out.append(nodes)
            else:
                out[i] = np.concatenate((out[i], nodes), axis=0)

            i_prev = i

    # out2 = np.array(out)

    return out  # tf.convert_to_tensor(out, dtype=tf.float32)


# node_trajectory_tr = rollout_builder(node_locations_tr, node_vels_tr)
def create_loss_ops(target_op, output_ops):
    """Create supervised loss operations from targets and outputs.

  Args:
    target_op: The target velocity tf.Tensor.
    output_ops: The list of output graphs from the model.

  Returns:
    A list of loss values (tf.Tensor), one per output op.
  """
    loss_ops = {}
    loss_ops['name'] = {'loss_ops'}
    loss_ops = [
        tf.reduce_mean(
            tf.reduce_sum((output_op.nodes - target_op[..., 0:2]) ** 2, axis=-1))
        for output_op in output_ops
    ]
    return loss_ops


def make_all_runnable_in_session(*args):
    """Apply make_runnable_in_session to an iterable of graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]


num_processing_steps_tr = 1
num_processing_steps_te = 1

num_training_iterations = 100
num_time_steps = 50
step_size = 0.005

model = models.EncodeProcessDecode(node_output_size=2)

parsed_data_tr = dataset_tr.map(_parse_data_func)
parsed_data_te = dataset_te.map(_parse_data_func)

node_locations_tr = []
node_connections_tr = []
node_vels_tr = []
node_boundaries_tr = []

static_graphs_tr = []

print('loading training data')

i = 0
for testcase in parsed_data_tr:
    node_locations_tr.append(tf.io.parse_tensor(testcase['location'], tf.float32))
    node_connections_tr.append(tf.io.parse_tensor(testcase['connections'], tf.int64))
    node_vels_tr.append(tf.io.parse_tensor(testcase['velocity'], tf.float32))

    graph, bounds = graph_builder(node_locations_tr[-1], node_connections_tr[-1])
    static_graphs_tr.append(graph)
    node_boundaries_tr.append(bounds)

    if i == 1:
        break
    else:
        i += 1

node_locations_te = []
node_connections_te = []
node_vels_te = []
node_boundaries_te = []

static_graphs_te = []

print('loading testing data')
i = 0
for testcase in parsed_data_te:
    node_locations_te.append(tf.io.parse_tensor(testcase['location'], tf.float32))
    node_connections_te.append(tf.io.parse_tensor(testcase['connections'], tf.int64))
    node_vels_te.append(tf.io.parse_tensor(testcase['velocity'], tf.float32))

    graph, bounds = graph_builder(node_locations_te[-1], node_connections_te[-1])
    static_graphs_te.append(graph)
    node_boundaries_te.append(bounds)

    if i == 1:
        break
    else:
        i += 1

base_graphs_tr = utils_tf.data_dicts_to_graphs_tuple(static_graphs_tr)

simulator = NSSimulator(step_size=step_size)

initial_conditions_tr, true_trajectory_tr = generate_trajectory(
    simulator,
    base_graphs_tr,
    num_time_steps,
    step_size,
    node_noise_level=0.04,
    edge_noise_level=5.0,
    global_noise_level=1.0)

t = tf.random.uniform([], minval=0, maxval=num_time_steps - 1, dtype=tf.int32)
input_graph_tr = initial_conditions_tr.replace(nodes=true_trajectory_tr[t])
target_nodes_tr = true_trajectory_tr[t + 1]
output_ops_tr = model(input_graph_tr, num_processing_steps_tr)

# testing data
base_graphs_te = utils_tf.data_dicts_to_graphs_tuple(static_graphs_te)
true_trajectory_te = rollout_builder(node_vels_te, node_boundaries_te, num_time_steps)
true_trajectory_te = tf.convert_to_tensor(true_trajectory_te)
_, predicted_trajectory_te = rollout_physics(
    lambda x: model(x, num_processing_steps_te),
    base_graphs_te, num_time_steps, step_size)

# Training loss.
loss_ops_tr = create_loss_ops(target_nodes_tr, output_ops_tr)
# Training loss across processing steps.
loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr

# Test/generalization loss: 4-mass.
loss_op_te = tf.reduce_mean(
    tf.reduce_sum(
        (predicted_trajectory_te[..., 0:2] -
         true_trajectory_te[..., 0:2]) ** 2,
        axis=-1))

learning_rate = 1e-3
optimizer = tf.train.AdamOptimizer(learning_rate)
"""
x = output_ops_tr[0].nodes


def create_loss():
    loss_ops = {}
    loss_ops['name'] = {'loss_ops'}
    loss_ops = tf.reduce_mean(
        tf.reduce_sum(
            tf.square(
                tf.subtract(x, target_nodes_tr[..., 0:2])),
            axis=-1))
    return loss_ops


step_op = optimizer.minimize(loss=create_loss(), var_list=[x])
"""
x = tf.Variable(output_ops_tr[0].nodes)
with tf.GradientTape(persistent=False) as t:
    # Loss function
    loss_ops = tf.reduce_mean(
        tf.reduce_sum(
            tf.square(
                tf.subtract(x, target_nodes_tr[..., 0:2])),
            axis=-1))
gradients = t.gradient(loss_ops, x)

# Apply the gradient
optimizer.apply_gradients(zip(gradients, output_ops_tr[0].nodes))

input_graph_tr = make_all_runnable_in_session(input_graph_tr)
base_graphs_te = make_all_runnable_in_session(base_graphs_te)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


last_iteration = 0
logged_iterations = []
losses_tr = []
losses_te = []

log_every_seconds = 20

print("# (iteration number), T (elapsed seconds), "
      "Ltr (training 1-step loss), "
      "Lte (test/generalization rollout loss)")

start_time = time.time()
last_log_time = start_time
for iteration in range(last_iteration, num_training_iterations):
    last_iteration = iteration
    train_values = sess.run({
        "step": step_op,
        "loss": loss_op_tr,
        "input_graph": input_graph_tr,
        "target_nodes": target_nodes_tr,
        "outputs": output_ops_tr
    })
    the_time = time.time()
    elapsed_since_last_log = the_time - last_log_time
    if elapsed_since_last_log > log_every_seconds:
        last_log_time = the_time
        test_values = sess.run({
            "loss_te": loss_op_te,
            "true_rollout_te": true_trajectory_te,
            "predicted_rollout_te": predicted_trajectory_te,
        })
        elapsed = time.time() - start_time
        losses_tr.append(train_values["loss"])
        losses_te.append(test_values["loss_te"])
        logged_iterations.append(iteration)
        print("# {:05d}, T {:.1f}, Ltr {:.4f}, Lte {:.4f}".format(
            iteration, elapsed, train_values["loss"], test_values["loss_4"]))

"""
print('train')
base_graphs_tr = utils_tf.data_dicts_to_graphs_tuple(static_graphs_tr)

node_trajectory_tr = rollout_builder(node_vels_tr, node_boundaries_tr)

a = []
for x in node_trajectory_tr:
    a.append(tf.convert_to_tensor(x, dtype=tf.float64))
node_trajectory_tr = a


t = tf.random.uniform([], minval=0, maxval=num_time_steps - 1, dtype=tf.int32)

input_graph_tr = base_graphs_tr.replace(nodes=node_trajectory_tr[t])

# TODO fix noise
# input_graph_tr = apply_noise(input_graph_tr)
target_nodes_tr = node_trajectory_tr[t + 1]
output_ops_tr = model(input_graph_tr, num_processing_steps_tr)

print('test')
base_graphs_te = utils_tf.data_dicts_to_graphs_tuple(static_graphs_te)
node_trajectory_te = rollout_builder(node_vels_te, node_boundaries_te)

b = []
for x in node_trajectory_te:
    b.append(tf.convert_to_tensor(x, dtype=tf.float64))
node_trajectory_te = b

input_graph_te = base_graphs_te.replace(nodes=node_trajectory_te[t])
true_nodes_rollout_te = node_trajectory_te[t]
predicted_nodes_rollout_te = model(input_graph_te, num_processing_steps_te)

# Training loss.
loss_ops_tr = create_loss_ops(target_nodes_tr, output_ops_tr)
# Training loss across processing steps.
loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr
# Test/generalization loss: 4-mass.
loss_op_te = tf.reduce_mean(
    tf.reduce_sum(
        (predicted_nodes_rollout_te[..., 2:4] -
         true_nodes_rollout_te[..., 2:4]) ** 2,
        axis=-1))

learning_rate = 1e-3
optimizer = tf.train.AdamOptimizer(learning_rate)
step_op = optimizer.minimize(loss_op_tr)

input_graph_tr = make_all_runnable_in_session(input_graph_tr)
initial_conditions_te = make_all_runnable_in_session(input_graph_te)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

last_iteration = 0
logged_iterations = []
losses_tr = []
losses_te = []

log_every_seconds = 20

print("# (iteration number), T (elapsed seconds), "
      "Ltr (training 1-step loss), "
      "Le (test/generalization rollout loss)")

start_time = time.time()
last_log_time = start_time
for iteration in range(last_iteration, num_training_iterations):
    last_iteration = iteration
    train_values = sess.run({
        "step": step_op,
        "loss": loss_op_tr,
        "input_graph": input_graph_tr,
        "target_nodes": target_nodes_tr,
        "outputs": output_ops_tr
    })
    the_time = time.time()
    elapsed_since_last_log = the_time - last_log_time
    if elapsed_since_last_log > log_every_seconds:
        last_log_time = the_time
        test_values = sess.run({
            "loss_te": loss_op_te,
            "true_rollout_te": true_nodes_rollout_te,
            "predicted_rollout_te": predicted_nodes_rollout_te
        })
        elapsed = time.time() - start_time
        losses_tr.append(train_values["loss"])
        losses_te.append(test_values["loss_te"])
        logged_iterations.append(iteration)
        print("# {:05d}, T {:.1f}, Ltr {:.4f}, Lge4 {:.4f}".format(
            iteration, elapsed, train_values["loss"], test_values["loss_4"]))
"""
