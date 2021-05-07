import numpy as np
import tensorflow as tf


def _get_connectivity(node_locations, node_connections):
    num_nodes = len(node_locations)
    # tf.print('node loc: ', node_locations[0], 'node con: ', node_connections[0], output_stream=sys.stdout)

    senders = []
    receivers = []

    for i in range(0, len(node_connections)):
        zero = node_connections[i][0]
        one = node_connections[i][1]
        two = node_connections[i][2]
        senders.append(zero)
        receivers.append(one)
        senders.append(one)
        receivers.append(zero)

        senders.append(zero)
        receivers.append(two)
        senders.append(two)
        receivers.append(zero)

        senders.append(one)
        receivers.append(two)
        senders.append(two)
        receivers.append(one)

    senders = np.array(senders)
    receivers = np.array(receivers)

    # senders = np.repeat(range(num_nodes), [len(a) for a in node_connections])
    # receivers = np.concatenate(node_connections, axis=0)
    return senders, receivers


def _get_connectivity_for_batch(node_locations, node_connections, n_nodes, n_con):
    nodes_per_graph_list = np.split(node_locations, np.cumsum(n_nodes[:-1]), axis=0)
    connections_per_graph_list = np.split(node_connections, np.cumsum(n_con[:-1]), axis=0)

    receivers_list = []
    senders_list = []
    n_edge_list = []
    num_nodes_in_previous_graph = 0

    for i in range(0, len(nodes_per_graph_list)):
        senders_graph_i, receivers_graph_i = _get_connectivity(nodes_per_graph_list[i], connections_per_graph_list[i])
        num_edges_graph_i = len(senders_graph_i)
        n_edge_list.append(num_edges_graph_i)

        receivers_list.append(receivers_graph_i + num_nodes_in_previous_graph)
        senders_list.append(senders_graph_i + num_nodes_in_previous_graph)

        num_nodes_graph_i = len(nodes_per_graph_list[i])
        num_nodes_in_previous_graph += num_nodes_graph_i

    senders = np.concatenate(senders_list, axis=0).astype(np.int32)
    receivers = np.concatenate(receivers_list, axis=0).astype(np.int32)
    n_edge = np.stack(n_edge_list).astype(np.int32)

    return senders, receivers, n_edge


def get_connectivity_for_batch_pyfunc(node_locations, node_connections, n_nodes, n_con):
    senders, receivers, n_edge = tf.py_function(
        _get_connectivity_for_batch,
        [node_locations, node_connections, n_nodes, n_con],
        [tf.int32, tf.int32, tf.int32])
    senders.set_shape([None])
    receivers.set_shape([None])
    n_edge.set_shape(n_nodes.get_shape())
    return senders, receivers, n_edge

"""    for node_graph_i, conn_graph_i in nodes_per_graph_list, connections_per_graph_list:
        senders_graph_i, receivers_graph_i = _get_connectivity(node_graph_i, conn_graph_i)

        num_edges_graph_i = len(senders_graph_i)
        n_edge_list.append(num_edges_graph_i)

        receivers_list.append(receivers_graph_i + num_nodes_in_previous_graph)
        senders_list.append(senders_graph_i + num_nodes_in_previous_graph)

        num_nodes_graph_i = len(node_graph_i)
        num_nodes_in_previous_graph += num_nodes_graph_i"""