import numpy as np
import tensorflow.compat.v1 as tf


def _compute_connectivity(node_connections, node_locations):
    num_nodes = len(node_locations)
    senders = np.repeat(range(num_nodes), [len(a) for a in node_connections])
    receivers = np.concatenate(node_connections, axis=0)
    return senders, receivers


def _compute_connectivity_for_batch(node_connections, node_locations):
    receivers_list = []
    senders_list = []
    n_edges_list = []
    num_nodes_in_previous_graphs = 0

    for node_con, node_loc in node_connections, node_locations:
        senders_graph, receivers_graph = _compute_connectivity(node_con, node_loc)

        num_edges_graph = len(senders_graph)
        n_edges_list.append(num_edges_graph)

        receivers_list.append(receivers_graph + num_nodes_in_previous_graphs)
        senders_list.append(senders_graph + num_nodes_in_previous_graphs)

        num_nodes_graph = len(node_loc)
        num_nodes_in_previous_graphs += num_nodes_graph

    senders = np.concatenate(senders_list, axis=0).astype(np.int32)
    receivers = np.concatenate(receivers_list, axis=0).astype(np.int32)
    n_edge = np.stack(n_edges_list).astype(np.int32)

    return senders, receivers, n_edge


def compute_connectivity_for_batch_pyfunc(node_connections, node_locations, n_node):
    senders, receivers, n_edge = tf.py_function(_compute_connectivity_for_batch,
                                                [node_connections, node_locations],
                                                [tf.int32, tf.int32])
    senders.set_shape([None])
    receivers.set_shape([None])
    #TODO check n_node
    n_edge.set_shape(n_node.get_shape())
    return senders, receivers, n_edge
