import graph_nets as gn
import networkx as nx
import sonnet as snt
import numpy as np
import tensorflow.compat.v1 as tf

from mesh import connectivity_utils
from mesh import graph_network

STD_EPSILON = 1e-8


class LearnedSimulator(snt.AbstractModule):
    def __init__(
            self,
            graph_network_kwargs,
            #boundaries,
            normalization_stats,
            name="LearnedSimulator"):
        super().__init__(name=name)

        #self._boundaries = boundaries
        self._normalization_stats = normalization_stats
        with self._enter_variable_scope():
            self._graph_network = graph_network.EncodeProcessDecode(
                output_size=2, **graph_network_kwargs)

    def _build(self, node_locations, node_connections, node_velocities, global_context=None):
        input_graphs_tuple = self._encoder_preprocessor(node_locations, node_connections, node_velocities,
                                                        global_context)
        normalized_acc = self._graph_network(input_graphs_tuple)

        next_position = self._decoder_postprocessor(normalized_acc, node_velocities)

        return next_position

    def _encoder_preprocessor(self, node_locations, node_connections, node_velocities, global_context):
        most_recent_vel = node_velocities[:, -1]
        most_recent_loc = node_locations[:, -1]
        most_recent_con = node_connections[:, -1]
        acc_sequence = time_diff(node_velocities)
        n_node = len(most_recent_loc)

        (senders, receivers, n_edge) = connectivity_utils.compute_connectivity_for_batch_pyfunc(most_recent_con,
                                                                                                most_recent_loc, n_node)

        node_features = []

        acc_stats = self._normalization_stats["acceleration"]
        normalized_acc_sequence = (acc_sequence - acc_stats.mean) / acc_stats.std

        flat_acc_sequence = snt.MergeDims(start=1, size=2)(
            normalized_acc_sequence)
        node_features.append(flat_acc_sequence)
        node_features.append(most_recent_vel)

        if global_context is not None:
            context_stats = self._normalization_stats["context"]
            # Context in some datasets are all zero, so add an epsilon for numerical
            # stability.
            global_context = (global_context - context_stats.mean) / tf.math.maximum(
                context_stats.std, STD_EPSILON)

        return gn.graphs.GraphsTuple(
            nodes=tf.concat(node_features, axis=-1),
            globals=global_context,
            n_node=n_node,
            n_edge=n_edge,
            senders=senders,
            receivers=receivers,
        )

    def _decoder_postprocessor(self, normalized_acc, vel_sequence):
        acc_stats = self._normalization_stats["acceleration"]
        acc = (normalized_acc * acc_stats.std) + acc_stats.mean

        most_recent_velocity = vel_sequence[:, -1]
        new_vel = most_recent_velocity + acc

        return new_vel

    def get_predicted_and_target_normalized_accelerations(
            self, node_locations, node_connections, next_vel, vel_sequence_noise, vel_sequence, global_context=None):
        noisy_vel_sequence = vel_sequence + vel_sequence_noise

        input_graphs_tuple = self._encoder_preprocessor(node_locations,
                                                        node_connections,
                                                        noisy_vel_sequence,
                                                        global_context)
        predicted_normalized_acceleration = self._graph_network(input_graphs_tuple)

        next_vel_adjusted = next_vel + vel_sequence_noise[:, -1]
        target_normalized_acceleration = self._inverse_decoder_postprocessor(
            next_vel_adjusted, noisy_vel_sequence)

        return predicted_normalized_acceleration, target_normalized_acceleration

    def _inverse_decoder_postprocessor(self, next_vel, vel_sequence):
        """Inverse of `_decoder_postprocessor`."""

        previous_vel = vel_sequence[:, -1]
        acceleration = next_vel - previous_vel

        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_acceleration = (acceleration - acceleration_stats.mean) / acceleration_stats.std
        return normalized_acceleration


def time_diff(input_sequence):
    return input_sequence[:, 1:] - input_sequence[:, :-1]
