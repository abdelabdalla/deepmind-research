import collections
import functools
import os
import sys

import tensorflow.compat.v1 as tf
from tensorflow.python import debug as tf_debug
import tree
from absl import app
from absl import flags

from last_time import noise_utils
from last_time import ns_simulator
from last_time import reading_utils

tf.enable_eager_execution()

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'eval_rollout'],
    help='Train model, one step evaluation or rollout evaluation.')
flags.DEFINE_enum('eval_split', 'test', ['train', 'valid', 'test'],
                  help='Split to use when running evaluation.')
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_integer('num_steps', int(2e7), help='Number of steps of training.')
flags.DEFINE_float('noise_std', 2e-2, help='The std deviation of the noise.')
flags.DEFINE_string('model_path', None,
                    help=('The path for saving checkpoints of the model. '
                          'Defaults to a temporary directory.'))
flags.DEFINE_string('output_path', None,
                    help='The path for saving outputs (e.g. rollouts).')

FLAGS = flags.FLAGS

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 6


def prepare_inputs(tensor_dict):
    vel = tensor_dict['velocity']
    vel = tf.transpose(vel, perm=[1, 0, 2])

    target_vel = vel[:, -1]

    tensor_dict['velocity'] = vel[:, :-1]

    # num_nodes = tf.shape(vel)[0]

    # tensor_dict['n_nodes_per_example'] = num_nodes[tf.newaxis]

    return tensor_dict, target_vel


def batch_concat(dataset, batch_size):
    windowed_ds = dataset.window(batch_size)

    initial_state = tree.map_structure(
        lambda spec: tf.zeros(
            shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype),
        dataset.element_spec
    )

    def reduce_window(inital_state, ds):
        return ds.reduce(inital_state, lambda x, y: tf.concat([x, y], axis=0))

    return windowed_ds.map(
        lambda *x: tree.map_structure(reduce_window, initial_state, x))


def get_input_fn(data_path, batch_size, mode, split):
    def input_fn():

        ds = tf.data.TFRecordDataset([os.path.join(data_path, f'{split}.tfrecord')])
        ds = ds.map(reading_utils.parse_serialized_simulation_example)
        if mode.startswith('one_step'):

            split_with_window = functools.partial(
                reading_utils.split_trajectory,
                window_length=INPUT_SEQUENCE_LENGTH + 1)
            ds = ds.flat_map(split_with_window)
            ds = ds.map(prepare_inputs)
            if mode == 'one_step_train':
                ds = ds.repeat()
                ds = ds.shuffle(batch_size * 50)
            # Custom batching on the leading axis.
            ds = batch_concat(ds, batch_size)
        return ds

    return input_fn


def _get_simulator(model_kwargs):
    simulator = ns_simulator.NSSimulator(
        graph_network_kwargs=model_kwargs
    )

    return simulator


def get_one_step_estimator_fn(noise_std,
                              latent_size=128,
                              hidden_size=128,
                              hidden_layers=2,
                              message_passing_steps=10):
    model_kwargs = dict(
        latent_size=latent_size,
        mlp_hidden_size=hidden_size,
        mlp_num_hidden_layers=hidden_layers,
        num_message_passing_steps=message_passing_steps)

    def estimator_fn(features, labels, mode):
        target_next_velocity = labels
        simulator = _get_simulator(model_kwargs)

        sampled_noise = noise_utils.get_random_walk_noise_for_velocity_sequence(
            features['velocity'], noise_std_last_step=noise_std)

        pred_target = simulator.get_predicted_and_target_normalized_accelerations(
            next_velocity=target_next_velocity,
            n_nodes=features['n_nodes'],
            n_conn=features['n_cons'],
            velocity_sequence=features['velocity'],
            node_locations=features['locations'],
            node_connections=features['connections'],
            velocity_sequence_noise=sampled_noise
        )
        pred_acceleration, target_acceleration = pred_target

        loss = tf.reduce_sum((pred_acceleration - target_acceleration) ** 2)
        global_step = tf.train.get_global_step()

        min_lr = 1e-6
        lr = tf.train.exponential_decay(learning_rate=1e-4 - min_lr,
                                        global_step=global_step,
                                        decay_steps=int(5e6),
                                        decay_rate=0.1) + min_lr
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = opt.minimize(loss, global_step)

        predicted_next_velocity = simulator(
            velocity_sequence=features['velocity'],
            n_nodes=features['n_nodes'],
            n_conn=features['n_cons'],
            node_locations=features['locations'],
            node_connections=features['connections'])

        predictions = {'predicted_next_velocity': predicted_next_velocity}

        eval_metrics_ops = {
            'loss_mse': tf.metrics.mean_squared_error(
                pred_acceleration, target_acceleration),
            'one_step_position_mse': tf.metrics.mean_squared_error(
                predicted_next_velocity, target_next_velocity)
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            train_op=train_op,
            loss=loss,
            predictions=predictions,
            eval_metric_ops=eval_metrics_ops)

    return estimator_fn


def main(_):
    if FLAGS.mode in ['train', 'eval']:
        estimator = tf.estimator.Estimator(
            get_one_step_estimator_fn(FLAGS.noise_std), model_dir=FLAGS.model_path)
        if FLAGS.mode == 'train':
            # Train all the way through.
            # sess = tf.Session()
            # hooks = [tf_debug.LocalCLIDebugHook(ui_type="readline")]
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            # sess.run(
            estimator.train(
                input_fn=get_input_fn(FLAGS.data_path, FLAGS.batch_size,
                                      mode='one_step_train', split='train'),
                max_steps=FLAGS.num_steps)  # , hooks=hooks)
        # )


if __name__ == '__main__':
    tf.disable_v2_behavior()
    app.run(main)
