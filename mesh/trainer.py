import collections
import functools
import json
import os
import pickle

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tree

from mesh import reading_utils
from mesh import learned_simulator
from mesh import noise_utils

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'eval_rollout'],
    help='Train model, one step evaluation or rollout evaluation.')
flags.DEFINE_enum('eval_split', 'test', ['train', 'valid', 'test'],
                  help='Split to use when running evaluation.')
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_integer('num_steps', int(2e7), help='Number of steps of training.')
flags.DEFINE_float('noise_std', 6.7e-4, help='The std deviation of the noise.')
flags.DEFINE_string('model_path', None,
                    help=('The path for saving checkpoints of the model. '
                          'Defaults to a temporary directory.'))
flags.DEFINE_string('output_path', None,
                    help='The path for saving outputs (e.g. rollouts).')

FLAGS = flags.FLAGS

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 6


# TODO this
def prepare_inputs(tensor_dict):
    """Prepares a single stack of inputs by calculating inputs and targets.

  Computes n_particles_per_example, which is a tensor that contains information
  about how to partition the axis - i.e. which nodes belong to which graph.

  Adds a batch axis to `n_particles_per_example` and `step_context` so they can
  later be batched using `batch_concat`. This batch will be the same as if the
  elements had been batched via stacking.

  Note that all other tensors have a variable size particle axis,
  and in this case they will simply be concatenated along that
  axis.



  Args:
    tensor_dict: A dict of tensors containing positions, and step context (
    if available).

  Returns:
    A tuple of input features and target positions.

  """
    # Position is encoded as [sequence_length, num_particles, dim] but the model
    # expects [num_particles, sequence_length, dim].
    pos = tensor_dict['position']
    pos = tf.transpose(pos, perm=[1, 0, 2])

    # The target position is the final step of the stack of positions.
    target_position = pos[:, -1]

    # Remove the target from the input.
    tensor_dict['position'] = pos[:, :-1]

    # Compute the number of particles per example.
    num_particles = tf.shape(pos)[0]
    # Add an extra dimension for stacking via concat.
    tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]

    if 'step_context' in tensor_dict:
        # Take the input global context. We have a stack of global contexts,
        # and we take the penultimate since the final is the target.
        tensor_dict['step_context'] = tensor_dict['step_context'][-2]
        # Add an extra dimension for stacking via concat.
        tensor_dict['step_context'] = tensor_dict['step_context'][tf.newaxis]
    return tensor_dict, target_position


# TODO this
def prepare_rollout_inputs(context, features):
    """Prepares an inputs trajectory for rollout."""
    out_dict = {**context}
    # Position is encoded as [sequence_length, num_particles, dim] but the model
    # expects [num_particles, sequence_length, dim].
    pos = tf.transpose(features['position'], [1, 0, 2])
    # The target position is the final step of the stack of positions.
    target_position = pos[:, -1]
    # Remove the target from the input.
    out_dict['position'] = pos[:, :-1]
    # Compute the number of nodes
    out_dict['n_particles_per_example'] = [tf.shape(pos)[0]]
    if 'step_context' in features:
        out_dict['step_context'] = features['step_context']
    out_dict['is_trajectory'] = tf.constant([True], tf.bool)
    return out_dict, target_position


def batch_concat(dataset, batch_size):
    """We implement batching as concatenating on the leading axis."""

    # We create a dataset of datasets of length batch_size.
    windowed_ds = dataset.window(batch_size)

    # The plan is then to reduce every nested dataset by concatenating. We can
    # do this using tf.data.Dataset.reduce. This requires an initial state, and
    # then incrementally reduces by running through the dataset

    # Get initial state. In this case this will be empty tensors of the
    # correct shape.
    initial_state = tree.map_structure(
        lambda spec: tf.zeros(  # pylint: disable=g-long-lambda
            shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype),
        dataset.element_spec)

    # We run through the nest and concatenate each entry with the previous state.
    def reduce_window(initial_state, ds):
        return ds.reduce(initial_state, lambda x, y: tf.concat([x, y], axis=0))

    return windowed_ds.map(
        lambda *x: tree.map_structure(reduce_window, initial_state, x))


def get_input_fn(data_path, batch_size, mode, split):
    """Gets the learning simulation input function for tf.estimator.Estimator.

  Args:
    data_path: the path to the dataset directory.
    batch_size: the number of graphs in a batch.
    mode: either 'one_step_train', 'one_step' or 'rollout'
    split: either 'train', 'valid' or 'test.

  Returns:
    The input function for the learning simulation model.
  """

    def input_fn():
        """Input function for learning simulation."""
        # Loads the metadata of the dataset.
        metadata = _read_metadata(data_path)
        # Create a tf.data.Dataset from the TFRecord.
        ds = tf.data.TFRecordDataset([os.path.join(data_path, f'{split}.tfrecord')])
        ds = ds.map(functools.partial(
            reading_utils.parse_serialized_simulation_example, metadata=metadata))
        if mode.startswith('one_step'):
            # Splits an entire trajectory into chunks of 7 steps.
            # Previous 5 velocities, current velocity and target.
            split_with_window = functools.partial(
                reading_utils.split_trajectory,
                window_length=INPUT_SEQUENCE_LENGTH + 1)
            ds = ds.flat_map(split_with_window)
            # Splits a chunk into input steps and target steps
            ds = ds.map(prepare_inputs)
            # If in train mode, repeat dataset forever and shuffle.
            if mode == 'one_step_train':
                ds = ds.repeat()
                ds = ds.shuffle(512)
            # Custom batching on the leading axis.
            ds = batch_concat(ds, batch_size)
        elif mode == 'rollout':
            # Rollout evaluation only available for batch size 1
            assert batch_size == 1
            ds = ds.map(prepare_rollout_inputs)
        else:
            raise ValueError(f'mode: {mode} not recognized')
        return ds

    return input_fn


def _combine_std(std_x, std_y):
    return np.sqrt(std_x ** 2 + std_y ** 2)


def _get_simulator(model_kwargs, metadata, acc_noise_std):
    cast = lambda v: np.array(v, dtype=np.float32)

    acceleration_stats = Stats(
        cast(metadata['acc_mean']),
        _combine_std(cast(metadata['acc_std']), acc_noise_std))
    normalization_stats = {'acceleration': acceleration_stats}

    simulator = learned_simulator.LearnedSimulator(
        graph_network_kwargs=model_kwargs,
        normalization_stats=normalization_stats
    )

    return simulator


def get_one_step_estimator_fn(data_path,
                              noise_std,
                              latent_size=128,
                              hidden_size=128,
                              hidden_layers=2,
                              message_passing_steps=10):
    metadata = _read_metadata(data_path)

    model_kwargs = dict(
        latent_size=latent_size,
        mlp_hidden_size=hidden_size,
        mlp_num_hidden_layers=hidden_layers,
        num_message_passing_steps=message_passing_steps)

    def estimator_fn(features, labels, mode):
        target_next_vel = labels  # TODO check deets
        simulator = _get_simulator(model_kwargs, metadata,
                                   vel_noise_std=noise_std,
                                   acc_noise_std=noise_std)

        sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
            features['velocity'], noise_std_last_step=noise_std)

        pred_target = simulator.get_predicted_and_target_normalized_accelerations(
            node_locations=features['location'],
            node_connections=features['connections'],
            next_vel=target_next_vel,
            vel_sequence_noise=sampled_noise,
            vel_sequence=features['velocity'],
            global_context=features.get('step_context'))
        pred_acceleration, target_acceleration = pred_target

        loss = (pred_acceleration - target_acceleration) ** 2

        global_step = tf.train.get_global_step()
        # Set learning rate to decay from 1e-4 to 1e-6 exponentially.
        min_lr = 1e-6
        lr = tf.train.exponential_decay(learning_rate=1e-4 - min_lr,
                                        global_step=global_step,
                                        decay_steps=int(5e6),
                                        decay_rate=0.1) + min_lr
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = opt.minimize(loss, global_step)

        predicted_next_vel = simulator(
            node_locations=features['location'],
            node_connections=features['connections'],
            node_velocities=features['velocity'],
            global_context=features.get('step_context')
        )

        predictions = {'predicted_next_vel': predicted_next_vel}

        eval_metrics_ops = {
            'loss_mse': tf.metrics.mean_squared_error(
                pred_acceleration, target_acceleration),
            'one_step_position_mse': tf.metrics.mean_squared_error(
                predicted_next_vel, target_next_vel)
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            train_op=train_op,
            loss=loss,
            predictions=predictions,
            eval_metric_ops=eval_metrics_ops)

    return estimator_fn


def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())


def main(_):
    """Train or evaluates the model."""

    if FLAGS.mode in ['train', 'eval']:
        estimator = tf.estimator.Estimator(
            get_one_step_estimator_fn(FLAGS.data_path, FLAGS.noise_std),
            model_dir=FLAGS.model_path)
        if FLAGS.mode == 'train':
            # Train all the way through.
            estimator.train(
                input_fn=get_input_fn(FLAGS.data_path, FLAGS.batch_size,
                                      mode='one_step_train', split='train'),
                max_steps=FLAGS.num_steps)
        else:
            # One-step evaluation from checkpoint.
            eval_metrics = estimator.evaluate(input_fn=get_input_fn(
                FLAGS.data_path, FLAGS.batch_size,
                mode='one_step', split=FLAGS.eval_split))
            logging.info('Evaluation metrics:')
            logging.info(eval_metrics)
    else:
        print('only training is supported atm')


if __name__ == '__main__':
    tf.disable_v2_behavior()
    app.run(main)
