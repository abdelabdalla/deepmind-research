import collections
import functools
import os
import pickle

import tensorflow.compat.v1 as tf
import tree
from absl import app
from absl import flags
from absl import logging

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

    return tensor_dict, target_vel


def prepare_rollout_inputs(context, features):
    out_dict = {**context}

    vel = tf.transpose(features['velocity'], [1, 0, 2])

    target_vel = vel[:, -1]

    out_dict['velocity'] = vel[:, :-1]
    out_dict['n_nodes'] = features['n_nodes']
    out_dict['n_cons'] = features['n_cons']
    out_dict['locations'] = features['locations']
    out_dict['connections'] = features['connections']

    out_dict['is_trajectory'] = tf.constant([True], tf.bool)

    return out_dict, target_vel


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

        elif mode == 'rollout':
            # Rollout evaluation only available for batch size 1
            assert batch_size == 1
            ds = ds.map(prepare_rollout_inputs)
        else:
            raise ValueError(f'mode: {mode} not recognized')
        return ds

    return input_fn


def rollout(simulator, features, num_steps):
    initial_vel = features['velocity'][:, 0:INPUT_SEQUENCE_LENGTH]
    ground_truth_vel = features['velocity'][:, INPUT_SEQUENCE_LENGTH:]
    global_context = features.get('step_context')

    def step_fn(step, current_vels, predictions):
        next_vel = simulator(velocity_sequence=current_vels,
                             n_nodes=features['n_nodes'],
                             n_conn=features['n_cons'],
                             node_locations=features['locations'],
                             node_connections=features['connections'])

        updated_predictions = predictions.write(step, next_vel)

        next_vels = tf.concat([current_vels[:, 1:],
                               next_vel[:, tf.newaxis]], axis=1)

        return step + 1, next_vels, updated_predictions

    predictions = tf.TensorArray(size=num_steps, dtype=tf.float32)
    _, _, predictions = tf.while_loop(
        cond=lambda step, state, prediction: tf.less(step, num_steps),
        body=step_fn,
        loop_vars=(0, initial_vel, predictions),
        back_prop=False,
        parallel_iterations=1)

    output_dict = {
        'initial_positions': tf.transpose(initial_vel, [1, 0, 2]),
        'predicted_rollout': predictions.stack(),
        'ground_truth_rollout': tf.transpose(ground_truth_vel, [1, 0, 2]),
        'particle_types': features['particle_type'],
    }

    return output_dict


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


def get_rollout_estimator_fn(latent_size=128,
                             hidden_size=128,
                             hidden_layers=2,
                             message_passing_steps=10):
    model_kwargs = dict(
        latent_size=latent_size,
        mlp_hidden_size=hidden_size,
        mlp_num_hidden_layers=hidden_layers,
        num_message_passing_steps=message_passing_steps)

    def estimator_fn(features, labels, mode):
        del labels  # Labels to conform to estimator spec.
        simulator = _get_simulator(model_kwargs)

        num_steps = 200 - INPUT_SEQUENCE_LENGTH
        rollout_op = rollout(simulator, features, num_steps=num_steps)
        squared_error = (rollout_op['predicted_rollout'] -
                         rollout_op['ground_truth_rollout']) ** 2
        loss = tf.reduce_mean(squared_error)
        eval_ops = {'rollout_error_mse': tf.metrics.mean_squared_error(
            rollout_op['predicted_rollout'], rollout_op['ground_truth_rollout'])}

        # Add a leading axis, since Estimator's predict method insists that all
        # tensors have a shared leading batch axis fo the same dims.
        rollout_op = tree.map_structure(lambda x: x[tf.newaxis], rollout_op)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            train_op=None,
            loss=loss,
            predictions=rollout_op,
            eval_metric_ops=eval_ops)

    return estimator_fn


def main(_):
    if FLAGS.mode in ['train', 'eval']:
        estimator = tf.estimator.Estimator(
            get_one_step_estimator_fn(FLAGS.noise_std), model_dir=FLAGS.model_path)
        if FLAGS.mode == 'train':
            estimator.train(
                input_fn=get_input_fn(FLAGS.data_path, FLAGS.batch_size,
                                      mode='one_step_train', split='train'),
                max_steps=FLAGS.num_steps)

        else:
            eval_metrics = estimator.evaluate(input_fn=get_input_fn(
                FLAGS.data_path, FLAGS.batch_size,
                mode='one_step', split=FLAGS.eval_split))
            logging.info('Evaluation metrics:')
            logging.info(eval_metrics)

    elif FLAGS.mode == 'eval_rollout':
        if not FLAGS.output_path:
            raise ValueError('A rollout path must be provided.')
        rollout_estimator = tf.estimator.Estimator(
            get_rollout_estimator_fn(FLAGS.data_path, FLAGS.noise_std),
            model_dir=FLAGS.model_path)

        # Iterate through rollouts saving them one by one.
        rollout_iterator = rollout_estimator.predict(
            input_fn=get_input_fn(FLAGS.data_path, batch_size=1,
                                  mode='rollout', split=FLAGS.eval_split))

        for example_index, example_rollout in enumerate(rollout_iterator):
            filename = f'rollout_{FLAGS.eval_split}_{example_index}.pkl'
            filename = os.path.join(FLAGS.output_path, filename)
            logging.info('Saving: %s.', filename)
            if not os.path.exists(FLAGS.output_path):
                os.mkdir(FLAGS.output_path)
            with open(filename, 'wb') as file:
                pickle.dump(example_rollout, file)


if __name__ == '__main__':
    tf.disable_v2_behavior()
    app.run(main)
