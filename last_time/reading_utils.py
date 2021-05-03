import functools
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

_FEATURE_DESCRIPTION = {
    'velocity': tf.io.VarLenFeature(tf.string)
}

_CONTEXT_FEATURES = {
    'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'locations': tf.io.VarLenFeature(tf.string),
    'connections': tf.io.VarLenFeature(tf.string)
}

_FEATURE_DTYPES = {
    'velocity': {
        'in': np.float32,
        'out': tf.float32
    }
}


def convert_to_tensor(x, encoded_dtype):
    if len(x) == 1:
        out = np.frombuffer(x[0].numpy(), dtype=encoded_dtype)
    else:
        out = []
        for el in x:
            out.append(np.frombuffer(el.numpy(), dtype=encoded_dtype))
    out = tf.convert_to_tensor(np.array(out))
    return out


def parse_serialized_simulation_example(example_proto):
    context, parsed_features = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=_CONTEXT_FEATURES,
        sequence_features=_FEATURE_DESCRIPTION)

    for feature_key, item in parsed_features.items():
        convert_fn = functools.partial(
            convert_to_tensor, encoded_dtype=_FEATURE_DTYPES[feature_key]['in'])
        print(item.values)
        parsed_features[feature_key] = tf.py_function(
            convert_fn, inp=[item.values], Tout=_FEATURE_DTYPES[feature_key]['out'])
        print(parsed_features)

    """context['locations'] = tf.py_function(
        functools.partial(convert_fn, encoded_dtype=np.float32),
        inp=[context['locations'].values],
        Tout=[tf.float32])
    context['connections'] = tf.py_function(
        functools.partial(convert_fn, encoded_dtype=np.int64),
        inp=[context['connections'].values],
        Tout=[tf.int64])"""
    return context, parsed_features


def split_trajectory(context, features, window_length=7):
    trajectory_length = features['velocity'].get_shape().as_list()[0]

    input_trajectory_length = trajectory_length - window_length + 1

    model_input_features = {}

    model_input_features['locations'] = tf.tile(
        tf.expand_dims(context['locations'], axis=0),
        [input_trajectory_length, 1])

    model_input_features['connections'] = tf.tile(
        tf.expand_dims(context['connections'], axis=0),
        [input_trajectory_length, 1])

    pos_stack = []

    for idx in range(input_trajectory_length):
        pos_stack.append(features['velocity'][idx:idx + window_length])

    model_input_features['velocity'] = tf.stack(pos_stack)

    return tf.data.Dataset.from_tensor_slices(model_input_features)


ds = tf.data.TFRecordDataset('/Volumes/Samsung/data/train.tfrecord')
ds = ds.map(parse_serialized_simulation_example)
