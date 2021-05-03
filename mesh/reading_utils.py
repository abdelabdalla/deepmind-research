# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Utilities for reading open sourced Learning Complex Physics data."""

import functools
import numpy as np
import tensorflow.compat.v1 as tf

# Create a description of the features.
_FEATURE_DESCRIPTION = {
    'velocity': tf.io.VarLenFeature(tf.string),
    'location': tf.io.VarLenFeature(tf.string),
    'connections': tf.io.VarLenFeature(tf.string),
}

# TODO check what step_context is
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT['step_context'] = tf.io.VarLenFeature(
    tf.string)

_FEATURE_DTYPES = {
    'velocity': {
        'in': np.float32,
        'out': tf.float32
    },
    'location': {
        'in': np.float32,
        'out': tf.float32
    },
    'connections': {
        'in': np.float32,
        'out': tf.float32
    },
    'step_context': {
        'in': np.float32,
        'out': tf.float32
    }
}

_CONTEXT_FEATURES = {
    'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
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


def parse_serialized_simulation_example(example_proto, metadata):
    """Parses a serialized simulation tf.SequenceExample.

  Args:
    example_proto: A string encoding of the tf.SequenceExample proto.
    metadata: A dict of metadata for the dataset.

  Returns:
    context: A dict, with features that do not vary over the trajectory.
    parsed_features: A dict of tf.Tensors representing the parsed examples
      across time, where axis zero is the time axis.

  """
    if 'context_mean' in metadata:
        feature_description = _FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT
    else:
        feature_description = _FEATURE_DESCRIPTION
    context, parsed_features = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=_CONTEXT_FEATURES,
        sequence_features=feature_description)
    for feature_key, item in parsed_features.items():
        convert_fn = functools.partial(
            convert_to_tensor, encoded_dtype=_FEATURE_DTYPES[feature_key]['in'])
        parsed_features[feature_key] = tf.py_function(
            convert_fn, inp=[item.values], Tout=_FEATURE_DTYPES[feature_key]['out'])

    # There is an extra frame at the beginning so we can calculate pos change
    # for all frames used in the paper.
    vel_shape = [metadata['sequence_length'] + 1, -1, metadata['dim']]

    # Reshape positions to correct dim:
    parsed_features['velocity'] = tf.reshape(parsed_features['velocity'],
                                             vel_shape)
    # Set correct shapes of the remaining tensors.
    sequence_length = metadata['sequence_length'] + 1
    parsed_features['location'] = tf.reshape(parsed_features['location'], sequence_length)
    parsed_features['connections'] = tf.reshape(parsed_features['connections'], sequence_length)

    return context, parsed_features


def split_trajectory(context, features, window_length=7):
    """Splits trajectory into sliding windows."""
    # Our strategy is to make sure all the leading dimensions are the same size,
    # then we can use from_tensor_slices.

    trajectory_length = features['velocity'].get_shape().as_list()[0]

    # We then stack window_length position changes so the final
    # trajectory length will be - window_length +1 (the 1 to make sure we get
    # the last split).
    input_trajectory_length = trajectory_length - window_length + 1

    model_input_features = {}
    # Prepare the context features per step.
    model_input_features['location'] = tf.tile(
        tf.expand_dims(context['location'], axis=0),
        [input_trajectory_length, 1])
    model_input_features['connections'] = tf.tile(
        tf.expand_dims(context['connections'], axis=0),
        [input_trajectory_length, 1])

    if 'step_context' in features:
        global_stack = []
        for idx in range(input_trajectory_length):
            global_stack.append(features['step_context'][idx:idx + window_length])
        model_input_features['step_context'] = tf.stack(global_stack)

    vel_stack = []
    for idx in range(input_trajectory_length):
        vel_stack.append(features['velocity'][idx:idx + window_length])
    # Get the corresponding positions
    model_input_features['velocity'] = tf.stack(vel_stack)

    return tf.data.Dataset.from_tensor_slices(model_input_features)
