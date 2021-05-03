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
"""Methods to calculate input noise."""

import tensorflow.compat.v1 as tf

from mesh import learned_simulator


def get_random_walk_noise_for_position_sequence(
        vel_sequence, noise_std_last_step):
    """Returns random-walk noise in the velocity applied to the position."""

    acc_sequence = learned_simulator.time_diff(vel_sequence)

    # We want the noise scale in the velocity at the last step to be fixed.
    # Because we are going to compose noise at each step using a random_walk:
    # std_last_step**2 = num_acc * std_each_step**2
    # so to keep `std_last_step` fixed, we apply at each step:
    # std_each_step `std_last_step / np.sqrt(num_input_velocities)`
    num_acc = acc_sequence.shape.as_list()[1]
    acc_sequence_noise = tf.random.normal(
        tf.shape(acc_sequence),
        stddev=noise_std_last_step / num_acc ** 0.5,
        dtype=vel_sequence.dtype)

    # Apply the random walk.
    acc_sequence_noise = tf.cumsum(acc_sequence_noise, axis=1)

    # Integrate the noise in the velocity to the positions, assuming
    # an Euler integrator and a dt = 1, and adding no noise to the very first
    # position (since that will only be used to calculate the first position
    # change).
    vel_sequence_noise = tf.concat([
        tf.zeros_like(acc_sequence_noise[:, 0:1]),
        tf.cumsum(acc_sequence_noise, axis=1)], axis=1)

    return vel_sequence_noise
