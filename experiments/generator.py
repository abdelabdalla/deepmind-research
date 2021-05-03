import fnmatch
import os

import meshio
import tensorflow as tf
import numpy as np


tf.enable_eager_execution()


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def file_format(point_vel, point_location, point_connections, key):

    vel_tensor = tf.convert_to_tensor(point_vel)
    vel_serialized = tf.io.serialize_tensor(vel_tensor)

    location_tensor = tf.convert_to_tensor(point_location)
    location_serialized = tf.io.serialize_tensor(location_tensor)

    connections_tensor = tf.convert_to_tensor(point_connections)
    connections_serialized = tf.io.serialize_tensor(connections_tensor)

    feature = {
        'key': _int64_feature(key),
        'velocity': _bytes_feature(vel_serialized),
        'location': _bytes_feature(location_serialized),
        'connections': _bytes_feature(connections_serialized)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


file_location = '/Users/abdelabdalla/Documents/Jet Updated'
record_file = '/Volumes/Samsung'

run_dir = os.listdir(file_location)
run_pattern = "run*"
run_list = []

for entry in run_dir:
    if fnmatch.fnmatch(entry, run_pattern):
        run_list.append(entry)

vel_array = np.array([]).reshape(0, 2)
loc_array = np.array([]).reshape(0, 2)
con_array = np.array([]).reshape(0, 3)
n_nodes_array = np.array([]).reshape(0, 1)
n_conns_array = np.array([]).reshape(0, 1)
n_timesteps_array = np.array([]).reshape(0, 1)

for run in run_list:
    print('folder ' + run)

    list_of_files = os.listdir(file_location + '/' + run)
    pattern = "flow_*.vtk"
    list_of_names = []

    for entry in list_of_files:
        if fnmatch.fnmatch(entry, pattern):
            list_of_names.append(entry)

    list.sort(list_of_names)

    list_of_paths = [file_location + '/' + run + '/' + s for s in list_of_names]
    mesh_first = meshio.read(list_of_paths[0])
    locs = np.delete(mesh_first.points, 2, 1)
    n_node = len(locs)
    cons = mesh_first.cells_dict['triangle']
    n_conn = len(cons)
    vels = []

    time_steps = 1
    for path in list_of_paths:
        print(path)
        mesh = meshio.read(path)
        vels.append(np.delete(mesh.point_data['Velocity'], 2, 1))
        time_steps += 1

    vel_array = np.concatenate((vel_array, np.array(vels)), axis=0)
    loc_array = np.concatenate((loc_array, np.array(locs)), axis=0)
    con_array = np.concatenate((con_array, np.array(cons)), axis=0)
    n_nodes_array = np.concatenate((n_nodes_array, np.array(n_node)), axis=0)
    n_conns_array = np.concatenate((n_conns_array, np.array(n_conn)), axis=0)
    n_timesteps_array = np.concatenate((n_timesteps_array, np.array(time_steps)), axis=0)

    train_


with tf.io.TFRecordWriter(record_file + '/train.tfrecord') as writer:

    for i in range(0, 200):
        print('writing train ' + str(i))
        tf_example = file_format(vel_array, loc_array, con_array, n_nodes)
        writer.write(tf_example.SerializeToString())

with tf.io.TFRecordWriter(record_file + '/test.tfrecord') as writer:

    for i in range(200, 220):
        print('writing test ' + str(i))
        tf_example = file_format(vel_list[i], loc_list[i], con_list[i], i)
        writer.write(tf_example.SerializeToString())

with tf.io.TFRecordWriter(record_file + '/validate.tfrecord') as writer:

    for i in range(200, 240):
        print('writing validate ' + str(i))
        tf_example = file_format(vel_list[i], loc_list[i], con_list[i], i)
        writer.write(tf_example.SerializeToString())
