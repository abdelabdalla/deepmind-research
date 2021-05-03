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

    location_tensor = tf.convert_to_tensor(point_location)
    location_serialized = tf.io.serialize_tensor(location_tensor)

    connections_tensor = tf.convert_to_tensor(point_connections)
    connections_serialized = tf.io.serialize_tensor(connections_tensor)

    context = {
        'key': _int64_feature(key),
        'locations': _bytes_feature(location_serialized),
        'connections': _bytes_feature(connections_serialized)
    }

    features = [tf.train.Feature(bytes_list=tf.train.BytesList(value=[val.tostring()])) for val in point_vel]
    featurelist = tf.train.FeatureList(feature=features)

    return tf.train.SequenceExample(context=tf.train.Features(feature=context),
                                    feature_lists=tf.train.FeatureLists(feature_list={'velocity': featurelist}))


file_location = '/Users/abdelabdalla/Documents/Jet Updated'
record_file = '/Volumes/Samsung/data_alternate'

run_dir = os.listdir(file_location)
run_pattern = "run*"
run_list = []

vel_list = []
loc_list = []
con_list = []

for entry in run_dir:
    if fnmatch.fnmatch(entry, run_pattern):
        run_list.append(entry)

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
    cons = mesh_first.cells_dict['triangle']
    vels = []

    time_steps = 1
    for path in list_of_paths:
        print(path)
        mesh = meshio.read(path)
        vels.append(np.delete(mesh.point_data['Velocity'], 2, 1))
        time_steps += 1

    vel_list.append(vels)
    loc_list.append(locs)
    con_list.append(cons)

with tf.io.TFRecordWriter(record_file + '/train.tfrecord') as writer:
    for i in range(0, 40):
        print('writing train ' + str(i))
        tf_example = file_format(vel_list[i], loc_list[i], con_list[i], i)
        writer.write(tf_example.SerializeToString())

with tf.io.TFRecordWriter(record_file + '/test.tfrecord') as writer:
    for i in range(41, 44):
        print('writing train ' + str(i))
        tf_example = file_format(vel_list[i], loc_list[i], con_list[i], i)
        writer.write(tf_example.SerializeToString())

with tf.io.TFRecordWriter(record_file + '/validate.tfrecord') as writer:
    for i in range(44, 48):
        print('writing train ' + str(i))
        tf_example = file_format(vel_list[i], loc_list[i], con_list[i], i)
        writer.write(tf_example.SerializeToString())