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


def file_format(point_vel, point_location, point_connections, n_nodes, n_cons, key):
    location_tensor = tf.convert_to_tensor(point_location)
    location_serialized = tf.io.serialize_tensor(location_tensor)

    connections_tensor = tf.convert_to_tensor(point_connections)
    connections_serialized = tf.io.serialize_tensor(connections_tensor)

    n_nodes_tensor = tf.convert_to_tensor(n_nodes)
    n_nodes_serialized = tf.io.serialize_tensor(n_nodes_tensor)

    n_cons_tensor = tf.convert_to_tensor(n_cons)
    n_cons_serialized = tf.io.serialize_tensor(n_cons_tensor)

    context = {
        'key': _int64_feature(key),
        'locations': _bytes_feature(location_serialized),
        'connections': _bytes_feature(connections_serialized),
        'n_nodes': _bytes_feature(n_nodes_serialized),
        'n_cons': _bytes_feature(n_cons_serialized)
    }

    features = [tf.train.Feature(bytes_list=tf.train.BytesList(value=[val.tobytes()])) for val in point_vel]
    featurelist = tf.train.FeatureList(feature=features)

    return tf.train.SequenceExample(context=tf.train.Features(feature=context),
                                    feature_lists=tf.train.FeatureLists(feature_list={'velocity': featurelist}))


file_location = 'D:\\Users\\abdel\\Documents\\Disso_New_Data_Updated'
record_file = 'D:\\Users\\abdel\\Documents\\new_dataset'

run_dir = os.listdir(file_location)
run_pattern = "run*"
run_list = []

vel_list = []
loc_list = []
con_list = []
n_nodes_list = []
n_cons_list = []

for entry in run_dir:
    if fnmatch.fnmatch(entry, run_pattern):
        run_list.append(entry)

for run in run_list:
    print('folder ' + run)

    list_of_files = os.listdir(file_location + '\\' + run)
    pattern = "flow_*.vtk"
    list_of_names = []

    for entry in list_of_files:
        if fnmatch.fnmatch(entry, pattern):
            list_of_names.append(entry)

    list.sort(list_of_names)

    list_of_paths = [file_location + '\\' + run + '\\' + s for s in list_of_names]
    mesh_first = meshio.read(list_of_paths[0])
    locs = np.delete(mesh_first.points, 2, 1)
    cons = np.array(mesh_first.cells_dict['triangle'])
    vels = []

    time_steps = 0
    for path in list_of_paths:
        if time_steps >= 200:
            break
        mesh = meshio.read(path)
        vels.append(np.delete(mesh.point_data['Velocity'], 2, 1).flatten().astype(np.float32))
        time_steps += 1

    vel_list.append(vels)
    loc_list.append(locs.flatten().astype(np.float32))
    con_list.append(cons.flatten().astype(np.int64))
    n_nodes_list.append(len(locs))
    n_cons_list.append(len(cons))



with tf.io.TFRecordWriter(record_file + '\\train.tfrecord') as writer:
    for i in range(0, 36):
        print('writing train ' + str(i))
        tf_example = file_format(vel_list[i], loc_list[i], con_list[i], n_nodes_list[i], n_cons_list[i], i)
        writer.write(tf_example.SerializeToString())

with tf.io.TFRecordWriter(record_file + '\\test.tfrecord') as writer:
    for i in range(36, 53):
        print('writing test ' + str(i))
        tf_example = file_format(vel_list[i], loc_list[i], con_list[i], n_nodes_list[i], n_cons_list[i], i)
        writer.write(tf_example.SerializeToString())

with tf.io.TFRecordWriter(record_file + '\\validate.tfrecord') as writer:
    for i in range(53, 70):
        print('writing validate ' + str(i))
        tf_example = file_format(vel_list[i], loc_list[i], con_list[i], n_nodes_list[i], n_cons_list[i], i)
        writer.write(tf_example.SerializeToString())
