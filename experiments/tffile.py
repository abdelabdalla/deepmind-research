import json

import tensorflow as tf
from google.protobuf.json_format import MessageToJson


def _parse_function(record):
    """Extracts features and labels.

    Args:
      record: File path to a TFRecord file
    Returns:
      A `tuple` `(labels, features)`:
        features: A dict of tensors representing the features
        labels: A tensor with the corresponding labels.
    """
    features = {
        "key": tf.VarLenFeature(dtype=tf.int64),  # terms are strings of varying lengths
        "velocity": tf.FixedLenFeature(shape=[1], dtype=tf.string)  # labels are 0 or 1
    }

    parsed_features = tf.parse_single_example(record, features)

    terms = parsed_features['key'].values
    labels = parsed_features['velocity']

    return {'terms': terms}, labels


# byte_list = []
#
# byte_translate = []

train_path = "/Volumes/Samsung/train.tfrecord"

for example in tf.python_io.tf_record_iterator(train_path):
    exjson = MessageToJson(tf.train.Example.FromString(example))
#     # ex = bytes(example)
#     # byte_list.append(ex)
#     ex = json.loads(exjson)
#     byties = bytes(ex['features']['feature']['particle_type']['bytesList']['value'][0])
#     # byte_list.append(byties)
#     # byte_translate.append(byties.decode('utf-8'))


# Create the Dataset object.
ds = tf.data.TFRecordDataset(train_path)
# Map features and labels with the parse function.
ds = ds.map(_parse_function)

# Make a one shot iterator
n = ds.make_one_shot_iterator().get_next()
sess = tf.Session()

end = sum(1 for _ in tf.python_io.tf_record_iterator(train_path))

output_features = []
output_labels = []
out_proper = []

for i in range(0, end):
    value = sess.run(n)
    output_features.append(value[0]['terms'])
    output_labels.append(value[1])
    out_proper.append(int.from_bytes(value[1][0], byteorder='big'))
