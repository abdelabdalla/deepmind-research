import functools
import json
import os
import tensorflow as tf

from learning_to_simulate import reading_utils


def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())


data_path = "/tmp/WaterDrop"
metadata = _read_metadata(data_path)
ds = tf.data.TFRecordDataset([os.path.join(data_path, 'test.tfrecord')])
ds = ds.map(functools.partial(
            reading_utils.parse_serialized_simulation_example, metadata=metadata))

n = ds.make_one_shot_iterator().get_next()
sess = tf.Session()

end = sum(1 for _ in tf.python_io.tf_record_iterator(os.path.join(data_path, 'test.tfrecord')))

value = []

for i in range(0, end):
    print(str(i))
    v = sess.run(n)
    value.append(v)
