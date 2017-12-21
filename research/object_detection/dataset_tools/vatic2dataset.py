import tensorflow as tf
from os import path, chdir

from object_detection.utils import dataset_util
import xml.etree.ElementTree as ET
from object_detection.protos import string_int_label_map_pb2
from google.protobuf import text_format
from sys import stdout

flags = tf.app.flags
flags.DEFINE_string('frames_dir', '', 'Dir of video frames')
flags.DEFINE_string('xml', '', 'vatic xml file')

flags.DEFINE_string('output_dir', '', 'output dir')
flags.DEFINE_string('records_file_name', 'data.tfrecords', 'output tfrecords file name')
flags.DEFINE_string('labels_file_name', 'label-map.pbtxt', 'output labels file name')
FLAGS = flags.FLAGS


def create_tf_example(file_url, frame_time, objects, file_content, img):
  height, width, _ = img.shape
  filename = file_url
  encoded_image_data = file_content
  image_format = b'jpeg'

  xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
  ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
  classes_text = []  # List of string class name of bounding box (1 per box)
  classes = []  # List of integer class id of bounding box (1 per box)

  for obj_name, obj_id, ((x0, y0), (x1, y1)) in objects:
    xmins.append(x0/width)
    ymins.append(y0/height)
    xmaxs.append(x1/width)
    xmaxs.append(y1/height)

    classes_text.append(obj_name)
    classes.append(obj_id)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(filename),
    'image/source_id': dataset_util.bytes_feature(filename),
    'image/encoded': dataset_util.bytes_feature(encoded_image_data),
    'image/format': dataset_util.bytes_feature(image_format),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def get_point(p):
  return int(p.find('x').text), int(p.find('y').text)


def get_rect(pts):
  return get_point(pts[0]), get_point(pts[2])


def main(_):
  tree = ET.parse(path.join(FLAGS.output_dir, FLAGS.xml))
  root = tree.getroot()

  file_url = tf.placeholder(dtype=tf.string)
  file_content = tf.read_file(file_url)
  img = tf.image.decode_jpeg(file_content)

  frames = []

  def add_frame_info(frame, rect, obj_name, obj_id):
    for i in range(len(frames), frame + 1):
      frames.append((i, []))
    i, objs = frames[frame]
    objs.append((obj_name, obj_id, rect))

  labels = {}
  for obj in root.findall('object'):
    obj_name = obj.find('name').text
    if obj_name not in labels:
      labels[obj_name] = len(labels) + 1
    obj_id = labels[obj_name]
    for p in obj.findall('polygon'):
      add_frame_info(int(p.find('t').text), get_rect(p.findall('pt')), obj_name, obj_id)

  records_file_name = path.join(FLAGS.output_dir, FLAGS.records_file_name)
  with tf.Session() as sess, tf.python_io.TFRecordWriter(records_file_name) as writer:
    print('Writing: ' + records_file_name)
    for frame_time, objects in frames:
      file_name = '{}.jpg'.format(frame_time)
      tf_example = create_tf_example(file_name,
                                     frame_time,
                                     objects,
                                     *sess.run([file_content, img],
                                               feed_dict={file_url: path.join(FLAGS.frames_dir, file_name)}))
      writer.write(tf_example.SerializeToString())
      stdout.write(('{} done from {}\r'.format(frame_time + 1, len(frames))))

  labels_file_name = path.join(FLAGS.output_dir, FLAGS.labels_file_name)

  with open(labels_file_name, 'w') as f:
    print('Writing: ' + labels_file_name)
    mp = string_int_label_map_pb2.StringIntLabelMap(item=[string_int_label_map_pb2.StringIntLabelMapItem(name=l, id=i)
                                                          for l, i in labels.items()])
    f.write(text_format.MessageToString(mp))


if __name__ == '__main__':
  tf.app.run()
