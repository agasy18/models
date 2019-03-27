###
# build:
# bazel build -c opt //im2txt/...
# pushd ..; protoc object_detection/protos/*.proto --python_out=.; popd
# example:
# bazel-bin/im2txt/camera_demo /Users/aghasy/GoogleDrive/SCI/PHD/demo/im2txt_orig/camera.config.json
# bazel-bin/im2txt/camera_demo /Users/aghasy/GoogleDrive/SCI/PHD/demo/im2txt+ssd/3/camera.config.json
# bazel-bin/im2txt/camera_demo /Users/aghasy/GoogleDrive/SCI/PHD/demo/im2txt+ssd/3/camera.config.json --source ~/tmp/demo.mp4
###



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../slim/')
sys.path.insert(0, '../object_detection')

import math
from time import time
import numpy as np
import argparse
import cv2
import tensorflow as tf

from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
import json
from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('--source', default=0)
parser.add_argument('--size', default='1280x960')

args = parser.parse_args()
with open(args.config) as f:
  config = json.load(f)
      

model_input_mapping = config['model']['input_mapping']  
model_output_mapping = config['model']['output_mapping']  

def abs_path(path):
    if os.path.exists(path):
          return os.path.abspath(path)
    path2 = os.path.join(os.path.dirname(args.config), path)
    if os.path.exists(path2):
          return os.path.abspath(path2)
    exit('File not found: ' + path)


PATH_TO_LABELS = abs_path(config['path_to_labels'])
NUM_CLASSES = config['num_classes']


label_map  = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


class CodeBlockTimer:
    def __init__(self, auto_reset=-1):
        self.count = 0
        self.totalDuration = 0.0
        self.maxTime = 0
        self.minTime = 0
        self._enter_time = 0
        self.auto_reset = auto_reset

    def __enter__(self):
        if 0 < self.auto_reset < self.count:
            self.reset()

        self._enter_time = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.count += 1
        t = time() - self._enter_time
        self.maxTime = max(t, self.maxTime)
        self.minTime = min(t, self.minTime)
        self.totalDuration += t

    @property
    def fps(self):
        if not self.count:
            return 0
        return 1.0 / self.mean_duration

    @property
    def mean_duration(self):
        if not self.count:
            return 0.0
        return self.totalDuration / self.count

    def reset(self):
        self.count = 0
        self.totalDuration = 0
        self.maxTime = 0
        self.minTime = 0

    def __str__(self):
        return "Mean: {:4.4f}, FPS: {:8.1f}, N: {:5d} T: {:4.4f}".format(self.mean_duration, self.fps, self.count, self.totalDuration)

class InferenceWrapper:
  def __init__(self, model_path):
    graph_def = tf.GraphDef()
    self.detection_timer = CodeBlockTimer(10)
    with tf.gfile.FastGFile(model_path, "rb") as f:
      graph_def.ParseFromString(f.read())
    self.image_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))
    self.results = {}
    tf.import_graph_def(graph_def, name="", input_map={model_input_mapping['float_image']: self.image_placeholder})

  def feed_image(self, sess, encoded_image):
    fetches = [
      model_output_mapping['lstm/initial_state'],
      model_output_mapping['num_detections'],
      model_output_mapping['detection_boxes'],
      model_output_mapping['detection_scores'],
      model_output_mapping['detection_classes']
    ]
    with self.detection_timer:
      self.results = sess.run(fetches=dict(zip(fetches, [f + ':0' for f in fetches])),
                              feed_dict={self.image_placeholder: encoded_image})
    print('Detection', self.detection_timer)
    initial_state = self.results[model_output_mapping['lstm/initial_state']]
    return initial_state

  def inference_step(self, sess, input_feed, state_feed):
    softmax_output, state_output = sess.run(
      fetches=[model_output_mapping["softmax"] + ":0", model_output_mapping["lstm/state"] + ":0"],
      feed_dict={
        model_input_mapping["input_feed"] + ":0": input_feed,
        model_input_mapping["lstm/state_feed"] + ":0": state_feed,
      })
    return softmax_output, state_output, None


vocab = vocabulary.Vocabulary(abs_path(config['vocab_file']))
model = InferenceWrapper(abs_path(config['model']['path']))

with tf.Session() as sess:
  generator = caption_generator.CaptionGenerator(model, vocab)

  cap = cv2.VideoCapture(args.source)
  total_timer = CodeBlockTimer(10)

  while True:
    t0 = time()
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, tuple(int(x) for x in args.size.split('x')))
    frame_f = (np.expand_dims(frame, 0) / 255.0).astype(np.float32)
    with total_timer:
      captions = generator.beam_search(sess, frame_f)
    spaceing = 15
    text_pos = 0
    text_pos += spaceing
    cv2.putText(frame,
                "FPS: {}".format(int(total_timer.fps * 100) / 100.0),
                (0, text_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                255)

    captions = [(" ".join([vocab.id_to_word(w) for w in caption.sentence[1:-1]]), math.exp(caption.logprob))
                for i, caption in enumerate(captions)]

    spaceing = 50

    sp = 0

    frame = vis_util.visualize_boxes_and_labels_on_image_array(
      frame,
      model.results[model_output_mapping['detection_boxes']][0],
      model.results[model_output_mapping['detection_classes']][0].astype(np.uint8),
      model.results[model_output_mapping['detection_scores']][0],
      category_index,
      instance_masks=None,
      use_normalized_coordinates=True,
      line_thickness=min(frame.shape[:-1]) // 100,
      fontsize=50,
      alignment='cc')

    sp = max(sp, sum(p for c, p in captions))
    for i, (c, p) in enumerate(captions):
      size = np.sqrt(p / sp)
      s = '{}) {}'.format(i + 1, c)
      text_pos += spaceing
      cv2.putText(frame,
                  s,
                  (10, text_pos),
                  cv2.FONT_HERSHEY_DUPLEX,
                  min(1.5 * size, 1.1),
                  (255, 0, 0))

    cv2.imshow('window', frame)
    print('Total', total_timer)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
