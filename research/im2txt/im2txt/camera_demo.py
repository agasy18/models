from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from time import time
import numpy as np
import argparse
import cv2
import tensorflow as tf
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('vocab_file')

args = parser.parse_args()


class InferenceWrapper:
  def __init__(self, model_path):
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(model_path, "rb") as f:
      graph_def.ParseFromString(f.read())
    self.image_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))
    tf.import_graph_def(graph_def, name="", input_map={'ExpandDims_1': self.image_placeholder})

  def feed_image(self, sess, encoded_image):
    initial_state = sess.run(fetches="lstm/initial_state:0",
                             feed_dict={self.image_placeholder: encoded_image})
    return initial_state

  def inference_step(self, sess, input_feed, state_feed):
    softmax_output, state_output = sess.run(
      fetches=["softmax:0", "lstm/state:0"],
      feed_dict={
        "input_feed:0": input_feed,
        "lstm/state_feed:0": state_feed,
      })
    return softmax_output, state_output, None


vocab = vocabulary.Vocabulary(args.vocab_file)
model = InferenceWrapper(args.model)

with tf.Session() as sess:
  generator = caption_generator.CaptionGenerator(model, vocab)

  cap = cv2.VideoCapture(0)

  last_fps = 0

  while True:
    t0 = time()
    ret, frame = cap.read()
    frame_f = (np.expand_dims(frame, 0) / 255.0).astype(np.float32)
    captions = generator.beam_search(sess, frame_f)

    last_fps += 1.0 / (time() - t0)
    last_fps /= 2.0
    spaceing = 15
    text_pos = 0
    text_pos += spaceing
    cv2.putText(frame,
                "FPS: {}".format(int(last_fps * 100) / 100.0),
                (0, text_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                255)

    captions = [(" ".join([vocab.id_to_word(w) for w in caption.sentence[1:-1]]), math.exp(caption.logprob))
                for i, caption in enumerate(captions)]

    spaceing = 50

    sp = 0

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
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
