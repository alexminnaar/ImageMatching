from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re

import numpy as np
import tensorflow as tf
import urllib2 as urllib
import boto3

s3 = boto3.resource('s3')


class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 model_dir,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.

        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


class InceptionImageClassifier:
    def __init__(self):
        # directory of inception model
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inception-2015-12-05")
        self.create_graph()

    def create_graph(self):
        """Creates a graph from saved GraphDef file and returns a saver."""
        # Creates graph from saved graph_def.pb.

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "inception-2015-12-05/classify_image_graph_def.pb"), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def run_inference_on_image(self, image, num_top_predictions=5):
        """Runs inference on an image.

        Args:
          image: Image file name.

        Returns:
          Nothing
        """
        # if not tf.gfile.Exists(image):
        #     tf.logging.fatal('File does not exist %s', image)
        # image_data = tf.gfile.FastGFile(image, 'rb').read()

        fd = urllib.urlopen(image)
        image_data = fd.read()

        # Creates graph from saved GraphDef.
        # self.create_graph()

        with tf.Session() as sess:
            # Some useful tensors:
            # 'softmax:0': A tensor containing the normalized prediction across
            #   1000 labels.
            # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
            #   float description of the image.
            # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
            #   encoding of the image.
            # Runs the softmax tensor by feeding the image_data as input to the graph.
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            # Creates node ID --> English string lookup.
            node_lookup = NodeLookup(self.model_dir)
            top_k = predictions.argsort()[-num_top_predictions:][::-1]
            tags = [(node_lookup.id_to_string(node_id), predictions[node_id]) for node_id in top_k]

            return tags


class CustomImageClassifier:
    def __init__(self):
        self.modelFullPath = os.path.dirname(os.path.abspath(__file__)) + "/model/output_graph_2.pb"
        self.labelsFullPath = os.path.dirname(os.path.abspath(__file__)) + "/model/output_labels_2.txt"
        self.create_graph()

    def create_graph(self):
        """Creates a graph from saved GraphDef file and returns a saver."""
        # Creates graph from saved graph_def.pb.
        with tf.gfile.FastGFile(self.modelFullPath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            del (graph_def.node[1].attr["dct_method"])
            _ = tf.import_graph_def(graph_def, name='')

    def run_inference_on_image(self, image_url):

        # Creates graph from saved GraphDef.
        image_data = urllib.urlopen(image_url).read()

        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
            f = open(self.labelsFullPath, 'rb')
            lines = f.readlines()
            labels = [str(w).replace("\n", "") for w in lines]

            return [labels[top_k[0]], predictions[top_k[0]]]



# def main():
#     image_url="https://images.viglink.com/product/250x250/images-eu-ssl-images-amazon-com/42ead83131510dcf3ba4b9666462f4da9a4c8782.jpg?url=https%3A%2F%2Fimages-eu.ssl-images-amazon.com%2Fimages%2FI%2F41Getv2arAL.jpg%2F518prb1KzVL.jpg"
#     my_classifier = CustomImageClassifier()
#     pred = my_classifier.run_inference_on_image(image_url)
#     print(pred)
#
#
# if __name__ == "__main__":
#     main()
