import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
from queue import Queue 
from threading import Thread


from client_facemask import FaceClient
from client_peoplenet import PeoplenetClient

from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

personQueue = Queue(maxsize=30)
faceQueue = Queue(maxsize=30)
flag = 0

class record():
    
    def __init__(self, width, height):

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.avi', self.fourcc, 30, (width,height))
    
    def write(self, image):
        self.out.write(image)
    
    def release(self):
        self.out.release()

re = record(1270, 720)

################### deep sort #########################

class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):

        self.session = tf.Session()
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % input_name)
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out

def create_box_encoder(model_filename, input_name="images",
                       output_name="features", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder

def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list

############################################

def pgie1(client, stream, queue_out, mode):
    if mode == "image":
        image = cv2.imread(stream)
        bbox_batch = client.getPrediction(image)
        queue_out.put(bbox_batch)

    if mode == "video":
        video_capture = cv2.VideoCapture(stream)
        while True:
            start = time.time()
            ret, image = video_capture.read()
            if not ret or image is None or queue_out.qsize() == 30:
                break
            bbox_batch = client.getPrediction(image)
            # queue_out.put([client.getImageResult(),bbox_batch])
            queue_out.put([image,bbox_batch])
            fps = 1/(time.time() - start)
            print(fps)

        flag = 1
        
def tracking(queue_in ,queue_out):
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", 0.2, None)

    tracker = Tracker(metric)
    encoder = create_box_encoder("./deep_sort/networks/mars-small128.pb", batch_size=32)

    while(True):
        buffer = queue_in.get()
        image = buffer[0]
        bbox = buffer[1]
        print(bbox)
        # re.write(image)
        # detections = 
        # tracker.predict()
        # tracker.update(detections)
        
        if flag:
            print("pgie1 stoped!")
            print(queue_in.qsize())
        if flag and queue_in.qsize() == 0:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        nargs='?',
                        help='Input file to load from in image or video mode')
    parser.add_argument('mode',
                        type=str,
                        nargs='?',
                        default="image",
                        help='Input file to load from in image or video mode')
    
    FLAGS = parser.parse_args()
    clientFacemask = FaceClient(model_name="facemask-tiny")
    clientPerson = PeoplenetClient(only_label = "person")

       
    Thread(target = pgie1, args =(clientPerson, FLAGS.input, personQueue, FLAGS.mode)).start()
    Thread(target = tracking, args =(personQueue, faceQueue)).start()

    # pipeline person detection -> tracking -> facemask detection -> data buffer