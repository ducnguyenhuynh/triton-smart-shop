import argparse
import numpy as np
import sys
import cv2
import time

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from labels import PeoplenetLabel
from people_utils import PeopleUtils
from render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS

class PeoplenetClient(object):
    def __init__(self, model_name = "peoplenet", url="0.0.0.0:8001", confidence=0.5, nms = 0.5, only_label = None):
        self.model_name = model_name
        self.url = url
        self.confidence = confidence
        self.nms = nms
        self.batch_size = 1
        self.module = PeopleUtils()
        self.only_label = only_label

        # Create server context
        try:
            self.triton_client = grpcclient.InferenceServerClient(
                url=url,
                verbose=False,
                ssl=False,
                root_certificates=None,
                private_key=None,
                certificate_chain=None)
        except Exception as e:
            print("context creation failed: " + str(e))
            sys.exit()
        
        # Health check
        if not self.triton_client.is_server_live():
            print("FAILED : is_server_live")
            sys.exit(1)

        if not self.triton_client.is_server_ready():
            print("FAILED : is_server_ready")
            sys.exit(1)
        
        if not self.triton_client.is_model_ready(self.model_name):
            print("FAILED : is_model_ready")
            sys.exit(1)


        self.inputs = []
        self.outputs = []
        
        self.inputs.append(grpcclient.InferInput('input_1', [self.batch_size, 3, 544, 960], "FP32"))
        self.outputs.append(grpcclient.InferRequestedOutput("output_bbox/BiasAdd"))
        self.outputs.append(grpcclient.InferRequestedOutput("output_cov/Sigmoid"))
        
    def getPrediction(self, input_image):
        
        self.image = input_image
        
        # normalize
        input_image_buffer = self.module._preprocess_(input_image, (544,960,3))
        # put to buffer
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        self.inputs[0].set_data_from_numpy(input_image_buffer)

        #get results
        results = self.triton_client.infer(model_name=self.model_name,
                                    inputs=self.inputs,
                                    outputs=self.outputs,
                                    client_timeout=None)

        result_boxes = results.as_numpy('output_bbox/BiasAdd')
        result_confs = results.as_numpy('output_cov/Sigmoid')

        # print(result_boxes, result_confs)
        

        bboxes, classes, scores = self.module._post_processing_(self.confidence, self.nms, [result_boxes[0], result_confs[0]])
        

        self.classes = [PeoplenetLabel(int(classID)).name for classID in classes]
        self.bboxes_batch = []
        for i, bbox in enumerate(bboxes):
            self.bboxes_batch.append([bbox[0],bbox[1],bbox[2], bbox[3], scores[i], classes[i]])

        if self.only_label:
            self.bboxes_batch = np.array(self.bboxes_batch)[np.array(self.classes) == self.only_label]
            self.classes = np.array(self.classes)[np.array(self.classes) == self.only_label]
        return self.bboxes_batch
    
    def getImageResult(self, image = None):

        if image is None:
            image = self.image

        
        for i, det in enumerate(self.bboxes_batch):
            # print(det)
            x1 = int(det[0])
            y1 = int(det[1])
            x2 = int(det[2])
            y2 = int(det[3])
            ratio_x = image.shape[1] / 960
            ratio_y = image.shape[0] / 544

            # box = int(x1), int(y1), int(x2), int(y2)
            
            box = x1*ratio_x, y1*ratio_y, x2*ratio_x, y2*ratio_y
            # print(box)
            confidence = det[4]
            
            # print(f"{self.classes[i]}: {confidence}")

            input_image = render_box(image, box, color=tuple(RAND_COLORS[i % 32].tolist()))
            size = get_text_size(input_image, f"{self.classes[i]}: {confidence:.2f}", normalised_scaling=0.6)
            result_image = render_filled_box(input_image, (box[0] - 3, box[1] - 3, box[0] + size[0], box[1] + size[1]), color=(220, 220, 220))
            image = render_text(result_image, f"{self.classes[i]}: {confidence:.2f}", (box[0], box[1]), color=(30, 30, 30), normalised_scaling=0.5)
            
            
        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        nargs='?',
                        help='Input file to load from in image or video mode')
    parser.add_argument('model_name',
                        type=str,
                        nargs='?',
                        default="peoplenet",
                        help='Input file to load from in image or video mode')
    FLAGS = parser.parse_args()

    client = PeoplenetClient(model_name=FLAGS.model_name)
    image = cv2.imread(str(FLAGS.input))
    client.getPrediction(image)
    image_output = client.getImageResult()
    cv2.imwrite("output.jpg", image_output)
    