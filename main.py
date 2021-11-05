import cv2
import time
import argparse
import numpy as np
from queue import Queue 
from threading import Thread

from labels import Facelabel
from client_facemask import FaceClient
from client_peoplenet import PeoplenetClient
from render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS

from deep_sort.deep_sort import DeepSort

personQueue = Queue(maxsize=30)
trackingQueue = Queue(maxsize=30)
faceQueue = Queue(maxsize=50)
flag = 0
history = {} # format 'id': image, labels, color




############################################

def pgie1(client, stream, queue_out, mode):
    frame_id = 0
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
            queue_out.put([image, frame_id, bbox_batch])
            fps = 1/(time.time() - start)
            print("pgie1: ",fps)
            frame_id += 1
        flag = 1
        
def tracking(queue_in ,queue_out):
    
    # Get tracker
    max_dist = 0.2
    min_confidence = 0.3
    nms_max_overlap = 0.7
    max_iou_distance = 0.7
    max_age = 70
    n_init = 3
    nn_budget = 100

    tracker = DeepSort(
        model_path='deep_sort/deep/checkpoint/ckpt.t7',
        max_dist=max_dist,
        min_confidence=min_confidence,
        nms_max_overlap=nms_max_overlap,
        max_iou_distance=max_iou_distance,
        max_age=max_age,
        n_init=n_init,
        nn_budget=nn_budget,
        use_cuda=use_cuda
    )

    while(True):
        start = time.time()
        buffer = queue_in.get()
        image = buffer[0]
        frame_id = buffer[1]
        bbox_batch = buffer[2]
        bboxes = bbox_batch[:,:-2].astype("int")
        # convert  x1,y1,x2,y2 -> x,y,w,h
        bboxes[:,2] = bboxes[:,2] - bboxes[:,0]
        bboxes[:,3] = bboxes[:,3] - bboxes[:,1]
        print(bboxes)
        tracks = tracker.update(bboxes, bbox_batch[:,-2], images, bbox_batch[:,-1])
        # encode detections and feed to tracker
        # features = encoder(image, bboxes)
        # detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        # Call the tracker
        # tracker.predict()
        # tracker.update(detections)
        fps = 1/(time.time() - start)
        print("tracking: ",fps)

        if flag:
            print("pgie1 stoped!")
            print(queue_in.qsize())
        if flag and queue_in.qsize() == 0:
            break

def pgie2(client, queue_in ,queue_out):
    '''
    queue in format
    image, frame_id, person_bboxes, id_objs
    bbox format is 'x1,y1,x2,y2'
    '''
    while True:
        start = time.time()
        buffer = queue_in.get()
        image, frame_id, person_bboxes, id_objs = buffer
        for i, id in enumerate(id_objs):
            # if have label in history -> continue
            # if history[str(id)][-1] is not None:
            #     continue
            box = person_bboxes[i]
            image_croped = image[box[1]:box[3],box[0]:box[2],:]
            # send to server
            bbox_batch = client.getPrediction(image_croped)
            [history.update({id:[Facelabel(int(bboxdata[-1])).name, tuple(RAND_COLORS[id % 32].tolist())]}) for bboxdata in bbox_batch]
            # (x1, y1, x2, y2, conf, class)
            queue_out.put([image, frame_id, person_bboxes, id_objs, bbox_batch])
            

        fps = 1/(time.time() - start)
        print("pgie2: ",fps)

        if flag:
            print("pgie1 stoped!")
            print(queue_in.qsize())
        if flag and queue_in.qsize() == 0:
            break

def draw_image_out(final_queue):
    while True:
        buffer = final_queue.get()
        image, frame_id, person_bboxes, id_objs, face_bboxes = buffer.get()
    
        for i,id  in enumerate(id_objs):
            clas, color = history[id]
            box_person = person_bboxes[i]
            box_face = face_bboxes[i]
            
            image = render_box(image, box_person, color=color)
            image = render_box(image, box_face, color=color)

            size = get_text_size(image, f"{id}: {clas}", normalised_scaling=0.6)
            image = render_filled_box(image, (box_person[0] - 3, box_person[1] - 3, box_person[0] + size[0], box_person[1] + size[1]), color=(220, 220, 220))
        
        cv2.imshow('image', image)
        cv2.waitKey(1)

        if flag:
            print("pgie1 stoped!")
            print(final_queue.qsize())
        if flag and final_queue.qsize() == 0:
            break

def record_out(final_queue):

    class record():
    
        def __init__(self, width, height):

            self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter('output.avi', self.fourcc, 30, (width,height))
        
        def write(self, image):
            self.out.write(image)
        
        def release(self):
            self.out.release()

    re = record(1270, 720)

    while True:
        
        buffer = final_queue.get()
        image, frame_id, person_bboxes, id_objs, face_bboxes = buffer.get()
        
        for i,id  in enumerate(id_objs):

            clas, color = history[id]
            box_person = person_bboxes[i]
            box_face = face_bboxes[i]
            
            image = render_box(image, box_person, color=color)
            image = render_box(image, box_face, color=color)
            size = get_text_size(image, f"{id}: {clas}", normalised_scaling=0.6)
            image = render_filled_box(image, (box_person[0] - 3, box_person[1] - 3, box_person[0] + size[0], box_person[1] + size[1]), color=(220, 220, 220))
        re.write(image)

        if flag:
            print("pgie1 stoped!")
            print(final_queue.qsize())
        if flag and final_queue.qsize() == 0:
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
    Thread(target = tracking, args =(personQueue, trackingQueue)).start()
    
    # pgie2
    # Thread(target = pgie2, arg= (clientFacemask, trackingQueue, faceQueue)).start()
    # Thread(target = pgie2, arg= (clientFacemask, trackingQueue, faceQueue)).start()
    # Thread(target = pgie2, arg= (clientFacemask, trackingQueue, faceQueue)).start()
    # Thread(target = pgie2, arg= (clientFacemask, trackingQueue, faceQueue)).start()
    # Thread(target = pgie2, arg= (clientFacemask, trackingQueue, faceQueue)).start()
    # draw to show or record

    # pipeline person detection -> tracking -> facemask detection -> data buffer