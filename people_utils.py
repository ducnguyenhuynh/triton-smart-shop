import cv2
import numpy as np
from utils import Module

class PeopleUtils(Module):
    def __init__(self):
        super(Module).__init__()
        model_h = 544
        model_w = 960
        stride = 16
        self.box_norm = 35.0

        self.grid_h = int(model_h / stride)
        self.grid_w = int(model_w / stride)
        self.grid_size = self.grid_h * self.grid_w

        self.grid_centers_w = []
        self.grid_centers_h = []

        for i in range(self.grid_h):
            value = (i * stride + 0.5) / self.box_norm
            self.grid_centers_h.append(value)

        for i in range(self.grid_w):
            value = (i * stride + 0.5) / self.box_norm
            self.grid_centers_w.append(value)


    def applyBoxNorm(self, o1, o2, o3, o4, x, y):
        """
        Applies the GridNet box normalization
        Args:
            o1 (float): first argument of the result
            o2 (float): second argument of the result
            o3 (float): third argument of the result
            o4 (float): fourth argument of the result
            x: row index on the grid
            y: column index on the grid

        Returns:
            float: rescaled first argument
            float: rescaled second argument
            float: rescaled third argument
            float: rescaled fourth argument
        """

        o1 = (o1 - self.grid_centers_w[x]) * -self.box_norm
        o2 = (o2 - self.grid_centers_h[y]) * -self.box_norm
        o3 = (o3 + self.grid_centers_w[x]) * self.box_norm
        o4 = (o4 + self.grid_centers_h[y]) * self.box_norm

        return o1, o2, o3, o4

    def _post_processing_(self, conf_thresh, nms_thresh, outputs, wh_format=False):
        """
        Postprocesses the inference output
        Args:
            outputs (list of float): inference output
            min_confidence (float): min confidence to accept detection
            analysis_classes (list of int): indices of the classes to consider

        Returns: list of list tuple: each element is a two list tuple (x, y) representing the corners of a bb
        """

        bboxes = []
        class_ids = []
        scores = []
        NUM_CLASSES = 3
        analysis_classes =  list(range(NUM_CLASSES))

        for c in analysis_classes:

            # x1_idx = c * 4 * self.grid_size
            # y1_idx = x1_idx + self.grid_size
            # x2_idx = y1_idx + self.grid_size
            # y2_idx = x2_idx + self.grid_size
            
            boxes = outputs[0][4*c:4*c+4]
            # print(boxes.shape)
            
            # print(result_confs.shape)
            for h in range(self.grid_h):
                for w in range(self.grid_w):
                    i = w + h * self.grid_w
                    
                    score = outputs[1][c][h][w]
                    
                    if score >= conf_thresh:
                        o1 = boxes[0][h][w]
                        o2 = boxes[1][h][w]
                        o3 = boxes[2][h][w]
                        o4 = boxes[3][h][w]

                        o1, o2, o3, o4 = self.applyBoxNorm(o1, o2, o3, o4, w, h)

                        xmin = int(o1)
                        ymin = int(o2)
                        xmax = int(o3)
                        ymax = int(o4)
                        if wh_format:
                            bboxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
                        else:
                            bboxes.append([xmin, ymin, xmax, ymax])
                        class_ids.append(c)
                        scores.append(float(score))

            indexes = cv2.dnn.NMSBoxes(bboxes, scores, conf_thresh, nms_thresh)
            
        return np.array(bboxes)[indexes], np.array(class_ids)[indexes], np.array(scores)[indexes]