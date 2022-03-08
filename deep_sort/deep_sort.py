import numpy as np
import torch

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker


__all__ = ['DeepSort']


def recur_trackid(track_id, prev_ids, class_id):
    if track_id in list(prev_ids.keys()) and class_id == prev_ids[track_id]:
        return track_id 
    elif track_id in list(prev_ids.keys()) and class_id != prev_ids[track_id]:
        track_id = track_id + 1
        return recur_trackid(track_id, prev_ids, class_id)
    else:
        return track_id

def area(x1, y1, x2, y2):
    width = abs(int(x2 - x1))
    height = abs(int(y2 - y1))
    return 0.5 * width * height

def compare_bboxes(current_box, current_trackid, current_classid,  previds_boxes):
    x1, y1, x2, y2 = map(int, current_box)
    current_area = area(x1, y1, x2, y2 )

    prevx1, prevy1, prevx2, prevy2 = previds_boxes[:4]
    prevbox_area = area(prevx1, prevy1, prevx2, prevy2)
    prev_classid = int(previds_boxes[-1])

    if int(current_classid) == prev_classid and current_area >= prevbox_area and prevx1 > x1:
        track_id = current_trackid
    elif int(current_classid) == prev_classid and current_area < prevbox_area:
        print("Less area")
        track_id = current_trackid + 1
    # elif int(current_classid) != prev_classid:
    #     print("SAME trackid but different class")
    #     track_id = current_trackid + 1
    else:
        track_id = current_trackid 
        
    return track_id

class DeepSort(object):
    def __init__(self, model_type, max_dist=0.2, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):

        self.extractor = Extractor(model_type, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

        self.previds_boxes = {}

    def update(self, bbox_xywh, confidences, classes, ori_img, prev_ids, frames_classes,frame_index, use_yolo_preds=True):
        self.height, self.width = ori_img.shape[:2]
        print("Initial Inputs", len(bbox_xywh))
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences)]

        # print("Features", len(features))
        # print("detections", [d.tlwh for d in detections])
        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            if use_yolo_preds:
                det = track.get_yolo_pred()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(det.tlwh)
            else:
                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            class_id = track.class_id

                
            # track_id = recur_trackid(track_id, prev_ids, class_id)
            
            # if len(self.previds_boxes) and track_id in list(self.previds_boxes.keys()) and class_id in [7, 4, 5]:
            #     prev_info = self.previds_boxes[track_id][-1]
            #     track_id = compare_bboxes([x1, y1, x2, y2], track_id, class_id, prev_info)


            # if not track_id in list(self.previds_boxes.keys()) :
            #     self.previds_boxes[track_id] = []
            # self.previds_boxes[track_id].append([x1, y1, x2, y2, class_id])

            # prev_ids.update(track_id = class_id)
            # print([x1, y1, x2, y2, track_id, class_id])

            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features
