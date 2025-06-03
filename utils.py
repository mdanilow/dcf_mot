from copy import copy

import numpy as np
import cv2


def draw_text_line(img, text, line=0, color=(0, 0, 0)):
    x_pos = 10
    y_pos = 20 + line * 20
    img = cv2.putText(img,
                      text,
                      (x_pos, y_pos),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.75,
                      color,
                      1,
                      cv2.LINE_AA)


def draw_bboxes(img, dets, color=(0, 0, 255), xywh_layout=False, id_to_color=None, id_to_trajectory=None, label_position='over', info_dict={}):
    # gt = dets.shape[-1] == 9
    # color = (0, 0, 255) if gt else (0, 255, 0)
    dets = copy(dets)
    for idx, det in enumerate(dets):
        obj_id = det[1]
        if id_to_color is not None and obj_id not in id_to_color:
            id_to_color[obj_id] = (np.random.randint(256), np.random.randint(256), np.random.randint(256))
        if not xywh_layout:
            # convert from x1y1x2y2 to xywh
            det[2:4] = det[2:4] - det[:2]
        xywh = det[:4]
        xywh = [int(x) for x in xywh]

        if id_to_trajectory is not None:
            if obj_id not in id_to_trajectory:
                id_to_trajectory[obj_id] = []
            center = (xywh[0] + xywh[2]//2, xywh[1] + xywh[3]//2)
            id_to_trajectory[obj_id].append(center) 
            for point in id_to_trajectory[obj_id]:
                img = cv2.circle(img, point, radius=1, color=id_to_color[obj_id], thickness=2)
        
        img = cv2.rectangle(img, (xywh[0], xywh[1]), (xywh[0]+xywh[2], xywh[1]+xywh[3]), id_to_color[obj_id] if id_to_color else color, 1)
        font_scale = 0.5
        line_thickness = 1
        y_pos = xywh[1] - 7 if label_position == "over" else xywh[1] + 14
        line_height = 17
        for i, (key, values) in enumerate(info_dict.items()):
            text = key + ": {}".format(values[idx])
            img = cv2.putText(img, text,
                            (xywh[0] + 3, y_pos + line_height * i),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            color,
                            line_thickness,
                            cv2.LINE_AA)
        

def draw_frame_info(img, trackers, detections, frame_number, show_conf=False, scale=2):
    trackers_bboxes = np.stack([np.squeeze(t.get_state()) for t in trackers]) if trackers else np.empty(shape=(0,4), dtype=int)
    if detections.size == 0:
        detections = np.empty(shape=(0, 4), dtype=int)
    if not show_conf and detections.size > 0:
        detections = np.array([det[:4] for det in detections])
    trackers_bboxes[:, :4] *= scale
    detections[:, :4] *= scale

    detections_info = {"idx": list(range(detections.shape[0]))}
    trackers_info = {"id": [t.id for t in trackers],
                     "age": [t.time_since_update for t in trackers],
                     "hit_streak": [t.hit_streak for t in trackers]}
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    draw_text_line(img, "Tracks", line=0, color=(255, 0, 0))
    draw_text_line(img, "Detections", line=1, color=(0, 0, 255))
    draw_text_line(img, "Frame: " + str(frame_number), line=2, color=(0, 255, 0))
    draw_bboxes(img, trackers_bboxes, color=(255, 0, 0), label_position="under", info_dict=trackers_info)
    draw_bboxes(img, detections, color=(0, 0, 255), info_dict=detections_info)
    return img


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, np.ndarray):
        np.clip(boxes[:, 0], 0, img_shape[1])
        np.clip(boxes[:, 1], 0, img_shape[0])
        np.clip(boxes[:, 2], 0, img_shape[1])
        np.clip(boxes[:, 3], 0, img_shape[0])
    else:
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2

    
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    coords = copy(coords)
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords