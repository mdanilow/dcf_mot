from copy import copy

import numpy as np
import cv2


def draw_text_line(img, text, line=0, offset=(10, 20), color=(0, 0, 0), font_scale=0.75):
    x_pos = offset[0]
    y_pos = offset[1] + line * 20
    img = cv2.putText(img,
                      text,
                      (x_pos, y_pos),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      font_scale,
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
        line_height = 17
        y_pos = xywh[1] - line_height * len(info_dict.keys()) if label_position == "over" else xywh[1] + 14
        for i, (key, values) in enumerate(info_dict.items()):
            text = key + ": {}".format(values[idx])
            img = cv2.putText(img, text,
                            (xywh[0] + 3, y_pos + line_height * i),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            color,
                            line_thickness,
                            cv2.LINE_AA)
        

def draw_frame_info(img, trackers, detections, frame_number, scale=2, dcf=False):
    trackers_bboxes = np.stack([np.squeeze(t.get_state()) for t in trackers]) if trackers else np.empty(shape=(0,4), dtype=int)
    detections = detections.copy()
    if detections.size == 0:
        detections = np.empty(shape=(0, 5), dtype=int)
    trackers_bboxes = trackers_bboxes.astype(float)
    detections = detections.astype(float)
    # if not show_conf and detections.size > 0:
    #     detections = np.array([det[:4] for det in detections])
    trackers_bboxes[:, :4] *= scale
    detections[:, :4] *= scale

    detections_info = {"idx": list(range(detections.shape[0])),
                       "conf": ["{:.2f}".format(d) for d in detections[:, 4]]}
    trackers_info = {"id": [t.id for t in trackers],
                     "sinc_upd": [t.time_since_update for t in trackers],
                     "hitstr": [t.hit_streak for t in trackers]}
    if dcf:
        dcf_info = {"psr": ["{:.1f}".format(t.dcf.psr) for t in trackers],
                    "m_res": ["{:.2f}".format(t.dcf.max_response) for t in trackers],
                    "sinc_det": [t.time_since_detected for t in trackers],
                    "d_hitstr": [t.detection_hit_streak for t in trackers]}
        trackers_info.update(dcf_info)
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    draw_text_line(img, "Tracks", line=0, color=(255, 0, 0))
    draw_text_line(img, "Detections", line=1, color=(0, 0, 255))
    draw_text_line(img, "Frame: " + str(frame_number), line=2, color=(0, 255, 0))
    draw_bboxes(img, trackers_bboxes, color=(255, 0, 0), label_position="under", info_dict=trackers_info)
    draw_bboxes(img, detections, color=(0, 0, 255), info_dict=detections_info)
    return img


def draw_frame_info_byte(img, trackers, lost_trackers, in_detections, frame_number, scale=1, dcf=False, det_conf_th=0):
    img = img.copy()
    trackers_bboxes = np.array([t.tlbr for t in trackers]) if trackers else np.empty(shape=(0,4), dtype=int)
    lost_trackers_bboxes = np.array([t.tlbr for t in lost_trackers]) if lost_trackers else np.empty(shape=(0,4), dtype=int)
    detections = in_detections.copy()
    if type(in_detections) == np.ndarray:
        if detections.size == 0:
            detections = np.empty(shape=(0, 5), dtype=int)
        else:
            detections = np.array([det for det in detections if det[4] >= det_conf_th])
    elif type(in_detections) == list:
        if len(detections) == 0:
            detections = np.empty(shape=(0, 5), dtype=int)
        else:
            detections = np.array([d.tlbr for d in detections if d.score >= det_conf_th])
    trackers_bboxes = trackers_bboxes.astype(float)
    lost_trackers_bboxes = lost_trackers_bboxes.astype(float)
    clip_coords(trackers_bboxes, img.shape)
    clip_coords(detections, img.shape)
    clip_coords(lost_trackers_bboxes, img.shape)
    detections = detections.astype(float)
    # if not show_conf and detections.size > 0:
    #     detections = np.array([det[:4] for det in detections])
    trackers_bboxes[:, :4] *= scale
    lost_trackers_bboxes[:, :4] *= scale
    detections[:, :4] *= scale

    if type(in_detections) == np.ndarray:
        detections_info = {"idx": [idx for idx in list(range(detections.shape[0])) if detections[idx, 4] >= det_conf_th],
                        "conf": ["{:.2f}".format(d) for d in detections[:, 4] if d >= det_conf_th]}
    else:
        detections_info = {"idx": [d.det_idx for d in in_detections if d.score >= det_conf_th],
                        "conf": ["{:.2f}".format(d.score) for d in in_detections if d.score >= det_conf_th]}
    trackers_info = {"id": [t.track_id for t in trackers],
                    #  "Ac": [t.is_activated for t in trackers],
                    #  "Occ": [t.is_occluded for t in trackers]
                    # "a": [int(t.area) for t in trackers]
                    # "lno": [track.last_not_occluded_frame for track in trackers]
                     }
    lost_trackers_info = {"id": [t.track_id for t in lost_trackers],
                        # "Ac": [t.is_activated for t in lost_trackers],
                        #  "Occ": [t.is_occluded for t in lost_trackers],
                        # "lno": [track.last_not_occluded_frame for track in lost_trackers],
                        # "lo": [track.last_occluded_frame for track in lost_trackers]
                        # "a": [int(t.tlwh[2] * t.tlwh[3]) for t in lost_trackers]
                        }
    if dcf:
        dcf_info = {"apce": ["{:.1f}".format(t.dcf.psr) for t in trackers],
                    # "m_res": ["{:.2f}".format(t.dcf.max_response) for t in trackers]
                    }
        lost_dcf_info = {"apce": ["{:.1f}".format(t.dcf.psr) for t in lost_trackers],
                        # "m_res": ["{:.2f}".format(t.dcf.max_response) for t in lost_trackers]
                        }
        trackers_info.update(dcf_info)
        lost_trackers_info.update(lost_dcf_info)
    if scale != 1:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    draw_text_line(img, "Tracks", line=0, color=(255, 0, 0))
    draw_text_line(img, "Detections", line=1, color=(0, 0, 255))
    draw_text_line(img, "Frame: " + str(frame_number), line=2, color=(0, 255, 0))
    draw_bboxes(img, lost_trackers_bboxes, xywh_layout=False, color=(0, 255, 255), label_position="under", info_dict=lost_trackers_info)
    draw_bboxes(img, trackers_bboxes, xywh_layout=False, color=(0, 255, 0), label_position="under", info_dict=trackers_info)
    draw_bboxes(img, detections, color=(0, 0, 255), info_dict=detections_info)
    return img


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, np.ndarray):
        boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])
        boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])
        boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])
        boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])
    else:
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2


# DEPRECATED
# def scale_f_coords(img1_shape, coords, img0_shape, ratio_pad=None):
#     coords = copy(coords)
#     # Rescale coords (xyxy) from img1_shape to img0_shape
#     if ratio_pad is None:  # calculate from img0_shape
#         gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
#         pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
#     else:
#         gain = ratio_pad[0][0]
#         pad = ratio_pad[1]

#     coords[:, [0, 2]] -= pad[0]  # x padding
#     coords[:, [1, 3]] -= pad[1]  # y padding
#     coords[:, :4] /= gain
#     clip_coords(coords, img0_shape)
#     return coords

# FOR YOLOX FEATURES
print('USING YOLOX FEATURE BBOX SCALING')
def scale_f_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    coords = copy(coords)
    img_h, img_w = img1_shape[0], img1_shape[1]
    scale = min(img0_shape[0] / float(img_h), img0_shape[1] / float(img_w))
    coords[:4] *= scale
    clip_coords(coords, img0_shape)
    return coords

# FOR YOLOv8n FEATURES
# print('USING YOLOv8n FEATURE BBOX SCALING')
# def scale_f_coords(img0_shape, coords, img1_shape):
#     coords = copy(coords)
#     h0, w0 = img0_shape[:2]
#     h1, w1 = img1_shape[:2]
#     # print(h0, w0, h1, w1)
#     x_gain = w1 / w0
#     y_gain = h1 / h0
#     coords[:, [0, 2]] *= x_gain
#     coords[:, [1, 3]] *= y_gain
#     clip_coords(coords, img1_shape)

#     return coords


def scale_coords(img0_shape, coords, img1_shape):
    coords = copy(coords)
    h0, w0 = img0_shape[:2]
    h1, w1 = img1_shape[:2]
    # print(h0, w0, h1, w1)
    x_gain = w1 / w0
    y_gain = h1 / h0
    coords[:, [0, 2]] *= x_gain
    coords[:, [1, 3]] *= y_gain
    clip_coords(coords, img1_shape)

    return coords


# completely outside image
def is_outside_image(img_shape, tlbr):
    img_h, img_w, _ = img_shape
    if tlbr[0] >= img_w - 1:
        return True
    if tlbr[1] >= img_h - 1:
        return True
    if tlbr[2] <= 0:
        return True
    if tlbr[3] <= 0:
        return True
    return False


def is_touching_img_borders(img_shape, tlbr):
    img_h, img_w, _ = img_shape
    if tlbr[0] <= 0:
        return True
    if tlbr[1] <= 0:
        return True
    if tlbr[2] >= img_w:
        return True
    if tlbr[3] >= img_h:
        return True
    return False
