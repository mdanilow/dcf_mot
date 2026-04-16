import argparse
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
from ultralytics import YOLO

from fasttracker.byte_tracker import BYTETracker


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument("--detector", help="Ultralytics model name", type=str, default='yolov8l.pt')
    parser.add_argument("--config", help="Path to config file", type=str, default='configs/bytetracker_demo.json')
    parser.add_argument("--output_dir", help="Path to the output dir", type=str, default="output/demo")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--debug_vis_scale", help="Scale the image for visualization (applicable if --debug)", type=float, default=1)
    parser.add_argument("--name", help="experiment name in output dir", type=str, default="")
    parser.add_argument("--det_score_division", help="Divide detection scores by this much", type=float, default=1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    camera = cv2.VideoCapture(0)
    ret, test_frame = camera.read()
    img_shape = test_frame.shape
    with open(args.config, "r") as config_file:
        config = json.load(config_file)
        tracker_config = config["tracker_config"]
    tracker = eval(config["meta_tracker"])(
        tracker_config=tracker_config,
        dcf_config=None,
        img_shape=img_shape,
        debug_vis_scale=args.debug_vis_scale,
        det_score_division=args.det_score_division,
        save_vis_frames=False
    )
    detector = YOLO(args.detector)


    while True:
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        
        start_time = time.time()

        # detect
        ret, img = camera.read()
        img_rgb = np.ascontiguousarray(img)
        detections = detector(img_rgb, verbose=False)
        # detections = detector("bus.jpg")
        img_det = detections[0].plot()

        # track
        tracker.update(
            output_results=detections[0].boxes.data,
            debug_img=img_rgb,
            debug=True
        )

        total_time = time.time() - start_time
        fps = 1 / total_time
        print('fps:', fps)

        # print(detections[0].boxes)
        # print(detections[0].boxes.data.cpu().numpy())

        cv2.imshow("demo", tracker.vis_frame)
        cv2.imshow("yolo", img_det)