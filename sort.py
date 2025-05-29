"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import sys
from os.path import join
from multiprocessing.pool import ThreadPool

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import glob
import time
import argparse
import cv2
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import json

from utils import draw_bboxes, scale_coords, draw_frame_info
from box_tracker import KalmanBoxTracker

np.random.seed(0)


def linear_assignment(cost_matrix, rows_indices=None, cols_indices=None):
#   try:
#     import lap
#     _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
#     return np.array([[y[i],i] for i in x if i >= 0]) #
#   except ImportError:
    # print('rowcol indices:', rows_indices, cols_indices)
    # print('matrix:')
    # print(cost_matrix)
    if rows_indices is not None and cols_indices is not None:
        cost_matrix = cost_matrix[np.ix_(rows_indices, cols_indices)]
    # print('submatrix:')
    # print(cost_matrix)

    x, y = linear_sum_assignment(cost_matrix)

    if rows_indices is not None and cols_indices is not None:
        x = [rows_indices[i] for i in x]
        y = [cols_indices[i] for i in y]

    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  



class Sort(object):
    def __init__(self, tracker_config, dcf_config=None, img_shape=None):
        """
        Sets key parameters for SORT
        """
        association_fn_lookup = {
            'iou': self.associate_iou,
            'cascaded': self.associate_cascaded
        }
        self.max_age = tracker_config['max_age']
        self.min_hits = tracker_config['min_hits']
        self.hits_to_be_confirmed = tracker_config['hits_to_be_confirmed']
        self.iou_threshold = tracker_config['iou_threshold']
        self.max_iou_for_new_target = tracker_config['max_iou_for_new_target']
        self.association_strategy = association_fn_lookup[tracker_config['strategy']]
        self.final_iou_assignment = tracker_config['final_iou_assignment']
        self.mask_cost_matrix_with_iou = tracker_config['mask_cost_matrix_with_iou']

        self.dcf_config = dcf_config
        self.trackers = []
        self.frame_count = 0
        self.img_shape = img_shape
        self.max_dcf_response = 0

    def update(self, dets=np.empty((0, 5)), features=None, debug_img=None, debug=None):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        # print('update, trackers:', len(self.trackers), 'detections:', dets.shape[0])
        # print(dets)
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        if self.dcf_config is not None:
            scaled_dets = scale_coords(self.img_shape, dets, features.shape[2:])
        else:
            scaled_dets = None

        matched, unmatched_dets, unmatched_trks = self.association_strategy(dets,
                                                                    self.trackers,
                                                                    features=features,
                                                                    scaled_dets=scaled_dets,
                                                                    debug_img=debug_img,
                                                                    debug=debug)

        # update matched trackers with assigned detections
        for m in matched:
            bbox = dets[m[0], :]
            scaled_bbox = None if scaled_dets is None else scaled_dets[m[0], :]
            self.trackers[m[1]].update(bbox, features=features, features_bbox=scaled_bbox)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :],
                                   self.hits_to_be_confirmed,
                                   self.dcf_config,
                                   features,
                                   None if scaled_dets is None else scaled_dets[i, :],
                                   debug=debug,
                                   index=len(self.trackers))
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))
    

    def compute_dcf_cost_matrix(self, scaled_dets, trackers, features, iou_matrix, debug):
        # response_matrix = np.array((len(trackers), detections.shape[0], self.dcf_config['roi_size'], self.dcf_config['roi_size']))
        response_matrix = np.zeros((scaled_dets.shape[0], len(trackers)))
        # response_matrix = np.zeros((len(trackers), len(trackers)))
        pairs_to_compute = np.where(iou_matrix > 0)

        # start = time.time()
        for tracker_idx in pairs_to_compute[1]:
            for detection_idx in pairs_to_compute[0]:
                dcf_response = trackers[tracker_idx].dcf.compute_response(features, scaled_dets[detection_idx], debug=debug, debug_idx=detection_idx)
                # print('resp:', np.max(dcf_response), 'selfcorr:', tracker.dcf.selfcorr)
                max_response = np.max(dcf_response)
                # if max_response > tracker.dcf.selfcorr:
                #     print('Frame {}: track {} correlation with detection {} bigger than self correlation:'.format(self.frame_count, tracker_idx, detection_idx), max_response, tracker.dcf.selfcorr)
                response_matrix[detection_idx, tracker_idx] = max_response

        local_max_response = np.max(response_matrix)
        if local_max_response > self.max_dcf_response:
            self.max_dcf_response = local_max_response
        # for i, result in enumerate(ThreadPool(8).imap(tracker_dcf_task, trackers)):
        #     response_matrix[:, i] = result
        # print('time:', time.time() - start)
        cost_matrix = -response_matrix

        return cost_matrix
    

    def associate_cascaded(self, detections, trackers, features=None, scaled_dets=None, debug_img=None, debug=None):
        """
        Assigns detections to tracked object (both represented as bounding boxes)

        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if debug:
            to_show = draw_frame_info(debug_img, trackers, detections, self.frame_count)
            print('debug:', to_show.shape)
            cv2.imshow('debug', to_show)
        if(len(trackers)==0):
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
        trackers_bboxes = np.stack([np.squeeze(t.get_state()) for t in trackers])
        iou_matrix = iou_batch(detections, trackers_bboxes)
        detection_indices_to_match = list(range(len(detections)))

        matches = np.empty(shape=(0,2), dtype=int)
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix >= self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matches = np.stack(np.where(a), axis=1)
            else:
                # cost matrix
                if self.dcf_config is not None:
                    cost_matrix = self.compute_dcf_cost_matrix(scaled_dets, trackers, features, iou_matrix, debug=debug)
                    if self.mask_cost_matrix_with_iou:
                        cost_matrix = np.where(iou_matrix <= self.iou_threshold, 1e+5, cost_matrix)
                else:
                    cost_matrix = -iou_matrix
                
                # if debug is not None:
                #     cv2.waitKey(0)
                # matching cascade
                for age in range(self.max_age):
                    if len(detection_indices_to_match) == 0:
                        break
                    tracker_indices_to_match = [i for i in range(len(trackers))
                                                if trackers[i].time_since_update == (age + 1)]
                    matched_indices = linear_assignment(cost_matrix,
                                                        rows_indices=detection_indices_to_match,
                                                        cols_indices=tracker_indices_to_match)
                    # filter matches by iou
                    matched_indices = np.array([m for m in matched_indices
                                                if iou_matrix[m[0], m[1]] >= self.iou_threshold])
                    if matched_indices.shape[0] > 0:
                        detection_indices_to_match = [i for i in detection_indices_to_match
                                                      if i not in matched_indices[:, 0]]
                        matches = np.concatenate([matches, matched_indices], axis=0)

                # final iou assignment
                if self.final_iou_assignment and self.dcf_config is not None:
                    tracker_indices_to_match = [i for i in range(len(trackers))
                                                if i not in matches[:, 1]]
                    matched_indices = linear_assignment(-iou_matrix,
                                                        rows_indices=detection_indices_to_match,
                                                        cols_indices=tracker_indices_to_match)
                    # filter matches by iou
                    matched_indices = np.array([m for m in matched_indices
                                                if iou_matrix[m[0], m[1]] >= self.iou_threshold])
                    if matched_indices.shape[0] > 0:
                        matches = np.concatenate([matches, matched_indices], axis=0)
                    
        # every unmatched detection will be considered a new track
        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in matches[:,0]):
                closest_track_iou = np.max(iou_matrix[d])
                if closest_track_iou < self.max_iou_for_new_target:
                    unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if(t not in matches[:,1]):
                unmatched_trackers.append(t)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


    def associate_iou(self, detections, trackers, features=None, scaled_dets=None, debug_img=None, debug=None):
        """
        Assigns detections to tracked object (both represented as bounding boxes)

        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if debug:
            draw_frame_info(debug_img, trackers, detections)
            cv2.imshow('debug', debug_img)
        if(len(trackers)==0):
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
        trackers_bboxes = np.stack([np.squeeze(t.get_state()) for t in trackers])
        # scaled_trackers_bboxes = scale_coords(debug_img.shape, trackers_bboxes, features.shape[2:])
        # trackers_bboxes = trackers
        iou_matrix = iou_batch(detections, trackers_bboxes)
        # print('iou matrix:', iou_matrix.shape)
        # print(iou_matrix)

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                # DCF matrix
                if self.dcf_config is not None:
                    cost_matrix = self.compute_dcf_cost_matrix(scaled_dets, trackers, features, iou_matrix, debug=debug)
                    cost_matrix = np.where(iou_matrix <= self.iou_threshold, 1e+5, cost_matrix)
                else:
                    cost_matrix = -iou_matrix
                # print('cost_matrix:', cost_matrix)
                matched_indices = linear_assignment(cost_matrix)

        else:
            matched_indices = np.empty(shape=(0,2))

        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in matched_indices[:,0]):
                closest_track_iou = np.max(iou_matrix[d])
                if closest_track_iou < self.max_iou_for_new_target:
                    unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if(t not in matched_indices[:,1]):
                unmatched_trackers.append(t)

        #filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if(iou_matrix[m[0], m[1]] < self.iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='neweval')
    parser.add_argument("--config", help="Path to config file", type=str, default='configs/base_dcf.json')
    parser.add_argument("--output_dir", help="Path to the output dir", type=str, default="output")
    parser.add_argument("--debug_images", help="Path to directory with sequences in mot format for visualization", type=str, default="")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--name", help="experiment name in output dir", type=str, default="")
    parser.add_argument("--single_sequence", help="Run only for this sequence", type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # all train
    args = parse_args()
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    name = phase + "_" + args.name if args.name != "" else phase
    output_dir = join(args.output_dir, name, 'data')
    key = None
    with open(args.config, "r") as config_file:
        config = json.load(config_file)
    tracker_config = config["tracker_config"]
    dcf_config = config["dcf_config"]
    use_dcf = dcf_config['use_conv_features'] != -1

    if args.name != "" and args.name != "test" and os.path.exists(output_dir):
        print('WARNING: output directory {} already exists, exiting...' )
        sys.exit()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(join(args.output_dir, name, 'args.json'), 'w') as args_file:
        json.dump({'args': vars(args), 'tracker_config': tracker_config, 'dcf_config': dcf_config}, args_file, indent=4)
    if args.single_sequence is not None and args.single_sequence != "None":
        seqnames = [args.single_sequence]
    else:
        seqnames = os.listdir(join(args.seq_path, phase))

    for seq in seqnames:
        if args.debug_images != "":
            img_shape = cv2.imread(join(args.debug_images, seq, 'img1', '%06d.jpg'%(1))).shape
        else:
            img_shape = None
        mot_tracker = Sort(tracker_config=tracker_config,
                            dcf_config=dcf_config if use_dcf else None,
                            img_shape=img_shape) #create instance of the SORT tracker
        seq_dets = np.loadtxt(join(args.seq_path, phase, seq, 'det', 'det.txt'), delimiter=',')
        seq_features_dir = join(args.seq_path, phase, seq, 'features')

        with open(os.path.join(output_dir, '%s.txt'%(seq)),'w') as out_file:
            print("Processing %s."%(seq), args.debug)
            pbar = tqdm(range(int(seq_dets[:,0].max())))
            for frame in pbar:
                if use_dcf:
                    frame_features_path = join(seq_features_dir, 'frame{}_f{}.npy'.format(frame, dcf_config["use_conv_features"]))
                    frame_features = np.load(frame_features_path)
                else:
                    frame_features = None

                frame += 1 #detection and frame numbers begin at 1
                # if seq == "KITTI-13" and frame == 50:
                #     args.debug = True
                dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
                dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                if args.debug and args.debug_images != "":
                    debug_img = cv2.imread(join(args.debug_images, seq, 'img1', '%06d.jpg'%(frame)))
                else:
                    debug_img = None

                start_time = time.time()
                trackers = mot_tracker.update(dets, features=frame_features, debug_img=debug_img, debug=args.debug)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)

                if args.debug and args.debug_images != "":
                    key = cv2.waitKey(0)
                    if key == ord('s') or key == ord('q'):
                        break

            print('Max dcf response in sequence:', mot_tracker.max_dcf_response)
        if key == ord('q'):
            break
        

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

