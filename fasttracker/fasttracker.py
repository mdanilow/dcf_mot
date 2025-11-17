import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
import json
import time

from .kalman_filter import KalmanFilter
import fasttracker.matching as matching
from .basetrack import BaseTrack, TrackState
from utils import draw_frame_info_byte
from box_tracker import DCF
from utils import scale_coords, scale_f_coords, is_outside_image


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    dcf_config = None
    tracker_config = None
    def __init__(self, tlwh, score, det_idx=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.not_matched = 0
        self.is_occluded = False
        self.occluded_len = 0
        self.last_occluded_frame = -1
        self.was_recently_occluded = False
        self.mean_history = []

        self.dcf = None
        self.dcf_updated_at_frame = None
        self.det_idx = det_idx

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks, features=None, debug=None):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                tlbr = mean[:4].copy()
                tlbr[2] *= tlbr[3]
                tlbr[:2] -= tlbr[2:] / 2
                tlbr[2:] += tlbr[:2]
                if is_outside_image(STrack.img_shape, tlbr):
                    stracks[i].mark_removed()
                area = mean[2] * mean[3] * mean[3]
                if area < STrack.tracker_config["min_box_area_to_report"]:
                    stracks[i].mark_removed()
                else:
                    stracks[i].mean = mean
                    stracks[i].covariance = cov

    def dcf_predict(self, features, debug=None):
        if STrack.dcf_config is not None:
            self.dcf.predict_displacement(features, self.tlbr, debug=debug)

    def activate(self, kalman_filter, frame_id, features=None, debug=None):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.mean_history.append(self.mean.copy())
        if len(self.mean_history) > 100:  # limit history length
            self.mean_history.pop(0)


        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

        if STrack.dcf_config is not None:
            self.dcf_updated_at_frame = frame_id
            self.dcf = DCF(dcf_config=STrack.dcf_config,
                           img_shape=STrack.img_shape,
                           features=features,
                           bbox=scale_f_coords(STrack.img_shape, np.expand_dims(self.tlbr, axis=0), features.shape[2:])[0],
                           debug=debug)

    def re_activate(self, new_track, frame_id, new_id=False, features=None, debug=None):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.mean_history.append(self.mean.copy())
        if len(self.mean_history) > 100:  # limit history length
            self.mean_history.pop(0)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        if STrack.dcf_config is not None:
            self.dcf_updated_at_frame = frame_id
            self.dcf.update_filter(
                features=features,
                bbox=scale_f_coords(STrack.img_shape, np.expand_dims(self.tlbr, axis=0), features.shape[2:])[0],
                debug=debug
            )

    def update(self, new_track, frame_id, features=None, debug=None):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        
        self.mean_history.append(self.mean.copy())
        if len(self.mean_history) > 100:  # limit history length
            self.mean_history.pop(0)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        if STrack.dcf_config is not None:
            self.dcf_updated_at_frame = frame_id
            self.dcf.update_filter(
                features=features,
                bbox=scale_f_coords(STrack.img_shape, np.expand_dims(self.tlbr, axis=0), features.shape[2:])[0],
                debug=debug
            )

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


def is_occluded_by(box_a, box_b, iou_thresh=0.7):
    """Returns True if box_a is significantly overlapped by box_b"""
    inter = (
        max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])) *
        max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    )
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    if area_a == 0:
        return False
    iou = inter / area_a
    return iou > iou_thresh

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)

class Fasttracker(object):
    def __init__(self, tracker_config,
                 dcf_config=None,
                 img_shape=None,
                 debug_vis_scale=1,
                 det_score_division=1,
                 frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_count = 0
        # self.args = args

        self.dcf_config = dcf_config
        self.use_dcf = dcf_config is not None
        if self.use_dcf:
            self.lost_psr_th = dcf_config["lost_psr_th"]
        STrack.dcf_config = dcf_config
        STrack.tracker_config = tracker_config
        STrack.img_shape = img_shape

        self.det_conf_thresholds = tracker_config["det_conf_thresholds"]
        self.match_thresholds = tracker_config["match_thresholds"]
        self.new_det_conf_th = tracker_config["new_det_conf_th"]
        self.buffer_size = int(frame_rate / 30.0 * tracker_config["track_buffer"])
        self.max_time_lost = self.buffer_size

        self.reset_velocity_offset_occ = tracker_config["reset_velocity_offset_occ"]
        self.reset_pos_offset_occ = tracker_config["reset_pos_offset_occ"]
        self.enlarge_bbox_occ = tracker_config["enlarge_bbox_occ"]
        self.dampen_motion_occ = tracker_config["dampen_motion_occ"]
        self.active_occ_to_lost_thresh = tracker_config["active_occ_to_lost_thresh"]
        self.init_iou_suppress = tracker_config["init_iou_suppress"]
        self.min_box_area_to_report = tracker_config["min_box_area_to_report"]
        self.not_matched_for_lost_th = tracker_config["not_matched_for_lost_th"]
        self.biou_buffer_sizes = tracker_config["biou_buffer_sizes"]
        self.handle_occlusion = tracker_config["handle_occlusion"]
        self.kalman_filter = KalmanFilter()

        # self.debug_modes = ["dcf_init", "dcf_update_det", "dcf_update_pred", "dcf_predict"]
        # self.debug_modes = ["dcf_update_pred", "dcf_predict"]
        self.debug_modes = []
        self.debug_history_afterupdate = []
        self.debug_history_itstart = []


    def update(self, output_results, features=None, debug_img=None, debug=None):
        self.frame_count += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        still_tracked = [] # for debug

        # output_results = output_results.cpu().numpy()
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        # img_h, img_w = img_info[0], img_info[1]
        # scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        # bboxes /= scale

        remain_inds = scores > self.det_conf_thresholds[1]
        inds_low = scores > self.det_conf_thresholds[0]
        inds_high = scores < self.det_conf_thresholds[1]

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, det_idx=list(range(len(remain_inds)))[i])
                          for i, (tlbr, s) in enumerate(zip(dets, scores_keep))]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        if debug:
            vis_img = draw_frame_info_byte(img=debug_img,
                                            trackers=self.tracked_stracks,
                                            lost_trackers=self.lost_stracks,
                                            detections=output_results,
                                            frame_number=self.frame_count,
                                            dcf=self.use_dcf)
            self.debug_history_itstart.append(vis_img)        

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool, features=features, debug=debug)
        strack_pool = [track for track in strack_pool if track.state != TrackState.Removed]
        dists = matching.iou_distance(strack_pool, detections, biou=self.biou_buffer_sizes[0])
        # if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresholds[0])

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet],
                             self.frame_count,
                             features=features,
                             debug="update trkid{} with det{}".format(track.track_id, track.det_idx) if
                                (debug is not None and "dcf_update_det" in self.debug_modes)
                                else None
                )
                activated_starcks.append(track)
            else:
                track.re_activate(det,
                                  self.frame_count,
                                  new_id=False,
                                  features=features,
                                  debug="re_activate trkid{} with det{}".format(track.track_id, track.det_idx) if
                                        (debug is not None and "dcf_update_det" in self.debug_modes)
                                        else None
                )
                refind_stracks.append(track)
            track.is_occluded = False
            track.not_matched = 0
            track.occluded_len = 0

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        # still_lost = [strack_pool[i] for i in u_track if strack_pool[i].state != TrackState.Tracked]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        if self.use_dcf:
            # try to recover with DCF
            for i in u_track:
                if strack_pool[i].state != TrackState.Tracked:
                    strack_pool[i].dcf_predict(features,
                                               debug="predict, trkid{}".format(track.track_id) if 
                                                    (debug is not None and "dcf_predict" in self.debug_modes)
                                                    else None
                    )
                    if strack_pool[i].dcf.psr >= self.lost_psr_th:
                        r_tracked_stracks.append(strack_pool[i])
        dists = matching.iou_distance(r_tracked_stracks, detections_second, biou=self.biou_buffer_sizes[1])
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=self.match_thresholds[1])
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det,
                             self.frame_count,
                             features=features,
                             debug="update trkid{} with det{}".format(track.track_id, track.det_idx) if
                                (debug is not None and "dcf_update_det" in self.debug_modes)
                                else None
                )
                activated_starcks.append(track)
            else:
                track.re_activate(det,
                                  self.frame_count,
                                  new_id=False,
                                  features=features,
                                  debug="re_activate trkid{} with det{}".format(track.track_id, track.det_idx) if
                                        (debug is not None and "dcf_update_det" in self.debug_modes)
                                        else None
                                 )
                refind_stracks.append(track)

            ## The tracklet is rematched with one DET, so it is not occluded    
            track.is_occluded = False
            track.not_matched = 0
            track.occluded_len = 0

        if self.handle_occlusion:
            self.occlusion_handling(u_track, r_tracked_stracks, activated_starcks, lost_stracks, features)
        else:
            for it in u_track:
                track = r_tracked_stracks[it]
                track.not_matched += 1
                if track.not_matched > self.not_matched_for_lost_th:
                    track.mark_lost()
                    lost_stracks.append(track)
                elif self.use_dcf:
                    track.dcf_predict(features, debug="predict, trkid{}".format(track.track_id) if 
                        (debug is not None and "dcf_predict" in self.debug_modes)
                        else None
                    )
                    if track.dcf.psr < self.lost_psr_th:
                        track.mark_lost()
                        lost_stracks.append(track)
                    else:
                        track.dcf.update_filter(
                            features=features,
                            bbox=scale_f_coords(STrack.img_shape, np.expand_dims(track.tlbr, axis=0), features.shape[2:])[0],
                            debug="update trkid{} with prediction".format(track.track_id) if
                                (debug is not None and "dcf_update_pred" in self.debug_modes)
                                else None
                        )


        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            track = unconfirmed[itracked]
            det = detections[idet]
            track.update(det,
                        self.frame_count,
                        features=features,
                        debug="update trkid{} with det{}".format(track.track_id, track.det_idx) if
                            (debug is not None and "dcf_update_det" in self.debug_modes)
                            else None
            )
            activated_starcks.append(track)
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_lost()
            lost_stracks.append(track)
            

        """ Step 4: Init new stracks (with IoU suppression) """
        # Gather active tracks *now* (already-updated ones + still-tracked ones)
        active_now = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_det_conf_th:
                continue

            # compute max IoU with any active track this frame
            det_box = STrack.tlwh_to_tlbr(track.tlwh)
            max_iou = 0.0
            for at in active_now:
                at_box = at.tlbr  # already tlbr
                max_iou = max(max_iou, _iou(det_box, at_box))
                if max_iou >= self.init_iou_suppress:
                    break

            # Only initialize if it does NOT heavily overlap an active track
            if max_iou < self.init_iou_suppress:
                track.activate(self.kalman_filter,
                               self.frame_count,
                               features=features,
                               debug="init from det{}".format(track.det_idx) if
                                    (debug is not None and "dcf_init" in self.debug_modes)
                                    else None
                )
                activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            recently_occluded = (
                track.was_recently_occluded and
                (self.frame_count - track.last_occluded_frame <= 40)  # configurable if needed
            )

            if not recently_occluded and (self.frame_count - track.end_frame > self.max_time_lost):
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        if debug:
            vis_img = draw_frame_info_byte(img=debug_img,
                                           trackers=self.tracked_stracks,
                                           lost_trackers=self.lost_stracks,
                                            # trackers=[t for t in self.tracked_stracks if t.track_id == 74],
                                            # lost_trackers=[t for t in self.lost_stracks if t.track_id == 74],
                                            detections=output_results,
                                            frame_number=self.frame_count,
                                            dcf=(self.dcf_config is not None),
                                            det_conf_th=self.det_conf_thresholds[0])
            self.debug_history_afterupdate.append(vis_img)
        

        # get scores of lost tracks
        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output = [np.array([t.tlwh[0], t.tlwh[1], t.tlwh[0] + t.tlwh[2], t.tlwh[1] + t.tlwh[3], t.track_id]) for t in self.tracked_stracks if t.is_activated]
        output = []
        for t in self.tracked_stracks:
            if t.is_activated:
                horizontal = t.tlwh[2] / t.tlwh[3] > 1.6
                if t.tlwh[2] * t.tlwh[3] > self.min_box_area_to_report and not horizontal:
                    output.append(np.array([t.tlwh[0], t.tlwh[1], t.tlwh[0] + t.tlwh[2], t.tlwh[1] + t.tlwh[3], t.track_id]))

        return output
    

    # u_track - unassigned tracks id
    # r_tracked_stracks - strack pool from 2nd assignment
    # activated_stracks - list of stracks activated in this frame
    # lost_stracks - list of stracks considered lost in this frame
    # features - optional conv features for dcf
    def occlusion_handling(self, u_track, r_tracked_stracks, activated_starcks, lost_stracks, features):
        ## occlusion handling version
        for it in u_track:
            track = r_tracked_stracks[it]
            track.not_matched += 1

            # Try detecting occlusion
            if not track.is_occluded:
                for other in activated_starcks:
                    if track.track_id == other.track_id:
                        continue
                    if not other.is_activated or other.is_occluded:
                        continue
                    if is_occluded_by(track.tlbr, other.tlbr):
                        # if debug is not None:
                        #     print("OCCLUSION, frame {}, trkid {} occluded by trkid {}".format(self.frame_count, track.track_id, other.track_id))
                        track.is_occluded = True
                        track.occluded_len += 1
                        track.last_occluded_frame = self.frame_count
                        track.was_recently_occluded = True

                        # Reset velocity
                        if len(track.mean_history) >= self.reset_velocity_offset_occ:
                            old_mean = track.mean_history[-self.reset_velocity_offset_occ]
                            track.mean[4:8] = old_mean[4:8]

                        # Reset position
                        if len(track.mean_history) >= self.reset_pos_offset_occ:
                            old_mean = track.mean_history[-self.reset_pos_offset_occ]
                            track.mean[0:4] = old_mean[0:4]

                        # Enlarge once
                        if track.occluded_len == 1:
                            track.mean[3] *= self.enlarge_bbox_occ  # increase height
                            # track.mean[2] = track.mean[2] / track.mean[3]  # adjust aspect ratio

                        # Dampen motion
                        track.mean[4:8] *= self.dampen_motion_occ
                        break
            else:
                track.occluded_len += 1

            if track.was_recently_occluded and (self.frame_count - track.last_occluded_frame > 40):
                track.was_recently_occluded = False

            if track.is_occluded:
                if track.occluded_len > self.active_occ_to_lost_thresh:
                    track.mark_lost()
                    lost_stracks.append(track)
            else:
                if track.not_matched > self.not_matched_for_lost_th:
                    track.mark_lost()
                    lost_stracks.append(track)
                elif self.use_dcf:
                    track.dcf_predict(features, debug="predict, trkid{}".format(track.track_id) if 
                                                                                                (debug is not None and "dcf_predict" in self.debug_modes)
                                                                                                else None
                    )
                    if track.dcf.psr < self.lost_psr_th:
                        track.mark_lost()
                        lost_stracks.append(track)
                    else:
                        track.dcf.update_filter(
                            features=features,
                            bbox=scale_f_coords(STrack.img_shape, np.expand_dims(track.tlbr, axis=0), features.shape[2:])[0],
                            debug="update trkid{} with prediction".format(track.track_id) if
                                                                                          (debug is not None and "dcf_update_pred" in self.debug_modes)
                                                                                          else None
                        )


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
