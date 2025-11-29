import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

from .kalman_filter import KalmanFilter
from . import matching
from .basetrack import BaseTrack, TrackState
from box_tracker import DCF
from utils import clip_coords, scale_coords, scale_f_coords, is_outside_image, is_touching_img_borders, draw_frame_info_byte


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, det_idx):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.mean_nowhpred = None
        self.mean_history = []
        self.is_activated = False

        self.not_matched = 0
        self.is_occluded = False
        self.left_occlusion = False
        self.last_not_occluded_state = None
        self.last_not_occluded_frame = -1
        self.last_occluded_frame = -1
        self.score = score
        self.tracklet_len = 0

        self.dcf = None
        self.det_idx = det_idx

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                # stracks[i].mean_nowhpred = stracks[i].mean.copy()
                # stracks[i].mean_nowhpred[:2] = mean[:2]
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def dcf_predict(self, features, debug=None):
        if STrack.dcf_config is not None:
            self.dcf.predict_displacement(features, self.tlbr, update_psr=True, debug=debug)


    def update_history(self, new_mean):
        self.mean_history.append(new_mean.copy())
        if len(self.mean_history) > 100:
            self.mean_history.pop(0)


    def activate(self, kalman_filter, frame_id, features=None, debug=None):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.update_history(self.mean)
        self.last_updated_state = self.mean.copy()

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
        self.update_history(self.mean)
        self.last_updated_state = self.mean.copy()
        self.not_matched = 0
        self.is_occluded = False
        self.occluded_len = 0
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
        self.not_matched = 0
        self.is_occluded = False
        self.occluded_len = 0
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.update_history(self.mean)
        self.last_updated_state = self.mean.copy()
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
    def tlwh_nowhpred(self):
        if self.mean_nowhpred is None:
            return self._tlwh.copy()
        ret = self.mean_nowhpred[:4].copy()
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
    
    @property
    def tlbr_nowhpred(self):
        ret = self.tlwh_nowhpred.copy()
        ret[2:] += ret[:2]
        return ret
    
    @property
    def area(self):
        return self.tlwh[2] * self.tlwh[3]
    
    @property
    def clipped_area(self):
        tlbr = self.tlbr.copy()
        clip_coords(np.expand_dims(tlbr, axis=0), STrack.img_shape)
        tlwh = STrack.tlbr_to_tlwh(tlbr)
        area = tlwh[2] * tlwh[3]
        return area
    
    @property
    def clipped_tlwh(self):
        tlbr = self.tlbr.copy()
        clip_coords(np.expand_dims(tlbr, axis=0), STrack.img_shape)
        tlwh = STrack.tlbr_to_tlwh(tlbr)
        return tlwh

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


class BYTETracker(object):
    def __init__(self,
                 tracker_config,
                 dcf_config=None,
                 img_shape=None,
                 debug_vis_scale=1,
                 det_score_division=1,
                 frame_rate=30):
        print('Tracker config:')
        print(tracker_config)
        if dcf_config is not None:
            print('DCF config:')
            print(dcf_config)

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.dcf_config = dcf_config
        self.use_dcf = dcf_config is not None
        if self.use_dcf:
            self.lost_psr_th = dcf_config["lost_psr_th"]
            self.use_dcf_gating = dcf_config["use_dcf_gating"]
            self.dcf_gating_th = dcf_config["dcf_gating_th"]
            self.dcf_gating_cost_th = dcf_config["dcf_gating_cost_th"]
            self.dcf_gating_candidate_cost_th = dcf_config["dcf_gating_candidate_cost_th"]
            self.use_dcf_reid = dcf_config["use_dcf_reid"]
            self.dcf_reid_th = dcf_config["dcf_reid_th"]
            self.dcf_min_h = img_shape[0] * 64 / 1080
            self.dcf_min_w = img_shape[1] * 24 / 1920
        STrack.dcf_config = dcf_config
        STrack.tracker_config = tracker_config
        STrack.img_shape = img_shape
        self.img_shape = img_shape

        # general tracking options
        self.frame_count = 0
        # self.args = args
        #self.det_thresh = args.track_thresh
        self.det_conf_thresholds = tracker_config["det_conf_thresholds"]
        self.det_thresh = tracker_config["new_det_conf_th"]
        self.match_thresholds = tracker_config["match_thresholds"]
        self.min_box_area = tracker_config["min_box_area"]
        self.init_iou_suppress = tracker_config["init_iou_suppress"]
        self.not_matched_for_lost_th = tracker_config["not_matched_for_lost_th"]
        self.biou_buffer_sizes = tracker_config["biou_buffer_sizes"]
        # occlusion options
        self.handle_occlusion = tracker_config["handle_occlusion"]
        if self.handle_occlusion:
            self.occ_velocity_rewind = tracker_config["occ_velocity_rewind"]
            self.occ_position_rewind = tracker_config["occ_position_rewind"]
            self.occ_overlap_thresh = tracker_config["occ_overlap_thresh"]
            self.occ_time_left_after_occlusion = tracker_config["occ_time_left_after_occlusion"]

        self.buffer_size = int(frame_rate / 30.0 * tracker_config["track_buffer"])
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        self.debug_vis_scale = debug_vis_scale
        self.debug_history_itstart = []
        self.debug_history_locpred = []
        self.debug_history_afterupdate = []
        self.debug_modes = []
        # self.debug_modes = ["dcf_update_det"]
        # self.debug_modes = ["dcf_gating"]

        self.dcf_histogram_data = []
        self.areas_to_psr = {}
        self.saved_idx = 0


    def update(self, output_results, features=None, debug_img=None, debug=None):
        self.frame_count += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
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
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, det_idx=[idx for idx, x in enumerate(remain_inds) if x][i])
                          for i, (tlbr, s) in enumerate(zip(dets, scores_keep))]
        else:
            detections = []
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, det_idx=[idx for idx, x in enumerate(inds_second) if x][i]) 
                                 for i, (tlbr, s) in enumerate(zip(dets_second, scores_second))]
        else:
            detections_second = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        if debug:
            print('frame:', self.frame_count)

            vis_img = draw_frame_info_byte(img=debug_img,
                                            trackers=self.tracked_stracks,
                                            lost_trackers=self.lost_stracks,
                                            in_detections=output_results,
                                            frame_number=self.frame_count,
                                            dcf=self.use_dcf)
            # from utils import draw_bboxes
            # draw_bboxes(vis_img, np.array([[0, 0, self.dcf_min_w, self.dcf_min_h]]))
            self.debug_history_itstart.append(vis_img)  

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # # EXPERIMENTS
        # if self.use_dcf:
        #     for track in strack_pool:
        #         if track.state == TrackState.Removed:
        #             print('dupa')
        #         area = track.clipped_area
        #         tlwh = track.clipped_tlwh
        #         if tlwh[2] > self.dcf_min_w and tlwh[3] > self.dcf_min_h:
        #             # self.histogram_areas.append(track.area)
        #             # print(track.tlbr)
        #             track.dcf_predict(features,
        #                             debug="predict, trkid{}".format(track.track_id) if 
        #                                 (debug is not None and "dcf_predict" in self.debug_modes)
        #                                 else None
        #             )
        #             if area in self.areas_to_psr:
        #                 self.areas_to_psr[area].append(track.dcf.psr)
        #             else:
        #                 self.areas_to_psr[area] = [track.dcf.psr]
        #             self.dcf_histogram_data.append(track.dcf.psr)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections, biou=self.biou_buffer_sizes[0])
        # if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresholds[0])

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det,
                             self.frame_count,
                             features=features,
                             debug="update trkid{} with det{}".format(track.track_id, det.det_idx) if
                                (debug is not None and "dcf_update_det" in self.debug_modes)
                                else None
                )
                activated_starcks.append(track)
            else:  
                track.re_activate(det,
                                  self.frame_count,
                                  new_id=False,
                                  features=features,
                                  debug="re_activate trkid{} with det{}".format(track.track_id, det.det_idx) if
                                        (debug is not None and "dcf_update_det" in self.debug_modes)
                                        else None
                )
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        if self.use_dcf and self.use_dcf_gating:
            dists = self.dcf_gated_iou_distance(r_tracked_stracks,
                                                detections_second,
                                                features=features,
                                                biou=self.biou_buffer_sizes[1],
                                                debug=debug if "dcf_gating" in self.debug_modes else None)
        else:
            dists = matching.iou_distance(r_tracked_stracks, detections_second, biou=self.biou_buffer_sizes[1])
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det,
                             self.frame_count,
                             features=features,
                             debug="update trkid{} with det{}".format(track.track_id, det.det_idx) if
                                (debug is not None and "dcf_update_det" in self.debug_modes)
                                else None
                )
                activated_starcks.append(track)
            else:
                track.re_activate(det,
                                  self.frame_count,
                                  new_id=False,
                                  features=features,
                                  debug="re_activate trkid{} with det{}".format(track.track_id, det.det_idx) if
                                        (debug is not None and "dcf_update_det" in self.debug_modes)
                                        else None
                )
                refind_stracks.append(track)

        '''Decide state of unassigned (but tracked, not lost) trackers'''
        for it in u_track:
            track = r_tracked_stracks[it]
            track.not_matched += 1
            clipped_tlwh = track.clipped_tlwh
            if not (track.state == TrackState.Lost) and (track.not_matched > self.not_matched_for_lost_th):
                track.mark_lost()
                lost_stracks.append(track)
            elif self.use_dcf and clipped_tlwh[2] > self.dcf_min_w and clipped_tlwh[3] > self.dcf_min_h:
                track.dcf_predict(features, debug="predict, trkid{}".format(track.track_id) if 
                    (debug is not None and "dcf_predict" in self.debug_modes)
                    else None
                )
                if track.dcf.psr < self.lost_psr_th:
                    track.mark_lost()
                    lost_stracks.append(track)
                else:
                    if debug is not None:
                        print('TRKID{} still tracked with dcf'.format(track.track_id))
                    track.update_history(track.mean)
                    track.dcf.update_filter(
                        features=features,
                        bbox=scale_f_coords(STrack.img_shape, np.expand_dims(track.tlbr, axis=0), features.shape[2:])[0],
                        debug="update trkid{} with prediction".format(track.track_id) if
                            (debug is not None and "dcf_update_pred" in self.debug_modes)
                            else None
                    )
                    # refind_stracks.append(track)
        if self.handle_occlusion:
            trackers_for_occlusion = [t for t in self.lost_stracks if t.state == TrackState.Lost]
            occluders = activated_starcks + refind_stracks
            # occluders = activated_starcks
            # if len(trackers_for_occlusion) > 0:
                # print('trackers_for_occlusion', [t.track_id for t in trackers_for_occlusion])
            self.occlusion_handling(trackers_for_occlusion, occluders)
            '''Try to reidentify recently occluded lost object with low score detections'''
            recently_occluded_lost_objects = [t for t in self.lost_stracks
                                              if (t.left_occlusion and self.frame_count - t.last_occluded_frame < 10)]
            # for t in recently_occluded_lost_objects:
            detections_for_occlusion_reid = [detections_second[i] for i in u_detection_second]
            dists = matching.iou_distance(recently_occluded_lost_objects, detections_for_occlusion_reid)
            # matching.print_cost_matrix(recently_occluded_lost_objects, detections_for_occlusion_reid, dists, masking_mode="1 or more")
            occ_matches, _, _ = matching.linear_assignment(dists, thresh=0.6)
            for itrack, idet in occ_matches:
                track = recently_occluded_lost_objects[itrack]
                det = detections_for_occlusion_reid[idet]
                # print("recently occluded track {} found with det {}".format(track.track_id, det.det_idx))
                track.re_activate(det,
                                  self.frame_count,
                                  new_id=False,
                                  features=features,
                                  debug="re_activate trkid{} with det{}".format(track.track_id, det.det_idx) if
                                        (debug is not None and "dcf_update_det" in self.debug_modes)
                                        else None
                )
                refind_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet],
                                        self.frame_count,
                                        features=features,
                                        debug="update trkid{} with det{}".format(track.track_id, det.det_idx) if
                                            (debug is not None and "dcf_update_det" in self.debug_modes)
                                            else None
            )
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        active_now = [t for t in (self.tracked_stracks + self.lost_stracks) if t.state == TrackState.Tracked]
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue

            det_box = track.tlbr
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
            if is_outside_image(self.img_shape, track.tlbr):
                track.mark_removed()
                removed_stracks.append(track)
                continue
            if self.frame_count - track.end_frame > self.max_time_lost + track.occluded_len:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks

        if debug:
            vis_img = draw_frame_info_byte(img=debug_img,
                                        #    trackers=self.tracked_stracks,
                                        #    lost_trackers=self.lost_stracks,
                                            # lost_trackers=[],
                                            trackers=[t for t in self.tracked_stracks if t.track_id in [2, 12, 18]],
                                            lost_trackers=[t for t in self.lost_stracks if t.track_id in [2, 12, 18]],
                                            # in_detections=output_results,
                                            in_detections=[det for det in (detections + detections_second) if det.det_idx == 14] if self.frame_count == 249 else [],
                                            # in_detections=[],
                                            frame_number=self.frame_count,
                                            scale=self.debug_vis_scale,
                                            dcf=(self.dcf_config is not None),
                                            det_conf_th=self.det_conf_thresholds[0])
            vis_Frames = [154, 179, 191, 199, 209, 219, 226, 239, 249]
            if self.frame_count in vis_Frames:
                cv2.imwrite("figures/occ{}.png".format(vis_Frames[self.saved_idx]), vis_img[435:678, 1024:1232, :])
                print("figures/occ{}.png".format(vis_Frames[self.saved_idx]), "saved")
                self.saved_idx += 1

            # if self.frame_count >= 116 and self.frame_count <= 128:
            self.debug_history_afterupdate.append(vis_img)

        output = []
        for t in self.tracked_stracks:
            if t.is_activated:
                horizontal = t.tlwh[2] / t.tlwh[3] > 1.6
                if t.tlwh[2] * t.tlwh[3] > self.min_box_area and not horizontal:
                    output.append(np.array([t.tlwh[0], t.tlwh[1], t.tlwh[0] + t.tlwh[2], t.tlwh[1] + t.tlwh[3], t.track_id]))

        return output


    def occlusion_handling(self, tracks, occluders):
        for track in occluders:
            occluded_by_any = False
            for other in occluders:
                if track.track_id == other.track_id:
                    continue
                if is_occluded_by(track.tlbr, other.tlbr, iou_thresh=0.1):
                    occluded_by_any = True
                    break
            if not occluded_by_any:
                track.last_not_occluded_state = track.mean.copy()
                track.last_not_occluded_frame = self.frame_count

        for track in tracks:
            if track.is_occluded:
                still_occluded = False
                for occluder in occluders:
                    if is_occluded_by(track.tlbr, occluder.tlbr, self.occ_overlap_thresh):
                        still_occluded = True
                        track.occluded_len += 1
                        track.last_occluded_frame = self.frame_count
                        break
                track.is_occluded = still_occluded
                if not still_occluded:
                    track.left_occlusion = True
                    track.occluded_len = self.occ_time_left_after_occlusion + self.frame_count - track.end_frame - self.max_time_lost
                    # print('trkid{} time left:'.format(track.track_id), self.frame_count - track.end_frame - self.max_time_lost - track.occluded_len)

                # if is_occluded_by_many(track, occluders, self.occ_overlap_thresh):
                #     track.is_occluded = True
                #     track.occluded_len += 1
                # else:
                #     track.is_occluded = False

            else:
                for occluder in occluders:
                    if is_occluded_by(track.tlbr, occluder.tlbr, self.occ_overlap_thresh):
                        # print('track {} was last not occluded {} frames ago on frame {}'.format(track.track_id, self.frame_count - track.last_not_occluded_frame, track.last_not_occluded_frame))
                        # reset and reduce velocity
                        # track.mean[4:8] = track.mean_history[-min(self.occ_velocity_rewind, len(track.mean_history))][4:8]
                        if track.last_not_occluded_state is not None:
                            track.mean[4:8] = track.last_not_occluded_state[4:8]
                            # track.mean = track.last_not_occluded_state
                        # rewind position
                        # track.mean[:4] = track.mean_history[-min(self.occ_position_rewind, len(track.mean_history))][:4]
                        # track.mean[:4] = track.last_updated_state[:4]

                        # Enlarge once
                        # if track.occluded_len == 0:
                        #     track.mean[3] *= self.occ_enlarge_bbox  # increase height
                        track.is_occluded = True
                        track.occluded_len = 1
                        track.last_occluded_frame = self.frame_count
                        track.left_occlusion = False
                        break
                # if is_occluded_by_many(track, occluders, self.occ_overlap_thresh, self.occ_many_min_iou):
                #     track.is_occluded = True
                #     track.occluded_len = 1
                #     # print('occluded, history len:', len(track.mean_history))
                #     # reset and reduce velocity
                #     track.mean[4:8] = track.mean_history[-min(self.occ_velocity_rewind, len(track.mean_history))][4:8]
                #     # rewind position
                #     track.mean[:4] = track.mean_history[-min(self.occ_position_rewind, len(track.mean_history))][:4]

    def dcf_gated_iou_distance(self, tracks, dets, features=None, biou=0, debug=None):
        cost_matrix = matching.iou_distance(tracks, dets, biou=biou)
        # get candidate pairs for dcf distance testing
        # biou_matrix = matching.iou_distance(tracks, dets, biou=0.3)
        candidates = np.array(np.where(cost_matrix < self.dcf_gating_candidate_cost_th)).transpose()
        # matching.print_cost_matrix(tracks, dets, biou_matrix, masking_mode="less than 1")
        response_matrix = np.zeros(cost_matrix.shape)
        # note: track_i != track_id and det_i != det_idx, it is local indexing only
        for track_i, det_i in candidates:
            clipped_tlwh = tracks[track_i].clipped_tlwh
            if clipped_tlwh[2] > self.dcf_min_w and clipped_tlwh[3] > self.dcf_min_h:
                _, psr = tracks[track_i].dcf.predict_displacement(features,
                                                        dets[det_i].tlbr,
                                                        update_psr=False,
                                                        debug="dcf response trkid{} with det{}".format(tracks[track_i].track_id, dets[det_i].det_idx)
                                                            if debug is not None else None)
                response_matrix[track_i, det_i] = psr
        gating_matrix = np.logical_or(cost_matrix < self.dcf_gating_cost_th, response_matrix >= self.dcf_gating_th)
        if debug is not None:
            matching.print_cost_matrix(tracks, dets, cost_matrix, masking_mode="1 or more")
            matching.print_cost_matrix(tracks, dets, response_matrix, masking_mode="zeros")
            matching.print_cost_matrix(tracks, dets, gating_matrix)
        cost_matrix = np.where(gating_matrix, cost_matrix, 9999)
        # cost_matrix = np.where(response_matrix >= 50, cost_matrix/response_matrix, cost_matrix)
        # matching.print_cost_matrix(tracks, dets, cost_matrix, masking_mode="1 or more")

        return cost_matrix
    

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


def is_occluded_by_many(track, occluders, overlap_thresh=0.7, min_iou=0.1):
    box_area = track.area
    box_tlwh = track.tlwh
    box_tlbr = track.tlbr
    occlusion_mask = np.zeros((int(box_tlwh[3]), int(box_tlwh[2])), dtype=np.uint8)
    occlusion_mask_area = np.prod(occlusion_mask.shape)
    current_overlap = 0
    # print('trkid', track.track_id, 'tlwh:', box_tlwh)
    for occluder in occluders:
        # print('occid', occluder.track_id)
        occ_tlbr = occluder.tlbr
        occ_tlwh = occluder.tlwh
        x_inter = max(0, min(box_tlbr[2], occ_tlbr[2]) - max(box_tlbr[0], occ_tlbr[0]))
        if x_inter == 0:
            continue
        y_inter = max(0, min(box_tlbr[3], occ_tlbr[3]) - max(box_tlbr[1], occ_tlbr[1]))
        if x_inter == 0:
            continue
        # print('inters:', x_inter, y_inter)
        intersection = x_inter * y_inter
        union = box_tlwh[2] * box_tlwh[3] + occ_tlwh[2] * occ_tlwh[3] - intersection
        if intersection / union < min_iou:
            # print('low iou!')
            continue
        
        x_inter = int(x_inter)
        y_inter = int(y_inter)
        if box_tlbr[0] > occ_tlbr[0] and box_tlbr[2] < occ_tlbr[2]: # fully ocluded in x axis
            if box_tlbr[1] > occ_tlbr[1] and box_tlbr[3] < occ_tlbr[3]: # also fully ocluded in y axis
                occlusion_mask[:, :] = 1
            elif box_tlbr[1] > occ_tlbr[1]: # from top
                occlusion_mask[:y_inter, :] = 1
            else: # from bottom
                occlusion_mask[-y_inter:, :] = 1
        elif box_tlbr[0] > occ_tlbr[0]: # track occluded from the left
            if box_tlbr[1] > occ_tlbr[1] and box_tlbr[3] < occ_tlbr[3]: # also fully ocluded in y axis
                occlusion_mask[:, :x_inter] = 1
            elif box_tlbr[1] > occ_tlbr[1]: # from top
                occlusion_mask[:y_inter, :x_inter] = 1
            else: # from bottom
                occlusion_mask[-y_inter:, :x_inter] = 1
        else: # track occluded from the right
            if box_tlbr[1] > occ_tlbr[1] and box_tlbr[3] < occ_tlbr[3]: # also fully ocluded in y axis
                occlusion_mask[:, -x_inter:] = 1
            elif box_tlbr[1] > occ_tlbr[1]: # from top
                occlusion_mask[:y_inter, -x_inter:] = 1
            else: # from bottom
                occlusion_mask[-y_inter:, -x_inter:] = 1
        
        current_overlap = occlusion_mask.sum() / occlusion_mask_area
        if current_overlap > overlap_thresh:
            break
    # print(occlusion_mask)
    # cv2.imshow('occlusion trkid{}'.format(track.track_id), occlusion_mask*255)
    return current_overlap

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
