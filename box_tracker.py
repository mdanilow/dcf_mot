from enum import Enum
import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
import torch
from torchvision.ops import roi_pool, roi_align
import matplotlib.pyplot as plt
import time

from utils import draw_bboxes, scale_coords, scale_f_coords, draw_text_line


class TrackerState(Enum):
    UNCERTAIN = 0
    CONFIRMED = 1
    ACTIVE = 2
    OCCLUDED = 3
    FOR_TERMINATION = 4


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h        #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((4,))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((5,))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    tracker_config = None
    frame_count = 0
    count = 0
    def __init__(self, bbox, dcf_config=None, img_shape=None, features=None, features_bbox=None, debug=None):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],    [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0      # since detected OR considered alive by the dcf
        self.time_since_detected = 0    # since detected
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0             # if detected OR considered alive by the dcf
        self.detection_hit_streak = 0   # if detected
        self.age = 0
        self.dcf_config = dcf_config
        self.predict_dcf_liveness = KalmanBoxTracker.tracker_config['predict_dcf_liveness']

        self.__tracker_state = TrackerState.UNCERTAIN

        if dcf_config is not None and features is not None and features_bbox is not None:
            self.dcf = DCF(dcf_config, img_shape, features, features_bbox, debug=debug)

    @property
    def tracker_state(self):
        if self.__tracker_state == TrackerState.UNCERTAIN:
            if (
                self.time_since_update > KalmanBoxTracker.tracker_config['max_time_since_update']
                or self.time_since_detected > KalmanBoxTracker.tracker_config['max_time_since_detected']
            ):
                self.__tracker_state = TrackerState.FOR_TERMINATION

            elif (
                self.time_since_update <= KalmanBoxTracker.tracker_config['max_time_since_update_to_report']
                and (self.hit_streak >= KalmanBoxTracker.tracker_config['min_hit_streak'] or KalmanBoxTracker.frame_count <= KalmanBoxTracker.tracker_config['min_hit_streak'])
                and self.time_since_detected <= KalmanBoxTracker.tracker_config['max_time_since_detected_to_report']
            ):
                self.__tracker_state = TrackerState.CONFIRMED
        
        elif self.__tracker_state == TrackerState.CONFIRMED:
            if (
                self.time_since_update > KalmanBoxTracker.tracker_config['max_time_since_update']
                or self.time_since_detected > KalmanBoxTracker.tracker_config['max_time_since_detected']
            ):
                self.__tracker_state = TrackerState.FOR_TERMINATION
        return self.__tracker_state

    def update(self, bbox, features=None, features_bbox=None, detected=False, debug=None):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        # if not self.dcf_config['predict_position']:
        self.kf.update(convert_bbox_to_z(bbox))
        if detected:
            self.time_since_detected = 0
            self.detection_hit_streak += 1
            if (features is not None) and (features_bbox is not None):
                self.dcf.update_filter(features, features_bbox, debug=debug)

    def predict(self, features=None, debug=None):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        if features is not None and self.dcf_config and self.predict_dcf_liveness:
            self.dcf.predict_displacement(features, self.get_state(), debug=debug)
        if features is not None and self.dcf_config and self.dcf_config['predict_position']:
            self.kf.x[:4] = convert_bbox_to_z(self.get_state() + self.dcf.predicted_displacement)
            
        self.history.append(convert_x_to_bbox(self.kf.x))
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        if(self.time_since_detected > 0):
            self.detection_hit_streak = 0
        self.time_since_detected += 1
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)
    

# peak_pos in (x, y) format
def compute_psr(x, peak_pos, peak_size):
    mask = np.zeros(x.shape)
    crop_left = np.clip(peak_pos[0] - peak_size // 2, 0, x.shape[1] - 1)
    crop_right = np.clip(peak_pos[0] + peak_size // 2, 0, x.shape[1] - 1)
    crop_top = np.clip(peak_pos[1] - peak_size // 2, 0, x.shape[0] - 1)
    crop_bottom = np.clip(peak_pos[1] + peak_size // 2, 0, x.shape[0] - 1)
    mask[crop_top : crop_bottom + 1, crop_left : crop_right + 1] = 1

    sidelobe = np.ma.array(x, mask=mask)
    sidelobe_mean = sidelobe.mean()
    sidelobe_std = sidelobe.std()
    peak = x[peak_pos[1], peak_pos[0]]
    psr = (peak - sidelobe_mean) / sidelobe_std

    return psr


def compute_apce(x, peak_pos, peak_size):
    xmax = np.max(x)
    xmin = np.min(x)
    dupa = np.mean((x - xmin)**2)
    return (xmax - xmin)**2 / dupa
    

class DCF():

    G = None
    hanning_window = None
    feature_pad_xy = (0, 0)

    def __init__(self, dcf_config, img_shape, features, bbox, debug=None):
        self.img_shape = img_shape
        self.roi_size = dcf_config['roi_size']
        self.sigma = dcf_config['sigma']
        self.search_region_scale = dcf_config['search_region_scale']
        self.crop_mode = dcf_config['crop_mode']
        self.lambd = dcf_config['lambd']
        self.lr = dcf_config['lr']
        self.update_strategy = dcf_config['update_strategy']
        self.normalize_features = dcf_config['normalize_features']
        self.psr_peak_crop_size = dcf_config['psr_peak_crop_size']
        self.resize_interp_mode = eval(dcf_config['resize_interp_mode'])
        self.liveness_fn = eval(dcf_config["liveness_fn"])
        self.shift_hanning_window_up = dcf_config["shift_hanning_window_up"]
        self.init_filter(features, bbox, debug=debug)
        self.psr = -1
        self.max_response = -1
        # self.selfcorr = np.max(self.compute_response(features, bbox))

    @staticmethod
    def init_constants(dcf_config):
        roi_size = dcf_config["roi_size"]
        DCF.G = np.fft.fft2(DCF.get_gauss_response(roi_size, dcf_config["sigma"]))
        DCF.hanning_window = window_func_2d(roi_size, roi_size, shift_up=dcf_config["shift_hanning_window_up"])


    @staticmethod
    def get_gauss_response(size, sigma):

        def linear_mapping(img):
            return (img - img.min()) / (img.max() - img.min())
       
        xx, yy = np.meshgrid(np.arange(size), np.arange(size))
        # get the center of the object...
        center_x = size // 2
        center_y = size // 2
        
        # cal the distance...
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * sigma)
        # get the response map...
        response = np.exp(-dist)
        # normalize...
        response = linear_mapping(response)

        return response
    

    def init_filter(self, features, bbox, debug=None):
        # start = time.time()
        template = self.crop_search_window(bbox, features, debug=debug)
        # crop = time.time()
        fi = self.pre_process(template)
        fftfi = np.fft.fft2(fi)
        self.Ai = DCF.G * np.conjugate(fftfi)
        self.Bi = fftfi * np.conjugate(fftfi) + self.lambd
        self.Bi = self.Bi.sum(axis=0)
        self.Hi = self.Ai / self.Bi
        # total = time.time()
        # print('init_filter total:', total - start, "crop:", (crop - start) * 100 / (total - start), "%")

    
    # features in CHW shape
    # bbox in format [x1,y1,x2,y2,score]
    def compute_response(self, features, bbox, debug=None):
        fi = self.crop_search_window(bbox, features, debug=debug)
        fi = self.pre_process(fi)
        fftfi = np.fft.fft2(fi)
        Gi = self.Hi * fftfi
        Gi = np.sum(Gi, axis=0)
        gi = np.real(np.fft.ifft2(Gi))
        # print('compte response debug:', debug)
        if debug is not None:
            cv2.imshow('response {}'.format(debug), gi)

        return gi
    

    def predict_displacement(self, features, bbox, update_psr=False, debug=None):
        # features_bbox = scale_f_coords(self.img_shape, np.expand_dims(bbox, axis=0), features.shape[2:])
        response = self.compute_response(features, bbox, debug=debug)
        max_value = np.max(response)
        self.max_response = max_value
        max_pos = np.where(response == max_value)
        max_pos = (int(np.mean(max_pos[1])), int(np.mean(max_pos[0])))

        # compute peak-to-sidelobe ratio (psr)
        psr = self.liveness_fn(response, max_pos, self.psr_peak_crop_size)
        if update_psr:
            self.psr = psr

        dx = max_pos[0] - response.shape[1] / 2
        dy = max_pos[1] - response.shape[0] / 2
        # scale from roi dimension to features dimension
        dx /= self.x_scale
        dy /= self.y_scale
        # scale from features dimension to image dimension
        displacement = scale_coords(DCF.unpadded_features_shape, np.array([[dx, dy, dx, dy]]), self.img_shape)[0]
        if update_psr:
            self.predicted_displacement = displacement[:2]

        if debug is not None:
            debug_response = ((response - np.min(response)) / (np.max(response) - np.min(response))) * 255
            debug_response = np.stack([debug_response] * 3, axis=2).astype(np.uint8)
            debug_response = cv2.circle(debug_response.copy(), max_pos, 2, (0, 0, 255), -1)
            draw_text_line(debug_response, "psr: " + "{:.1f}".format(self.psr), offset=(2, 10), color=(0, 255, 0), font_scale=0.25)
            cv2.imshow('response {}'.format(debug), debug_response)

        return displacement, psr
    

    def update_filter(self, features, bbox, debug=None):
        assert self.update_strategy in ["init", "average", "none"]
        if self.update_strategy == "init":
            self.init_filter(features, bbox, debug=debug)
        elif self.update_strategy == "average":
            fi = self.crop_search_window(bbox, features, debug=debug)
            fi = self.pre_process(fi)
            fftfi = np.fft.fft2(fi)
            self.Ai = self.lr * (DCF.G * np.conjugate(fftfi)) + (1 - self.lr) * self.Ai
            self.Bi = self.lr * (np.sum(fftfi * np.conjugate(fftfi) + self.lambd, axis=0)) + (1 - self.lr) * self.Bi
        elif self.update_strategy == "none":
            pass

    
    # features in CHW shape
    # bbox in format [x1,y1,x2,y2,score]
    def crop_search_window(self, bbox, features, debug=None):
        # start = time.time()
        if len(features.shape) == 4:
            features = features[0]

        # scale bbox from img dimensions to features dimension, take feature padding into account
        bbox = scale_f_coords(self.img_shape,
            np.expand_dims(bbox, axis=0),
            DCF.unpadded_features_shape[1:]
        )[0]

        xmin, ymin, xmax, ymax = bbox[:4]
        width = xmax - xmin
        height = ymax - ymin

        if self.search_region_scale != 1:
            x_offset = (width * self.search_region_scale - width) / 2
            y_offset = (height * self.search_region_scale - height) / 2
            xmin = xmin - x_offset
            xmax = xmax + x_offset
            ymin = ymin - y_offset
            ymax = ymax + y_offset
        
        # scaling between features dimension and roi dimension - to calculate displacement later
        self.x_scale = self.roi_size / (xmax - xmin)
        self.y_scale = self.roi_size / (ymax - xmin)

        # x_pad = int(width * self.search_region_scale)
        # y_pad = int(height * self.search_region_scale)
        # # to HWC
        # features = features.transpose(1, 2, 0)
        # pad_start = time.time()
        # features = cv2.copyMakeBorder(features, y_pad, y_pad, x_pad, x_pad, cv2.BORDER_REFLECT)
        # pad_end = time.time()
        # xmin += x_pad
        # xmax += x_pad
        # ymin += y_pad
        # ymax += y_pad
        # # to CHW
        # features = features.transpose(2, 0, 1)
        xmin += DCF.feature_pad_xy[0]
        xmax += DCF.feature_pad_xy[0]
        ymin += DCF.feature_pad_xy[1]
        ymax += DCF.feature_pad_xy[1]

        box = np.array([[xmin, ymin, xmax, ymax]]).astype(float)
        # box = [int(el) for el in box]

        if self.crop_mode == "roi_pool":
            f = torch.from_numpy(np.expand_dims(features, axis=0).astype(float))
            b = [torch.from_numpy(box)]
            window = roi_pool(f, b, self.roi_size).numpy()[0]
        elif self.crop_mode == "crop_resize":
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            window = features[:, ymin:ymax, xmin:xmax]
            window = window.transpose(1, 2, 0)
            window = cv2.resize(window, (self.roi_size, self.roi_size), interpolation=self.resize_interp_mode)
            window = window.transpose(2, 0, 1)
        # total = time.time() - start
        # pad = pad_end - pad_start
        # print('crop total:', total, "pad:", pad * 100 / total, "%")

        if debug is not None:
            for i in range(11, 12):
                ch = features[i]
                test = ((ch - np.min(ch)) / (np.max(ch) - np.min(ch))) * 255
                test = np.stack([test] * 3, axis=2)
                debug_window = window[i]
                debug_window = ((debug_window - np.min(debug_window)) / (np.max(debug_window) - np.min(debug_window))) * 255
                debug_window = np.stack([debug_window] * 3, axis=2)
                draw_bboxes(test, np.array([[xmin, ymin, xmax, ymax]]))
                cv2.imshow('{}'.format(debug), test.astype(np.uint8))
                cv2.imshow('window {}'.format(debug), debug_window.astype(np.uint8))

        return window

    def pre_process(self, img):
        channels, height, width = img.shape
        # print(type(img), img.shape)
        # img = np.log(img + 1)
        # print('img:', img)
        # img = (img - np.mean(img)) / (np.std(img) + 1e-5)
        if self.normalize_features:
            img = img + np.min(img)
            img = img / (np.max(img) + 1e-5)

        # window = window_func_2d(height, width)
        img = img * self.hanning_window

        return img
    

def window_func_2d(height, width, shift_up=0):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)

    win = mask_col * mask_row
    if shift_up != 0:
        new_win = np.zeros(win.shape)
        abs_shift = int(shift_up * height)
        new_win[:(height - abs_shift), :] = win[abs_shift:, :]
        win = new_win

    # test = test = ((win - np.min(win)) / (np.max(win) - np.min(win))) * 255
    # test = np.stack([test] * 3, axis=2)
    # cv2.imshow("window", test.astype(np.uint8))

    return win

    