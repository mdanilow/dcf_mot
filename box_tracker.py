import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
import torch
from torchvision.ops import roi_pool, roi_align
import matplotlib.pyplot as plt

from utils import draw_bboxes, scale_coords, draw_text_line


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
    count = 0
    def __init__(self, bbox, hits_to_be_confirmed=3, dcf_config=None, img_shape=None, features=None, features_bbox=None, debug=None):
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
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.hits_to_be_confirmed = hits_to_be_confirmed
        self.dcf_config = dcf_config
        self.predict_dcf_liveness = KalmanBoxTracker.tracker_config['predict_dcf_liveness']

        if dcf_config is not None and features is not None and features_bbox is not None:
            self.dcf = DCF(dcf_config, img_shape, features, features_bbox, debug=debug)

    def is_confirmed(self):
        return self.hits >= self.hits_to_be_confirmed

    def update(self, bbox, features=None, features_bbox=None):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        # if not self.dcf_config['predict_position']:
        self.kf.update(convert_bbox_to_z(bbox))
        if features is not None and features_bbox is not None:
            self.dcf.update_filter(features, features_bbox)

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
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
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
    

class DCF():

    G = None

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
        if DCF.G is None:
            DCF.G = np.fft.fft2(self.get_gauss_response(self.roi_size))

        self.init_filter(features, bbox, debug=debug)
        self.psr = -1
        self.max_response = -1
        # self.selfcorr = np.max(self.compute_response(features, bbox))
    
    def init_filter(self, features, bbox, debug=None):
        template = self.crop_search_window(bbox, features, debug=debug)
        fi = self.pre_process(template)
        fftfi = np.fft.fft2(fi)
        self.Ai = DCF.G * np.conjugate(fftfi)
        self.Bi = fftfi * np.conjugate(fftfi) + self.lambd
        self.Bi = self.Bi.sum(axis=0)
        self.Hi = self.Ai / self.Bi

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
    

    def predict_displacement(self, features, bbox, debug=None):
        features_bbox = scale_coords(self.img_shape, np.expand_dims(bbox, axis=0), features.shape[2:])
        response = self.compute_response(features, features_bbox[0], debug=None)
        max_value = np.max(response)
        self.max_response = max_value
        max_pos = np.where(response == max_value)
        max_pos = (int(np.mean(max_pos[1])), int(np.mean(max_pos[0])))

        # compute peak-to-sidelobe ratio (psr)
        self.psr = compute_psr(response, max_pos, self.psr_peak_crop_size)

        dx = max_pos[0] - response.shape[1] / 2
        dy = max_pos[1] - response.shape[0] / 2
        # scale from roi dimension to features dimension
        dx /= self.x_scale
        dy /= self.y_scale
        # scale from features dimension to image dimension
        displacement = scale_coords(features.shape[2:], np.array([[dx, dy, dx, dy]]), self.img_shape)[0]
        self.predicted_displacement = displacement

        if debug is not None:
            debug_response = ((response - np.min(response)) / (np.max(response) - np.min(response))) * 255
            debug_response = np.stack([debug_response] * 3, axis=2).astype(np.uint8)
            debug_response = cv2.circle(debug_response.copy(), max_pos, 2, (0, 0, 255), -1)
            draw_text_line(debug_response, "psr: " + "{:.1f}".format(self.psr), offset=(2, 10), color=(0, 255, 0), font_scale=0.25)
            cv2.imshow('response {}'.format(debug), debug_response)

        return displacement
    

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

    def get_gauss_response(self, size):

        def linear_mapping(img):
            return (img - img.min()) / (img.max() - img.min())
       
        xx, yy = np.meshgrid(np.arange(size), np.arange(size))
        # get the center of the object...
        center_x = size // 2
        center_y = size // 2
        
        # cal the distance...
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.sigma)
        # get the response map...
        response = np.exp(-dist)
        # normalize...
        response = linear_mapping(response)

        return response
    
    # features in CHW shape
    # bbox in format [x1,y1,x2,y2,score]
    def crop_search_window(self, bbox, features, debug=None):
        if len(features.shape) == 4:
            features = features[0]
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

        x_pad = int(width * self.search_region_scale)
        y_pad = int(height * self.search_region_scale)
        # to HWC
        features = features.transpose(1, 2, 0)
        features = cv2.copyMakeBorder(features, y_pad, y_pad, x_pad, x_pad, cv2.BORDER_REFLECT)
        xmin += x_pad
        xmax += x_pad
        ymin += y_pad
        ymax += y_pad
        # to CHW
        features = features.transpose(2, 0, 1)

        box = np.array([[xmin, ymin, xmax, ymax]]).astype(float)
        # box = [int(el) for el in box]

        if self.crop_mode == "roi_pool":
            f = torch.from_numpy(np.expand_dims(features, axis=0).astype(float))
            b = [torch.from_numpy(box)]
            window = roi_pool(f, b, self.roi_size).numpy()[0]
        elif self.crop_mode == "crop_resize":
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            window = features[:, ymin:ymax, xmin:xmax]
            window = cv2.resize(window, (self.roi_size, self.roi_size))

        if debug is not None:
            for i in range(14, 15):
                ch = features[i]
                test = ((ch - np.min(ch)) / (np.max(ch) - np.min(ch))) * 255
                test = np.stack([test] * 3, axis=2)
                draw_bboxes(test, np.array([[xmin, ymin, xmax, ymax]]))
                cv2.imshow('features{} {}'.format(i, debug), test)
                cv2.imshow('features{} window {}'.format(i, debug), window.transpose(1, 2, 0)[:, :, i])

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

        window = window_func_2d(height, width)
        img = img * window

        return img
    

def window_func_2d(height, width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)

    win = mask_col * mask_row

    return win

    