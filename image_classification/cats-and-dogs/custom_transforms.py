'''
Mini transforms libary for Pytorch
-----------------------------
- 2D and 3D agnostic

- Expects all input items to be numpy ndarrys 

Can convert between tensors, ndarrys, and PIL images using torchvision transforms
'''

from scipy.ndimage import zoom, rotate
import numpy as np


class Normalise:
    '''
    Normalisation intensity values between i and j, given an
    input (i,j)

    Default: (0,1)
    '''
    def __init__(self, intensity_range=(0, 1)):
        self.intensity_range = intensity_range

    def __call__(self, sample: np.ndarray):
        low = self.intensity_range[0]
        high = self.intensity_range[1]
        _min = np.min(sample)
        _max = np.max(sample)
        _range = _max - _min
        x = high - (((high - low) * (_max - sample)) / _range)
        
        return x

class Windsorise:
    '''
    Clips image intensities at nth percentile
    '''
    def __init__(self, intensity_percentile: int):
        self.int_perc = intensity_percentile

    def __call__(self, sample: np.ndarray):
        sample[sample > np.percentile(sample, self.int_perc)] = np.percentile(sample, self.int_perc)

        return sample


class Resize_zero_pad:
    '''

    Resizes ndimage 
    - Keeps proportional image dimensions
    - Minimal zoom to obtain desired dimension
    - Zeros pads other dimensions

    Number of channels stay the same

    - Use a spline order of 0 for nearest neighbour interpolation and a spline order of 1 for linear

    '''
    def __init__(self, resize: tuple, order=1):
        self.resize = resize
        self.order = order
        self.is2D = None
        self.is3D = None

        if len(resize) == 2:
            self.is2D = True

        elif len(resize) == 3:
            self.is3D = True
        else:
            raise TypeError('Resize Tuple must be 2D or 3D')
    
    def __call__(self, sample: np.ndarray):
        
        img_shape = np.shape(sample)

        if len(img_shape) != 3 and len(img_shape) != 4:
            print('Image dimensions: ', np.shape(sample))
            raise TypeError('Input image must be 3d or 4d array e.g. width, height, channels or width, height, depth, channels')
        
        if self.is2D:
            x = np.zeros((self.resize[0], self.resize[1], img_shape[2]), dtype='float32')
            zoomx = self.resize[0] / img_shape[0]
            zoomy = self.resize[1] / img_shape[1]
            zoom_param = np.min([zoomx, zoomy])            
            rs_img_data = zoom(sample, (zoom_param, zoom_param, 1), order=self.order)
            rsids = rs_img_data.shape
            x[:rsids[0], :rsids[1], :rsids[2]] = rs_img_data
        
        if self.is3D:
            x = np.zeros((self.resize[0], self.resize[1], self.resize[2], img_shape[3]), dtype='float32')
            zoomx = self.resize[0] / img_shape[0]
            zoomy = self.resize[1] / img_shape[1]
            zoomz = self.resize[2] / img_shape[2]
            zoom_param = np.min([zoomx, zoomy, zoomz])
            rs_img_data = zoom(sample, (zoom_param, zoom_param, zoom_param, 1), order=self.order)
            rsids = rs_img_data.shape
            x[:rsids[0], :rsids[1], :rsids[2], :rsids[3]] = rs_img_data
        
        return x


class RandomRotationAboutZ:
    '''
    Rotation range: degrees (0-180)

    Applies a random rotation about the z-axis
    '''
    def __init__(self, rotation_range: float, order=1):
        self.rotation_range = rotation_range
        self.order = order

    def __call__(self, sample: np.ndarray):
        # random value - degrees
        theta = np.random.uniform(-self.rotation_range, self.rotation_range)

        if len(np.shape(sample)) == 3 or len(np.shape(sample)) == 4:
            x = rotate(sample, theta, axes=(1,0), order=self.order, reshape=False)
        else:
            raise TypeError('Input image must be 3d or 4d array e.g. width, height, channels or width, height, depth, channels')
        
        return x