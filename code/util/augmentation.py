import numpy as np
from PIL import Image
import random

class CTWindowing:
    def __init__(self, opt, jitter=False, return_PIL=True):
        self.opt = opt
        self.jitter = jitter
        self.return_PIL = return_PIL
            
    def __call__(self, np_array):
        assert isinstance(np_array, np.ndarray) # (H, W)
        minv, maxv = np_array.min(), np_array.max()
        if minv >= -1024 and maxv <= -769:
            image = np_array + 1024
        elif minv >= 0 and maxv <= 255:
            image = np_array
        else:
            channels = []
            for i in range(3):
                ww = self.opt.ww_wl_list[i][0]
                wl = self.opt.ww_wl_list[i][1]
                if self.jitter:
                    c_ww = random.sample(range(int(ww - 0.1 * ww), int(ww + 0.1 * ww)),1)[0]
                    c_wl = random.sample(range(int(wl - 0.1 * wl), int(wl + 0.1 * wl)),1)[0]
                    ww, wl = c_ww, c_wl
                minv = wl - ww / 2
                maxv = wl + ww / 2
                dn = maxv - minv
                channel = np.clip(np_array, minv, maxv)
                channel = (channel - minv) * 255.0 / dn
                channels.append(channel)
            image = np.stack(channels, axis=-1)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1).repeat(3, axis=-1)

        image = np.clip(image, 0, 255).astype(np.uint8)
        if self.return_PIL:
            return Image.fromarray(image)
        else:
            return image
        
    def __repr__(self):
        return self.__class__.__name__