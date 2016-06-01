from params import params
import numpy as np
import cv2

from functools import partial
import util
from multiprocessing.pool import Pool

class Augmenter():
    def __init__(self, multiprocess=True):
        #Determine the center to rotate around
        self.center_shift = np.array((params.PIXELS, params.PIXELS)) / 2. - 0.5
        self.multiprocess = multiprocess

        if self.multiprocess:
            self.pool = Pool(4)

    def augment(self, Xb):
        # Random number 0-1 whether we flip or not
        random_flip = np.random.randint(2) == 1

        # Translation shift
        shift_x = np.random.uniform(*params.AUGMENTATION_PARAMS['translation_range'])
        shift_y = np.random.uniform(*params.AUGMENTATION_PARAMS['translation_range'])

        # Rotation, zoom
        rotation = np.random.uniform(*params.AUGMENTATION_PARAMS['rotation_range'])
        log_zoom_range = [np.log(z) for z in params.AUGMENTATION_PARAMS['zoom_range']]
        zoom = np.exp(np.random.uniform(*log_zoom_range))

        # Color AUGMENTATION_PARAMS
        random_hue = np.random.uniform(*params.AUGMENTATION_PARAMS['hue_range'])
        random_saturation = np.random.uniform(*params.AUGMENTATION_PARAMS['saturation_range'])
        random_value = np.random.uniform(*params.AUGMENTATION_PARAMS['value_range'])

        return self.augment_with_params(Xb, shift_x, shift_y, rotation, random_flip, zoom, random_hue, random_saturation, random_value)

    def augment_with_params(self, Xb, shift_x, shift_y, rotation, random_flip, zoom, hue, saturation, value):


            # Define affine matrix
            # TODO: Should be able to incorporate flips directly instead of through an extra call
            M = cv2.getRotationMatrix2D((self.center_shift[0], self.center_shift[1]), rotation, zoom)
            M[0, 2] += shift_x
            M[1, 2] += shift_y

            augment_partial = partial(augment_image,
                                        M=M,
                                        random_flip=random_flip,
                                        random_hue=hue,
                                        random_saturation=saturation,
                                        random_value=value)

            if self.multiprocess:
                l = self.pool.map(augment_partial, Xb)
                Xbb = np.array(l)
            else:
                Xbb = np.zeros(Xb.shape, dtype=np.float32)
                for i in xrange(Xb.shape[0]):
                    Xbb[i] = augment_partial(Xb[i])


            return Xbb

# Augments a single image, singled out for easier profiling
def augment_image(original_image, M=0, random_flip=0,
                    random_hue=0, random_saturation=0, random_value=0):
        im = cv2.warpAffine(original_image.transpose(1, 2, 0), M, (params.PIXELS, params.PIXELS))

        # im is now RGB 01c

        if random_flip:
            im = cv2.flip(im, 1)

        if params.COLOR_AUGMENTATION:
            im = util.hsv_augment(im, random_hue, random_saturation, random_value)

        # Back to c01
        return im.transpose(2, 0, 1)
