from params import params as P
import numpy as np
from scipy.ndimage.interpolation import rotate, shift, zoom, affine_transform
from skimage.transform import warp, AffineTransform

def augment(images):

    random_flip = P.AUGMENTATION_PARAMS['flip'] and np.random.randint(2) == 1

    # Translation shift
    shift_x = np.random.uniform(*P.AUGMENTATION_PARAMS['translation_range'])
    shift_y = np.random.uniform(*P.AUGMENTATION_PARAMS['translation_range'])
    rotation_degrees = np.random.uniform(*P.AUGMENTATION_PARAMS['rotation_range'])
    #zoom_factor = np.random.uniform(*P.AUGMENTATION_PARAMS['zoom_range'])

    #zoom_x = np.random.uniform(*P.AUGMENTATION_PARAMS['zoom_range'])
    #zoom_y = np.random.uniform(*P.AUGMENTATION_PARAMS['zoom_range'])

    for image in images:

        center = image.shape[0]/2.-0.5

        if random_flip:
            image = image.transpose(1,0)
            image[:,:] = image[::-1,:]
            image = image.transpose(1,0)

        rotate(image, rotation_degrees, reshape=False, output=image)
        #affine_transform(image, np.array([[zoom_x,0], [0,zoom_x]]), output=image)
        #z = AffineTransform(scale=(2,2))
        #image = warp(image, z.params)
        shift(image, [shift_x,shift_y], output=image)

    return images
