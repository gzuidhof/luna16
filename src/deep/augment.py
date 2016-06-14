from params import params as P
import numpy as np


try:
    import cv2
    CV2_AVAILABLE=True
    print "OpenCV 2 available, using that for augmentation"
except:
    from scipy.ndimage.interpolation import rotate, shift, zoom, affine_transform
    from skimage.transform import warp, AffineTransform
    CV2_AVAILABLE=False
    print "OpenCV 2 NOT AVAILABLE, using skimage/scipy.ndimage instead"

def augment(images):
    pixels = images[0].shape[1]
    center = pixels/2.-0.5

    random_flip = P.AUGMENTATION_PARAMS['flip'] and np.random.randint(2) == 1

    # Translation shift
    shift_x = np.random.uniform(*P.AUGMENTATION_PARAMS['translation_range'])
    shift_y = np.random.uniform(*P.AUGMENTATION_PARAMS['translation_range'])
    rotation_degrees = np.random.uniform(*P.AUGMENTATION_PARAMS['rotation_range'])
    zoom_factor = np.random.uniform(*P.AUGMENTATION_PARAMS['zoom_range'])

    if CV2_AVAILABLE:
        M = cv2.getRotationMatrix2D((center, center), rotation_degrees, zoom_factor)
        M[0, 2] += shift_x
        M[1, 2] += shift_y

    for i in range(len(images)):
        image = images[i]

        if CV2_AVAILABLE:
            #image = image.transpose(1,2,0)
            image = cv2.warpAffine(image, M, (pixels, pixels))
            if random_flip:
                image = cv2.flip(image, 1)
            #image = image.transpose(2,0,1)
            images[i] = image
        else:
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
