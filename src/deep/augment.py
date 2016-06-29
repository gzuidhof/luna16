from params import params as P
import numpy as np


try:
    import cv2
    CV2_AVAILABLE=True
    print "OpenCV 2 available, using that for augmentation"
    from scipy.ndimage.interpolation import rotate, shift, zoom, affine_transform
    from skimage.transform import warp, AffineTransform
except:
    from scipy.ndimage.interpolation import rotate, shift, zoom, affine_transform
    from skimage.transform import warp, AffineTransform
    CV2_AVAILABLE=False
    print "OpenCV 2 NOT AVAILABLE, using skimage/scipy.ndimage instead"

def augment(images):
    pixels = images[0].shape[1]
    center = pixels/2.-0.5

    random_flip_x = P.AUGMENTATION_PARAMS['flip'] and np.random.randint(2) == 1
    random_flip_y = P.AUGMENTATION_PARAMS['flip'] and np.random.randint(2) == 1

    # Translation shift
    shift_x = np.random.uniform(*P.AUGMENTATION_PARAMS['translation_range'])
    shift_y = np.random.uniform(*P.AUGMENTATION_PARAMS['translation_range'])
    rotation_degrees = np.random.uniform(*P.AUGMENTATION_PARAMS['rotation_range'])
    zoom_factor = np.random.uniform(*P.AUGMENTATION_PARAMS['zoom_range'])
    #zoom_factor = 1 + (zoom_f/2-zoom_f*np.random.random())
    if CV2_AVAILABLE:
        M = cv2.getRotationMatrix2D((center, center), rotation_degrees, zoom_factor)
        M[0, 2] += shift_x
        M[1, 2] += shift_y

    for i in range(len(images)):
        image = images[i]

        if CV2_AVAILABLE:
            #image = image.transpose(1,2,0)
            image = cv2.warpAffine(image, M, (pixels, pixels))
            if random_flip_x:
                image = cv2.flip(image, 0)
            if random_flip_y:
                image = cv2.flip(image, 1)
            #image = image.transpose(2,0,1)
            images[i] = image
        else:
            if random_flip_x:
                #image = image.transpose(1,0)
                image[:,:] = image[::-1,:]
                #image = image.transpose(1,0)
            if random_flip_y:
                image = image.transpose(1,0)
                image[:,:] = image[::-1,:]
                image = image.transpose(1,0)

            rotate(image, rotation_degrees, reshape=False, output=image)
            #image2 = zoom(image, [zoom_factor,zoom_factor])
            image2 = crop_or_pad(image, pixels, -3000)
            shift(image2, [shift_x,shift_y], output=image)
            #affine_transform(image, np.array([[zoom_x,0], [0,zoom_x]]), output=image)
            #z = AffineTransform(scale=(2,2))
            #image = warp(image, z.params)
            images[i] = image



    return images

def crop_or_pad(image, desired_size, pad_value):
    if image.shape[0] < desired_size:
        offset = (desired_size-image.shape[0])//2
        image = np.pad(image, offset, 'constant', constant_values=pad_value)
        if image.shape[0] != desired_size:
            new_image = np.full((image.shape[0]+1,image.shape[1]+1),fill_value=pad_value)
            new_image[:image.shape[0],:image.shape[1]]=image
            image = new_image
    if image.shape[0] > desired_size:
        offset = (image.shape[0]-desired_size)//2
        image = image[offset:offset+desired_size,offset:offset+desired_size]

    return image


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

OPTS = [[False,False,False], [False, False, True], [False, True, False], [False, True, True],
        [True, False, False], [True, False, True], [True, True, False], [True, True, True]]


def testtime_augmentation(image, label):
    labels = []
    images = []
    rotations = [0]
    flips = [[0,0],[1,0],[0,1],[1,1]]
    shifts = [[0,0]]
    zooms = [1]

    for r in rotations:
        for f in flips:
            for s in shifts:
                for z in zooms:
                    image2 = np.array(image)
                    if f[0]:
                        image2[:,:] = image2[::-1,:]
                    if f[1]:
                        image2 = image2.transpose(1,0)
                        image2[:,:] = image2[::-1,:]
                        image2 = image2.transpose(1,0)
                    #rotate(image2, r, reshape=False, output=image2)
                    #image3 = zoom(image2, [z,z])
                    #image3 = crop_or_pad(image3, P.INPUT_SIZE, 0)
                    #image2 = image3
                    # #shift(image2, [s[0],s[1]], output=image2)
                    images.append([image2]) #Adds color channel dimension!
                    labels.append(label)

    return images, labels

def flip_given_axes(image, opt):
    offset = 0
    if image.shape[0] == 1: #Has color channel
        offset = 1

    for i in range(3):
        if opt[i]:
            flip_axis(image, i+offset)
    return image

def get_all_flips_3d(image):
    flippos = []
    offset = 0
    if image.shape[0] == 1: #Has color channel
        offset = 1

    for opt in OPTS[1:]: #All opts except first (no flips)
        im = np.copy(image)
        for i in range(3):
            if opt[i]:
                flip_axis(im, i+offset)
        flippos.append(im)

    return flippos
