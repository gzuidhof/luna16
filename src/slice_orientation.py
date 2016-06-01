import numpy as np
import image_read_write
import matplotlib.pyplot as plt

#orientation can be: x,y,z
def show_slice(image,coordinate,orientation):
    if orientation == "x":
        return image[:,coordinate,:]

    elif orientation == "y":
        return image[:,:,coordinate]

    else:
        return image[coordinate,:,:]

if __name__ == "__main__":
    #image,_,_= image_read_write.load_itk_image("../data/subset0/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd")
    image,_,_= image_read_write.load_itk_image("../data/subset0/hoi.mhd")
    print image.shape
    show_slice(image,150,"z")


