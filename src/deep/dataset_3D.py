import numpy as np
import SimpleITK as sitk
import pandas as pd

def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    image = np.transpose(sitk.GetArrayFromImage(itkimage))
    origin = np.array(itkimage.GetOrigin())
    spacing = np.array(itkimage.GetSpacing())
    return image, origin, spacing

def world_2_voxel(world_coord, origin, spacing):
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return voxel_coord

#this function creates an a list of 3D sub-image, where the candidate is in the middel of the image
def giveSubImage(image_path, coordinateList, size):

    image, origin, spacing = load_itk(image_path)
    output = []
    offset = size//2
    #padd images with values -3000, such that also candidates at the edge will remain in the middel of the image
    image_padded = np.pad(image,size,'constant',constant_values=-3000)
    #loop over all candidates and take the subimage and save this in the ouput list.
    for coordinate in coordinateList:
        center_pixel = np.floor(world_2_voxel(coordinate,origin,spacing)) + offset
        center_pixel = map(int, center_pixel)
        sub_image = image_padded[center_pixel[0]:center_pixel[0]+size,
                                    center_pixel[1]:center_pixel[1]+size,
                                    center_pixel[2]:center_pixel[2]+size]
        sub_image = np.expand_dims(sub_image, axis=0)
        output.append(sub_image)

    del image
    del origin
    del spacing
    del image_padded
    return output


#if __name__ == "__main__":
#    imagePath = "data/original/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd"
#    candidates = pd.read_csv('csvfiles/candidates.csv')
#    possibleAnnotations = candidates[candidates['seriesuid'] == '1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260']
#    x = possibleAnnotations.coordX
#    y = possibleAnnotations.coordY
#    z = possibleAnnotations.coordZ
#    coordinates = np.column_stack((x, y, z))
#    size = 96
#    giveSubImage(imagePath, coordinates, size)
