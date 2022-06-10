"""
    Module for image reading/writing based on ITK library

    :functions
    imagereader: returns ITK image
    imagewriter: writes image with the specified format
"""

# Public module
import itk
import numpy as np

# Private module
from ImageProcessing import filter_image
from ImageProcessing import rescale_image


def imagereader(image_file_path, image_type, image_filter='None', image_bits=0, compute_info=False):
    """
        Read, filter and obtain info (spacing, origin, size, volume centre) of an image with ITK
        https://itk.org/Doxygen/html/classitk_1_1ImageFileReader.html

        :param str image_file_path: file path of the image that is to be read
        :param itk.F image_type: ITK image type made up out of [pixel type, image dimensions]
        :param str image_filter: filter of ImageProcessing module that is to be applied on the image; none as default
        :param int image_bits: rescales intensities to certain bit size; zero i.e., inactive by default
        :param bool compute_info: if True computes spacing, origin, size, volume centre; False by default
        :return itk.F image: image in specified ITK type
        :return dict image_info: (optional) dictionary with Keys (Spacing, Origin, Size, Volume_center), if compute info
    """
    print(f'\nFile to be read: {image_file_path}')
    # create ITK reader
    image_reader = itk.ImageFileReader[image_type].New()
    image_reader.SetFileName(image_file_path)

    # save image in variable
    image_reader.Update()
    image = image_reader.GetOutput()

    # rescale intensities; can be necessary for semantic-based registration
    if image_bits > 0:
        # rescale image
        image = rescale_image(image, image_type, image_type, image_bits)

    # avoid unnecessary use of filtering function
    if not image_filter == 'None':
        image = filter_image(image, image_type, image_filter)

    # compute image information such origin, size, spacing and volume centre when desired
    if compute_info:
        # retrieve numerous characteristics
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        size = image.GetBufferedRegion().GetSize()
        volume_centre = np.asarray(origin) + np.multiply(spacing, np.divide(size, 2.)) - np.divide(spacing, 2.)

        # store characteristics in variable
        image_info = {'Spacing': spacing, 'Origin': origin, 'Size': size, 'Volume_centre': volume_centre}

        return image, image_info
    else:
        return image


def imagewriter(image, image_type, image_file_name, image_bits=0, image_filter='None', extension='.mha'):
    """
        Write and optionally filter an image with ITK
        https://itk.org/Doxygen/html/classitk_1_1ImageFileWriter.html

        :param itk.F image: image to be saved
        :param itk.F image_type: ITK image type made up out of [pixel type, image dimensions]
        :param str image_file_name: name and path of image that is to be saved
        :param int image_bits: number of bits the image should be written as; zero/inactive by default
        :param str image_filter: filter of ImageProcessing module that is to be applied on the image; none as default
        :param str extension: file extension that image is to be stored as; .mha as default to accommodate filtering
        :return str image_file_name: (optional) in case of a filter is specified the image name is saved for matching
    """

    image, image_file_name = filter_image(image, image_type, image_filter, image_file_name)

    if image_bits > 0:
        image_type_original = image_type
        if extension != '.mha':
            # most file formats such as .jpeg, .tif et cetera do not support float images
            image_type = itk.Image[itk.UC, 2]
        # CannyEdge images are rescaled within filter
        if image_filter != 'CannyEdge':
            # rescale image
            image = rescale_image(image, image_type, image_type_original, image_bits)
    # CannyEdge is returned as unsigned character thus pixel type should align
    if image_filter == 'CannyEdge':
        image_type = itk.Image[itk.UC, 2]

    # create ITK writer
    image_writer = itk.ImageFileWriter[image_type].New()

    # update filename with extension
    image_file_name = image_file_name + extension

    # set filename and save image
    image_writer.SetFileName(image_file_name)
    image_writer.SetInput(image)
    image_writer.Update()

    print(f'File saved as: {image_file_name}\n')

    # return file name to save information
    return image_file_name
