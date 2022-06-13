#  PYTHON MODULES
import itk

"""
    Application of numerous image (feature) filters included in the Insight Toolkit (ITK).
    Written by J. Hogenboom in December 2021 using ITK version 5.2.1post.
    
    Documentation of the ITKImageFeature module can be found at:
    https://itk.org/Doxygen/html/group__ITKImageFeature.html
"""


def equalise_histogram(image, image_type):
    """
        Equalises the histogram of the provided ITK image and returns it.
        https://itk.org/Doxygen/html/classitk_1_1AdaptiveHistogramEqualizationImageFilter.html

        :param itk.F image: ITK image that is to be altered
        :param itk.F image_type: ITK image type made up out of [pixel type, image dimensions]
        :return itk.F image: filtered ITK image
     """
    print('Applying adaptive histogram equalisation.')
    # create image type
    histogram_equalisation_image_filter_type = itk.AdaptiveHistogramEqualizationImageFilter[image_type]

    # create new image instance using specified type
    equalised_image = histogram_equalisation_image_filter_type.New()
    equalised_image.SetInput(image)
    equalised_image.SetAlpha(1)
    equalised_image.SetBeta(0.5)
    equalised_image.SetRadius(0)

    # pass image on
    equalised_image.Update()
    image = equalised_image.GetOutput()
    return image


def match_histogram(image, image_type, reference_image):
    """
        Matches the histogram of the provided ITK image with a reference ITK image and returns it.
        https://itk.org/Doxygen/html/classitk_1_1HistogramMatchingImageFilter.html

        :param itk.F image: ITK image that is to be altered
        :param itk.F image_type: ITK image type made up out of [pixel type, image dimensions]
        :param itk.F reference_image: ITK image of which the histogram serves as reference for process
        :return itk.F image: filtered ITK image
     """
    print('Applying histogram match.')
    # create image type
    histogram_matching_image_filter_type = itk.HistogramMatchingImageFilter[image_type, image_type]

    # create new image instance using specified type
    matched_image = histogram_matching_image_filter_type.New()
    matched_image.SetInput(image)
    matched_image.SetReferenceImage(reference_image)
    matched_image.SetNumberOfHistogramLevels(1024)
    matched_image.SetNumberOfMatchPoints(7)
    matched_image.ThresholdAtMeanIntensityOn()

    # pass image on
    matched_image.Update()
    image = matched_image.GetOutput()
    return image


def rescale_image(image, image_type, image_type_original, image_bits):
    """
        Rescales image to specified number of bits
        https://itk.org/Doxygen/html/classitk_1_1RescaleIntensityImageFilter.html

        :param itk.F image: ITK image that is to be altered
        :param itk.F image_type: ITK image type made up out of [pixel type, image dimensions]
        :param itk.F image_type_original: ITK image type
        :param int image_bits: the number of bits that the image is to be rescaled to
        :return itk.F image: filtered ITK image
     """
    print(f'Rescaling image to {image_bits}-bits.')
    if image_bits == 16:
        maximum = 65535
    elif image_bits == 8:
        maximum = 255
    else:
        return image

    # create image type
    rescale_filter_type = itk.RescaleIntensityImageFilter[image_type_original, image_type]

    # create new image instance using specified type
    rescaler = rescale_filter_type.New()
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(maximum)
    rescaler.SetInput(image)

    # return image
    rescaler.Update()
    image = rescaler.GetOutput()
    return image


def filter_info(form_list=False):
    """
        Simple function that returns a hardcoded set of available filters
        if desired returns a list, otherwise a set

        :param bool form_list: determines whether output of filter info is list; True returns list
        :return set filters: returns a list of available filters
    """
    filters = ['Derivative', 'CannyEdge', 'Hessian', 'Inverse',
               'LoG', 'Sobel', 'Threshold', 'None']
    # filters = ['Derivative', 'Inverse', 'LoG', 'Sobel', 'Threshold', 'None']

    if form_list:
        available_filters = filters
    else:
        available_filters = set(filters)

    return available_filters


def filter_image(image, image_type, image_filter, image_file_name=''):
    """
        Selects the filter from a list of available filters, passes image to specified filter function and
        returns the filtered image.
        In case a filename is provided the filename will be updated with the filter that is applied to the image.

        :param itk.F image : ITK image that is to be altered
        :param itk.F image_type : ITK image type made up out of [pixel type, image dimensions]
        :param str image_filter : filter that is to be applied to the image
        :param str image_file_name : filename of the image that is to be filtered
        :return itk.F image: filtered ITK image
        :return str image_file_name: updated filename containing original name and applied filter
     """
    # ensure chosen filter is available
    available_filters = filter_info()
    if image_filter == '' or image_filter not in available_filters:
        print('No available filter was specified.\n Currently available filters: \n', available_filters)
        image_filter = input('Please enter the filter you would like to use:\n')

    if image_filter == 'CannyEdge' or image_filter == 'cannyedge_filter':
        image = cannyedge_filter(image, image_type)
        if not image_file_name == '':
            image_file_name = f'{image_file_name}_filter_CannyEdge'
    elif image_filter == 'Derivative' or image_filter == 'derivative_filter':
        image = derivative_filter(image, image_type)
        if not image_file_name == '':
            image_file_name = f'{image_file_name}_filter_Derivative'
    elif image_filter == 'Hessian' or image_filter == 'hessian_filter':
        # noinspection PyTypeChecker
        image = hessian_filter(image, image_type)
        if not image_file_name == '':
            image_file_name = f'{image_file_name}_filter_Hessian'
    elif image_filter == 'Inverse' or image_filter == 'inverse_filter':
        image = inverse_filter(image, image_type)
        if not image_file_name == '':
            image_file_name = f'{image_file_name}_filter_Inverse'
    elif image_filter == 'LoG' or image_filter == 'loG_filter':
        image = log_filter(image, image_type)
        if not image_file_name == '':
            image_file_name = f'{image_file_name}_filter_LoG'
    elif image_filter == 'Sobel' or image_filter == 'sobel_filter':
        image = sobel_filter(image, image_type)
        if not image_file_name == '':
            image_file_name = f'{image_file_name}_filter_Sobel'
    elif image_filter == 'Threshold' or image_filter == 'threshold_filter':
        image = threshold_filter(image, image_type)
        if not image_file_name == '':
            image_file_name = f'{image_file_name}_filter_Threshold'
    elif image_filter == 'None':
        if not image_file_name == '':
            image_file_name = f'{image_file_name}_filter_None'
    else:
        exit('No available filter was specified.\nExiting program.')

    if image_file_name != '':
        return image, image_file_name
    else:
        return image


def cannyedge_filter(image, image_type, lower_threshold=0, upper_threshold=255, variance=7):
    """
        Applies a Canny Edge image filter on the provided ITK image and returns it.
        https://itk.org/Doxygen/html/classitk_1_1CannyEdgeDetectionImageFilter.html

        :param itk.F image: ITK image that is to be altered
        :param itk.F image_type: ITK image type made up out of [pixel type, image dimensions]
        :param int lower_threshold: lower threshold for CannyEdge output
        :param int upper_threshold: upper threshold for CannyEdge output
        :param int variance: variance for CannyEdge filter
        :return itk.F image: filtered ITK image
     """
    print('Applying CannyEdge filter.')
    # create image type
    canny_edge_image_filter_type = itk.CannyEdgeDetectionImageFilter[image_type, image_type]

    # create new image instance using specified type and set parameters
    canny_edge_image = canny_edge_image_filter_type.New()
    canny_edge_image.SetInput(image)
    canny_edge_image.SetLowerThreshold(lower_threshold)
    canny_edge_image.SetUpperThreshold(upper_threshold)
    canny_edge_image.SetVariance(variance)

    # pass image to rescale; necessary for CannyEdgeDetectionImageFilter
    canny_edge_image.GetOutput()
    image_type_ce = itk.Image[itk.UC, 2]
    rescaler = itk.RescaleIntensityImageFilter[image_type, image_type_ce].New()
    rescaler.SetInput(canny_edge_image.GetOutput())
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(255)

    # return image
    rescaler.Update()
    image = rescaler.GetOutput()
    return image


def derivative_filter(image, image_type):
    """
        Applies a derivative filter on the provided ITK image and returns it.
        https://itk.org/Doxygen/html/classitk_1_1DerivativeImageFilter.html

        :param itk.F image: ITK image that is to be altered
        :param itk.F image_type: ITK image type made up out of [pixel type, image dimensions]
        :return itk.F image: filtered ITK image
     """
    print('Applying derivative image filter.')
    # create image type
    derivative_image_filter_type = itk.DerivativeImageFilter[image_type, image_type]

    # create new image instance using specified type
    derivative_image = derivative_image_filter_type.New()
    derivative_image.SetInput(image)

    # pass image to filter selector
    derivative_image.Update()
    image = derivative_image.GetOutput()
    return image


def hessian_filter(image, image_type):
    """
        Applies a Hessian filter on the provided ITK image and returns it.
        https://itk.org/Doxygen/html/classitk_1_1HessianToObjectnessMeasureImageFilter.html

        credits to 'Fivethousand' on the itk community site
        https://discourse.itk.org/t/is-it-possible-to-compute-2d-hessian-value-for-each-slice-in-a-3d-medical-array/4565

        :param image: ITK image that is to be altered
        :param image_type : ITK image type made up out of [pixel type, image dimensions]
        :return image: filtered ITK image
     """
    print('Applying Hessian image filter.')

    sigma_minimum = 0.2
    sigma_maximum = 1  # 3.
    number_of_sigma_steps = 8

    dimension = image.GetImageDimension()
    hessian_pixel_type = itk.SymmetricSecondRankTensor[itk.D, dimension]
    hessian_image_type = itk.Image[hessian_pixel_type, dimension]

    objectness_filter = itk.HessianToObjectnessMeasureImageFilter[hessian_image_type, image_type].New()
    objectness_filter.SetBrightObject(True)
    objectness_filter.SetScaleObjectnessMeasure(True)
    objectness_filter.SetAlpha(0.5)
    objectness_filter.SetBeta(1.0)
    objectness_filter.SetGamma(5.0)

    multi_scale_filter = itk.MultiScaleHessianBasedMeasureImageFilter[image_type, hessian_image_type, image_type].New()
    multi_scale_filter.SetInput(image)
    multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
    multi_scale_filter.SetSigmaStepMethodToLogarithmic()
    multi_scale_filter.SetSigmaMinimum(sigma_minimum)
    multi_scale_filter.SetSigmaMaximum(sigma_maximum)
    multi_scale_filter.SetNumberOfSigmaSteps(number_of_sigma_steps)

    multi_scale_filter.Update()
    image = multi_scale_filter.GetOutput()
    return image


def inverse_filter(image, image_type):
    """
        Applies a inverse filter on the provided ITK image and and returns it.
        https://itk.org/ITKExamples/src/Filtering/ImageIntensity/InvertImage/Documentation.html

        :param itk.F image: ITK image that is to be altered
        :param itk.F image_type: ITK image type made up out of [pixel type, image dimensions]
        :return itk.F image: filtered ITK image
     """
    print('Applying inverse image filter.')
    # create image type
    inverse_image_filter_type = itk.InvertIntensityImageFilter[image_type, image_type]

    # create new image instance using specified type
    inverse_image = inverse_image_filter_type.New()
    inverse_image.SetInput(image)
    inverse_image.SetMaximum(255)

    # pass image to filter selector
    inverse_image.Update()
    image = inverse_image.GetOutput()
    return image


def log_filter(image, image_type):
    """
        Applies a Laplacian over Gaussian filter on the provided ITK image and and returns it.
        https://itk.org/Doxygen/html/classitk_1_1LaplacianRecursiveGaussianImageFilter.html

        :param itk.F image: ITK image that is to be altered
        :param itk.F image_type: ITK image type made up out of [pixel type, image dimensions]
        :return itk.F image: filtered ITK image
     """
    print('Applying Laplacian over Gaussian image filter.')
    # create image type
    log_image_filter_type = itk.LaplacianRecursiveGaussianImageFilter[image_type, image_type]

    # create new image instance using specified type
    log_image = log_image_filter_type.New()
    log_image.SetInput(image)

    # export image to filter selector
    log_image.Update()
    image = log_image.GetOutput()
    return image


def sobel_filter(image, image_type):
    """
        Applies a Sobel filter on the provided ITK image and and returns it.
        https://itk.org/Doxygen/html/classitk_1_1SobelEdgeDetectionImageFilter.html

        :param itk.F image: ITK image that is to be altered
        :param itk.F image_type: ITK image type made up out of [pixel type, image dimensions]
        :return itk.F image: filtered ITK image
     """
    print('Applying Sobel image filter.')
    # create image type
    sobel_image_filter_type = itk.SobelEdgeDetectionImageFilter[image_type, image_type]

    # create new image instance using specified type
    sobel_image = sobel_image_filter_type.New()
    sobel_image.SetInput(image)

    # export image to filter selector
    sobel_image.Update()
    image = sobel_image.GetOutput()
    return image


def threshold_filter(image, image_type, minimum=-750, maximum=1e+6, outside=0):
    """
        Applies a threshold filter on the provided ITK image and returns it.
        https://itk.org/Doxygen/html/classitk_1_1ThresholdImageFilter.html

        :param itk.F image: ITK image that is to be altered
        :param itk.F image_type: ITK image type made up out of [pixel type, image dimensions]
        :param int minimum:  lower threshold value
        :param int maximum: upper threshold value
        :param int outside: value that pixels outside of minimum and maximum should be set to
        :return image: filtered ITK image
     """
    print('Applying threshold image filter.')

    # create image type
    threshold_image_filter_type = itk.ThresholdImageFilter[image_type]

    # create new image instance using specified type
    threshold_image = threshold_image_filter_type.New()
    threshold_image.SetInput(image)
    threshold_image.ThresholdOutside(minimum, maximum)
    threshold_image.SetOutsideValue(outside)

    # pass image to filter selector
    threshold_image.Update()
    image = threshold_image.GetOutput()
    return image
