""""
    J. Hogenboom - Clinical Data Science Maastricht Maastro Clinic - June 2022

    Inspired by the work of Joris Guérin.
    https://github.com/jorisguerin/pretrainedCNN_clustering/blob/master/Utils/Feature_extractor.py
    DOI: 10.1016/j.neucom.2020.10.068
"""

# Public module
import cv2
import itk

import numpy as np
# functions
from tensorflow.keras import Model
# pre-processing
from keras.applications.vgg19 import preprocess_input as prpc_vgg
from keras.applications.xception import preprocess_input as prpc_xce_inc
from keras.applications.resnet import preprocess_input as prpc_res
# models
import keras.applications.resnet as res
import keras.applications.vgg16 as vgg16
import keras.applications.vgg19 as vgg19
import keras.applications.xception as xce

# Private module
import ReadWriteImageModule as rw


class FeatureExtractor:

    def __init__(self, extractors, available_extractors={'ResNet50', 'VGG16', 'VGG19', 'Xception'}):
        # retrieve extractors that are to be used
        self.Extractors = extractors

        # initialise variables
        self.ImageSizes = []
        self.Models = []
        self.PrPc = []
        self.ModelIndexes = []
        self.EmptyFeatures = []

        # check availability of extractor and initiate
        self.ExtractorToUse = available_extractors.intersection(self.Extractors)
        if len(self.ExtractorToUse) == 0:
            exit(f'None of chosen extractors is available.\n Available extractors: {available_extractors}')
        else:
            print(f'The following extractors were available and selected:\n{self.ExtractorToUse}')
            self.initialise_extractors()
            self.initialise_negative_control()

    def initialise_extractors(self):
        """
            Initialise the extractors that are specified in __innit__
        """

        # TODO adaptive layer selection

        # build all supported and selected models and place them in a list
        if 'ResNet50' in self.ExtractorToUse:
            # include necessities as image size and keras preprocessing
            self.ImageSizes.append(224)
            self.PrPc.append(prpc_res)

            # build model
            base_res50 = res.ResNet50(weights='imagenet')

            base_res50.summary()
            res50model = Model(inputs=base_res50.input, outputs=base_res50.get_layer('conv5_block3_2_relu').output)
            self.Models.append(res50model)

        if 'VGG16' in self.ExtractorToUse:
            # include necessities as image size and keras preprocessing
            self.ImageSizes.append(224)
            self.PrPc.append(prpc_vgg)

            # build model
            base_vgg16 = vgg16.VGG16(weights='imagenet')
            vgg16model = Model(inputs=base_vgg16.input, outputs=base_vgg16.get_layer('block5_conv3').output)
            self.Models.append(vgg16model)

        if 'VGG19' in self.ExtractorToUse:
            # include necessities as image size and keras preprocessing
            self.ImageSizes.append(224)
            self.PrPc.append(prpc_vgg)

            # build model
            base_vgg19 = vgg19.VGG19(weights='imagenet')
            vgg19model = Model(inputs=base_vgg19.input, outputs=base_vgg19.get_layer('block5_conv3').output)
            self.Models.append(vgg19model)

        if 'Xception' in self.ExtractorToUse:
            # include necessities as image size and keras preprocessing
            self.ImageSizes.append(299)
            self.PrPc.append(prpc_xce_inc)

            # build model
            base_xce = xce.Xception(weights='imagenet')
            xcemodel = Model(inputs=base_xce.input, outputs=base_xce.get_layer('add_11').output)
            self.Models.append(xcemodel)

        if len(self.ImageSizes) != len(self.PrPc) or len(self.ImageSizes) != len(self.Models):
            exit('Mismatch in number of models and their necessities\nExiting program.')
        else:
            # start with -1 so that index 0 can be retrieved
            model_index = -1
            for number_of_models in range(len(self.Models)):
                model_index = model_index + 1
                self.ModelIndexes.append(model_index)

    def initialise_negative_control(self):
        """
            Generate a list of feature arrays of empty arrays (to serve as control)
        """
        # create an empty array
        empty_array = np.zeros(shape=[1024, 1024, 1], dtype=np.uint8)

        # reshape and retrieve features
        self.EmptyFeatures = self.extract_features(empty_array, True)

    def open_and_convert(self, image, pixel_type=itk.UC, dimension=2):
        """
            Generates an ITK image that is converted to an array.

            :param str image: filepath and image name of image that needs processing
            :param itk.UC pixel_type: ITK image type
            :param int dimension: number of image dimensions
            :return: np.ndarray input_image_array: returns an array of image
        """
        # OpenCV functions do not function with float images thus unsigned character is advisable
        image_type = itk.Image[pixel_type, dimension]

        # load standard image using ITK
        input_image = rw.imagereader(image, image_type)
        # convert to array for further use
        input_image_array = itk.array_from_image(input_image)
        return input_image_array

    def process_image(self, input_image_array):
        """
            Adjusts and adds appropriate size and colour space information, and applies model specific preprocessing.

            :param np.ndarray input_image_array: image array
            :return: np.ndarray image_arrays: returns an array of all processed images
        """
        # create list for arrays
        image_arrays = []

        for model_index in self.ModelIndexes:
            # convert to 1, height, width, 3 shape for CNN
            # resize to proper height and width
            processed_arr = cv2.resize(input_image_array, (self.ImageSizes[model_index], self.ImageSizes[model_index]))

            # add colour space information
            processed_arr = cv2.cvtColor(processed_arr, cv2.COLOR_GRAY2BGR)

            # finalise shape and convert to array
            processed_arr = [processed_arr]
            processed_arr = np.asarray(processed_arr, dtype=np.float64)

            # Keras specific processing
            processed_arr = self.PrPc[model_index](processed_arr)

            # add array to list of arrays
            image_arrays.append(processed_arr)

        return image_arrays

    def determine_penalty(self, features, acceptance_threshold=15, feature_threshold=0, penalty_factor=1.1):
        """
            Inspects whether the feature array has less than acceptable entries smaller that
            are larger than or equal to -feature threshold and smaller than or equal to +feature threshold.

            :param numpy.ndarray features: extracted image features
            :param int acceptance_threshold: per cent of entries that are accepted to be smaller than feature threshold
            :param int feature_threshold: value that array entries will be compared the extracted feature with
            :param int penalty_factor: the factor used as penalty, is multiplied per percentage of emptiness
            :return: int penalty
        """

        similarity_to_empty = self.compare_content(features, self.EmptyFeatures)

        # determine percentage
        per_cent_denied = (similarity_to_empty / features.size) * 100

        # increase penalty as emptiness increases
        penalty = 1 + (per_cent_denied / 100) * penalty_factor

        # return penalty if larger than accepted threshold else return zero penalty
        if acceptance_threshold <= per_cent_denied:
            print(f'Number of feature elements that do not deviate more than {feature_threshold} from is '
                  f'{np.round(per_cent_denied, 2)} per cent of total.\n'
                  f'This is smaller or equal than the acceptable percentage of {acceptance_threshold}.\n'
                  f'Contents are penalised.')
            return penalty
        else:
            return 1

    def compute_loss(self, features_a, features_b, penalty_a=1, penalty_b=1):
        """
            Compute the content loss between a and b ndarray

            :param numpy.ndarray features_a: extracted image features of image a
            :param numpy.ndarray features_b: extracted image feature of image b
            :param int penalty_a: penalty term for features of a
            :param int penalty_b: penalty term for features of b
            :return: int content_loss: the difference between a and b
        """

        for c, c_result in zip(features_a, features_b):
            # compute difference
            content_loss = self.compare_content(c, c_result)

            # use highest penalty to penalise comparison
            penalty = 1

            # multiply loss with penalty
            content_loss = content_loss * penalty
            print(f'content loss {content_loss}')

        return content_loss

    def compare_content(self, contents_a, contents_b):
        """
            Computes mean squared error between contents a and b

            :param numpy.ndarray contents_a: extracted image contents of a
            :param numpy.ndarray contents_b: extracted image contents of b
            :return:
        """

        content_loss = 0
        # compute difference
        content_loss += np.mean(np.square((contents_a - contents_b)))
        return content_loss

    def extract_features(self, image, array=False):
        """
            Extract image features with the defined models

            :param str image: filepath and image name of image that needs processing
            :param bool array: boolean used for when array is fed to extractor instead of string
            :return list features_list: a list of numpy.ndarrays that contain the extracted features
        """

        # create list for extracted features
        features_list = []

        if not array:
            image = self.open_and_convert(image)

        # prepare first image
        image_array = self.process_image(image)

        for model_index in self.ModelIndexes:
            # retrieve model from list
            extraction_model = self.Models[model_index]

            # extract feature
            features = extraction_model.predict(image_array[model_index])

            # add extracted features to list
            features_list.append(features)

        # return list of extracted features
        return features_list
