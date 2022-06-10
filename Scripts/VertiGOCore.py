# Public module
import itk
import numpy as np
import os
import time

# Private module
import ReadWriteImageModule as rw
import RigidMotionModule as rm
import ProjectorsModule as pm
import ImageProcessing as ip
import FeatureExtraction as fe
import DataVisualisation as dv


class VertiGOCore:
    """
        J. Hogenboom - Clinical Data Science Maastricht Maastro Clinic - June 2022

        A framework for 2D/3D registration of an x-ray fluoroscopic image in a CT-scan.

        The Core, Projector, and RigidMotion module and the SiddonGpu library were cloned or inspired by the work of:
        Fabio D'Isidoro - ETH Zurich - March 2018
        Original to be found at https://github.com/fabio86d/HipHop_2D3Dregistration/blob/master/modules/HipHop.py

    """

    def __init__(self, project_directory, patient_info, analysis_info, projector_info, camera_info):
        """
            Initialises numerous parameters and ensures output folders are present.
            Initialises projector or neural network depending on TypeOfOperation.

            :param dict project_directory: file(path) in sample data is stored
            :param dict patient_info: patient parameters formatted as:
                {PatientID, 3DModel, TrialName}
            :param dict analysis_info: analysis parameters formatted as:
                {TypeOfOperation, InitialPose, SpecificCamera, PixelType,
                ImageDimension, InputExtension, OutputExtension, ImageFilter}
            :param dict projector_info projector parameters formatted as:
                {Name, threadsPerBlock_x, threadsPerBlock_y, threadsPerBlock_z}
        """
        # retrieve sample information
        self.PatientID = patient_info['PatientID']
        self.TrialName = patient_info['TrialName']
        self.Model3D = patient_info['3Dmodel']

        # retrieve analysis settings
        self.SpecificPose = analysis_info['InitialPose']
        self.ImageFilter = analysis_info['ImageFilter']

        self.VolumeExtension = analysis_info['InputExtension']
        self.ImageExtension = analysis_info['OutputExtension']
        self.ImageBits = analysis_info['OutputBits']

        self.ImageType = itk.Image[analysis_info['PixelType'], analysis_info['ImageDimension']]

        # Initialize variables
        self.DRR_counter = 1

        # Prepare and define directories
        self.ProjectDir = project_directory
        self.PatientDir = f'{self.ProjectDir}\\{self.PatientID}'

        # ensure folder is present and repetition thereafter for different folders
        if not os.path.isdir(f'{self.PatientDir}\\fluoroscopy\\ready\\'):
            os.makedirs(f'{self.PatientDir}\\fluoroscopy\\ready\\')
        self.FluoroscopyDir = f'{self.PatientDir}\\fluoroscopy\\ready\\'

        if not os.path.isdir(f'{self.FluoroscopyDir}processed\\'):
            os.makedirs(f'{self.FluoroscopyDir}processed\\')
        self.FluoroscopyProcessDir = f'{self.FluoroscopyDir}processed\\'

        if not os.path.isdir(f'{self.PatientDir}\\digitally_reconstructed_radiographs\\'):
            os.makedirs(f'{self.PatientDir}\\digitally_reconstructed_radiographs\\')
        self.DRRDir = f'{self.ProjectDir}\\{self.PatientID}\\digitally_reconstructed_radiographs\\'

        if not os.path.isdir(f'{self.PatientDir}\\analyses\\'):
            os.makedirs(f'{self.PatientDir}\\analyses\\')
        self.AnalysesDir = f'{self.PatientDir}\\analyses\\'

        if analysis_info['TypeofOperation'] == 'preoperative' or analysis_info['TypeofOperation'] == 'optimise filter':
            self.Projector = []
            self.InitialPose = []
            self.evaluated_parameters = []
            self.current_parameter = []

            # landscape parameters
            self.range_tx = []
            self.range_ty = []
            self.range_tz = []
            self.range_rotx = []
            self.range_roty = []
            self.range_rotz = []
            self.centres = []
            self.parameters_ranges = []

            # initialise projector
            self.initialise_projector(projector_info, camera_info)

        if analysis_info['TypeofOperation'] == 'intraoperative' or \
                analysis_info['TypeofOperation'] == 'optimise filter':
            # retrieve semantic-based registration specific settings
            self.IntraOpLowThreshold = analysis_info['IntraoperativeLowerThreshold']
            self.IntraOpUpThreshold = analysis_info['IntraoperativeUpperThreshold']
            self.Extractors = analysis_info['FeatureExtractors']

            # initialise semantic-based registration and perform built-in control
            self.FluoroscopySlide = []

            t1 = time.perf_counter()
            self.feature_extractor = fe.FeatureExtractor(self.Extractors)
            t2 = time.perf_counter()
            print(f'\nExtractors initialised. Time elapsed for initialisation: {t2 - t1} seconds'
                  f'\n\n-----------------------\n'
                  f'Evaluating positive and negative control')

            self.feature_extractor_control()

    def initialise_projector(self, projector_info, camera_info):
        """
            initialises the projector using the ProjectorModule and compiled SiddonGpuLibrary

            :param dict projector_info: projector parameters formatted as:
                {Name, threadsPerBlock_x, threadsPerBlock_y, threadsPerBlock_z}
            :param dict camera_info: camera parameters formatted as:
                {focal_length, spacing_x, spacing_y, principal_point_x, principal_point_y, size_x, size_y}
        """

        # Load camera intrinsic parameters
        projector_info['focal_length'] = camera_info['focal_length']
        projector_info['DRRspacing_x'] = camera_info['spacing_x']
        projector_info['DRRspacing_y'] = camera_info['spacing_y']
        projector_info['DRR_ppx'] = camera_info['principal_point_x']
        projector_info['DRR_ppy'] = camera_info['principal_point_y']
        projector_info['DRRsize_x'] = camera_info['size_x']
        projector_info['DRRsize_y'] = camera_info['size_y']

        # Create Projector attribute of class HipHop
        model_filepath = f'{self.PatientDir}\\3D_models\\ready\\{self.PatientID}_{self.Model3D}{self.VolumeExtension}'
        self.Projector = pm.projector_factory(projector_info, model_filepath)

    def initialise_initial_pose(self, initial_pose_file, euler_sequence='zxy'):
        """
            Initializes the initial pose i.e. the centre around which the landscape is to revolve

            :param str initial_pose_file: file(path) in which the initial pose is stored
            :param str euler_sequence: the sequence of the parameters as used in the RigidMotionModule
        """

        # Load initial pose
        initial_pose_transform = np.loadtxt(initial_pose_file, delimiter=';')

        euler_zxy_g, e2 = rm.get_euler_zxy(initial_pose_transform[:3, :3])
        rotx_ip = np.deg2rad(euler_zxy_g[1])
        roty_ip = np.deg2rad(euler_zxy_g[2])
        rotz_ip = np.deg2rad(euler_zxy_g[0])

        t = initial_pose_transform[:3, 3]
        tx_ip = t[0]
        ty_ip = t[1]
        tz_ip = t[2]

        # Create dict for initial pose
        self.InitialPose = {'EulerSequence': euler_sequence, 'Rotx': rotx_ip, 'Roty': roty_ip, 'Rotz': rotz_ip,
                            'Translx': tx_ip, 'Transly': ty_ip, 'Translz': tz_ip}

        # Compute DRR and metric for ground truth pose
        transform_parameters_ip = np.array([rotx_ip, roty_ip, rotz_ip, tx_ip, ty_ip, tz_ip])

        return transform_parameters_ip

    def delete_projector(self):
        """
            Deletes the projector and clears the GPU memory it has taken up
        """

        self.Projector.delete()

    def generate_drr(self, transform_parameters, image_file_name, image_filter='None'):
        """
            Generates a DRR for the given parameters (used by default as ZXY an Euler sequence) and saves it.

            :param list transform_parameters: list of the parameters for a specific position
            :param str image_file_name: filename of the DRR
            :param str image_filter: denotes the filter that is to be applied from the ImageProcessing module
            :return itk.F image: a DRR in ITK-format
        """

        # Compute DRR
        drr = self.Projector.compute(transform_parameters)

        # Save DRR
        image_file = rw.imagewriter(drr, self.ImageType, f'{self.DRRDir}{image_file_name}', self.ImageBits,
                                    image_filter, self.ImageExtension)

        return image_file

    # noinspection PyTypeChecker
    def landscape_initialise(self, domain_parameters):
        """
            Computes the range of pose parameters for the computations of the landscape

            :param dict domain_parameters: dictionary containing a centre, width and NSamples per parameter
            :return dict parameter_range: dictionary containing the ranges per parameter
        """

        t0 = time.perf_counter()

        # Assign centres from initial pose
        domain_parameters['Translx']['Centre'] = self.InitialPose['Translx']
        domain_parameters['Transly']['Centre'] = self.InitialPose['Transly']
        domain_parameters['Translz']['Centre'] = self.InitialPose['Translz']
        domain_parameters['Rotx']['Centre'] = self.InitialPose['Rotx']
        domain_parameters['Roty']['Centre'] = self.InitialPose['Roty']
        domain_parameters['Rotz']['Centre'] = self.InitialPose['Rotz']

        # Generate ranges linspace (-width, + width, NSamples+1) making sure that the centre is included
        # translation x
        a1_tx = np.linspace(domain_parameters['Translx']['Centre'] - domain_parameters['Translx']['Width'],
                            domain_parameters['Translx']['Centre'],
                            int(round(domain_parameters['Translx']['NSamples'] / 2.)), endpoint=False)
        a2_tx = np.linspace(domain_parameters['Translx']['Centre'],
                            domain_parameters['Translx']['Centre'] + domain_parameters['Translx']['Width'],
                            int(round(domain_parameters['Translx']['NSamples'] / 2.) + 1))
        self.range_tx = np.concatenate([a1_tx, a2_tx])

        # translation y
        a1_ty = np.linspace(domain_parameters['Transly']['Centre'] - domain_parameters['Transly']['Width'],
                            domain_parameters['Transly']['Centre'],
                            int(round(domain_parameters['Transly']['NSamples'] / 2.)), endpoint=False)
        a2_ty = np.linspace(domain_parameters['Transly']['Centre'],
                            domain_parameters['Transly']['Centre'] + domain_parameters['Transly']['Width'],
                            int(round(domain_parameters['Transly']['NSamples'] / 2.) + 1))
        self.range_ty = np.concatenate([a1_ty, a2_ty])

        # translation z
        a1_tz = np.linspace(domain_parameters['Translz']['Centre'] - domain_parameters['Translz']['Width'],
                            domain_parameters['Translz']['Centre'],
                            int(round(domain_parameters['Translz']['NSamples'] / 2.)), endpoint=False)
        a2_tz = np.linspace(domain_parameters['Translz']['Centre'],
                            domain_parameters['Translz']['Centre'] + domain_parameters['Translz']['Width'],
                            int(round(domain_parameters['Translz']['NSamples'] / 2.) + 1))
        self.range_tz = np.concatenate([a1_tz, a2_tz])

        # rotation x
        a1_rx = np.linspace(domain_parameters['Rotx']['Centre'] - domain_parameters['Rotx']['Width'],
                            domain_parameters['Rotx']['Centre'],
                            int(round(domain_parameters['Rotx']['NSamples'] / 2.)), endpoint=False)
        a2_rx = np.linspace(domain_parameters['Rotx']['Centre'],
                            domain_parameters['Rotx']['Centre'] + domain_parameters['Rotx']['Width'],
                            int(round(domain_parameters['Rotx']['NSamples'] / 2.) + 1))
        self.range_rotx = np.concatenate([a1_rx, a2_rx])

        # rotation y
        a1_ry = np.linspace(domain_parameters['Roty']['Centre'] - domain_parameters['Roty']['Width'],
                            domain_parameters['Roty']['Centre'],
                            int(round(domain_parameters['Roty']['NSamples'] / 2.)), endpoint=False)
        a2_ry = np.linspace(domain_parameters['Roty']['Centre'],
                            domain_parameters['Roty']['Centre'] + domain_parameters['Roty']['Width'],
                            int(round(domain_parameters['Roty']['NSamples'] / 2.) + 1))
        self.range_roty = np.concatenate([a1_ry, a2_ry])

        # rotation z
        a1_rz = np.linspace(domain_parameters['Rotz']['Centre'] - domain_parameters['Rotz']['Width'],
                            domain_parameters['Rotz']['Centre'],
                            int(round(domain_parameters['Rotz']['NSamples'] / 2.)), endpoint=False)
        a2_rz = np.linspace(domain_parameters['Rotz']['Centre'],
                            domain_parameters['Rotz']['Centre'] + domain_parameters['Rotz']['Width'],
                            int(round(domain_parameters['Rotz']['NSamples'] / 2.) + 1))
        self.range_rotz = np.concatenate([a1_rz, a2_rz])

        # generates centres
        self.centres = np.array([domain_parameters['Rotx']['Centre'],
                                 domain_parameters['Roty']['Centre'],
                                 domain_parameters['Rotz']['Centre'],
                                 domain_parameters['Translx']['Centre'],
                                 domain_parameters['Transly']['Centre'],
                                 domain_parameters['Translz']['Centre']])
        t1 = time.perf_counter()
        print(f'Landscape initialised.\nTime elapsed for initialisation: {t1 - t0} seconds\n')

        return [self.range_rotx, self.range_roty, self.range_rotz, self.range_tx, self.range_ty, self.range_tz]

    def landscape_run(self, domain_parameters, save_info=True):
        """
            Computes a landscape relative to the initial pose

            :param dict domain_parameters: dictionary containing a centre, width and NSamples per parameter
            :param bool save_info: determines whether landscape parameters are to be exported to a csv-file
        """

        print('Generation of Landscape \n')

        # use initial pose per patient or universal for left and right side
        if self.SpecificPose:
            initial_pose_file = f'{self.ProjectDir}\\{self.PatientID}\\necessities\\pose' \
                f'\\{self.PatientID}_{self.Model3D}_initial_pose.csv'
        else:
            initial_pose_file = f'{os.getcwd()}\\necessities\\pose' \
                f'\\{self.TrialName}_{self.Model3D}_initial_pose.csv'

        initial_pose = self.initialise_initial_pose(initial_pose_file)

        angles = np.round(np.rad2deg(initial_pose[0:3]), decimals=2)
        translation = initial_pose[3:6]
        print(f'Initial pose parameters: \n Rotation x: {angles[0]} y: {angles[1]} z: {angles[2]}\n '
              f'Translation x: {translation[0]} y: {translation[1]} z: {translation[2]}')

        # Initialize landscape for current fixed image
        self.parameters_ranges = self.landscape_initialise(domain_parameters)

        # create list of available domain parameters
        parameters_to_test = list(domain_parameters.keys())
        parameter_list = parameters_to_test.copy()

        # only include parameters of which the landscape width is larger than zero
        for parameter in list(domain_parameters.keys()):
            if domain_parameters[str(parameter)]['Width'] == 0:
                parameters_to_test.remove(parameter)

        # initialise vector for parameters (at the centres)
        # important to get a copy, otherwise self.centres will be the same as self.parameters
        self.current_parameter = self.centres.copy()

        no_parameters_to_test = len(parameters_to_test)

        if no_parameters_to_test == 0:
            exit('No domain width was set for any parameter.\n'
                 'Ensure that the object has a defined space to move in, if a landscape is to be generated.\n'
                 'Exiting program.')
        else:
            # determine parameter index and range
            zero, zero_range = self.landscape_collect_parameter(0, parameters_to_test, parameter_list)

            # iterate through the set points in space
            for parameter_zero in zero_range:
                # update parameter of interest in parameter set
                self.current_parameter[zero] = parameter_zero

                # include more parameters if defined so or generate drr; repetition therefore no comments hereafter
                if no_parameters_to_test < 2:
                    self.landscape_generate_drr(save_info)
                else:
                    one, one_range = self.landscape_collect_parameter(1, parameters_to_test, parameter_list)
                    for parameter_one in one_range:
                        self.current_parameter[one] = parameter_one
                        if no_parameters_to_test < 3:
                            self.landscape_generate_drr(save_info)
                        else:
                            two, two_range = self.landscape_collect_parameter(2, parameters_to_test, parameter_list)
                            for parameter_two in two_range:
                                self.current_parameter[two] = parameter_two
                                if no_parameters_to_test < 4:
                                    self.landscape_generate_drr(save_info)
                                else:
                                    three, three_range = self.landscape_collect_parameter(3, parameters_to_test,
                                                                                          parameter_list)
                                    for parameter_three in three_range:
                                        self.current_parameter[three] = parameter_three
                                        if no_parameters_to_test < 5:
                                            self.landscape_generate_drr(save_info)
                                        else:
                                            four, four_range = self.landscape_collect_parameter(4, parameters_to_test,
                                                                                                parameter_list)
                                            for parameter_four in four_range:
                                                self.current_parameter[four] = parameter_four
                                                if no_parameters_to_test < 6:
                                                    self.landscape_generate_drr(save_info)
                                                else:
                                                    five, five_range = self.landscape_collect_parameter(5,
                                                                                                        parameters_to_test,
                                                                                                        parameter_list)
                                                    for parameter_five in five_range:
                                                        self.current_parameter[five] = parameter_five
                                                        self.landscape_generate_drr(save_info)
                                                        # number of parameters cannot exceed six thus end of nesting

        # save and close landscape info if so desired
        if save_info:
            self.landscape_save_info('save', '', None, None)

        print('Landscape successfully generated.\n')

    def landscape_collect_parameter(self, parameter_index, parameters_to_test, parameter_list):
        """
            Collects the different orientations from the initialised landscape for a specific parameter

            :param int parameter_index: index of parameter in list only containing a width higher than zero
            :param list parameters_to_test: list only containing parameters with a width higher than zero
            :param list parameter_list: complete list of parameters
            :return int parameter: index of parameter in the complete list of parameters
            :return tuple parameter_range: tuple of different orientations for the specific parameter
        """

        # select position of first parameter to test in current parameters
        parameter = parameter_list.index(str(parameters_to_test[parameter_index]))

        # collect the list of points in space in which the object is to be moved
        parameter_range = self.parameters_ranges[parameter].copy(order='C')

        # ensure that the number of samples in which the object can move is larger than one
        assert (len(parameter_range) > 1), 'Although a space for the object to move in was defined,' \
                                           'the number of samples that should be taken from this range' \
                                           ' was not provided.\nExiting program.'
        return parameter, parameter_range

    def landscape_generate_drr(self, save_info):
        """
            Pushes parameters to DRR generation function whilst transforming the parameters for info file if so desired

            :param bool save_info: determines whether transformed parameters are passed to save function
        """

        # convert and round rotation to degrees for readability
        rotation = np.round(np.rad2deg(self.current_parameter[0:3]), decimals=2)
        translation = self.current_parameter[3:6]

        print(f'Parameters: \n Rotation x: {rotation[0]} y: {rotation[1]} z: {rotation[2]}\n '
              f'Translation x: {translation[0]} y: {translation[1]} z: {translation[2]}')

        # determine filename under which the drr will be stored
        image_file_name = f'{self.PatientID}_DRR{str(self.DRR_counter)}_{self.Model3D}' \
            f'_Rx_{str(rotation[0])}_Ry_{str(rotation[1])}_Rz_{str(rotation[2])}' \
            f'_Tx_{str(translation[0])}_Ty_{str(translation[1])}_Tz_{str(translation[2])}'

        # generate drr
        image_file = self.generate_drr(self.current_parameter, image_file_name, self.ImageFilter)

        # push information to saving function
        if save_info:
            if self.DRR_counter == 1:
                self.landscape_save_info('initialise', image_file, rotation, translation)
            elif self.DRR_counter >= 2:
                self.landscape_save_info('add', image_file, rotation, translation)

        self.DRR_counter = self.DRR_counter + 1

    def landscape_save_info(self, task, image_file, angles, translation):
        """
            Saves filename, angles and translation of the image in a csv-file

            :param str task: determines whether the info file is to be created, appended or saved
            :param str image_file: filename of the DRR associated with the parameters
            :param angles: collection of angle parameters that is to be saved
            :param translation: collection of translation parameters that is to be saved
        """

        # create numpy array for data storage
        if task == 'initialise':
            self.evaluated_parameters = np.array([['Index', 'filename',
                                                   'Rotation x', 'Rotation y', 'Rotation z',
                                                   'Translation x', 'Translation y', 'Translation z',
                                                   'content loss']])

            # add initial drr to array
            self.evaluated_parameters = np.append(self.evaluated_parameters, [[self.DRR_counter, str(image_file),
                                                                               angles[0], angles[1], angles[2],
                                                                               translation[0], translation[1],
                                                                               translation[2],
                                                                               't.b.d.']], axis=0)

        # add the tested parameters to array
        if task == 'add':
            self.evaluated_parameters = np.append(self.evaluated_parameters, [[self.DRR_counter, str(image_file),
                                                                               angles[0], angles[1], angles[2],
                                                                               translation[0], translation[1],
                                                                               translation[2],
                                                                               't.b.d.']], axis=0)

        # save array as csv file
        if task == 'save':
            filename = f'{self.PatientID}_evaluated_parameters_{self.Model3D}_filter_{self.ImageFilter}'
            print(f'Evaluated parameters saved in csv file entitled: {filename}\n')
            # noinspection PyTypeChecker
            np.savetxt(f'{self.AnalysesDir}{filename}.csv', self.evaluated_parameters, delimiter=',', fmt='%s')

    def feature_extractor_control(self, positive_threshold=1, negative_threshold=10):
        """
         Initialises image pre-processing and forwards a positive and negative control to feature evaluator

        :param int positive_threshold: Maximum similarity score for positive control to pass
        :param int negative_threshold: Minimum similarity score for negative control to pass
        """
        # the image that is to be compared to a list of images is read here for direct image processing
        image = rw.imagereader(f'{os.getcwd()}\\necessities\\feature_extractor\\controls\\'
                               f'control_base_pt_1_fluoroscopy_filter_None.jpg',
                               self.ImageType)

        print('\nInitialising image processing features')
        self.feature_extractor_preprocessing(image, self.ImageFilter, f'Control')

        # forward image as array for semantic-based registration without processing to ensure proper evaluation
        # evaluate controls
        control_base = f'{os.getcwd()}\\necessities\\feature_extractor\\controls\\'\
                       f'control_base_pt_1_fluoroscopy_filter_None.jpg'

        control_positive = f'{os.getcwd()}\\necessities\\feature_extractor\\controls\\'\
                           f'control_positive_pt_1_fluoroscopy_filter_None.jpg'
        control_negative = f'{os.getcwd()}\\necessities\\feature_extractor\\controls\\'\
                           f'control_negative_RA-82042_EHBK_edit.jpg'

        # extract features for each image and store as list
        features_control_base = self.feature_extractor.extract_features(control_base)
        features_control_positive = self.feature_extractor.extract_features(control_positive)
        features_control_negative = self.feature_extractor.extract_features(control_negative)

        # compute loss of base control to positive and negative control
        content_loss_positive = self.feature_extractor.compute_loss(features_control_base, features_control_positive)
        content_loss_negative = self.feature_extractor.compute_loss(features_control_base, features_control_negative)

        if content_loss_positive <= positive_threshold:
            positive = 'passed'
        else:
            positive = 'failed'
        if content_loss_negative >= negative_threshold:
            negative = 'passed'
        else:
            negative = 'failed'

        print(f'Control successfully performed.\n'
              f'Positive control: {positive} with feature loss {content_loss_positive}, '
              f'negative control: {negative} with feature loss {content_loss_negative}'
              f'\n-----------------------\n')
        input('pauze')

    def feature_extractor_evaluate(self, drr_info_file='', exit_key='terminate', refresh_key='refresh', plot_info=True):
        """
            Forwards the files to pre-processing, feature matching network and plotting if so desired

            :param str drr_info_file: filename of file containing DRR's that are to be compared
            :param str exit_key: keyword to be used to exit evaluation loop
            :param str refresh_key: keyword to be used for refreshing the list of available fluoroscopy images
            :param bool plot_info: determines whether the evaluated data is to be plotted if so desired
        """

        # load preoperative components
        if drr_info_file == '':
            drr_info_file = f'{self.PatientID}_evaluated_parameters_{self.Model3D}_filter_{self.ImageFilter}'
        output = np.genfromtxt(f'{self.AnalysesDir}\\{drr_info_file}.csv', dtype=str, delimiter=',')

        # remove unnecessary information
        files = list(output[1:, 1])

        # create empty list of DRR features
        features_drr = []

        # extract features for each image and store as list
        t0 = time.perf_counter()
        for drr in files:
            t2 = time.perf_counter()
            features_single_drr = self.feature_extractor.extract_features(drr)
            t3 = time.perf_counter()
            features_drr.append(features_single_drr)
            print(f'Time elapsed for extraction: {t3-t2}')
        t1 = time.perf_counter()
        print(f'Total time elapsed for landscape extraction: {t1-t0}\nMean time: {(t1-t0)/len(features_drr)}')

        # retrieve slide to analyse after initialisation for enhanced intraoperative use
        fluoroscopy_slides = list(os.listdir(self.FluoroscopyDir))
        fluoroscopy_slides.remove('processed')
        print(f'Available fluoroscopy slides: \n {fluoroscopy_slides}')
        self.FluoroscopySlide = input(f'Enter the slide you would like to match\n'
                                      f'Enter: "{exit_key}" to stop the program, '
                                      f'enter "{refresh_key}" to refresh the list of fluoroscopy slides\n')

        # allow exit of while-loop with keyword
        while self.FluoroscopySlide != exit_key:
            if self.FluoroscopySlide not in fluoroscopy_slides and self.FluoroscopySlide != refresh_key:
                print(f'"{self.FluoroscopySlide}" was not found, be aware that selection is case sensitive.')
                self.FluoroscopySlide = refresh_key
            # allow updating of folder within terminal
            if self.FluoroscopySlide == refresh_key:
                fluoroscopy_slides = list(os.listdir(self.FluoroscopyDir))
                fluoroscopy_slides.remove('processed')
                print(f'\nRe-evaluating folder.\nAvailable fluoroscopy slides: \n {fluoroscopy_slides}')
                self.FluoroscopySlide = input(f'Enter the slide you would like to match\n'
                                              f'Enter: "{exit_key}" to stop the program, '
                                              f'enter "{refresh_key}" to refresh the list of fluoroscopy slides\n')

            if self.FluoroscopySlide == exit_key:
                exit()
            # include path to selected file
            fluoroscopy_file = f'{self.FluoroscopyDir}{self.FluoroscopySlide}'

            # retrieve filter that was used
            image_filter_raw = self.misc_extract_filter(output)

            # load image
            image = rw.imagereader(fluoroscopy_file, self.ImageType, 'None', self.ImageBits)

            # remove file extension for new file
            fluoroscopy_filename = f"{self.FluoroscopySlide[0: self.FluoroscopySlide.index('.')]}"

            # pass image to pre-processing
            image_filter = self.feature_extractor_preprocessing(image, image_filter_raw,
                                                                f'{self.FluoroscopyProcessDir}{fluoroscopy_filename}')

            # evaluate data
            fluoroscopy_file_full = f'{self.FluoroscopyProcessDir}{fluoroscopy_filename}_'\
                                    f'filter_{image_filter}{self.ImageExtension}'
            features_fluoroscopy = self.feature_extractor.extract_features(fluoroscopy_file_full)

            # create empty list for loss
            content_loss = []

            # evaluate loss per DRR
            t0 = time.perf_counter()
            for drr_feature in features_drr:
                loss = self.feature_extractor.compute_loss(drr_feature, features_fluoroscopy)
                content_loss.append(loss)
            t1 = time.perf_counter()
            print(f'Total time elapsed for comparison: {t1-t0}\nMean time per comparison: {(t1-t0)/len(features_drr)}')

            output[1:, 8] = content_loss

            filename = f'{self.PatientID}_evaluated_parameters_{self.Model3D}_filter_{image_filter_raw}'
            print(f'Evaluated parameters updated in csv file entitled: {filename}\n')
            # noinspection PyTypeChecker
            np.savetxt(f'{self.AnalysesDir}{filename}.csv', output, delimiter=',', fmt='%s')

            if plot_info:
                self.feature_extractor_plot(output, image_filter)

            # reset while-loop
            self.FluoroscopySlide = refresh_key

    def feature_extractor_preprocessing(self, image, image_filter_raw, filename):
        """
            Performs numerous image processing steps on the supplied image to increase similarity with a generated DRR

            :param itk.F image: ITK image that is to be processed
            :param str image_filter_raw: image filter that was used in the data that is to be analysed
            :param str filename: filename by which the processed image is to be saved
            :return data_array image_array: processed array from an ITK image
            :return str image_filter: filter that was applied
        """
        # undesirable to inverse an inverse
        if image_filter_raw == 'Inverse':
            image_filter = 'None'
        elif image_filter_raw == 'None':
            image_filter = 'Inverse'
        else:
            image_filter = image_filter_raw

        # remove surgical equipment and undesired soft tissue
        if self.IntraOpLowThreshold != 0 or self.IntraOpUpThreshold != 0:
            image = ip.threshold_filter(image, self.ImageType,
                                        self.IntraOpLowThreshold, self.IntraOpUpThreshold, self.IntraOpUpThreshold)

        # inverse image prior to further processing if desired
        if image_filter != 'None':
            # apply filter prior to processing
            image = ip.filter_image(image, self.ImageType, 'Inverse')
            # load reference DRR with a white object and a dark background
            reference_image = rw.imagereader(f'{os.getcwd()}\\necessities\\feature_extractor\\reference_image\\'
                                             f'Patient_1_DRR_filter_None.mha', self.ImageType)
        else:
            # load reference DRR with a dark object and a white background
            reference_image = rw.imagereader(f'{os.getcwd()}\\necessities\\feature_extractor\\reference_image\\'
                                             f'Patient_1_DRR_filter_Inverse.mha', self.ImageType)

        # equalise histogram for higher similarity
        image = ip.equalise_histogram(image, self.ImageType)
        image = ip.match_histogram(image, self.ImageType, reference_image)

        # avoid saving a processed control
        if filename != 'Control':
            if image_filter == 'Inverse':
                rw.imagewriter(image, self.ImageType, f'{filename}', self.ImageBits, 'None', self.ImageExtension)
            else:
                rw.imagewriter(image, self.ImageType, f'{filename}', self.ImageBits, image_filter, self.ImageExtension)
            print(f'Image preprocessing successful.\n-----------------------')
            return image_filter
        else:
            print(f'\nImage preprocessing successful.\n-----------------------')

    def feature_extractor_plot(self, data_array, image_filter):
        """
            Plots the feature evaluation output in a plot appropriate for the evaluation

            :param numpy.ndarray data_array: output array after feature evaluation
            :param str image_filter: filter that was applied to intraoperative image
        """
        # allow post-hoc data visualisation
        if not isinstance(data_array, np.ndarray):
            data_array = np.genfromtxt(f'{self.AnalysesDir}\\{data_array}.csv', dtype=str, delimiter=',')

        # retrieve filter that was used
        image_filter_drr = self.misc_extract_filter(data_array)

        if image_filter == '':
            image_filter = image_filter_drr

        if image_filter == 'None':
            image_filter = 'unfiltered'

        title = f"{self.PatientID.replace('_', ' ')} {self.Model3D.replace('_', ' ')}" \
            f"\nPreoperative - {image_filter_drr} DRR n-D landscape | " \
            f"Intraoperative - {image_filter} of {self.FluoroscopySlide[0: self.FluoroscopySlide.index('.')]}"
        filename = title.replace('|', '-')
        filename = filename.replace('\n', ' - ')

        # configure axes depending on the number of axes with variation
        axes = dv.format_axes(data_array)

        dv.select_plot(title, axes, data_array, f'{self.AnalysesDir}{filename}')

    def misc_extract_filter(self, data_array, image_extension=''):
        """
            Determines the filter that was used in the array that is passed to the function.

            :param numpy.ndarray data_array: landscape output array; n x 8 array as str
            :param str image_extension: image extension that was used in the data array, default: self.ImageExtension
            :return str image_filter: image filter as string
        """
        # force extension if not provided
        if image_extension == '':
            image_extension = self.ImageExtension

        file = data_array[1, 1]
        image_filter = file[file.index('filter_') + 7: file.index(image_extension)]

        return image_filter

    def optimise_filter(self, domain_parameters):
        """
            Sequential landscape generation and feature evaluation using all available filters

            :param dict domain_parameters: dictionary containing a centre, width and NSamples per parameter
        """
        # retrieve all currently available filters as list to maintain fixed order
        available_filters = ip.filter_info(True)

        # retrieve slide to analyse after initialisation for enhanced intraoperative use
        print(f'Available fluoroscopy slides: \n {os.listdir(self.FluoroscopyDir)}')
        self.FluoroscopySlide = input('Enter the slide you would like to match\n')

        for filters in available_filters:
            # alter filter
            self.ImageFilter = filters

            # run landscape generation
            self.landscape_run(domain_parameters)

            # reset counter
            self.DRR_counter = 1

            landscape_info = f'{self.PatientID}_evaluated_parameters_{self.Model3D}_filter_{self.ImageFilter}'
            self.feature_extractor_evaluate(self.FluoroscopySlide, landscape_info)
