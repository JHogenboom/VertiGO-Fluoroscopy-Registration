"""
    J. Hogenboom - Clinical Data Science Maastricht Maastro Clinic - June 2022

    A framework for 2D/3D registration of an x-ray fluoroscopic image in a CT-scan.
"""

from VertiGOCore import *
import os.path

ProjectDir = input('Enter the project filepath: \n')

# Input of user
print('Available patients: \n', os.listdir(ProjectDir))
patient = input('Enter the PatientID of the patient you would like to match: \n')
type_of_operation = input('Enter the type of analysis you would like to perform '
                          '("preoperative", "intraoperative"): \n')
side = input('Enter the side of the patient you would like to analyse ("left_ear" or "right_ear"): \n')

# Info patient and analysis, InitialPose
Patient_info = {'PatientID': patient,
                '3Dmodel': side,
                'TrialName': 'vertigo'}

Analysis_info = {'TypeofOperation': type_of_operation,
                 'InitialPose': False,
                 'PixelType': itk.F,
                 'ImageDimension': 2,
                 'InputExtension': '.nii',
                 'OutputExtension': '.tif',
                 'OutputBits': 8,
                 'ImageFilter': 'Inverse',
                 'IntraoperativeLowerThreshold': 0,
                 'IntraoperativeUpperThreshold': 0,
                 'FeatureExtractors': {'Xception'}}

# Define projector for generation of DRR from 3D model (Digitally Reconstructed Radiographs)
projector_info = {'Name': 'SiddonGpu',
                  'threadsPerBlock_x': 16,
                  'threadsPerBlock_y': 16,
                  'threadsPerBlock_z': 1}

camera_info = {'focal_length': 953,
               'spacing_x': 0.288,
               'spacing_y': 0.288,
               'principal_point_x': 1,
               'principal_point_y': 1,
               'size_x': 1024,
               'size_y': 1024}

domain_info = {'Rotx': {'Centre': 0, 'Width': np.deg2rad(10), 'NSamples': 20},
               'Roty': {'Centre': 0, 'Width': np.deg2rad(0), 'NSamples': 0},
               'Rotz': {'Centre': 0, 'Width': np.deg2rad(0), 'NSamples': 0},
               'Translx': {'Centre': 0, 'Width': 0, 'NSamples': 0},
               'Transly': {'Centre': 0, 'Width': 0, 'NSamples': 0},
               'Translz': {'Centre': 0, 'Width': 50, 'NSamples': 100}}

t1 = time.perf_counter()
# Create registration object
NewRegistration = VertiGOCore(ProjectDir,
                         Patient_info,
                         Analysis_info,
                         projector_info,
                         camera_info)

if type_of_operation == 'preoperative':
    t1_1 = time.perf_counter()
    NewRegistration.landscape_run(domain_info)
    t2_1 = time.perf_counter()
    print(f'Total DRR generation time: {t2_1-t1_1}')
elif type_of_operation == 'intraoperative':
    NewRegistration.feature_extractor_evaluate()
    # NewRegistration.feature_extractor_plot('G:\\Joost_(Joshi)\\\VertiGO\\Patient_1\\analysis\registration\\HipHop\\OptInfo\\Patient_1_evaluated_parameters_left_ear.csv')
elif type_of_operation == 'optimise filter':
    NewRegistration.optimise_filter(domain_info)

t2 = time.perf_counter()
print(f'Total execution time: {t2-t1} ')

