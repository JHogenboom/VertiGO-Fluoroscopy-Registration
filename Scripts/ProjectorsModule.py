"""Module for generation of Digitally Reconstructed Radiographs (DRR).

This module includes classes for generation of DRRs from either a volumetric image (CT,MRI) 
or a STL model, and a projector class factory.

Classes:
    SiddonGpu: GPU accelerated (CUDA) DRR generation from CT or MRI scan.  

Functions:
    projector_factory: returns a projector instance.
    
New projectors can be plugged-in and added to the projector factory
as long as they are defined as classes with the following methods:
    compute: returns a 2D image (DRR) as a numpy array.
    delete: eventually deletes the projector object (only needed to deallocate memory from GPU) 
"""

# Public module
import numpy as np
import time
import sys
import itk

# Private module
import ReadWriteImageModule as rw
import RigidMotionModule as rm

# Python wrapped C library for GPU accelerated DRR generation
sys.path.append('../wrapped_modules/')
from SiddonGpuPy import pySiddonGpu


def projector_factory(projector_info,
                      moving_image_file_name,
                      pixel_type=itk.F,
                      dimension=3):
    """Generates instances of the specified projectors.

    Args:
        :param projector_info: includes camera intrinsic parameters and projector-specific parameters
        :param moving_image_file_name: cost function returning the metric value
        :param dimension:
        :param pixel_type:

    Returns:
        opt: instance of the specified projector class.

    """

    if projector_info['Name'] == 'SiddonGpu':
        p = SiddonGpu(projector_info,
                      moving_image_file_name,
                      pixel_type,
                      dimension)

        return p


class SiddonGpu:
    """
       GPU accelerated DRR generation from volumetric image (CT or MRI scan).

       This class renders a DRR from a volumetric image, with an accelerated GPU algorithm
       from a Python wrapped library (SiddonGpuPy), written in C++ and accelerated with Cuda.
       Implementation is based both on the description found in the “Improved Algorithm” section in Jacob’s paper (1998):
       researchgate.net/publication/2344985_A_Fast_Algorithm_to_Calculate_the_Exact_Radiological_Path_Through_a_Pixel_Or_Voxel_Space
       and on the implementation suggested in Greef et al 2009:
       ncbi.nlm.nih.gov/pubmed/19810482

       Methods:
            compute (function): returns a 2D image (DRR) as a numpy array.
            delete (function): deletes the projector object (needed to deallocate memory from GPU)
    """

    def __init__(self, projector_info,
                 moving_image_file_name,
                 pixel_type,
                 dimension):
        """Reads the moving image and creates a siddon projector
           based on the camera parameters provided in projector_info (dict)
        """

        # ITK: Instantiate types
        self.Dimension = dimension
        self.ImageType = itk.Image[pixel_type, dimension]
        self.ImageType2D = itk.Image[pixel_type, 2]
        self.RegionType = itk.ImageRegion[dimension]

        # image of physical coordinates
        phy_image_type = itk.Image[itk.Vector[itk.F, dimension], dimension]

        # Read moving image (CT or MRI scan)
        mov_image, mov_image_info = rw.imagereader(moving_image_file_name, self.ImageType, 'Threshold', 0, True)
        self.movDirection = mov_image.GetDirection()

        # Calculate side planes
        x0 = mov_image_info['Volume_centre'][0] - mov_image_info['Spacing'][0] * mov_image_info['Size'][0] * 0.5
        y0 = mov_image_info['Volume_centre'][1] - mov_image_info['Spacing'][1] * mov_image_info['Size'][1] / 2.0
        z0 = mov_image_info['Volume_centre'][2] - mov_image_info['Spacing'][2] * mov_image_info['Size'][2] / 2.0

        # Get 1d array for moving image
        # ravel does not generate a copy of the array (it is faster than flatten)
        # mov_img_array_1d = np.ravel(itk.PyBuffer[self.ImageType].GetArrayFromImage(mov_image), order='C')
        mov_img_array_1d = np.ravel(itk.GetArrayFromImage(mov_image), order='C')

        # Set parameters for GPU library SiddonGpuPy
        num_threads_per_block = np.array([projector_info['threadsPerBlock_x'], projector_info['threadsPerBlock_y'],
                                          projector_info['threadsPerBlock_z']])
        drr_size_for_gpu = np.array([projector_info['DRRsize_x'], projector_info['DRRsize_y'], 1])
        mov_size_for_gpu = np.array([mov_image_info['Size'][0], mov_image_info['Size'][1], mov_image_info['Size'][2]])
        mov_spacing_for_gpu = np.array(
            [mov_image_info['Spacing'][0], mov_image_info['Spacing'][1], mov_image_info['Spacing'][2]]).astype(
            np.float32)

        # Define source point at its initial position (at the origin = moving image center)
        self.source = [0] * dimension
        self.source[0] = mov_image_info['Volume_centre'][0]
        self.source[1] = mov_image_info['Volume_centre'][1]
        self.source[2] = mov_image_info['Volume_centre'][2]

        # Set drr image at initial position (at +focal length along the z direction)
        drr = self.ImageType.New()
        self.DRRregion = self.RegionType()

        drr_start = itk.Index[dimension]()
        drr_start.Fill(0)

        self.DRRsize = [0] * dimension
        self.DRRsize[0] = projector_info['DRRsize_x']
        self.DRRsize[1] = projector_info['DRRsize_y']
        self.DRRsize[2] = 1

        self.DRRregion.SetSize(self.DRRsize)
        self.DRRregion.SetIndex(drr_start)

        self.DRRspacing = itk.Point[itk.F, dimension]()
        self.DRRspacing[0] = projector_info['DRRspacing_x']
        self.DRRspacing[1] = projector_info['DRRspacing_y']
        self.DRRspacing[2] = 1.

        self.DRRorigin = itk.Point[itk.F, dimension]()
        self.DRRorigin[0] = mov_image_info['Volume_centre'][0] - projector_info['DRR_ppx'] - self.DRRspacing[0] * (
                self.DRRsize[0] - 1.) / 2.
        self.DRRorigin[1] = mov_image_info['Volume_centre'][1] - projector_info['DRR_ppy'] - self.DRRspacing[1] * (
                self.DRRsize[1] - 1.) / 2.
        self.DRRorigin[2] = mov_image_info['Volume_centre'][2] + projector_info['focal_length']

        drr.SetRegions(self.DRRregion)
        drr.Allocate()
        drr.SetSpacing(self.DRRspacing)
        drr.SetOrigin(self.DRRorigin)
        self.movDirection.SetIdentity()
        drr.SetDirection(self.movDirection)

        # Get array of physical coordinates for the drr at the initial position
        physical_point_imagefilter = itk.PhysicalPointImageSource[phy_image_type].New()
        physical_point_imagefilter.SetReferenceImage(drr)
        physical_point_imagefilter.SetUseReferenceImage(True)
        physical_point_imagefilter.Update()
        source_drr = physical_point_imagefilter.GetOutput()

        # self.sourceDRR_array_to_reshape =
        # itk.PyBuffer[phy_image_type].GetArrayFromImage(source_drr)[0].copy(order = 'C')
        # array has to be reshaped for matrix multiplication
        self.sourceDRR_array_to_reshape = itk.GetArrayFromImage(source_drr)[0]
        # array has to be reshaped for matrix multiplication

        t_gpu1 = time.perf_counter()

        # Generate projector object
        self.projector = pySiddonGpu(num_threads_per_block,
                                     mov_img_array_1d,
                                     mov_size_for_gpu,
                                     mov_spacing_for_gpu,
                                     x0.astype(np.float32), y0.astype(np.float32), z0.astype(np.float32),
                                     drr_size_for_gpu)

        t_gpu2 = time.perf_counter()

        print(f'\nSiddon object initialised. Time elapsed for initialisation: {t_gpu2 - t_gpu1}\n')

    def compute(self, transform_parameters):
        """Generates a DRR given the transform parameters.

           Args:
               transform_parameters (list of floats): rotX, rotY,rotZ, transX, transY, transZ
 
        """
        t_drr1 = time.perf_counter()

        # Get transform parameters
        rotx = transform_parameters[0]
        roty = transform_parameters[1]
        rotz = transform_parameters[2]
        tx = transform_parameters[3]
        ty = transform_parameters[4]
        tz = transform_parameters[5]

        # compute the transformation matrix and its inverse (itk always needs the inverse)
        tr = rm.get_rigid_motion_mat_from_euler(rotz, 'z', rotx, 'x', roty, 'y', tx, ty, tz)
        inv_t = np.linalg.inv(tr)  # very important conversion to float32, otherwise the code crashes

        # Move source point with transformation matrix
        source_transformed = np.dot(inv_t, np.array([self.source[0], self.source[1], self.source[2], 1.]).T)[0:3]
        source_for_gpu = np.array([source_transformed[0], source_transformed[1], source_transformed[2]],
                                  dtype=np.float32)

        # Instantiate new 3D DRR image at its initial position (at +focal length along the z direction)
        new_drr = self.ImageType.New()

        new_drr.SetRegions(self.DRRregion)
        new_drr.Allocate()
        new_drr.SetSpacing(self.DRRspacing)
        new_drr.SetOrigin(self.DRRorigin)
        self.movDirection.SetIdentity()
        new_drr.SetDirection(self.movDirection)

        # Get 3d array for DRR (where to store the final output, in the image plane that in fact does not move)
        # new_drr_array = itk.PyBuffer[self.ImageType].GetArrayFromImage(new_drr)
        new_drr_array = itk.GetArrayViewFromImage(new_drr)

        # Get array of physical coordinates of the transformed DRR
        source_drr_array_reshaped = self.sourceDRR_array_to_reshape.reshape(
            (self.DRRsize[0] * self.DRRsize[1], self.Dimension), order='C')

        source_drr_array_transformed = np.dot(inv_t, rm.augment_matrix_coord(source_drr_array_reshaped))[0:3].T
        # apply inverse transform to detector plane, augmentation is needed for multiplication with rigid motion matrix

        source_drr_array_transf_to_ravel = source_drr_array_transformed.reshape(
            (self.DRRsize[0], self.DRRsize[1], self.Dimension), order='C')

        drr_phy_array = np.ravel(source_drr_array_transf_to_ravel, order='C').astype(np.float32)

        # Generate DRR
        output = self.projector.generateDRR(source_for_gpu, drr_phy_array)

        # Reshape copy
        # output_reshaped = np.reshape(output, (self.DRRsize[2], self.DRRsize[1], self.DRRsize[0]), order='C')
        # no guarantee about memory contiguity
        output_reshaped = np.reshape(output, (self.DRRsize[1], self.DRRsize[0]), order='C')
        # no guarantee about memory contiguity

        # Re-copy into original image array, hence into original image (since the former is just a view of the latter)
        new_drr_array.setfield(output_reshaped, new_drr_array.dtype)

        # Redim filter to convert the DRR from 3D slice to 2D image (necessary for further metric comparison)
        filter_redim = itk.ExtractImageFilter[self.ImageType, self.ImageType2D].New()
        filter_redim.InPlaceOn()
        filter_redim.SetDirectionCollapseToSubmatrix()

        # important, otherwise the following filterRayCast.GetOutput().GetLargestPossibleRegion() returns an empty image
        new_drr.UpdateOutputInformation()

        size_input = new_drr.GetLargestPossibleRegion().GetSize()
        start_input = new_drr.GetLargestPossibleRegion().GetIndex()

        size_output = [0] * self.Dimension
        size_output[0] = size_input[0]
        size_output[1] = size_input[1]
        size_output[2] = 0

        slice_number = 0
        start_output = [0] * self.Dimension
        start_output[0] = start_input[0]
        start_output[1] = start_input[1]
        start_output[2] = slice_number

        desired_region = self.RegionType()
        desired_region.SetSize(size_output)
        desired_region.SetIndex(start_output)

        filter_redim.SetExtractionRegion(desired_region)

        filter_redim.SetInput(new_drr)

        t_drr2 = time.perf_counter()

        filter_redim.Update()

        print(f'Time elapsed for generation of DRR: {t_drr2 - t_drr1}')

        return filter_redim.GetOutput()

    def delete(self):
        """
        Deletes the projector object >>> GPU is freed <<<
        """

        self.projector.delete()
