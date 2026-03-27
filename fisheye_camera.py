from __future__ import annotations
# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla
from carla import ColorConverter as cc
import numpy as np
import weakref
from scipy.spatial.transform import Rotation as R
import cv2
from typing import Tuple
import abc
import sys

def smoothstep(x, edge0, edge1):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

class BaseProjection(abc.ABC):
    """This is base projection class which should be inherit from."""

    @classmethod
    @abc.abstractmethod
    def from_fov(cls, width: int, height: int, fov: float,  k0: float, k1: float, k2: float, k3: float, k4: float)-> None:
        """Constructor from fov."""

    @abc.abstractmethod
    def projection(self, points3d: np.ndarray)-> np.ndarray:
        """The camera projection model from 3d points to normalized coordinates.
        
        Args:
            points3d: the 3d points array of the shape [..., 3]

        Returns:
            normalized_coordinates: with z=1 of the shape [..., 3]
        """

    @abc.abstractmethod
    def inverse_projection(self, normalized_coords: np.ndarray)-> np.ndarray:
        """Iverse projection from normalized camera coordinates to 3d rays on the unit sphere.
        
        
        Args:
            normalized_coords: normmalized camera coordinates (z=1) of the shape [..., 3]
        
        Returns:
            rays3d: 3d rays on the unit sphere of the shape [..., 3]
        """

    @abc.abstractmethod
    def from_3d_to_2d(self, points3d: np.ndarray)-> np.ndarray:
        """The camera projection from 3D to 2D image.
        
        Args:
            points3d: the 3d points array of the shape [..., 3]
        
        Returns:
            points2d: the 2d pixels positions on the image [..., 2]
        """
    

    @abc.abstractmethod
    def from_2d_to_3d(self, pixels_coords: np.ndarray)-> np.ndarray:
        """The inverse projection from 2d image to 3d space.
        

        Args:
            pixels_coords: the pixels coordinates of the shape [..., 2]
        
        Returns:
            rays3d: the unit rays of the shape [..., 3]
        """

class EquidistantProjection(BaseProjection):
    """This is implementation of the fish eye equidistant camera projection.

    Note: The derived model equasions could be found in paper:
        Steffen Abraham, Wolfgang Förstner. Fish-Eye-Stereo Calibration and Epipolar Rectification, (2005)
        http://www.ipb.uni-bonn.de/pdfs/Steffen2005Fish.pdf
    
    Args:
        fx: focal length in x direction
        fy: focal lenght in y direction
        cx: principle point x coordinate
        cy: principle point y coordinate
    """

    def __init__(self, fx: float, fy: float, cx: float, cy: float, k0: float, k1: float, k2: float, k3: float, k4: float)-> None:
        """Init."""
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.intrinsic_matrix = np.asarray([[fx, 0.0, cx],
                                            [0.0, fy, cy],
                                            [0.0, 0.0, 1.0]])
        
        self.inv_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)
        self.lense_distortion = LenseDistortion(fx=fx, fy=fy, k0=k0, k1=k1, k2=k2, k3=k3, k4=k4)

    @classmethod
    def from_fov(cls, width: int, height: int, fov: float,  k0: float, k1: float, k2: float, k3: float, k4: float)-> None:
        """Creates calibration from fov.
        
          Note: estimate calibration for the equidistant projection
        The relation is r = f * theta
        Here r = width / 2
        theta = np.deg2rad(FOV / 2.0 ) or FOV * pi /360
        for more info see: 
        https://www.researchgate.net/publication/6899685_A_Generic_Camera_Model_and_Calibration_Method_for_Conventional_Wide-Angle_and_Fish-Eye_Lenses

        Args:
            width: width of the image
            height: height of the image
            fov: horizontal field of view
        """

        calibration = np.identity(3)
        calibration[0, 2] = float(width) / 2.0 
        calibration[1, 2] = float(height) / 2.0 
        calibration[0, 0] =calibration[1, 1] =  float(width) / (2.0 * float(fov) * np.pi / 360.0)
        return cls(fx=calibration[0,0], fy=calibration[1,1], cx=calibration[0,-1], cy=calibration[1,-1], k0=k0, k1=k1, k2=k2, k3=k3, k4=k4)


    def projection(self, points3d: np.ndarray)-> np.ndarray:
        """The camera projection model from 3d points to normalized coordinates.
        
        Args:
            points3d: the 3d points array of the shape [..., 3]

        Returns:
            normalized_coordinates: with z=1 of the shape [..., 3]
        """
        shape = points3d.shape
    
        if shape[-1] != 3:
            raise ValueError(f"Incorrect channels shape should be 3 but it is {shape[-1]}!")

        points3d = points3d.reshape(-1, 3)

        # First find sqrt(X**2 + Y**2)
        norm_xy = np.linalg.norm(points3d[:, :2], axis=-1)

        # Find theta angle between optical axis and ray
        theta = np.arctan2(norm_xy, points3d[:, 2])
        # Compute normalized coordinates here we add small epsilon to avoid division by zero
        x_normalized_coordinate = points3d[:, 0] * theta / (norm_xy + sys.float_info.epsilon) 
        y_normalized_coordinate = points3d[:, 1] * theta/ (norm_xy + sys.float_info.epsilon)

        ones = np.ones_like(x_normalized_coordinate)
        # Here homogeneous coordinates of the shape  [..., 3]
        normalized_coordinates = np.concatenate((x_normalized_coordinate[:, None], y_normalized_coordinate[:, None], ones[:, None]), axis=-1)
        return normalized_coordinates.reshape(shape)


    def inverse_projection(self, normalized_coords: np.ndarray)-> np.ndarray:
        """Iverse projection from normalized camera coordinates to 3d rays on the unit sphere.
        
        
        Args:
            normalized_coords: normmalized camera coordinates (z=1) of the shape [..., 3]
        
        Returns:
            rays3d: 3d rays on the unit sphere of the shape [..., 3]
        """
        shape = normalized_coords.shape
    
        if shape[-1] != 3:
            raise ValueError(f"Incorrect channels shape should be 3 but it is {shape[-1]}!")
        # Get sqrt(x_norm**2 + y_norm**2)
        theta = np.linalg.norm(normalized_coords[:, :2], axis=-1)
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        X = normalized_coords[:, 0] * sin_t/ (theta + sys.float_info.epsilon)
        Y = normalized_coords[:, 1] * sin_t  / (theta + sys.float_info.epsilon)
        Z = cos_t

        # begin new
        # phi = np.pi/2 - theta
        # lam = np.arctan2(normalized_coords[:, 1], normalized_coords[:, 0])
        # X = np.cos(phi) * np.cos(lam)
        # Y = -np.cos(phi) * np.sin(lam)
        # Z = np.sin(phi)
        # end new

        rays3d = np.concatenate((X[:, None], Y[:, None], Z[:, None]), axis=-1)
        return rays3d.reshape(shape)


    def from_3d_to_2d(self, points3d: np.ndarray)-> np.ndarray:
        """The camera projection from 3D to 2D image.
        
        Args:
            points3d: the 3d points array of the shape [..., 3]
        
        Returns:
            points2d: the 2d pixels positions on the image [..., 2]
        """
       
        # 1. Apply fisheye projection
        # Aplly equidistant projection to get normalized camera coordinates (z=1)
        homogeneous_coords = self.projection(points3d)
        shape = homogeneous_coords.shape
        homogeneous_coords = homogeneous_coords.reshape(-1, 3)

        # 2. Apply distortion
        undist_x, undist_y, ones = homogeneous_coords[:, 0], homogeneous_coords[:, 1], homogeneous_coords[:, 2]
        dist_x, dist_y =  self.lense_distortion.distortion(undist_x, undist_y)
        # Here distorted homoheneous coords z=1 of the shape [3, ...]
        dist_homogeneous_coords = np.concatenate(( dist_x[None, :], dist_y[None, :], ones[None, :]), axis=0)

        # 3. Apply pinhole projection
        # Here pixels coordinates of the shape [..., 3]
        pixels_coords = (self.intrinsic_matrix @ dist_homogeneous_coords).T
        pixels_coords = np.reshape(pixels_coords, shape)

        return pixels_coords[..., :2]


    def from_2d_to_3d(self, pixels_coords: np.ndarray)-> np.ndarray:
        """The inverse projection from 2d image to 3d space.
        
        Note: to get original 3d point cloud you need to multiply final rays by depth.
        e.g. rays3d * Depth where Depth = sqrt(X**2 + Y**2 + Z**2) and X, Y, Z original position
        of the 3d point.

        Args:
            pixels_coords: the pixels coordinates of the shape [..., 2]
        
        Returns:
            rays3d: the unit rays of the shape [..., 3]
        """
        shape = pixels_coords.shape
    
        if shape[-1] != 2:
            raise ValueError(f"Incorrect channels shape should be 2 but it is {shape[-1]}!")
        
        # 1. Here inverse pinhole projection
        pixels_coords = pixels_coords.reshape(-1, 2)
        # Homogeneous coordinates of the shape [3, ...]
        homogeneous_coords = np.concatenate((pixels_coords, np.ones_like(pixels_coords[..., 0][..., None])), axis=-1).T
        #  Normalized coordinates of the shape [..., 3]
        normalized_coords = (self.inv_intrinsic_matrix @ homogeneous_coords).T
        
        # 2. Apply undistortion 
        dist_x, dist_y, ones = normalized_coords[:, 0], normalized_coords[:, 1], normalized_coords[:, 2]
        undist_x, undist_y = self.lense_distortion.undistortion(dist_x, dist_y)
        undist_homogeneous_coords = np.concatenate(( undist_x[:, None], undist_y[:, None], ones[:, None]), axis=-1)
        undist_homogeneous_coords = undist_homogeneous_coords.reshape((*shape[:-1], 3))

        # 3. Apply inverse fisheye projection
        rays3d = self.inverse_projection(undist_homogeneous_coords)
        return rays3d

class StereographicProjection(BaseProjection):
    """This is implementation of the fish eye stereographic camera projection.

    Note: The derived model equasions could be found in paper:
        Steffen Abraham, Wolfgang Förstner. Fish-Eye-Stereo Calibration and Epipolar Rectification, (2005)
        http://www.ipb.uni-bonn.de/pdfs/Steffen2005Fish.pdf
    
    Args:
        fx: focal length in x direction
        fy: focal lenght in y direction
        cx: principle point x coordinate
        cy: principle point y coordinate
    """

    def __init__(self, fx: float, fy: float, cx: float, cy: float, k0: float, k1: float, k2: float, k3: float, k4: float)-> None:
        """Init."""
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.intrinsic_matrix = np.asarray([[fx, 0.0, cx],
                                            [0.0, fy, cy],
                                            [0.0, 0.0, 1.0]])
        
        self.inv_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)
        self.lense_distortion = LenseDistortion(fx=fx, fy=fy, k0=k0, k1=k1, k2=k2, k3=k3, k4=k4)

    @classmethod
    def from_fov(cls, width: int, height: int, fov: float,  k0: float, k1: float, k2: float, k3: float, k4: float)-> None:
        """Creates calibration from fov.
        
          Note: estimate calibration for the stereographic projection
        The relation is r = f * tan(theta /2)
        Here r = width / 2
        theta = r /  np.tan(np.deg2rad(FOV / 4.0 )) 
        for more info see: 
        https://www.researchgate.net/publication/6899685_A_Generic_Camera_Model_and_Calibration_Method_for_Conventional_Wide-Angle_and_Fish-Eye_Lenses

        Args:
            width: width of the image
            height: height of the image
            fov: horizontal field of view
        """

        calibration = np.identity(3)
        calibration[0, 2] = float(width) / 2.0 
        calibration[1, 2] = float(height) / 2.0 
        calibration[0, 0] = calibration[1, 1] =  float(width) / (2.0 * np.tan(np.deg2rad(float(fov) / 4.0 ))) 
        return cls(fx=calibration[0,0], fy=calibration[1,1], cx=calibration[0,-1], cy=calibration[1,-1], k0=k0, k1=k1, k2=k2, k3=k3, k4=k4)


    def projection(self, points3d: np.ndarray)-> np.ndarray:
        """The camera projection model from 3d points to normalized coordinates.
        
        Args:
            points3d: the 3d points array of the shape [..., 3]

        Returns:
            normalized_coordinates: with z=1 of the shape [..., 3]
        """
        shape = points3d.shape
    
        if shape[-1] != 3:
            raise ValueError(f"Incorrect channels shape should be 3 but it is {shape[-1]}!")

        points3d = points3d.reshape(-1, 3)

        # First find sqrt(X**2 + Y**2 + Z**2) + Z
        denominator = np.linalg.norm(points3d, axis=-1) +  points3d[:, -1]

        # Compute normalized coordinates
        x_normalized_coordinate = points3d[:, 0] / denominator
        y_normalized_coordinate = points3d[:, 1] / denominator
        ones = np.ones_like(x_normalized_coordinate)
        # Here homogeneous coordinates of the shape  [..., 3]
        normalized_coordinates = np.concatenate((x_normalized_coordinate[:, None], y_normalized_coordinate[:, None], ones[:, None]), axis=-1)
        return normalized_coordinates.reshape(shape)


    def inverse_projection(self, normalized_coords: np.ndarray)-> np.ndarray:
        """Iverse projection from normalized camera coordinates to 3d rays on the unit sphere.
        
        
        Args:
            normalized_coords: normmalized camera coordinates (z=1) of the shape [..., 3]
        
        Returns:
            rays3d: 3d rays on the unit sphere of the shape [..., 3]
        """
        shape = normalized_coords.shape
    
        if shape[-1] != 3:
            raise ValueError(f"Incorrect channels shape should be 3 but it is {shape[-1]}!")

        # Get x**2 + y**2
        squared_xy = normalized_coords[:, 0]** 2 + normalized_coords[:, 1] ** 2

        X = 2.0 * normalized_coords[:, 0] / (1.0  +  squared_xy)
        Y = 2.0 *  normalized_coords[:, 1]  / (1.0 + squared_xy)
        Z = (1.0 - squared_xy) /  (1.0 + squared_xy)

        rays3d = np.concatenate((X[:, None], Y[:, None], Z[:, None]), axis=-1)
        return rays3d.reshape(shape)


    def from_3d_to_2d(self, points3d: np.ndarray)-> np.ndarray:
        """The camera projection from 3D to 2D image.
        
        Args:
            points3d: the 3d points array of the shape [..., 3]
        
        Returns:
            points2d: the 2d pixels positions on the image [..., 2]
        """
       
        # 1. Apply fisheye projection
        # Aplly equidistant projection to get normalized camera coordinates (z=1)
        homogeneous_coords = self.projection(points3d)
        shape = homogeneous_coords.shape
        homogeneous_coords = homogeneous_coords.reshape(-1, 3)

        # 2. Apply distortion
        undist_x, undist_y, ones = homogeneous_coords[:, 0], homogeneous_coords[:, 1], homogeneous_coords[:, 2]
        dist_x, dist_y =  self.lense_distortion.distortion(undist_x, undist_y)
        # Here distorted homoheneous coords z=1 of the shape [3, ...]
        dist_homogeneous_coords = np.concatenate(( dist_x[None, :], dist_y[None, :], ones[None, :]), axis=0)

        # 3. Apply pinhole projection
        # Here pixels coordinates of the shape [..., 3]
        pixels_coords = (self.intrinsic_matrix @ dist_homogeneous_coords).T
        pixels_coords = np.reshape(pixels_coords, shape)

        return pixels_coords[..., :2]


    def from_2d_to_3d(self, pixels_coords: np.ndarray)-> np.ndarray:
        """The inverse projection from 2d image to 3d space.
        
        Note: to get original 3d point cloud you need to multiply final rays by depth.
        e.g. rays3d * Depth where Depth = sqrt(X**2 + Y**2 + Z**2) and X, Y, Z original position
        of the 3d point.

        Args:
            pixels_coords: the pixels coordinates of the shape [..., 2]
        
        Returns:
            rays3d: the unit rays of the shape [..., 3]
        """
        shape = pixels_coords.shape
    
        if shape[-1] != 2:
            raise ValueError(f"Incorrect channels shape should be 2 but it is {shape[-1]}!")
        
        # 1. Here inverse pinhole projection
        pixels_coords = pixels_coords.reshape(-1, 2)
        # Homogeneous coordinates of the shape [3, ...]
        homogeneous_coords = np.concatenate((pixels_coords, np.ones_like(pixels_coords[..., 0][..., None])), axis=-1).T
        #  Normalized coordinates of the shape [..., 3]
        normalized_coords = (self.inv_intrinsic_matrix @ homogeneous_coords).T
        
        # 2. Apply undistortion 
        dist_x, dist_y, ones = normalized_coords[:, 0], normalized_coords[:, 1], normalized_coords[:, 2]
        undist_x, undist_y = self.lense_distortion.undistortion(dist_x, dist_y)
        undist_homogeneous_coords = np.concatenate(( undist_x[:, None], undist_y[:, None], ones[:, None]), axis=-1)
        undist_homogeneous_coords = undist_homogeneous_coords.reshape((*shape[:-1], 3))

        # 3. Apply inverse fisheye projection
        rays3d = self.inverse_projection(undist_homogeneous_coords)
        return rays3d

class LenseDistortion:
    """Lense distortion.
    
    Note: see for mode details
        https://people.cs.rutgers.edu/~elgammal/classes/cs534/lectures/CameraCalibration-book-chapter.pdf


    Args:
        fx: focal length in x direction
        fy: focal length in y direction
        k0: radial distortion first coefficient
        k1: radial distortion second coefficient
        k2: tangential distortion (decentering distortion) first coefficient
        k3: tangential distortion (decentering distortion) second coefficient
        k4: radial distortion third coefficient
    """

    def __init__(self, fx: np.float64, fy: np.float64, k0: np.float64, k1: np.float64, k2: np.float64, k3: np.float64, k4: np.float64, max_iterations: int = 100, max_delta: np.float64 = 0.001)-> None:
        """Init."""
        self.k0 = k0
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4


        self.squared_fx = fx ** 2
        self.squared_fy = fy ** 2

        self.max_iterations = max_iterations
        self.max_delta = max_delta

    def distortion(self,  undist_x: np. ndarray, undist_y : np.ndarray)-> Tuple[np.ndarray,np.ndarray] :
        """Apply distortion on the undistorted set of rays.

        Args:
            undist_x: undistorted set of rays x coordinates in camera coordinates where z=1.
            undist_y:  undistorted set of rays y coordinates in camera coordinates where z=1.

        Returns:
            dist_x: distorted set of rays in x coordinates in camera coordinates where z=1.
            dist_y: distorted set of rays in y coordinates in camera coordinates where z=1.
        """

        undist_x = undist_x.astype(np.float64)
        undist_y = undist_y.astype(np.float64)

        x2=undist_x * undist_x
        y2=undist_y * undist_y
        r2=x2 + y2
        r4=r2*r2
        r6=r4*r2

        dist_x = undist_x * (1.0 + self.k0*r2 + self.k1*r4 + self.k4*r6) + 2.0 * self.k2 * undist_x * undist_y + self.k3 * (r2 + 2.0 * x2)
        dist_y = undist_y * (1.0 + self.k0*r2 + self.k1*r4 + self.k4*r6) + 2.0 * self.k3 * undist_x * undist_y + self.k2 * (r2 + 2.0 * y2) 


        return dist_x, dist_y
    
    def undistortion(self, dist_x:  np.ndarray, dist_y : np.ndarray)-> Tuple[np.ndarray,np.ndarray] :
        """Apply undistortion on distorted set of rays.
        
        Note: Due to complexity to derive close form solution from high order polynomial,
        here using Newton method instead.
        
        Args:
            dist_x: distorted set of rays in x coordinates in camera coordinates where z=1.
            dist_y: distorted set of rays in y coordinates in camera coordinates where z=1.
        
        Returns:
            undist_x: undistorted set of rays x coordinates in camera coordinates where z=1.
            undist_y:  undistorted set of rays y coordinates in camera coordinates where z=1.
        """
        dist_x = dist_x.astype(np.float64)
        dist_y = dist_y.astype(np.float64)
        
        undist_x, undist_y = dist_x.copy(), dist_y.copy()

        num_iterations = 0.0
        delta = np.ones_like(undist_x) +  self.max_delta

        while (num_iterations < self.max_iterations) & (delta > self.max_delta).any():
            mask = delta > self.max_delta

            updated_x, updated_y = self.distortion(undist_x, undist_y)

            delta_x = (updated_x - dist_x)
            delta_y = (updated_y - dist_y)


            undist_x[mask] -=  delta_x[mask] 
            undist_y[mask] -=  delta_y[mask] 

            # compute delta in pixels coordinates
            delta[mask] = (delta_x * delta_x * self.squared_fx + delta_y * delta_y  * self.squared_fy)[mask]
            num_iterations += 1.0
        return undist_x, undist_y

def process_image(image: carla.libcarla.Image)-> np.ndarraty:
    """ The callback function which gets raw image and convert it to an array."""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    return array[:, :, ::-1] 


class PinholeCamera:
    """Simulate simple pinhole camera in carla.
 
    Args:
        parent_actor: vehicle actor to attach the camera
        width: width of the image
        height: height of the image
        fov: field of view in degrees
        tick: simulation seconds between sensor captures (ticks).
        x: x position with respect to the ego vehicle in meters
        y: y position with respect to the ego vehicle in meters
        z: z position with respect to the ego vehicle in meters
        roll: roll angle in degrees
        pitch: pitch angle in degrees
        yaw: yaw angle in degrees
        camera_type: can be: 'sensor.camera.rgb', 'sensor.camera.semantic_segmentation' or 'sensor.camera.depth'
    """
    def __init__(self, parent_actor: carla.Actor, width: int, height: int, fov: int=90, tick: float=0.0,
                 x: float=-6.5, y:float=0.0, z:float=2.7,
                 roll:float=0.0, pitch:float=0.0, yaw:float=0.0,
                 camera_type: str ='sensor.camera.rgb')-> None:
        """Init."""
        if camera_type not in [ 'sensor.camera.rgb', 'sensor.camera.semantic_segmentation', 'sensor.camera.depth']:
            raise ValueError(f"Camera type {camera_type} is not supported!")
        # Carla related parameters
        self._parent = parent_actor
        # Visualization related parameters
        self.camera_type = camera_type
        self.image = None
        self.frame = 0
        # Set up the sensor
        blueprint = self._parent.get_world().get_blueprint_library().find(camera_type)
        # Modify the attributes of the blueprint to set image resolution and field of view.
        blueprint.set_attribute('image_size_x', str(width))
        blueprint.set_attribute('image_size_y', str(height))
        blueprint.set_attribute('fov', str(fov))
        # Set the time in seconds between sensor captures
        blueprint.set_attribute('sensor_tick', str(tick))

        # Provide the position of the sensor relative to the vehicle.
        transform  = carla.Transform(carla.Location(x=x, y=y, z=z),
                                        carla.Rotation(roll=roll, pitch=pitch, yaw=yaw))

        # Tell the world to spawn the sensor, don't forget to attach it to your vehicle actor.
        self.sensor = self._parent.get_world().spawn_actor(blueprint, transform, attach_to=self._parent)

        # Estimate intrinsic matrix for the camera
        # For pinhole camera the relation:
        # r = f * tan(theta)
        # theta is angle between principle point and incoming ray
        # r is the distance from principle point to incoming ray
        # r = width / 2.0; theta = np.deg2rad(FOV / 2.0) or FOV * pi /360
        # for more info see: 
        # https://www.researchgate.net/publication/6899685_A_Generic_Camera_Model_and_Calibration_Method_for_Conventional_Wide-Angle_and_Fish-Eye_Lenses
        calibration = np.identity(3)
        calibration[0, 2] = float(width) / 2.0 
        calibration[1, 2] = float(height) / 2.0 
        calibration[0, 0] = calibration[1, 1] = float(width) / (2.0 * np.tan(float(fov) * np.pi / 360.0))
        self.sensor.calibration = calibration

        # We need to pass the lambda a weak reference to self to avoid
        # circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: self._parse_image(weak_self, image))

    def destroy(self)-> None:
        """Destroys camera."""
        self.sensor.destroy()

    @staticmethod
    def _parse_image(weak_self: PinholeCamera, image: carla.libcarla.Image)-> None:
        """Parse image and postprocess it."""
        self = weak_self()
        if not self:
            return
        if self.camera_type=='sensor.camera.depth':
            image.convert(cc.LogarithmicDepth) #  'Camera Depth (Logarithmic Gray Scale)'     
        elif self.camera_type== 'sensor.camera.semantic_segmentation':
            image.convert(cc.CityScapesPalette)
        else:         
            image.convert(cc.Raw)
            
        self.image = process_image(image)
        self.frame += 1



class FisheyeCamera:
    """ FisheyeCamera class that simulates equidistant projection fish eye camera.
    
    Args:
        parent_actor: parent carla actor (e.g. vehicle) to attach the camera
        width: image width
        height: image height
        fov: field of view in degrees
        tick: simulation seconds between sensor captures (ticks=0.0 maximum possible).
        x: x position with respect to the ego vehicle in meters
        y: y position with respect to the ego vehicle in meters
        z: z position with respect to the ego vehicle in meters
        roll: roll angle in degrees
        pitch: pitch angle in degrees
        yaw: yaw angle in degrees
        camera_type: can be: 'sensor.camera.rgb', 'sensor.camera.semantic_segmentation' or 'sensor.camera.depth'
    """
    def __init__(self, parent_actor: carla.Actor, camera_model: BaseProjection, width: int=640, height: int=640, tick:float=0.0,
                 x: float=-6.5, y: float=0.0, z:float=2.7, roll:float=0.0, pitch:float=0.0, yaw: float=0.0, fx=0, fy=0, cx=0, cy=0, k0: float=0.0, k1: float=0.0, k2: float=0.0, k3: float=0.0, k4: float=0.0,
                 max_angle=0, camera_type='sensor.camera.rgb')-> None:
        # Carla parameters
        self._parent = parent_actor # vehicle where camera will be attached
        self.image = None
        self._five_pinhole_image = None
        self.frame = 0
      
        # initialize fisheye camera projection
        # self.projection_model = camera_model.from_fov(width=width, height=height, fov=240, k0=k0, k1=k1, k2=k2, k3=k3, k4=k4)
        self.projection_model = camera_model(fx=fx, fy=fy, cx=cx, cy=cy, k0=k0, k1=k1, k2=k2, k3=k3, k4=k4)

        # Create cube from 5 pinhole cameras for reprojection to fish eye

        # We create pinhole with the same focal length as fish eye camera and FOV = 90
        # From the formula r = f * tan(theta) we can get 
        #  width / 2 = f * tan(FOV/2) = f * tan(45 deg); width = 2.0 * f (tan(45 deg) = 1), also we assume width = height
        pinhole_width = int(2.0 *  self.projection_model.fx)
        pinhole_height = int(2.0 *  self.projection_model.fy)

        # initialize all cameras
        main_rot = R.from_euler('xyz',[roll, pitch, yaw], degrees=True).as_matrix()
        self._front_pinhole = PinholeCamera(self._parent, width=pinhole_width, height=pinhole_height, fov=90, tick=tick,
                 x=x, y=y, z=z,roll=roll, pitch=pitch, yaw=yaw,camera_type=camera_type)

        # First we rotate camera 90 degrees to the left
        # Then chain it to the main rotation to the vehicle
        left_local_rot =  R.from_euler('xyz',[0.0, 0.0, -90], degrees=True).as_matrix()
        left_rot = R.from_matrix( main_rot @ left_local_rot).as_euler('xyz', degrees=True)
        self._left_pinhole = PinholeCamera(self._parent, width=pinhole_width, height=pinhole_height, fov=90, tick=tick,
                 x=x, y=y, z=z,roll=left_rot[0], pitch=left_rot[1], yaw=left_rot[2],camera_type=camera_type)

        # Second we rotate camera 90 degrees to the right
        # Then chain it to the main rotation to the vehicle
        right_local_rot =  R.from_euler('xyz',[0.0, 0.0, 90], degrees=True).as_matrix()
        right_rot = R.from_matrix(main_rot @ right_local_rot).as_euler('xyz', degrees=True)
        self._right_pinhole = PinholeCamera(self._parent, width=pinhole_width, height=pinhole_height, fov=90, tick=tick,
                 x=x, y=y, z=z,roll=right_rot[0], pitch=right_rot[1], yaw=right_rot[2],camera_type=camera_type)

        # Third we rotate camera 90 degrees to the top
        # Then chain it to the main rotation to the vehicle
        top_local_rot =  R.from_euler('xyz',[0.0, 90.0, 0.0], degrees=True).as_matrix()
        top_rot = R.from_matrix(main_rot @ top_local_rot).as_euler('xyz', degrees=True)
        self._top_pinhole = PinholeCamera(self._parent, width=pinhole_width, height=pinhole_height, fov=90, tick=tick,
                 x=x, y=y, z=z,roll=top_rot[0], pitch=top_rot[1], yaw=top_rot[2],camera_type=camera_type)

        # Fourth we rotate camera 90 degrees to the bottom
        # Then chain it to the main rotation to the vehicle
        bottom_local_rot =  R.from_euler('xyz',[0.0, -90.0, 0.0], degrees=True).as_matrix()
        bottom_rot = R.from_matrix(main_rot @ bottom_local_rot).as_euler('xyz', degrees=True)
        self._bottom_pinhole = PinholeCamera(self._parent, width=pinhole_width, height=pinhole_height, fov=90, tick=tick,
                 x=x, y=y, z=z,roll=bottom_rot[0], pitch=bottom_rot[1], yaw=bottom_rot[2],camera_type=camera_type)
        

        # For all 5 image matrices intrinsc will be the same therefore will take from one
        self.pinhole_intrisic_matrix = self._front_pinhole.sensor.calibration

        # Compute mapping
        self.maptable = self.compute_mapping(fisheye_width=width, fisheye_height=height, projection_model=self.projection_model, pinhole_intrisic_matrix=self.pinhole_intrisic_matrix)
        
        # Compute circular mask
        self.circular_mask = circular_mask = np.zeros((height, width), dtype=np.float64)
        for y in range(height):
            for x in range(width):
                r = np.sqrt(((x - cx - .5) / fx) ** 2 + ((y - cy - .5) / fy) ** 2)
                theta = r
                theta_f = max_angle * np.pi / 360.0
                circular_mask[y, x] = max(0, 1 - smoothstep(theta, theta_f - .2, theta_f))


    def compute_mapping(self, fisheye_width: int, fisheye_height: int, projection_model: BaseProjection, pinhole_intrisic_matrix: np.ndarray)-> np.ndarray:
        """Compute mapping for inverse warping between 5 pinhole to fish eye."""

        # Get image coordinates
        y, x = np.meshgrid(range(fisheye_height), range(fisheye_width), indexing='ij')

        # Here pixels coords of the shape [height, width, 2]
        fisheye_image_coords = np.concatenate((x[..., None], y[..., None]), axis=-1)
        shape = fisheye_image_coords.shape
        fisheye_image_coords = fisheye_image_coords.reshape(-1, 2)
        maptable = np.zeros_like(fisheye_image_coords).T

        fisheye_rays = projection_model.from_2d_to_3d(fisheye_image_coords)
        fisheye_rays = fisheye_rays.T  

        # Get coords from front given the fact that front camera FOV 90 for horizontal and vertical
        # we can compute the x and y coords        
        pinhole_width =  int(2.0 * projection_model.fx)
        pinhole_height = int(2.0 * projection_model.fy)

        # Front camera 
        front_camera_mask = np.ones((shape[0]*shape[1])).astype(np.bool_)
        front_camera_mask, front_cam_img_coords = self.get_coordinates_for_five_pinhole_image(fisheye_rays=fisheye_rays, pinhole_width=pinhole_width, pinhole_height=pinhole_height, pinhole_intrisic_matrix=pinhole_intrisic_matrix, camera_mask=front_camera_mask, camera_direction="front")
        maptable[:, front_camera_mask] = front_cam_img_coords

        # Left camera
        left_camera_mask = (fisheye_image_coords[:, 0] <=fisheye_width / 2.0)
        left_camera_mask, left_cam_img_coords = self.get_coordinates_for_five_pinhole_image(fisheye_rays=fisheye_rays, pinhole_width=pinhole_width, pinhole_height=pinhole_height, pinhole_intrisic_matrix=pinhole_intrisic_matrix, camera_mask=left_camera_mask, camera_direction="left")
        maptable[:, left_camera_mask] = left_cam_img_coords

        # Right camera  
        right_camera_mask =  (fisheye_image_coords[:, 0] >fisheye_width / 2.0)
        right_camera_mask, right_cam_img_coords = self.get_coordinates_for_five_pinhole_image(fisheye_rays=fisheye_rays, pinhole_width=pinhole_width, pinhole_height=pinhole_height, pinhole_intrisic_matrix=pinhole_intrisic_matrix, camera_mask=right_camera_mask,camera_direction="right")
        maptable[:, right_camera_mask] = right_cam_img_coords

        # Top camera
        top_camera_mask = (fisheye_image_coords[:, 1] <=fisheye_height /2.0) 
        top_camera_mask, top_cam_img_coords = self.get_coordinates_for_five_pinhole_image(fisheye_rays=fisheye_rays, pinhole_width=pinhole_width, pinhole_height=pinhole_height, pinhole_intrisic_matrix=pinhole_intrisic_matrix, camera_mask=top_camera_mask, camera_direction="top")
        maptable[:, top_camera_mask] = top_cam_img_coords

        # Bottom camera
        bottom_camera_mask = (fisheye_image_coords[:, 1] >fisheye_height /2.0) 
        bottom_camera_mask, bottom_cam_img_coords = self.get_coordinates_for_five_pinhole_image(fisheye_rays=fisheye_rays, pinhole_width=pinhole_width, pinhole_height=pinhole_height, pinhole_intrisic_matrix=pinhole_intrisic_matrix, camera_mask=bottom_camera_mask, camera_direction="bottom")
        maptable[:, bottom_camera_mask] = bottom_cam_img_coords
        return maptable.T.reshape(shape).astype(np.float32)

    def get_coordinates_for_five_pinhole_image(self, fisheye_rays: np.ndarray, pinhole_width: int, pinhole_height: int, pinhole_intrisic_matrix: np.ndarray, camera_mask: str, camera_direction: str, margin: float = 1.5)-> Tuple[np.ndarray, np.ndarray]:
        """Gets coordinates for the box image for given camera in a maptable."""

        if camera_direction == "front":
            cam_transform = np.eye(3)
            box_idx = np.asarray([2 * pinhole_width, 0.0])[:, None]
        if camera_direction == "left":
            cam_transform = R.from_euler('xyz',[0.0, 90, 0.0], degrees=True).as_matrix()
            box_idx = np.asarray([0.0, 0.0])[:, None]

        if camera_direction == "right":
            cam_transform = R.from_euler('xyz',[0.0, -90, 0.0], degrees=True).as_matrix()
            box_idx = np.asarray([4 * pinhole_width, 0.0])[:, None]

        if camera_direction == "top":
            cam_transform = R.from_euler('xyz',[-90, 0.0, 0.0], degrees=True).as_matrix()
            box_idx = np.asarray([pinhole_width, 0.0])[:, None]
    
        if camera_direction == "bottom":
            cam_transform = R.from_euler('xyz',[90, 0.0, 0.0], degrees=True).as_matrix()
            box_idx = np.asarray([3 * pinhole_width, 0.0])[:, None]

        fisheye_rays = fisheye_rays[:, camera_mask].copy()

        transform = pinhole_intrisic_matrix @ cam_transform
        cam_img_coords = transform @ fisheye_rays
        cam_img_coords = cam_img_coords[:2, :] / cam_img_coords[2][None, :]

        mask_image_coords = (cam_img_coords[0]>=-margin) & (cam_img_coords[0]< pinhole_width + margin)  & (cam_img_coords[1]>=-margin) & (cam_img_coords[1]< pinhole_height +margin) 
        cam_img_coords = cam_img_coords[:, mask_image_coords]    

        # Fix for floating point coordinates by using margin
        # This part removes black dots on the borders of the five pinhole cube
        cam_img_coords[0, cam_img_coords[0, :] >= pinhole_width -1.0] = pinhole_width -1.0
        cam_img_coords[1, cam_img_coords[1, :] >= pinhole_height -1.0] = pinhole_height -1.0
        cam_img_coords[0, cam_img_coords[0, :] <= 0.0] = 0.0
        cam_img_coords[1, cam_img_coords[1, :] <=0.0] = 0.0

        cam_img_coords += box_idx
        camera_mask[camera_mask] = mask_image_coords


        return camera_mask, cam_img_coords


    def destroy(self)-> None:
        """Delete all cameras."""
        actors = [
            self._front_pinhole,
            self._left_pinhole,
            self._right_pinhole,
            self._top_pinhole,
            self._bottom_pinhole,
        ]
        for actor in actors:
            if actor is not None:
                actor.destroy()

    def create_fisheye_image(self)->None:
        """Creates fisheye image.
        
        Note: this function should be called in order to update image for fish eye camera.
        """
        if any([self._front_pinhole.image is None, self._left_pinhole.image is None, self._right_pinhole.image is None, self._top_pinhole.image is None, self._bottom_pinhole.image is None]):
            return

        self._five_pinhole_image = np.hstack((self._left_pinhole.image,
                              self._top_pinhole.image,
                             self._front_pinhole.image,
                            self._bottom_pinhole.image,                                
                            self._right_pinhole.image)).astype(np.float32)

        remapped_img = cv2.remap(self._five_pinhole_image, self.maptable[..., 0], self.maptable[..., 1], cv2.INTER_NEAREST)

        if len(remapped_img.shape) == 2:
            return

        remapped_img = np.multiply(remapped_img, self.circular_mask[..., None])
        
        self.image = remapped_img.astype('uint8')

        self.frame += 1.0
