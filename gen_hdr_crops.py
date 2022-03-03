import os
import envmap
from ezexr import imread, imwrite
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from tqdm import tqdm

def genLDRimage(hdrim, 
                intensityMultiplier=None, 
                putMedianIntensityAt=0.5, 
                gamma=1./2.2,
                returnIntensityMultiplier=False):
    """
    Produce a LDR tonemapping from HDR using gamma compression. The returned
    image pixels are within the [0, 1] range.
    This function has two modes: 
    - If `intensityMultiplier` is provided, then this function returns
        (intensityMultiplier * hdrim)^gamma
    - If `intensityMultiplier` is NOT provided (that is, None value), then the
        `putMedianIntensityAt` parameter is used, and the function computes the
        multiplier that, once applied to the HDR image followed by the gamma
        compression, puts the median of the LDR image at `putMedianIntensityAt`.
    
    `gamma` must be provided in both cases (note that it should be provided as
    an exponent, that is 1/2.2 and not 2.2 directly).
    This function returns the LDR image, clipped between 0.5/255 and 1., to match
    the dynamic range of a standard image.
    If `returnIntensityMultiplier` is True, returns a tuple, the second element 
    being the multiplier value explained above.
    """
    hdrim = hdrim.astype('float32')
    if intensityMultiplier is None:
        # Get the intensity and compute its median
        intensity = 0.299 * hdrim[...,0] + \
                    0.587 * hdrim[...,1] + \
                    0.114 * hdrim[...,2]
        med = np.median(intensity)

        if med > 0.:
            intensityMultiplier = putMedianIntensityAt**(1./gamma) / med
        else:
            intensityMultiplier = 1
    
    imLDR = np.clip((intensityMultiplier*hdrim)**gamma, 0.5/255., 1.)
    if returnIntensityMultiplier:
        return imLDR, intensityMultiplier
    else:
        return imLDR

def extractImage(envmap, viewing_angles, output_height, vfov=50, output_width=320, interp_order=1):
    """
    Extract an image from an environment map.
    :envmap: an environment map (image)
    :viewing_angles: phi (elevation), lambda (azimuth), and theta (roll) in radians 
    :output_height: The height of the output image
    :vfov: vertical field of view (in degrees)
    :ratio: ratio width/height
    """
    # Source of the formulaes : mathworld.wolfram.com/GnomonicProjection.html
    def rectilinear2latlong(x, y, phi0, lambda0):
        rho = np.sqrt(x**2 + y**2)
        c = np.arctan(rho)

        return (np.arcsin(np.cos(c) * np.sin(phi0) + y * np.sin(c) * np.cos(phi0) / rho),                              # phi (elevation)
                lambda0 + np.arctan2(x * np.sin(c), rho * np.cos(phi0) * np.cos(c) - y * np.sin(phi0) * np.sin(c)))    # lambda (azimuth)
                
    # 37.8 fov == 35mm lens
    elevation = azimuth = roll = 0
    if len(viewing_angles) > 0:
        elevation = viewing_angles[0]
    if len(viewing_angles) > 1:
        azimuth = viewing_angles[1]
    if len(viewing_angles) > 2:
        roll = viewing_angles[2]
        
    # ratiohw = 1./ratio
    fovRad = np.radians(vfov)
    fovY = np.tan(fovRad / 2.)
    # fovX = fovY / ratiohw
    fovX = np.tan(fovRad / 2.)
    

    nb_channels = envmap.shape[-1]

    # We produce a rectilinear image, cropped from the latlong envmap
    croppedSize = (output_height, output_width)
    # if any([cs < 1 for cs in croppedSize]):
    #     print("Warning! negative resolution {}x{} (from aspect ratio {})".format(croppedSize[1],croppedSize[0],ratiohw))
    xcoords, ycoords = np.meshgrid( np.linspace(-fovX, fovX, croppedSize[1]),
                                    np.linspace(-fovY, fovY, croppedSize[0]),
                                    indexing='xy')

    # apply roll in the image plane before doing the gnomonic projection
    flatxcoords = np.reshape(xcoords,(1,np.product(np.shape(xcoords))))
    flatycoords = np.reshape(ycoords,(1,np.product(np.shape(xcoords))))
    i = np.stack((flatxcoords,flatycoords),axis=0)
    xform = np.array([[np.cos(roll),-np.sin(roll)],[np.sin(roll),np.cos(roll)]])
    rolled = (np.mat(i).T*np.mat(xform)).T
    xcoordsrolled = np.array(np.reshape(rolled[0],croppedSize))
    ycoordsrolled = np.array(np.reshape(rolled[1],croppedSize))

    elev, azimuth = rectilinear2latlong(xcoordsrolled, ycoordsrolled, elevation, azimuth)
    azimuth[azimuth > np.pi] -= 2*np.pi
    azimuth[azimuth < -np.pi] += 2*np.pi
    azimuthPix = azimuth / np.pi * envmap.shape[1] / 2 + envmap.shape[1] / 2
    elevPix = elev / (np.pi / 2) * envmap.shape[0] / 2 + envmap.shape[0] / 2

    outimg = np.empty(croppedSize + (nb_channels,), dtype='float32')
    coordinates = np.stack((elevPix, azimuthPix), axis=0)

    for c in range(nb_channels):  # Color channels
        map_coordinates(envmap[..., c], coordinates, outimg[..., c], order=interp_order, mode="wrap")

    return outimg

input_folder = '/root/datasets_ssd/LavalIndoor/1942x971/test'
output_folder = '/root/datasets_ssd/LavalIndoor/crops_hdr/test'

# list all .exr files in input_folder
input_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.exr')])

# remove output folder if it exists
if os.path.exists(output_folder):
    os.system('rm -rf {}'.format(output_folder))
os.makedirs(output_folder)

for input_file in tqdm(input_files):
    gt_envmap = imread(input_file)
    elevation = 0
    azimuth = 0
    cropHeight = 220
    cropWidth = 260
    vfov = 50
    gamma = 1/2.4
    envmap_data = envmap.EnvironmentMap(gt_envmap, 'latlong')
    crop = extractImage(envmap_data.data, [elevation, azimuth], cropHeight, vfov=vfov, output_width = cropWidth)
    _, reexpose_scale_factor = genLDRimage(crop, putMedianIntensityAt=0.45, returnIntensityMultiplier=True, gamma=gamma)
    # save the cropped envmap
    imwrite(os.path.join(output_folder, os.path.basename(input_file)), crop * reexpose_scale_factor)