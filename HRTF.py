# %% --------------import packages-------------------------------------
import pyfar
from pyfar import Signal
from pyfar.spatial.samplings import sph_great_circle
from pyfar.coordinates import Coordinates
import numpy as np
import cmath

# %% ---------------Off Ear HRTF ------------------------------------
def hrtf(a, r, r_0, theta, f, c, threshold):    
    """calculates head-realated impulse responses (HRIRs) of a spherical head
        model with offset ears using the formulation from according to [1]. HRIRs
        are calculated by dividing the pressure on the sphere by the free field
        essure of a point source in the origin of coordinates.
       
    Parameters
    ----------  
        a: float   
            - radius of the spherical head in m (default = 0.0875)
        
        r: float
            - distance of the free-field point source in m to the center of 
            of the sphere

        r_0: float
            - distance of the free-field point source in m that used as
            reference (by default the radius from the sampling grid is
            taken: r_0 = sg[0,2])
        
        theta: float       
            - great circle distance between each point on the sampling grid
        
        f: list of float        
            - f = list(np.arange(0, fs/2 + fs/Nsamples, fs/Nsamples)) 
        
        c: int
            - speed of sound [m/s] (default = 343)
        
        threshold: int  
            - order of spherical harmonics
        

    Returns
    -------
        H: ndarray
            - 2D array containing data with `float` type
                                    
    References
    ----------
    .. [1]  Duda, R. and W. Martens. “Range dependence of the response of a 
            spherical head model.” Journal of the Acoustical Society of America 
            104 (1998): 3048-3058.
    """

    # create matrix for HRTF
    H = np.zeros((len(f), len(theta)))

    # normalized distance - Eq. (5) in [1]
    r_unique, idx = np.unique(r, return_index=True)
    rho = [r / a for r in r_unique]
    rho_0 = r_0 / a
    
    # normalized frequency - Eq. (4) in [1]
    mu = np.array([(2 * np.pi * freq * a) / c for freq in f])
    
    for i, angle in enumerate(theta):
        x = np.cos(angle)

        if len(rho) == 1: 
            zr = 1 / (1j * mu * rho)
            za = 1 / (1j * mu)
        
        else:
            zr = 1 / (1j * mu * rho[idx[i]])
            za = 1 / (1j * mu)

        qr2 = zr
        qr1 = zr * (1 - zr)
        qa2 = za
        qa1 = za * (1 - za)

        # initialize legendre Polynom for order m=0 (P2) and m=1 (P1)
        p2 = 1
        p1 = x

        # initialize the sum - Eq. (A10) in [1]
        summ = 0

        # calculate the sum for m=0
        term = zr / (za * (za - 1))
        summ = summ + term

        # calculate sum for m=1
        if threshold > 0:
            term = (3 * x * zr * (zr - 1)) / (za * (2 * (za ** 2) - 2 * za + 1))
            summ = summ + term

        for m in range(2, threshold+1):
            # recursive calculation of the Legendre polynomial of order m (see doc legendreP)
            p = ((2 * m - 1) * x * p1 - (m - 1) * p2) / m

            # recursive calculation of the Hankel fraction
            qr = - (2 * m - 1) * zr * qr1 + qr2
            qa = - (2 * m - 1) * za * qa1 + qa2
            
            # update the sum and recursive terms
            term = ((2 * m + 1) * p * qr) / ((m + 1) * za * qa - qa1)  # might become NaN for low frequencies
        
            summ = summ + term  
            
            qr2 = qr1
            qr1 = qr
            qa2 = qa1
            qa1 = qa
            p2 = p1
            p1 = p

        if len(rho) == 1: 
            H[:,i] = rho_0 * np.exp(1j * (mu * rho - mu * rho_0 - mu)) * summ / (1j * mu)
        else:
            H[:,i] = rho_0 * np.exp(1j * (mu * rho[idx[i]] - mu * rho_0 - mu)) * summ / (1j * mu)
 
    return H

# %% AKsphericalHead
def AKsphericalHead(sg = sph_great_circle(), ear = [85, -13], offCenter = False, r_0 = None, a = 0.0875, Nsh = 100, Nsamples = 1024, fs = 44100, c = 343):
    """calculates head-realated impulse responses (HRIRs) of a spherical head
        model with offset ears using the formulation from according to [1]. HRIRs
        are calculated by dividing the pressure on the sphere by the free field
        essure of a point source in the origin of coordinates.
       
    Parameters
    ----------
        sg: Coordinates       
            - Sampling positions as Coordinate object
            - [N x 3] matrix with the spatial sampling grid, where the
            first column specifies the azimuth (0 deg. = front,
            90 deg. = left), the second column the elevation
            (90 deg. = above, 0 deg. = front, -90 deg. = below), and the
            third column the radius [m]. 
        
        ear: list of int      
            - four element vector that specifies position of left and right
            [azimuth_l, elevation_l, azimuth_r, elevation_r]. If only
            two values are passed, symmetrical ears are assumed.
            (default = [85, -13])

        offCenter: bool or list of float
            - False: the spherical head is centered in the coordinate
                    system (default)    
            - True: the interaural axis (i.e., the connection between
                    the two ears) is centered. This is done be
                    averaging the ear azimuth and elevation, and
                    and a translation the sampling grid
            - [x, y, z]: x/y/z coordinates of the center of the sphere [m].
                    E.g., [-4e3, 1e3, 2e3] moves the spherical head 4 mm
                    to the back, 1 mm to the left (left ear away from
                    the origin of coordinates) and 2 mm up.
        
        a: float   
            - radius of the spherical head in m (default = 0.0875)
        
        r_0: None or float
            - distance of the free-field point source in m that used as
            reference (by default the radius from the sampling grid is
            taken: r_0 = sg[0,2])
        
        Nsh: int       
            - spherical harmonics order (default = 100)
        
        Nsamples: int  
            - length in samples (default = 1024)
        
        fs: int        
            - sampling rate in Hz (default = 44100)
        
        c: int
            - speed of sound [m/s] (default = 343)

    Returns
    -------
        shtf: Signal       
            - spherical HRTF as Signal object
        
        offCenterParameter: dict 
            - spherical head model parameters after translation and 
            changing the ear position (if applied)

            ear: new ear position (see above)
            sg: new sampling grid (see above)
            r: radius for each point of sg
            azRot: rotation above z-axis (azimuth) that was
                    applied to get the new ear azimuth
            elRot: rotation above x-axis (elevation) that was
                    applied to get the new ear elevation
            xTrans: translation of the spherical head in x-
                    direction, that was applied to center the
                    interaural axis
            zTrans: translation of the spherical head in z-
                    direction, that was applied to center the
                    interaural axis
                                    
    References
    ----------
    .. [1]  Duda, R. and W. Martens. “Range dependence of the response of a 
            spherical head model.” Journal of the Acoustical Society of America 
            104 (1998): 3048-3058.
    """

    # check format of spherical grid
    if not isinstance(sg, Coordinates):
        raise ValueError("The spherical grid needs to be a pyfar.coordinates object.")
    
    # check format of ear vector
    if len(ear) < 2 or len(ear) == 3 or len(ear) > 4:
        raise ValueError("Ear needs to have either 2 entries if symmetrical ears are assumed, or four elements [azimuth_l elevation_l azimuth_r elevation_r].")
    elif len(ear) == 2:
        ear = ear + [360 - ear[0], ear[1]]

    # check format of spherical grid
    if offCenter is not True and offCenter is not False and len(offCenter) != 3:
        raise ValueError("offCenter must be either True, False, or given as x/y/z coordinates off the center of the sphere [m].")

    # check reference radius 
    if r_0 is None:
        r_0 = sg.get_sph()[0,2]
    
    if any(sg.get_sph()[:,2] < a) or r_0 < a:
        raise ValueError("Source is inside the head. sg(:,3), and r_0 must be larger than a.")

    # dict for offCenter parameter
    offCenterParameter = {}

    # rotate and translate the spherical head
    # center the interaural axis
    if isinstance(offCenter, list):

        # sampling grid in carthesian coordinates
        cart_coor = sg.get_cart(convention='right', unit='met')
        cart_coor = cart_coor.reshape((-1,3))

        # translate the sampling grid
        sg.set_cart(cart_coor[0] - offCenter[0], cart_coor[1] - offCenter[1], cart_coor[2] - offCenter[2])

        # translated sampling grid in spherical coordinates
        sph_coor          = sg.get_sph(convention='top_elev', unit='deg')
        sph_coor = np.round(sph_coor, 5)    
        sph_coor[:,0] = [x%360 for x in sph_coor[:,0]]
        
        # save offCenterParameter
        offCenterParameter['sg'] = sph_coor
        
    elif offCenter == True: 
        # check if the ear azimuths are symmetrical
        if ear[0] != 360-ear[2]:
            earAz = np.mean([ear[0], 360-ear[2]])
            offCenterParameter['azimuthRotation'] = earAz - ear[0]
            for i, el in zip([0,2], [earAz, 360-earAz]):
                ear[i] = el
        else:
            offCenterParameter['azimuthRotation'] = 0
        
        # check if the ear elevations are symmetrical
        if ear[1] != ear[3]:
            earEl = np.mean([ear[1], ear[3]])
            offCenterParameter['elevation']         = earEl
            offCenterParameter['elevationRotation'] = earEl - ear[1]
            for i in [1,3]:
                ear[i] = earEl
        else:
            offCenterParameter['elevationRotation'] = 0
        
        #sampling grid in carthesian coordinates
        cart_coor = sg.get_cart(convention='right', unit='met')
        doTranslate   = False
        
        # check for translation in x-direction (front/back)
        if ear[0] != 90:
            offCenterParameter['xTranslation'] = np.sin(ear[0] - 90) * a
            doTranslate = True
            sg.set_cart(cart_coor[0] - offCenterParameter['xTranslation'], cart_coor[1], cart_coor[2])

        else:
            offCenterParameter['xTranslation'] = 0
        
        # check for translation in z-direction (up/down)
        if ear[1] != 0:
            offCenterParameter['zTranslation'] = np.sin(-ear[1]) * a
            doTranslate = True
            sg.set_cart(cart_coor[0], cart_coor[1], cart_coor[2] - offCenterParameter['zTranslation'])
        else:
            offCenterParameter['zTranslation'] = 0
        
        # transform grid to spherical coordinates again
        if doTranslate:
            sph_coor = sg.get_sph(convention='top_elev', unit='deg')

            sph_coor = np.round(sph_coor, 5)    
            sph_coor[:,0] = [x%360 for x in sph_coor[:,0]]

        offCenterParameter['sg']  = sg
        offCenterParameter['ear'] = ear
        
    elif offCenter == False:
        offCenterParameter = False 
        sph_coor = sg.get_sph(convention='top_elev', unit='deg')

    # spherical head model
    # calculate great circle distances between the sampling grid and the ears
    arr1 = np.arccos(np.sin(sph_coor[:, 1]) * np.sin(ear[1]) + np.cos(sph_coor[:, 1]) * np.cos(ear[1]) * np.cos(sph_coor[:, 0] - ear[0]))
    arr1 = np.reshape(arr1, (len(arr1),1))

    arr2 = np.arccos(np.sin(sph_coor[:, 1]) * np.sin(ear[3]) + np.cos(sph_coor[:, 1]) * np.cos(ear[3]) * np.cos(sph_coor[:, 0] - ear[2]))
    arr2 = np.reshape(arr2, (len(arr2),1))

    gcd = np.vstack((arr1, arr2))
   
    rep = 2
    radii = np.reshape(np.repeat(sph_coor[:, 2], rep), (rep*len(sph_coor[:, 2]),1)) 
   
    # get unique list of great circle distances and radii
    gcd_sg = np.hstack((gcd, radii))
    GCD, gcdID = np.unique(gcd_sg, axis=0, return_inverse=True)  

    # gcd = reshape(GCD(gcdID), size(gcd))
    r   = GCD[:,1]
    GCD = GCD[:,0]

    # get list of frequencies to be calculated
    f = list(np.arange(0, fs/2 + fs/Nsamples, fs/Nsamples)) 

    # calculate complex the transfer function in the frequency domain
    H = hrtf(a, r, r_0, GCD/180*np.pi, f, c, Nsh)  

    # set 0 Hz bin to 1 (0 dB)
    H[0,:] = 1

    # make sure bin at fs/2 is real
    if f[-1] == fs/2:
        H[-1,:] = abs(H[-1,:])

    H = AKsingle2bothSidedSpectrum(H, 1 - Nsamples%2)
    shtf = Signal(H, fs, Nsamples) 

    # add delay to shift the pulses away from the very start
    shtf.time = np.roll(shtf.time, round(1.5e-3*fs), axis=0)

    # resort to match the desired sampling grid
    h = np.zeros((Nsamples, sg.cshape[0], 2))
    h[:,:,0] = shtf.time[:, gcdID[0:sg.cshape[0] ]]
    h[:,:,1] = shtf.time[:, gcdID[sg.cshape[0] :]]

    shtf = Signal(h, fs, Nsamples) 

    return shtf, offCenterParameter
    
# %% AKsingle2bothSidedSpectrum
def AKsingle2bothSidedSpectrum(single_sided, is_even=1):
    """
    Parameters
    ----------
        single-sided: ndarray
            - single sided spectrum , of size [N, M, C], where N is the
            number of frequency bins, M the number of measurements
            and C the number of channels. N must correspond to frequencies of 
            0 <= f <= fs/2, where f is the sampling frequency
        is_even: int       
            - true if both sided spectrum had even number of taps (default).
            - if is_even > 1, it denotes the number of samples of the both 
            sided spectrum (default = 1)
    Returns
    -------
        single-sided: ndarray
            - mirrored spectrum
    """
    
    if is_even>1:
        is_even = 1 - is_even % 2 

    N = single_sided.shape[0] 

    if is_even:
        # make sure that the bin at nyquist frequency is real
        # there might be rounding errors that produce small immaginary parts
        single_sided[-1,:] = [float(i) for i in single_sided[-1,:]]

        # mirror the spectrum
        both_sided = np.vstack((single_sided, np.flipud(np.conj(single_sided[1:N-1,:])))) 

    else:
        # mirror the spectrum
        both_sided = np.vstack((single_sided, np.flipud(np.conj(single_sided[1:N+1,:]))))

    # make sure that the bin at 0 Hz is real
    # there might be rounding errors that produce small immaginary parts
    both_sided[0,:] = [float(i) for i in both_sided[0,:]]

    return both_sided