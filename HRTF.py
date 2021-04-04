# %% --------------import packages-------------------------------------

from pyfar import Signal
import numpy as np
import numpy.matlib
import math 
import cmath
# import sfs

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# %% ---------------Define Variables------------------------------------
a = 0.0875     # radius of the sphere (m)
r = [1e16, 0.8, 0.4, 0.2, 0.15, 0.125]      # distance from the center of the sphere to the source (m)
r_0 = 100 * a  # distance of the source to the origin of co-coordinates
theta = [0, 150/180*np.pi] # angle of incidence (rad) 
c = 343    # ambient speed of sound (m/s)
threshold = 100 # min 50 

# %% ---------------OLD HRTF Duda 1998------------------------------------
def hrtf(a, r, theta, f, c, threshold):
    
    h_upp_dB = []

    # normalized distance - Eq. (5) in [1]
    rho = [radius / a for radius in r]

    # normalized frequency - Eq. (4) in [1]
    norm_freq = (2 * np.pi * f * a) / c
    
    for angle in theta:
        x = np.cos(angle)
        for r in rho:
            for mu in norm_freq:
                zr = 1 / (1j * mu * r)
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
                    
                    # check for NaNs
                    if np.isnan(term) == True:
                        print(term, angle, r, mu)

                    summ = summ + term  
                    
                    qr2 = qr1
                    qr1 = qr
                    qa2 = qa1
                    qa1 = qa
                    p2 = p1
                    p1 = p

                pressure = (r * cmath.exp(-1j * mu) * summ) / (1j * mu)
                dB = 20*np.log(np.abs(pressure)/(1))
                h_upp_dB.append(dB)  
    return h_upp_dB

# %% ---------------Off Ear HRTF ------------------------------------
def new_hrtf(a, r, r_0, theta, f, c, threshold):    
    '''
    Right now, rho_0 = rho, see line 152
    '''
    
    h_upp_dB = []

    # normalized distance - Eq. (5) in [1]
    rho = [radius / a for radius in r]
    # rho_0 = r_0 / a
    
    # normalized frequency - Eq. (4) in [1]
    norm_freq = (2 * np.pi * f * a) / c
    
    for angle in theta:
        x = np.cos(angle)
        for r in rho:
            for mu in norm_freq:
                zr = 1 / (1j * mu * r)
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
                    
                    # check for NaNs
                    if np.isnan(term) == True:
                        print(term, angle, r, mu)

                    summ = summ + term  
                    
                    qr2 = qr1
                    qr1 = qr
                    qa2 = qa1
                    qa1 = qa
                    p2 = p1
                    p1 = p

                pressure = r * cmath.exp(1j * (mu * r - mu * r - mu)) * summ / (1j * mu)
                dB = 20*np.log(np.abs(pressure)/(1))
                h_upp_dB.append(dB)  
    return h_upp_dB


# %% -------------------info prints----------------------------------
f_low = round((0.1*c)/(2*np.pi*a), 2)
print(f'f bei 0.1 mu: {f_low} Hz')

f_hi = round((100*c)/(2*np.pi*a), 2)  # isn't this too high?!
print(f'f bei 100 mu: {f_hi} Hz')

mu_20k = (2*np.pi*20000*a)/c
print(f'mu bei 20 kHz: {mu_20k}')

# create frequency / mu vector
freq_vec = np.linspace(f_low, f_hi, 2000)
mu_freq_vec = (2*np.pi*freq_vec*a)/c

# calculate HRTF
HRTF_freq_dB = hrtf(a, r, theta, freq_vec, c, threshold)
off_ear_freq_dB = new_hrtf(a, r, r_0, theta, freq_vec, c, threshold)

print(f'HRTF_freq_dB max: {round(max(HRTF_freq_dB),1)} dB, HRTF_freq_dB min: {round(min(HRTF_freq_dB),1)} dB')
print(f'off_ear_freq_dB max: {round(max(off_ear_freq_dB),1)} dB, off_ear_freq_dB min: {round(min(off_ear_freq_dB),1)} dB')

# %% -----------------------Plotting Duda 1998--------------------------
len_plot = int(len(HRTF_freq_dB)/(len(r)*len(theta)))
l_size = 3

plt.figure(figsize=(20,10))
plt.semilogx(mu_freq_vec, HRTF_freq_dB[:len_plot], '-', linewidth=l_size, label=r'$ \theta = 0°, \rho = 1e16$')
plt.semilogx(mu_freq_vec, HRTF_freq_dB[len_plot:len_plot*2], '-', linewidth=l_size, label=r'$ \theta = 0°, \rho = 8$')
plt.semilogx(mu_freq_vec, HRTF_freq_dB[len_plot*2:len_plot*3], '-', linewidth=l_size, label=r'$ \theta = 0°, \rho = 4$')
plt.semilogx(mu_freq_vec, HRTF_freq_dB[len_plot*3:len_plot*4], '-', linewidth=l_size, label=r'$ \theta = 0°, \rho = 2$')
plt.semilogx(mu_freq_vec, HRTF_freq_dB[len_plot*4:len_plot*5], '-', linewidth=l_size, label=r'$ \theta = 0°, \rho = 1.5$')
plt.semilogx(mu_freq_vec, HRTF_freq_dB[len_plot*5:len_plot*6], '-', linewidth=l_size, label=r'$ \theta = 0°, \rho = 1.25$')
plt.semilogx(mu_freq_vec, HRTF_freq_dB[len_plot*6:len_plot*7], '--', linewidth=l_size, label=r'$ \theta = 150°, \rho = 1e16$')
plt.semilogx(mu_freq_vec, HRTF_freq_dB[len_plot*7:len_plot*8], '-.', linewidth=l_size, label=r'$ \theta = 150°, \rho = 8$')
plt.semilogx(mu_freq_vec, HRTF_freq_dB[len_plot*8:len_plot*9], '--', linewidth=l_size, label=r'$ \theta = 150°, \rho = 4$')
plt.semilogx(mu_freq_vec, HRTF_freq_dB[len_plot*9:len_plot*10], '-.', linewidth=l_size, label=r'$ \theta = 150°, \rho = 2$')
plt.semilogx(mu_freq_vec, HRTF_freq_dB[len_plot*10:len_plot*11], '--', linewidth=l_size, label=r'$ \theta = 150°, \rho = 1.5$')
plt.semilogx(mu_freq_vec, HRTF_freq_dB[len_plot*11:len_plot*12], '-.', linewidth=l_size, label=r'$ \theta = 150°, \rho = 1.25$')
plt.xlim([0.1,100])

# vertical lines
position1 = np.arange(0.1, 1.1, step=0.1)
position2 = np.arange(1, 11, step=1)
position3 = np.arange(10, 110, step=10)
position = list(position1) + list(position2) + list(position3)
for tick in position:
    plt.vlines(tick, -65, 45, colors='k', linestyle=':', linewidth=1)
    
plt.vlines(mu_20k, -65, 45, colors='r', linestyle='-', linewidth=1.5, label='20 kHz')
plt.grid(color='k', linestyle=':', linewidth=2)

# plt.ylim(-130, -35)
plt.title('Effect of range on the magnitude response - Duda 1998', fontsize=20)
plt.ylabel('Response (dB)', fontsize=15)
plt.xlabel(r'Normierte Frequenz: $\mu = \frac{2\pi fa}{c}$', fontsize=15)
plt.legend(prop={'size': 18})
plt.show()

# %% -----------------------Plotting Off Ear--------------------------
len_plot = int(len(off_ear_freq_dB)/(len(r)*len(theta)))
l_size = 3

plt.figure(figsize=(20,10))
plt.semilogx(mu_freq_vec, off_ear_freq_dB[:len_plot], '-.', linewidth=l_size, label=r'$ \theta = 0°, \rho = 1e16$')
plt.semilogx(mu_freq_vec, off_ear_freq_dB[len_plot:len_plot*2], '--', linewidth=l_size, label=r'$ \theta = 0°, \rho = 8$')
plt.semilogx(mu_freq_vec, off_ear_freq_dB[len_plot*2:len_plot*3], '-.', linewidth=l_size, label=r'$ \theta = 0°, \rho = 4$')
plt.semilogx(mu_freq_vec, off_ear_freq_dB[len_plot*3:len_plot*4], '--', linewidth=l_size, label=r'$ \theta = 0°, \rho = 2$')
plt.semilogx(mu_freq_vec, off_ear_freq_dB[len_plot*4:len_plot*5], '-.', linewidth=l_size, label=r'$ \theta = 0°, \rho = 1.5$')
plt.semilogx(mu_freq_vec, off_ear_freq_dB[len_plot*5:len_plot*6], '-.', linewidth=l_size, label=r'$ \theta = 0°, \rho = 1.25$')
plt.semilogx(mu_freq_vec, off_ear_freq_dB[len_plot*6:len_plot*7], '--', linewidth=l_size, label=r'$ \theta = 150°, \rho = 1e16$')
plt.semilogx(mu_freq_vec, off_ear_freq_dB[len_plot*7:len_plot*8], '-.', linewidth=l_size, label=r'$ \theta = 150°, \rho = 8$')
plt.semilogx(mu_freq_vec, off_ear_freq_dB[len_plot*8:len_plot*9], '--', linewidth=l_size, label=r'$ \theta = 150°, \rho = 4$')
plt.semilogx(mu_freq_vec, off_ear_freq_dB[len_plot*9:len_plot*10], '-.', linewidth=l_size, label=r'$ \theta = 150°, \rho = 2$')
plt.semilogx(mu_freq_vec, off_ear_freq_dB[len_plot*10:len_plot*11], '--', linewidth=l_size, label=r'$ \theta = 150°, \rho = 1.5$')
plt.semilogx(mu_freq_vec, off_ear_freq_dB[len_plot*11:len_plot*12], '-.', linewidth=l_size, label=r'$ \theta = 150°, \rho = 1.25$')
plt.xlim([0.1,100])

# vertical lines
position1 = np.arange(0.1, 1.1, step=0.1)
position2 = np.arange(1, 11, step=1)
position3 = np.arange(10, 110, step=10)
position = list(position1) + list(position2) + list(position3)
for tick in position:
    plt.vlines(tick, -65, 45, colors='k', linestyle=':', linewidth=1)
    
plt.vlines(mu_20k, -65, 45, colors='r', linestyle='-', linewidth=1.5, label='20 kHz')
plt.grid(color='k', linestyle=':', linewidth=2)

# plt.ylim(-130, 200)
plt.title('Effect of range on the magnitude response - Off Ear', fontsize=20)
plt.ylabel('Response (dB)', fontsize=15)
plt.xlabel(r'Normierte Frequenz: $\mu = \frac{2\pi fa}{c}$', fontsize=15)
plt.legend(prop={'size': 18})
plt.show()


# %% quadarea
def quadarea(lat1, lon1, lat2, lon2):
    h = np.sin(lat2)-np.sin(lat1)
    Az = 2 * np.pi * h
    Aq = Az * (lon2-lon1)/(2*np.pi)

    A = (np.sin(np.deg2rad(lat2))-np.sin(np.deg2rad(lat1))) * np.deg2rad(lon2-lon1) / (4*np.pi)
    print(Aq/(4*np.pi))
    return A

quadarea(-90,-180,-89,180)
# %% AKgreatCircleGrid
def AKgreatCircleGrid(el=list(range(90, -92, -2)), max_ang=2, fit=90, do_plot=0, res_ang=1):

    # check input format
    # el = reshape(el, [numel(el) 1]);

    if fit == 0:
        fit = 360

    if 1 % res_ang:
        print('AKgreatCircleGrid:Input', 'results can contain errors if 1/res_ang is NOT an integer numer')

    # calculate delta phi to meet the criterion
    # (according to Bovbjerg et al. 2000: Measuring the head related transfer
    # functions of an artificial head with a high directional resolution,
    # R.W. Sinnott, "Virtues of the Haversine", Sky and Telescope, vol. 68, no.
    # 2, 1984, p. 159)

    d_phi = [np.rad2deg(2*np.arcsin(np.sin(np.deg2rad(max_ang/2))/np.cos(np.deg2rad(i)))) for i in el]

    # correct values at the poles
    abs_el = np.array([abs(i) for i in el])
    idx_90 = np.where(abs_el == 90)
    d_phi = [360 if element ==90 else d_phi[i] for i, element in enumerate(abs_el)]

    # round to desired angular resolution
    d_phi = [int(i / res_ang) * res_ang for i in d_phi]   # floor / int

    # adjust delta phi to assure equal spacing on sphere -> mod(360, d_phi)!=0
    # or quarter sphere -> mod(90, d_phi)!=0
    # (if equally spaced on on quarter sphere (fit = 90), median, horizontal
    # and frontal plane are included in measurements).
    # this operation is easier in degrees than in radians...
    for n in range(len(d_phi)):
        if abs(el[n]) != 90:
            while fit % d_phi[n]:
                d_phi[n] = round((d_phi[n] - res_ang) / res_ang) * res_ang
        else:
            # irregularity at north and south pole
            d_phi[n] = 360
    
    act_ang = d_phi

    # calculate great circle angle that is actually used in the grid
    # (R.W. Sinnott, "Virtues of the Haversine", Sky and Telescope, vol. 68, no. 2, 1984, p. 159)
    act_ang_GCD = [np.rad2deg(2*np.arcsin(np.sqrt(np.cos(np.deg2rad(e))**2*np.sin(np.deg2rad(phi/2))**2))) for phi, e in zip(d_phi,el)]

    # construct pre-grid
    hrtf_grid = [] 

    m = 0
    for n in range(len(d_phi)):
        tmp = np.arange(0, 360-d_phi[n]+1, d_phi[n])
        for i in tmp:
            hrtf_grid.append([i, el[n]])
        m += len(tmp)

    #final grid in degree
    hrtf_grid_deg = hrtf_grid
    # estimated area weights using lat-long rectangles
    weights = np.zeros((len(hrtf_grid_deg),1)) 
    el_sort = np.sort(el)

    for n in range(len(act_ang)):
        if len(act_ang) == 1:
            weight = 1

        elif el_sort[n] == -90:
            el_range = [-90, np.mean(el_sort[n:n+2])]  # + 2 because of difference in indexing 
            weight   = quadarea(el_range[0], -180, el_range[1], 180)  

        elif el_sort[n] == 90:
            el_range = [90, np.mean(el_sort[n-1:n+1])] 
            weight   = quadarea(el_range[0], -180, el_range[1], 180)

        else:
            if n == 0:
                el_diff  = ( el_sort[n+1]-el_sort[n] ) / 2
                el_range = el_sort[n] + [-el_diff, el_diff]
                weight   = quadarea(el_range[0], 0, el_range[1], act_ang[n])
            elif n == len(act_ang):
                el_diff  = ( el_sort[n]-el_sort[n-1] ) / 2
                el_range = el_sort[n] + [-el_diff, el_diff]
                weight   = quadarea(el_range[0], 0, el_range[1], act_ang[n])
            else:
                # print(el_sort[n-1:n+1], el_sort[n:n+2])
                el_range = [np.mean(el_sort[n-1:n+1]), np.mean(el_sort[n:n+2])]  # + 2 because of difference in indexing 
                weight   = quadarea(el_range[0], 0, el_range[1], act_ang[n])
        
        hrtf_grid_deg = np.array(hrtf_grid_deg)
        for i, element in enumerate(hrtf_grid_deg[:, 1]):
            if element == el_sort[n]:
                weights[i] = weight

    weights = weights / sum(weights)

    return [hrtf_grid_deg, act_ang_GCD, act_ang, weights]

AKgreatCircleGrid()

# %% AKsphericalHead
def AKsphericalHead(sg = AKgreatCircleGrid(el=list(range(90, -92, -2)), max_ang=2, fit=90, do_plot=0, res_ang=1), ear = [85, -13], offCenter = False, a = 0.0875, r_0 = 100*0.0875, Nsh = 100, Nsamples = 1024, fs = 44100, c = 343):
    '''calculates head-realated impulse responses (HRIRs) of a spherical head
        model with offset ears using the formulation from according to [1]. HRIRs
        are calculated by dividing the pressure on the sphere by the free field
        essure of a point source in the origin of coordinates.
        See AKsphericalHeadDemo.m for example use cases
        Additional information on the model itself can be found in
        2_Tools/SphericalHarmonics/AKsphericalHead.pdf
    Params:
        sg        - [N x 3] matrix with the spatial sampling grid, where the
                    first column specifies the azimuth (0 deg. = front,
                    90 deg. = left), the second column the elevation
                    (90 deg. = above, 0 deg. = front, -90 deg. = below), and the
                    third column the radius [m]. If only two columns are given
                    the radius is set to a*100 (see below)
                    (default: AKgreatCircleGrid(90:-10:-90, 10, 90) )
        ear       - four element vector that specifies position of left and right
                    ear: [azimuth_l elevation_l azimuth_r elevation_r]. If only
                    two values are passed, symmetrical ears are assumed.
                    (defualt = [85 -13], average values from [2], Tab V,
                    condition All, O-A) 
        offCenter - false   : the spherical head is centered in the coordinate
                            system (default)
                    true    : the interaural axis (i.e., the connection between
                            the two ears) is centered. This is done be
                            averaging the ear azimuth and elevation, and
                            and a translation the sampling grid
                    [x y z] : x/y/z coordinates of the center of the sphere [m].
                            E.g., [-4e3 1e3 2e3] moves the spherical head 4 mm
                            to the back, 1 mm to the left (left ear away from
                            the origin of coordinates) and 2 mm up.
        a         - radius of the spherical head in m (default = 0.0875)
        r_0       - distance of the free-field point source in m that used as
                    reference (by default the radius from the sampling grid is
                    taken: r_0 = sg(1,3) )
        Nsh       - spherical harmonics order (default = 100)
        Nsamples  - length in samples (default = 1024)
        fs        - sampling rate in Hz (default = 44100)
        c         - speed of sound [m/s] (default = 343)

    Returns:
        h                  - spherical head impulse responses given in matrix of
                            size [Nsamples x N x 2]: Left ear = h(:,:,1),
                            right ear = h(:,:,2)
        offCenterParameter - spherical head model parameters after translation
                            and changing the ear position (if applied)

                            ear    : new ear position (see above)
                            sg     : new sampling grid (see above)
                            r      : radius for each point of sg
                            azRot  : rotation above z-axis (azimuth) that was
                                    applied to get the new ear azimuth
                            elRot  : rotation above x-axis (elevation) that was
                                    applied to get the new ear elevation
                            xTrans : translation of the spherical head in x-
                                    direction, that was applied to center the
                                    interaural axis
                            zTrans : translation of the spherical head in z-
                                    direction, that was applied to center the
                                    interaural axis'''
    
    offCenterParameter = {}

    #  --- set default parameters ---
    if sg.shape[1] < 3:
        r_0 = 100*a
    else:
        r_0 = sg[0,2]

    # check format of the sampling grid
    if sg.shape[1] < 3:
        print(sg.shape)
        sg = np.hstack((sg, r_0 * np.ones((sg.shape[0], 1))))  

    # check format of ear vector
    if len(ear) == 2:
        ear = ear + [360 - ear[0], ear[1]]

    # rotate and translate the spherical head
    # center the interaural axis
    if offCenter == True:
        # sampling grid in carthesian coordinates
        sgX, sgY, sgZ = sfs.util.sph2cart(sg[:,0]/180*np.pi, sg[:,1]/180*np.pi, sg[:,2])
        
        # translate the samplingv grid
        sgX = sgX - offCenter[0]
        sgY = sgY - offCenter[1]
        sgZ = sgZ - offCenter[2]
        
        # translated sampling grid in spherical coordinates
        sgAz, sgEl, sgR   = sfs.util.cart2sph(sgX, sgY, sgZ)
        sg                = [sgAz/np.pi*180, sgEl/np.pi*180, sgR]
        sg                = round(sg*10000) / 10000
        sg[:,0]           = sg[:,0]%360
        
        # save parameter
        offCenterParameter['sg'] = sg
        
        del [sgX, sgY, sgZ, sgAz, sgEl, sgR]

    elif offCenter == False: 
        # check if the ear azimuths are symmetrical
        if ear[0] != 360-ear[2]:
            earAz = np.mean([ear[0], 360-ear[2]])
            offCenterParameter['azimuthRotation'] = earAz - ear[0]
            for i, el in zip([0,2], [earAz, 360-earAz]):
                ear[i] = el
            del earAz
        else:
            offCenterParameter['azimuthRotation'] = 0

        
        # check if the ear elevations are symmetrical
        if ear[1] != ear[3]:
            earEl = np.mean([ear[1], ear[3]])
            offCenterParameter['elevation']         = earEl
            offCenterParameter['elevationRotation'] = earEl - ear[1]
            for i in [1,3]:
                ear[i] = earEl
            del earEl
        else:
            offCenterParameter['elevationRotation'] = 0
        
        #sampling grid in carthesian coordinates
        sgX, sgY, sgZ = sfs.util.sph2cart(sg[:,0]/180*np.pi, sg[:,1]/180*np.pi, sg[:,2])
        doTranslate   = False
        
        # check for translation in x-direction (front/back)
        if ear[0] != 90:
            offCenterParameter['xTranslation'] = np.sin(ear[0] - 90) * a
            doTranslate = True
            sgX         = sgX - offCenterParameter['xTranslation']
        else:
            offCenterParameter['xTranslation'] = 0
        
        # check for translation in z-direction (up/down)
        if ear[1] != 0:
            offCenterParameter['zTranslation'] = np.sin(-ear[1]) * a
            doTranslate = True
            sgZ         = sgZ - offCenterParameter['zTranslation']
        else:
            offCenterParameter['zTranslation'] = 0
        
        # transform grid to spherical coordinates again
        if doTranslate:
            sgAz, sgEl, sgR   = sfs.util.cart2sph(sgX, sgY, sgZ)
            sg                = [sgAz/np.pi*180, sgEl/np.pi*180, sgR]
            sg                = round(sg*10000) / 10000
            sg[:,0]           = sg[:,0] % 360
        
        offCenterParameter['sg']  = sg
        offCenterParameter['ear'] = ear

        del [earAz, earEl, sgX, sgY, sgZ, sgAz, sgEl, sgR, doTranslate]
        
    else:
        offCenterParameter = False 

    # check parameter values
    if any(sg[:,2] < a) or r_0 < a:
        print('AKsphericalHead:Input', 'sg(:,3), and r_0 must be larger than a')

    # spherical head model
    # calculate great circle distances between the sampling grid and the ears
    gcd = np.matrix([[np.arccos(np.sin(sg[:, 1]) * np.sin(ear[1]) + np.cos(sg[:, 1]) * np.cos(ear[1]) * np.cos(sg[:, 0] - ear[0]))], 
                     [np.arccos(np.sin(sg[:, 1]) * np.sin(ear[3]) + np.cos(sg[:, 1]) * np.cos(ear[3]) * np.cos(sg[:, 0] - ear[2]))]])

    # get unique list of great circle distances and radii
    gcd_sg = np.hstack((gcd, sg[:, 2]))
    GCD, gcdID = np.unique(gcd_sg, axis=0, return_inverse=True)  
    # gcd = reshape(GCD(gcdID), size(gcd))
    r   = GCD[:,1]
    GCD = GCD[:,0]

    # get list of frequencies to be calculated
    f = list(range(0, fs/2, fs/Nsamples)) 

    # calculate complex the transfer function in the frequency domain
    H = new_hrtf(a, r, r_0, GCD/180*np.pi, f, c, Nsh)

    # set 0 Hz bin to 1 (0 dB)
    H[0,:] = 1

    # make sure bin at fs/2 is real
    if f[end] == fs/2:
        H[end,:] = abs(H[end,:])

    # mirror the spectrum
    H = AKsingle2bothSidedSpectrum(H, 1 - Nsamples%2)

    # get the impuse responses
    hUnique = np.fft.ifft(H)

    # add delay to shift the pulses away from the very start
    hUnique = np.roll(hUnique, round(1.5e-3*fs), axis=0)

    # resort to match the desired sampling grid
    h = np.zeros((Nsamples, sg.shape[0], 2))
    h[:,:,0] = hUnique[:, gcdID[1:sg.shape[0]+1]]
    h[:,:,1] = hUnique[:, gcdID[sg.shape[0]+1:end]]

AKsphericalHead()
# %%

