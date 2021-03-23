# %% --------------import packages-------------------------------------

from pyfar import Signal
import numpy as np
import cmath
from scipy.special import legendre, hankel1

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


# %%
