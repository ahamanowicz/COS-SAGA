import numpy as np
import sys
import matplotlib.pyplot as plt
import spectro as spec
from astropy.table import Table
from astropy import convolution
import fnmatch
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
import os
def plot_line(n=0, wav='', flux='', vel=0, veldown=-200, velup=600, sightline='' ):
    # n - the line index from the line_list.txt
    # wavelength and flux tables, vel - velocity of the system
    # velcut - the cut in velocity around the spectrum +/- velcut

    lines = np.loadtxt("../line_list.txt", dtype='str')
    # 1) choose the line


    print(lines.T[1][n], lines.T[2][n], lines.T[3][n])
    linename=lines.T[1][n]
    line= float(lines.T[2][n])
    line_f = lines.T[3][n]
    
    # 2) convert to velocity, correct to LSR  and move to the galaxy rest frame
    
    v_helio = (wav - line)/line * 3e5
    v_lsr = spec.helio_to_lsr(v_helio,ra,dec ) #- vel
    
    #3) catout the spectrum  +/- x around the line -check for the continuum

    i_min, i_max = min(np.where(v_lsr>veldown)[0]), max(np.where(v_lsr<velup)[0])
    #limits as for contfit
    line_cut_vel = v_lsr[i_min:i_max]
    line_cut_flux = flux[i_min:i_max]
    
    # 4) plot the line
    plt.figure(1,figsize=(12,6))
    plt.plot(line_cut_vel, line_cut_flux, c='k')
    plt.annotate( linename+ " " + str(round(line,2)), (0.7, 0.9), xycoords='axes fraction', fontsize=18 )
    plt.annotate( sightline, (0.2, 0.9), xycoords='axes fraction', fontsize=18 )
    plt.axvline(0, ls='--', color='r')
    plt.xlim([veldown,velup])
    plt.ylim([min(line_cut_flux),1.2*max(line_cut_flux)])

    plt.show()

v_IC1613, v_SexA, v_WLM, v_LeoP = -233, 324, -130, 264



# choose sigthile - m - number form the table
M = int(sys.argv[1])

sightlines=['IC1613-61331', 'IC1613-62024', 'IC1613-64066', 'IC1613-67559', 'IC1613-67684', 'IC1613-A13', 'IC1613-B11', 'IC1613-B2', 'IC1613-B3', 'IC1613-B7', 'IC1613-C10','LEO-P-ECG26','SEXTANS-A-OB321', 'SEXTANS-A-OB326', 'SEXTANS-A-OB521','SEXTANS-A-OB523','SEXTANS-A-s3', 'WLM-A11', 'WLM-A15']
ra_hex = ['01:05:0.200', '01:05:0.6460','01:05.20700', '01:05:4.7670','01:05:4.900', '01:05:6.25', '01:04:43.8', '01:05:3.0680', '01:05:6.3700','01:05:01.97','01:04:43.39','10:21:45.10', '10:11:0.6600', '10:10:53.800','10:11:5.38','10:11:6.047','10:10:58.19', '00:01:59.97','00:02:00.533']
dec_hex=['+02:09:13.10', '+02:08:49.26','+02:09:28.10', '+02:09:23.19', '+02:09:32.60', '+02:10:43.0', '+02:06:44.75','+02:10:4.54','+02:09:31.34', '+02:08:05.10','+02:10:22.20', '18:05:16.93', '-04:40:44.30', '-04:41:13', '-04:42:40.10', '-04:42:11.37', '-04:43:18.4','-15:28:19.20','-15:29:52.41']
      
c = SkyCoord(ra=ra_hex[M], dec=dec_hex[M],  unit=(u.hourangle, u.deg))

## coordinates
vel = v_SexA
ra,dec = c.ra.degree, c.dec.degree

sightline = sightlines[M]

##spectrum file
file='/Users/ahamanowicz/Box/METALZ/COADDS/'+sightline+'_COS_coadd.fits'
print(file)
hdul = fits.open(file)
data=hdul[1].data

wav = data['WAVELENGTH']
fx = np.array(data['FLUX'])
err=np.array(data['ERROR'])
flux =fx#convolution.convolve(fx, convolution.Gaussian1DKernel(2))

#0 - wave, 1 - flux, 2 - error
#apply the lsr correction

N=int(sys.argv[2])

plot_line(n=N,wav=wav, flux=flux, vel=vel, veldown=-600, velup=1500, sightline=sightline)

