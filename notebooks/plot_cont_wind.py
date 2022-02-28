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

    lines = np.loadtxt("../linelist_cos.txt", dtype='str', skiprows=1)
    # 1) choose the line


    print(lines.T[0][n], lines.T[1][n], lines.T[2][n])
    linename=lines.T[0][n]
    line= float(lines.T[1][n])
    line_f = lines.T[2][n]
    
    # 2) convert to velocity, correct to LSR  and move to the galaxy rest frame
    
    v = (wav - line)/line * 3e5 - vel
    
    #3) catout the spectrum  +/- x around the line -check for the continuum

    i_min, i_max = min(np.where(v>veldown)[0]), max(np.where(v<velup)[0])
    #limits as for contfit
    line_cut_vel = v[i_min:i_max]
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

N=int(sys.argv[2])
M = int(sys.argv[1])

qso = Table.read("../85746-QSOdata.txt", format='csv')

sightline =qso['qso'][M]
print(sightline)
file='/Users/ahamanowicz/Dropbox/COS-SAGA/Targets/'+sightline+"/Data/"+sightline+"_nbin3_coadd.fits"
hdul = fits.open(file)

wav = hdul['WAVELENGTH'].data
flux = hdul['FLUX'].data

vel = 1457.0
#0 - wave, 1 - flux, 2 - error

plot_line(n=N,wav=wav, flux=flux, vel=vel, veldown=-2500, velup=2500, sightline=sightline)



