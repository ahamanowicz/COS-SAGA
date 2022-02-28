import spectro as spec
import numpy as np
import matplotlib.pyplot as plt
import scipy as scy
from astropy import convolution
import numpy.ma as ma
from scipy.optimize import curve_fit,least_squares
from scipy.integrate import quad
from astropy.io import fits, ascii
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from spectral_cube import SpectralCube
from astropy.convolution import convolve, Box1DKernel

from scipy.special import legendre


def continuum_fit_err(sightline_no,vel_sys,line_no, vmin, vmax, sightlines_table, lines, order=3, save_spec=False, smooth=1):e
   
    cont_min, cont_max = -600, 600
    
    smoothing=convolution.Gaussian1DKernel(smooth) #COS smoothing
    m = sightline_no
    n = line_no
    ## coordinates
    vel = vel_sys

    ra,dec = sightlines_table['RA [deg]'][m], sightlines_table['Dec [deg]'][m]
    sightline = sightlines_table['sightline'][m]
    
    ##spectrum file
    file='/Users/ahamanowicz/Dropbox/COS-SAGA/Targets/Data/'+sightline+"_nbin3_coadd.fits"
    hdul = fits.open(file)
    data=hdul[1].data

    w = data['WAVELENGTH']
    fx = np.array(data['FLUX'])
    es=np.array(data['ERROR'])

    fs= fx#convolution.convolve(fx, smoothing)
    
    # get the systemi velocity for the sightline from SII lines
    lsii = 1250.578
    #determinethe continuum near the sii line, normalize the spectrum, get the velocity components from the S II 1250 A line
    vel_sii, fn_sii, rmsn_sii, vc_sii, fc_sii = spec.find_components(w, fx, lsii, 'HI-FIT/'+sightline + '_sii_windows.dat',  smooth = 5, outname ='HI-FIT/'+sightline+ '_sii_components', main_only=True, vsys = vel, outdir  = './')
    vel = vc_sii[1]
    
    #get the line parameters 
    linename=lines.T[1][n]
    line= float(lines.T[2][n])
    line_f = float(lines.T[3][n])
    
    # continuum  fit
    #use plot_line_vel.py n to set the windows for fitting

    nanarray = np.isnan(fs) #mask nans in lfux
    notnan = ~ nanarray

    flux = fs[notnan]
    wav = w[notnan]
    err = es[notnan]
    
    cont = spec.cont_fit(w=wav, f=flux, line=line, window_file=sightline+'_line_'+str(n)+'_cont_win.txt',  degree = order, smooth = 1, outname = sightline+'line_'+str(n)+'_fitting_cont_fit', plt_vmin = cont_min, plt_vmax =cont_max, outdir = './',spline=False, show=True)
    cont_err, cont_err_norm = cont_fit_err(w=wav,f=flux,vsys=vel, line=line,window_file=sightline+'_line_'+str(n)+'_cont_win.txt', nord=order, smooth=1, vlim_down=cont_min, vlim_up=cont_max )
    print(sightline+'_line_'+str(n)+'_cont_win.txt')
    # continuum substraciton and spectral line cut
    ## continuum corrected
    wav_cont, cont_fit=cont[0], cont[1]
    
    #move to the velocity of the line
    flux_cont = flux/cont_fit
    err_norm = err/cont_fit
    vel_cut,flux_cut = cut_spec_window(wav, flux, line=line, vel=vel, veldown=cont_min,velup=cont_max, ra=ra,dec=dec )
#     plt.figure(figsize=(12,7))
#     plt.plot(vel_cut, flux_cut, 'k-')
    
    
#     #cut for EW
    vel_cut,flux_cont_cut = cut_spec_window(wav, flux_cont, line=line, vel=0, veldown=cont_min,velup=cont_max,ra=ra,dec=dec )
    vel_cut,flux_err_cut = cut_spec_window(wav, err_norm, line=line, vel=0, veldown=cont_min,velup=cont_max, ra=ra,dec=dec )
    
    print(np.size(cont_err_norm), np.size(flux_cont_cut))

    if np.size(cont_err_norm) > np.size(flux_cont_cut):
        cont_err_norm = cont_err_norm[1:]
        print(np.size(cont_err_norm), np.size(flux_cont_cut))
    if np.size(cont_err_norm) < np.size(flux_cont_cut):
        flux_cont_cut = flux_cont_cut[1:]
        vel_cut =vel_cut[1:]
        flux_err_cut = flux_err_cut[1:]
        print(np.size(cont_err_norm), np.size(flux_cont_cut))
        
#     vel_cont_f,flux_cont_f = cut_spec_window(wav_cont, cont_fit, line=line, vel=vel, velcut=500, ra=ra,dec=dec )
#     plt.plot(vel_cont_f, flux_cont_f)
    wav_cut = (vel_cut)*line/3.e5 + line
    data = np.stack((wav_cut,flux_cont_cut,flux_err_cut, cont_err_norm), axis=-1)
    ascii.write(data, sightline+'_'+linename+'_'+str(line)+".dat", overwrite=True, names=('wave','flux_norm','flux_err', "cont_err"))
    
    plt.figure(figsize=(12,7))
    
    plt.errorbar(vel_cut, flux_cont_cut, yerr=cont_err_norm,fmt='bo')
    plt.axhline(1.0,c='gray', ls='--') 
    plt.xlim([cont_min, cont_max])
    plt.ylim([0,1.5])
    
    plt.figure(figsize=(12,7))
    
    plt.plot(vel_cut, flux_cont_cut, 'k-')
    plt.axhline(1.0,c='gray', ls='--') 
    plt.xlim([cont_min, cont_max])
    plt.ylim([0,1.5])
    
#     fig=plt.figure( figsize=(12,7))
    
#     #plt.errorbar(wav_cut, flux_cont_cut, fmt='ko')
#     #### EqW calculations
#     #vmin,vmax - EW limits
#     eqw=eqw_simple(wav=vel_cut, flux=flux_cont_cut, vmax=vmax,vmin=vmin, nbins=100,line=line, plot=True)
    
#     fig=plt.figure( figsize=(8,8))
#     plt.step(vel_cut, flux_cont_cut, c='k')
#     i_min, i_max = min(np.where(vel_cut>vmin)[0]), max(np.where(vel_cut<vmax)[0])

#     cut_vel = vel_cut[i_min:i_max]
#     cut_flux = flux_cont_cut[i_min:i_max]
#     plt.fill_between(cut_vel, cut_flux,1,  fc='#b20000', step='pre')
#     plt.annotate(sightline, (0.1, 0.85), xycoords='axes fraction', fontsize=16 )
#     plt.annotate(linename+" "+str(line), (0.65, 0.9), xycoords='axes fraction', fontsize=16 )
#     plt.annotate("EW = "+str(round(eqw,3)), (0.65, 0.8), xycoords='axes fraction', fontsize=16 )
#     plt.xlim([cont_min, cont_max])
#     plt.ylim([0.2, 1.3])
#     plt.axhline(1, ls='--', c='gray')
#     plt.xlabel("Velocity [km/s]", fontsize=15)
#     plt.ylabel("Normalized flux", fontsize=15)
#     fig.savefig(sightline+"_"+linename+"_"+str(line)+"_eqw.pdf")  
#     eq=eqw
#     eq_err=0.00
    
#     #AOD N
#     dv = vel_cut[1]-vel_cut[0]
#     logN = AOD(vel=vel_cut, I_obs_v=flux_cont_cut, dv=dv, log_f_lam=line_f, vmin=vmin, vmax=vmax, linename=linename, linewav=line,sightline=sightline)
#     N_err=0.00
    
    print(sightline, round(vel,3),linename, line, line_f) #, round(eq,3), eq_err,logN, N_err)
    
    return sightline, np.round(vel,3),linename, line, line_f#, np.round(eq,3), eq_err,np.round(logN,3), N_err
