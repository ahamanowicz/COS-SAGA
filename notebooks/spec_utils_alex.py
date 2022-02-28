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
def gaussian(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def cut_spec_window(wav,flux, line,vel, veldown=-500, velup=500 , ra=0,dec=0):
    
    v_helio = (wav - line)/line * 3e5
    vel_lsr = spec.helio_to_lsr(v_helio,ra,dec )-vel
    i_min, i_max = min(np.where(vel_lsr>veldown)[0]), max(np.where(vel_lsr<=velup)[0])

    line_cut_vel = vel_lsr[i_min:i_max]
    line_cut_flux = flux[i_min:i_max]
    
    return(line_cut_vel, line_cut_flux)

def eqw_simple(wav, flux, vmax, vmin,  nbins=100,line=0, plot=True):
    
    #returns the EQW in mA requires choice of the integration limits
    #takes in cut spectral window with the line at vel = 0
    
    
    #width of the bin in km/s
    dx = (vmax-vmin)/nbins #km/s
    #print(dx)
    grid = np.arange(vmin, vmax, dx)
    if plot == True:
        plt.figure(figsize=(10,7))
        plt.plot(wav, flux, c='k')
        plt.axvline(vmax, c='r')
        plt.axvline(vmin, c='r')
        plt.axhline(1.0, c='gray', ls='--')
        plt.annotate("limits: "+str(vmin)+" "+str(vmax), (0.05,0.9),xycoords='axes fraction', fontsize=18)
        plt.show()
    # sum the area of the line -> sum(1-f(x')) over x' being the value of the flux at the wavelength x'. sum in flux units
    eqw_sum=0
    x_ind=0

    for x in grid:

        for i in range(np.size(wav)-1):

             if x>= wav[i] and x <wav[i+1]:
                x_ind = i
                eqw_sum+= (1-flux[x_ind])


    eqw_field = eqw_sum*dx
    
    #height of the rectangle == continuum level c =1
    eqw = eqw_field/1.
    eqw = eqw/3.e5 *line*1.e3
    return eqw

def eqw_automatic_limits(wav='', flux='', cont_err='',line=0, nbins=100, velmax=250, velmax_sampl=50, mean=1, sigmacut=5.,linename='', linewav='',sightline=''):
    
    #automatically chooses the integration limits
    #returns the EQW in mA 
    #takes in cut spectral window with the line at vel = 0
    #velmax - choose the max velocity tested for integration limits
    #mean = initial limit for the gaussian fit, manipulate if fit doesnt converge
    c = 3.e5
    lambda0=linewav 
    
    fig1 = plt.figure(1, figsize=(19,6))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    EQ=np.array([0])
    DER=np.array([0])
    limit=0
    VEL=np.linspace(0,velmax,velmax_sampl) # 
    for (vmax,j) in zip(np.linspace(1,velmax, velmax_sampl), range(velmax_sampl)):
        eqw = eqw_simple(wav, flux, vmax=vmax, vmin=-vmax, nbins=nbins, line=linewav, plot=False)
        #print(eqw)
        EQ=np.append(EQ,eqw)

        ax1.plot(vmax, eqw, 'bo')
        if j>0:
            der = EQ[j]-EQ[j-1]
            DER = np.append(DER,der)
            ax2.plot(vmax, der, 'bo')
            #print("la",der) 
    ax2.set_xlabel("integration limit [km/s]", fontsize=16)
    ax1.set_ylabel("EW [mA]",fontsize=16)
    ax2.set_ylabel(r"$\Delta$",fontsize=16)
    ax2.axhline(0, c='r')

    #fit the gaussian tot the derivative
    x = VEL
    y = DER
    n = len(x)                          #the number of data
    mean = mean
    #print(mean)#note this correction
    sigma = sum(y*(x-mean)**2)/n  
    #print(sigma)

    popt,pcov = curve_fit(gaussian,x,y,p0=[1,mean,sigma])

    ax3 = plt.subplot(133)

    plt.plot(VEL, DER, 'ob')
    plt.plot(VEL,gaussian(VEL,*popt),'ro:',label='fit')
    cut=popt[1]+sigmacut*popt[2]
    #print(popt)
    plt.axvline(cut, c="k")
    plt.xlabel("integration limit [km/s]", fontsize=16)
    plt.ylabel(r"$\Delta$",fontsize=16)
    plt.annotate("limit = "+str(round(cut,2)) + " km/s", (0.4, 0.8), xycoords='axes fraction', fontsize=18 )
    #plt.show()
    
    final_eqw = eqw_simple(wav, flux, line=linewav, vmax=cut, vmin=-cut, nbins=nbins, plot=False)
  
    #error
    E_sum = cont_err**2
    nx0 = np.size(flux)
    sigma = np.sqrt(np.sum((flux-1)**2)/(nx0-3-1))
    #chose the Eqw region
    i_min, i_max = min(np.where(wav>-cut)[0]), max(np.where(wav<cut)[0])

    cut_vel = wav[i_min:i_max]
    cut_flux = flux[i_min:i_max]

    #calculate the errors within the eqw range
    xs = max (np.abs(cut_vel))
    Xm = cut_vel/xs
    nx = np.size(Xm)
    dxi = (max(Xm)-min(Xm))/nx
    sigma_sum=0
    for i in range(i_min, i_max+1):
            
            sigma_sum+=E_sum[i] * (flux[i]**2)**2 * dxi**2 +   dxi**2    
            
    sigma_total = np.sqrt(sigma_sum) * sigma * xs * lambda0/c

    W_err = sigma_total*1e3
    print("W = "+str(round(final_eqw,3))+"+/-"+str(round( W_err,3)))
    fig1.savefig(sightline+"_"+linename+"_"+str(linewav)+"_limit-eqw.pdf")

    fig = plt.figure(3, figsize=(8,8))
    plt.step(wav, flux, c='k')
    
    plt.fill_between(cut_vel, cut_flux,1,  fc='#b20000', step='pre')
    plt.annotate(sightline, (0.1, 0.95), xycoords='axes fraction', fontsize=16 )
    plt.annotate(linename+" "+str(linewav), (0.65, 0.95), xycoords='axes fraction', fontsize=16 )
    plt.annotate("W = "+str(round(final_eqw,1))+" +/- "+str(round( W_err,1)), (0.65, .9), xycoords='axes fraction', fontsize=16 )
    plt.axvline(cut, ls='--', c='gray')
    plt.axvline(-cut, ls='--', c='gray')
    plt.xlim([-250,250])
    plt.ylim([min(cut_flux)*0.5, 1.3])
    plt.xlabel("Velocity [km/s]", fontsize=15)
    plt.ylabel("Normalized flux", fontsize=15)
    fig.savefig(sightline+"_"+linename+"_"+str(linewav)+"_eqw.pdf")
    #run eqw again for the chosen limit value
    return final_eqw, W_err, cut


def plot_line(n=0, wav='', flux='', vel=0, velcut=700, ra=0,dec=0, sightline='', linename='', linewav='', save=True ):
    # n - the line index from the line_list.txt
    # wavelength and flux tables, vel - velocity of the system
    # velcut - the cut in velocity around the spectrum +/- velcut
    
    lines = np.loadtxt("line_list.txt", dtype='str')
    # 1) choose the line


    print(lines.T[1][n], lines.T[2][n], lines.T[3][n])
    linename=lines.T[1][n]
    line= float(lines.T[2][n])
    line_f = lines.T[3][n]
    
    # 2) convert to velocity, correct to LSR  and move to the galaxy rest frame
    
    v_helio = (wav - line)/line * 3e5
    v_lsr = spec.helio_to_lsr(v_helio,ra,dec )-vel
    
    #3) catout the spectrum  +/- x around the line -check for the continuum

    i_min, i_max = min(np.where(v_lsr>-velcut)[0]), max(np.where(v_lsr<velcut)[0])

    line_cut_vel = v_lsr[i_min:i_max]
    line_cut_flux = flux[i_min:i_max]
    
    # 4) plot the line
    fig=plt.figure(figsize=(8,6))
    plt.step(line_cut_vel, line_cut_flux, c='k')
    plt.annotate( linename+ " " + str(round(line,2)), (0.7, 0.9), xycoords='axes fraction', fontsize=18 )
    plt.annotate(sightline, (0.1, 0.85), xycoords='axes fraction', fontsize=16 )

    plt.axvline(0, ls='--', c='r')
    plt.xlim([-velcut,velcut])
    plt.ylim([min(line_cut_flux),1.2*max(line_cut_flux)])
    if save==True:
        fig.savefig(sightline+"_"+linename+"_"+str(linewav)+"_line.pdf")  

def AOD(vel, I_obs_v,dv, log_f_lam, vmin=-100, vmax=100, sightline='', linename='', linewav=''):
        
    #AOD Savage+91 Na(v). Spectrum MUST be normalized to 1
    #returns the array of Na_
    i_min, i_max = min(np.where(vel>vmin)[0]), max(np.where(vel<vmax)[0])

    vel = vel[i_min:i_max]
    I_obs_v = I_obs_v[i_min:i_max]

    
    #in - the flux of the absorption line, the wavelengths of the absorpiton line and the oscilator strength (log f*lam)
    log_Na_v = np.log10(np.log(1./I_obs_v)) - log_f_lam + 14.576
    log_Na_v=ma.masked_invalid(log_Na_v)
    Na_v = np.power(10,log_Na_v)
    N = np.log10(np.sum(Na_v)*dv)
    
    fig = plt.figure(figsize=(7,7))
    plt.step(vel, Na_v, c='k')
    plt.xlabel("Velocity offset [km/s]", fontsize=18)
    plt.ylabel("Na", fontsize=18)
    plt.annotate(sightline, (0.1, 0.85), xycoords='axes fraction', fontsize=16 )
    plt.annotate(linename+" "+str(linewav), (0.65, 0.85), xycoords='axes fraction', fontsize=16 )
    plt.annotate("log(N) = "+str(round(N,3)), (0.65, 0.75), xycoords='axes fraction', fontsize=16 )

    fig.savefig(sightline+"_"+linename+"_"+str(linewav)+"_aod.pdf")
    #integrate over velocities
    
    
    return N #returns the log of N - integarted Na over the whole spec

def line_EW_AOD(sightline_no,vel_sys,line_no, vmin, vmax, sightlines_table, lines, order=3, save_spec=False, smooth=1):
    
    """
    ## lines=line table
    ##
    ## choose sightline - m - number form the table
    ##
    ## choose the line 
    ## Fe II: 0,1,2,20,21,30,31
    ## S II: 9,10, 11
    ##
    ## vmin, vmax - EW limits
    
    continuum fit velocity limits: 
    Fe II 1142, 1143, 1144:
     IC1613 -600 1200
     Sex A -500 1500
    """
   
    cont_min, cont_max = -600, 1200
    
    smoothing=convolution.Gaussian1DKernel(smooth) #COS smoothing
    m = sightline_no
    n = line_no
    ## coordinates
    vel = vel_sys

    ra,dec = sightlines_table['RA [deg]'][m], sightlines_table['Dec [deg]'][m]
    sightline = sightlines_table['sightline'][m]
    
    ##spectrum file
    file='/Users/ahamanowicz/Box/METALZ/COADDS/'+sightline+'_COS_coadd.fits'
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

def cont_fit_err(w,f, line,window_file,  vsys,nord=3, smooth=1, vlim_down=-500, vlim_up=500):
    
    vel = (w-line)/line*3.e5-vsys

    cont_win =  ascii.read(window_file)
    vmin = cont_win['col1'].data
    vmax = cont_win['col2'].data

    if smooth > 1:
        fs = convolve(flux, Box1DKernel(smooth))
    else:
        fs=f

    mask = np.isnan(fs)==True #mask nans in lfux
    cont_index = spec.get_cont_index(vmin, vmax, vel, mask = mask)
    spec_to_fit = fs[cont_index]
    vel_to_fit  = vel[cont_index]
    w_to_fit = w[cont_index]

    X,Y = vel_to_fit, spec_to_fit 
   
    xs = max(np.abs(X))
    Xm = X/xs
    # params
    lambda0 = line # Angstrom
    c = 3.e5
    nx = np.size(Xm)
    dxi = (max(Xm)-min(Xm))/nx

    # array of Legandre polynomials coefficients
    p = np.zeros((nord+1, np.size(X)))
    for k in range(0,nord+1):
        Pk=legendre(k)
        #evaluate for data points
        Pkx=Pk(Xm)  
        p[k] = Pkx 

    ncoeff = nord+1
    alpha=np.zeros((nord+1, nord+1)) #minimalisation chi^2, curvature matrix
    beta = np.zeros(nord+1)

    for i in range(nord+1):
        beta[i] = np.matmul(Y, p[i])

    for j in range(nord+1):
         for k in range(nord+1):
            alpha[j][k] = np.matmul(p[j],p[k])

    eps=np.linalg.inv(alpha) # error matrix E

    #fitting the continuum
    a=np.matmul(beta, eps) #fit coefficitents
    ncoeff = np.size(a)
    #coefs = np.polynomial.legendre.legfit(Xm, Y, nord) #x - vel, y - spec

    cont = np.polynomial.legendre.legval(Xm, a) 
    
    print(np.size(vel))
    #### continuum fit errors

    i_min, i_max = min(np.where(vel> vlim_down)[0]), max(np.where(vel<vlim_up)[0])
    vel_to_plot=vel[i_min:i_max]
    spec_to_plot =fs[i_min:i_max]
    
    print(np.size(spec_to_plot))
    
    coefs2 = np.polynomial.legendre.legfit(vel_to_fit, spec_to_fit, nord)
    cont2 = np.polynomial.legendre.legval(vel_to_plot, coefs2)
    #sigma
    yfit = cont2
    
    Xm = vel_to_plot/np.max(abs(vel_to_plot))
    nx = np.size(Xm)
    sigma = np.sqrt(np.sum((spec_to_plot-yfit)**2)/(nx-nord-1))

    sigma2_ahk = sigma**2 * eps #matrix of variances and covariances

    sigma_c = np.zeros(nx)

    for i in range(nx):
        summa= 0
        for l in range(nord+1):
            Pl =legendre(l)
            Plx = Pl(Xm[i])
            for k in range(nord+1):
                Pk =legendre(k)
                Pkx = Pk(Xm[i])
                summa += sigma2_ahk[l][k] * Plx * Plx
        sigma_c[i] =np.sqrt(summa)               
    sigma_cx = sigma_c/cont2
    if line == 1250.578:
        sigma_cx = sigma_cx[1:]
    return sigma_c, sigma_cx




def get_hi_spectrum(ra_in, dec_in, hi_file):
    
    #hi_file+="_NA_ICL001.fits"
    if isinstance(ra_in, str):
        c = SkyCoord(ra_in, dec_in, frame = 'icrs')
        ra = c.ra.deg
        dec = c.dec.deg
    else:
        ra = ra_in
        dec = dec_in

    coords = SkyCoord(ra*u.deg, dec*u.deg, frame = 'icrs')

#     t = ascii.read('/astro/dust_kg/jduval/GASS+EBHIS/cubes_eq_tan.dat')
#     rac = t['rac'].data
#     decc = t['decc'].data

#     dist = np.sqrt((ra-rac)**2 + (dec-decc)**2)
#     ind = np.argmin(dist)

#     hi_file = t['file'].data[ind]

    print("HI FILE ", hi_file)
    
    cube = fits.open("/Users/ahamanowicz/Desktop/METAL-Z/METAL-Z-I/HI_maps/" +hi_file)
    
    wcs = WCS(cube[0].header)
    wcs = wcs.dropaxis(3)
    wcs = wcs.dropaxis(2)
    print(wcs)
    x ,y= wcs.all_world2pix(ra,dec,0)

    
    spec_cube = SpectralCube.read("/Users/ahamanowicz/Desktop/METAL-Z/METAL-Z-I/HI_maps/"+ hi_file)
    velocity = spec_cube.spectral_axis*u.km/u.m
    velocity = np.array(velocity/1000.)

    spectrum = cube[0].data[:,:,int(np.round(y)), int(np.round(x))]
    
    return(velocity, spectrum)

# read in the spectrum, get the rms and get out the velocity limits
def hi_vel_limits(vel, so, noise_lim=[30,60], nsigma=1):
    #returns the velocity limits for HI in LSR km/s with zero at Sun 
    
    n = nsigma
    ## select the rms limits
    left,right = noise_lim[0], noise_lim[1] #noise limit - exclude the emission, prowide the leftmost andr rightmost indexes
    #define the noise spec
    noise=np.append(so[:left],so[right:])
    sigma = np.std(noise)

    #get the index for n*sigma velocities
    k = np.where(so == np.max(so))
    
    # define nsigma
    a=np.where(so <= n*sigma)

    ##find first index from the max hitting n*sigma line
    m=np.min(abs(a[0]-k[0]))
    #limits
    hi_lim = k-m, k+m
    
    plt.plot( vel, so)
    plt.axvline(vel[hi_lim[0]], ls='--', c='k')
    plt.axvline(vel[hi_lim[1]], ls='--',c='k')

    return(vel[hi_lim[0]], vel[hi_lim[1]])

def clump_velocity(vel_lsr, flux, vel_gal, plot=True):
    ## find the clump velocity
    ##find the local velocity
    #isolate the line based on general galaxy velocity
    a = np.where(abs(vel_lsr- vel_gal) <=100 )[0]
    amin=np.where(flux[a] == np.min(flux[a]))[0]

    if plot == True: 
        plt.plot( vel_lsr[a], flux[a])
        plt.plot(vel_lsr[a][amin], np.min(flux[a]), 'ro')
    print("clump velocity", vel_lsr[a][amin] )
    return vel_lsr[a][amin]

def eqw_with_err(wav='', flux='', cont_err='', vellim=100 , velmask=0,linename='', linewav='',sightline='', nbins=100, xlim=250, plot=True):
    
    #automatically chooses the integration limits
    #returns the EQW in mA 
    #takes in cut spectral window with the line at vel = 0
    #velmax - choose the max velocity tested for integration limits
    #mean = initial limit for the gaussian fit, manipulate if fit doesnt converge
    c = 3.e5
    lambda0=linewav 
    cut=vellim
    if velmask == 0:
        final_eqw = eqw_simple(wav, flux, line=linewav, vmax=cut, vmin=-cut, nbins=nbins, plot=False)

    elif velmask > 0:
        final_eqw = eqw_simple(wav, flux, line=linewav, vmax=velmask, vmin=-cut, nbins=nbins, plot=False)
    elif velmask < 0:
        final_eqw = eqw_simple(wav, flux, line=linewav, vmax=cut, vmin=velmask, nbins=nbins, plot=False)
    cont_err[np.isnan(cont_err)] = 0
    #error
    E_sum = cont_err**2
    nx0 = np.size(flux)
    sigma = np.sqrt(np.sum((flux-1)**2)/(nx0-3-1))
    #chose the Eqw region
    i_min, i_max = min(np.where(wav>-cut)[0]), max(np.where(wav<cut)[0])

    cut_vel = wav[i_min:i_max]
    cut_flux = flux[i_min:i_max]

    #calculate the errors within the eqw range
    xs = max (np.abs(cut_vel))
    Xm = cut_vel/xs
    nx = np.size(Xm)
    dxi = (max(Xm)-min(Xm))/nx
    sigma_sum=0
    for i in range(i_min, i_max+1):

            sigma_sum+=E_sum[i] * (flux[i]**2)**2 * dxi**2 +   dxi**2    

    sigma_total = np.sqrt(sigma_sum) * sigma * xs * lambda0/c

    W_err = sigma_total*1e3
    print("W = "+str(round(final_eqw,3))+"+/-"+str(round( W_err,3)))
    if plot == True:
        fig = plt.figure( figsize=(8,8))
        plt.step(wav, flux, c='k')

        plt.fill_between(cut_vel, cut_flux,1,  fc='#b20000', step='pre')
        plt.annotate(sightline, (0.1, 0.95), xycoords='axes fraction', fontsize=16 )
        plt.annotate(linename+" "+str(linewav), (0.65, 0.95), xycoords='axes fraction', fontsize=16 )
        plt.annotate("W = "+str(round(final_eqw,1))+" +/- "+str(round( W_err,1)), (0.65, .9), xycoords='axes fraction', fontsize=16 )
        plt.axvline(cut, ls='--', c='gray')
        plt.axvline(-cut, ls='--', c='gray')
        plt.xlim([-xlim,xlim])
        plt.ylim([min(cut_flux)*0.5, 1.3])
        plt.xlabel("Velocity [km/s]", fontsize=15)
        plt.ylabel("Normalized flux", fontsize=15)
        fig.savefig(sightline+"_"+linename+"_"+str(linewav)+"_eqw.pdf")

    #run eqw again for the chosen limit value
    return final_eqw, W_err

def hilimits(velhi,spect, vel_gal, alp=-1, pos=True,sigmacut=5.):
    #alp coeff adjusts how we calculate the derivative, invertingit sometimes help to get the fit
    so=spect[0]
    m = np.where(so==np.max(so))[0]
    dvel = velhi[m] - vel_gal
    velcorr = velhi - vel_gal - dvel
    
  
    #dreviative apprach to HI   
    velmax=120
    if pos == True:
        a = np.where((velcorr<velmax) & (velcorr > 0) )[0]
    else:
        a = np.where((velcorr<velmax) & (velcorr < 0) )[0]
    VEL=velcorr[a]

    DER=np.array([0])
    m = np.where(so==np.max(so))[0]
    m=m[0]
    so = so/np.mean(so)
    sof = convolve(so, Box1DKernel(5))
    m = np.where(sof==np.max(sof))[0]
    
    fig = plt.figure(1, figsize=(23,6))
    plt.subplot(131)
    plt.plot(velcorr, sof, c='navy')
    plt.axvline(velcorr[m], c='k')
    plt.xlabel("Velocity [km/s]")
    plt.ylabel("Flux")
    
    for j in a:

        if j >0:
            der = alp*(sof[j-1] - sof[j])
            DER = np.append(DER,der)
    
    DER=DER[1:]
    
    plt.subplot(132)
    plt.xlabel("integration limit [km/s]", fontsize=16)
    plt.ylabel(r"$\Delta$",fontsize=16)
    plt.plot(VEL, DER, 'bo')

    #fit the gaussian to the derivative
    x = VEL
    y = DER
    n = len(x)                         
    mean = 0
    #print(mean)#note this correction
    sigma = sum(y*(x-mean)**2)/n 
    sigmacut=sigmacut
    popt,pcov = curve_fit(gaussian,x,y,p0=[0,mean,sigma])

    plt.subplot(133)

    plt.plot(VEL, DER, 'ob')
    plt.plot(VEL,gaussian(VEL,*popt),'ro:',label='fit')
    cut=popt[1]+sigmacut*popt[2]
    #print(popt)
    plt.axvline(cut, c="k")
    plt.xlabel("integration limit [km/s]", fontsize=16)
    plt.ylabel(r"$\Delta$",fontsize=16)
    plt.annotate("limit = "+str(round(cut,2)) + " km/s", (0.4, 0.8), xycoords='axes fraction', fontsize=18 )

    limhi = cut
    return limhi,fig

def plot_lsf(lsf,lsfcoeff, wav, f, velclump, lim, line, line0, sightline):
    
        #find the lsf corresponding to that line
        k = np.where(abs(lsf[0] - int(line0)) <= 2)[0]
        print(k)
        lsf0 = lsf.T[k][0][1:]
    
        fig=plt.figure(figsize=(6,8))
        plt.subplot(211)
        plt.step(wav/(1+velclump/3.e5)-line0, f, c='k')
        plt.annotate(sightline, (0.1, 0.9), xycoords='axes fraction', fontsize=16 )
        plt.annotate(line+" "+str(line0), (0.65, 0.9), xycoords='axes fraction', fontsize=16 )
        limwav = lim/3.e5 *line0
        plt.axvline(limwav, ls='--', c='gray')
        plt.axvline(-limwav, ls='--', c='gray')
        plt.xlim([-1.5,1.5])
        plt.ylim([min(f)*0.5, 1.3])
        plt.ylabel("Normalized flux", fontsize=15)
        
        plt.subplot(212)
        x = range(np.size(lsf0)) * lsfcoeff

        j = np.where(lsf0 == np.max(lsf0))[0]
        mx = np.max(x)
        xx = x- x[j]

        #integrate the flux in lsf outside the 
        outind = np.where((xx < -limwav) | (xx > limwav))
        intout = np.sum(lsf0[outind]) 

        lsf = round((intout),2)

        plt.plot(xx,np.log10(lsf0), c='navy', lw=4)
        plt.xlabel(r"$\Delta \lambda$ from the line centre", fontsize=15)
        plt.axvline(limwav, ls='--', c='gray')
        plt.axvline(-limwav, ls='--', c='gray')
        #plt.subplots_adjust(hspace=0.05)
        fig.savefig(sightline+"_"+line+"_"+str(line0)+"_lsf.pdf")
        return lsf