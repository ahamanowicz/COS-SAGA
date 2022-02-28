#Load modules
import matplotlib.pyplot as plt

import numpy as np
from importlib import reload
from rbvfit import guess_profile_parameters_interactive as g
from linetools.spectra.xspectrum1d import XSpectrum1D  
from rbvfit import vfit_mcmc as mc
from rbvfit import model as m
from GUIs import rb_spec as r 

from pkg_resources import resource_filename
import pickle
from IGM import rb_setline as setline
from astropy.convolution import Gaussian1DKernel, convolve, Box1DKernel
from astropy.io import ascii, fits
from astropy import convolution
import os


def get_cont_index(vmin, vmax, vel, mask = []):

    nwin = len(vmin)
    
    
    for i in range(nwin):
        if len(mask)>0:
            this_ind = np.where((vel > vmin[i]) & (vel < vmax[i])& (mask==False))
        else:
            this_ind = np.where((vel > vmin[i]) & (vel < vmax[i]))
        this_ind = this_ind[0]
        if i == 0:
            cont_index = this_ind
        else:
            cont_index = np.concatenate((cont_index, this_ind))

    return(cont_index)

def cut_spec_window(wav,flux, line, velcut=500):
    
    vel = (wav - line)/line * 3e5

    i_min, i_max = min(np.where(vel>-velcut)[0]), max(np.where(vel<=velcut)[0])

    line_cut_vel = vel[i_min:i_max]
    line_cut_flux = flux[i_min:i_max]
    line_cut_wav=wav[i_min:i_max]
    return(line_cut_vel, line_cut_flux, line_cut_wav)

def cont_fit(w, f, line, window_file,  degree = 3, smooth = 1, outname = 'nhi_fitting_cont_fit', plt_vmin = -200., plt_vmax = 600., outdir = './', spline=False, show=True):

    target = outname.split("_")[0]
    
    vel = (w-line)/line*3.e5
    if smooth >= 1:
        fs = convolve(f, Box1DKernel(smooth))
    else:
        fs = f
        
    if (os.path.isfile(window_file)==False):
        print("CREATE THE WINDOW FILE ", window_file)
        plt.clf()
        plt.close()
        plt.plot(vel, fs, 'k-')
        plt.xlim([plt_vmin, plt_vmax])
        plt.ylim([0., 3.*np.median(fs[np.where((np.isnan(fs)==False) & (np.isinf(fs)==False))])])
        plt.show()
        plt.clf()
        plt.close()
        
    cont_win =  ascii.read(window_file)
    vmin = cont_win['col1'].data
    vmax = cont_win['col2'].data

    mask = np.isnan(f)==True

    cont_index = get_cont_index(vmin, vmax, vel, mask = mask)
    spec_to_fit = fs[cont_index]
    vel_to_fit  = vel[cont_index]
    w_to_fit = w[cont_index]

    if spline==False:
        print("USING LEGENDRE")
        coefs = np.polynomial.legendre.legfit(vel_to_fit, spec_to_fit, degree)
        cont = np.polynomial.legendre.legval(vel, coefs)

        
    else:
        print("USING SPLINE")
        spl= UnivariateSpline(vel_to_fit, spec_to_fit, k=5,check_finite=True)
        cont = spl(vel)  
        
    rms = np.std((spec_to_fit - cont[cont_index]))
    rmsn = np.std((spec_to_fit/cont[cont_index]) - 1.)

    fn = f/cont

    plt.clf()
    plt.close()
    fig = plt.figure(figsize = (15, 8))
    plt.plot(vel, fs, 'k-', label = 'Spectrum')
    plt.plot(vel_to_fit, spec_to_fit, 'b.', label = 'Range to fit', alpha = 0.5)
    plt.plot(vel, cont, 'r-', label = 'Continuum')
    plt.xlim([plt_vmin, plt_vmax])
    plt.ylim([0, 2.*np.median(spec_to_fit)])
    plt.xlabel("Heliocentric Velocity (km s" + r'$^{-1}$)', fontsize = 18)
    plt.ylabel("Flux (" + r'erg s$^{-1}$' + ' ' + r'cm$^{-2}$' + ' ' + r'A$^{-1}$' + ')' , fontsize = 18)
    plt.text(0, np.median(spec_to_fit)*1.6, str(line) + ' A', fontsize = 18)
    plt.text(0, np.median(spec_to_fit)*1.8, target, fontsize = 18)
    plt.legend(fontsize = 16)
    fig.tight_layout()
    fig =plt.gcf()
    fig.savefig(outdir + outname + '.pdf', format = 'pdf', dpi = 1000)
    if show:
        plt.show()
    plt.clf()
    plt.close()


    return(vel, cont, rms, fn, rmsn)


def compute_EW(lam,flx,wrest,lmts,flx_err,plot=False,**kwargs):
    #------------------------------------------------------------------------------------------
    #   Function to compute the equivalent width within a given velocity limits lmts=[vmin,vmax]
    #           [Only good for high resolution spectra]
    #  Caveats:- Not automated, must not include other absorption troughs within the velocity range.
    # 
    #   Input:- 
    #           lam         :- Observed Wavelength vector (units of Angstrom)
    #           flx         :- flux vector ( same length as wavelgnth vector, preferably continuum normalized)
    #           wrest       :- rest frame wavelength of the line [used to make velcity cuts]
    #           lmts        :- [vmin,vmax], the velocity window within which equivalent width is computed.
    #           flx_err     :- error spectrum [same length as the flux vector]
    #
    #   OPTIONAL :-
    #           f0=f0       :- fvalue of the transition 
    #           zabs=zabs   :- absorber redshift
    #           plot        :- plot keyword, default = no plots plot=0
    #                           plot=1 or anything else will plot the corresponding spectrum 
    #                            and the apparent optical depth of absorption. 
    #
    #
    #
    # Output:-  In a Python dictionary format
    #           output['ew_tot']      :- rest frame equivalent width of the absorpiton system [Angstrom]
    #           output['err_ew_tot']  :- error on rest fram equivalent width 
    #           output['col']         :- AOD column denisty 
    #           output['colerr']      :- 1 sigma error on AOD column density 
    #           output['n']           :- AOD column density as a function of velocity
    #           output['Tau_a']       :- AOD as a function of velocity
    #           output['med_vel']     :- velocity centroid (Median Equivalent Width weighted velocity within lmts)
    #           output['vel_disp']    : 1 sigma velocity dispersion
    #           output['vel50_err']   : error on velocity centroid
    #
    #
    #   Written :- Rongmon Bordoloi                             2nd November 2016
    #-  I translated this from my matlab code compute_EW.m, which in turn is from Chris Thom's eqwrange.pro. 
    #   This was tested with COS-Halos/Dwarfs data. 
    #   Edit:  RB July 5 2017. Output is a dictionary. Edited minor dictionary arrangement
    #          RB July 25 2019. Added med_vel
    #          RB April 28, 2021, changed med_vel to weight be EW & vel_disp
    #------------------------------------------------------------------------------------------
    defnorm=1.0;
    spl=2.9979e5;  #speed of light
    if 'zabs' in kwargs:
        zabs=kwargs['zabs']
    else:
        zabs=0.

    if 'sat_limit' in kwargs:
        sat_limit=kwargs['sat_limit']
    else:
        sat_limit=0.10 #  Limit for saturation (COS specific). Set to same as fluxcut for now. WHAT SHOULD THIS BE???
    vel = (lam-wrest*(1.0 + zabs))*spl/(wrest*(1.0 + zabs));
    lambda_r=lam/(1.+zabs);

    

    norm=defnorm

    norm_flx=flx/norm;
    flx_err=flx_err/norm;
    sq=np.isnan(norm_flx);
    tmp_flx=flx_err[sq]
    norm_flx[sq]=tmp_flx
    #clip the spectrum. If the flux is less than 0+N*sigma, then we're saturated. Clip the flux array(to avoid inifinite optical depth) and set the saturated flag
    q=np.where(norm_flx<=sat_limit);
    tmp_flx=flx_err[q]
    norm_flx[q]=tmp_flx
    q=np.where(norm_flx<=0.);
    tmp_flx=flx_err[q]+0.01
    norm_flx[q]=tmp_flx;


    del_lam_j=np.diff(lambda_r);
    del_lam_j=np.append([del_lam_j[0]],del_lam_j);


    pix = np.where( (vel >= lmts[0]) & (vel <= lmts[1]));
    Dj=1.-norm_flx

    # Equivalent Width Per Pixel
    ew=del_lam_j[pix]*Dj[pix];


    sig_dj_sq=(flx_err)**2.;
    err_ew=del_lam_j[pix]*np.sqrt(sig_dj_sq[pix]);
    err_ew_tot=np.sqrt(np.sum(err_ew**2.));
    ew_tot=np.sum(ew);

    #compute the velocity centroid of ew weighted velcity.
    ew50=np.cumsum(ew)/np.max(np.cumsum(ew))
    vel50=np.interp(0.5,ew50,vel[pix])
    vel16=np.interp(0.16,ew50,vel[pix])
    vel_disp=np.abs(vel50-vel16)
    vel50_err = vel_disp/np.sqrt(len(ew))



    print('W_lambda = ' + np.str('%.3f' % ew_tot) + ' +/- ' + np.str('%.3f' % err_ew_tot)  +'  \AA   over [' + np.str('%.1f' % np.round(lmts[0]))+' to ' +np.str('%.1f' % np.round(lmts[1])) + ']  km/s')
    output={}
    output["ew_tot"]=ew_tot
    output["err_ew_tot"]=err_ew_tot
    output["vel_disp"]=vel_disp
    output['vel50_err']=vel50_err


    if 'f0' in kwargs:
        f0=kwargs['f0']
        #compute apparent optical depth
        Tau_a =np.log(1./norm_flx);
        
        



        # REMEMBER WE ARE SWITCHING TO VELOCITY HERE
        del_vel_j=np.diff(vel);
        del_vel_j=np.append([del_vel_j[0]],del_vel_j)
        
        # Column density per pixel as a function of velocity
        nv = Tau_a/((2.654e-15)*f0*lambda_r);# in units cm^-2 / (km s^-1), SS91 
        n = nv* del_vel_j# column density per bin obtained by multiplying differential Nv by bin width 
        tauerr = flx_err/norm_flx;
        nerr = (tauerr/((2.654e-15)*f0*lambda_r))*del_vel_j; 
        col = np.sum(n[pix]);
        colerr = np.sum((nerr[pix])**2.)**0.5; 
        print('Direct N = ' + np.str('%.3f' % np.log10(col))  +' +/- ' + np.str('%.3f' % (np.log10(col+colerr) - np.log10(col))) + ' cm^-2')
        output["col"]=col
        output["colerr"]=colerr
        output["Tau_a"]=Tau_a
        output["med_vel"]=vel50
        




    # If plot keyword is  set start plotting
    if plot is not False:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1=fig.add_subplot(111)
        ax1.step(vel,norm_flx)
        ax1.step(vel,flx_err,color='r')
        #plt.xlim([lmts[0]-2500,lmts[1]+2500])
        plt.xlim([-600,600])
        plt.ylim([-0.02,1.8])
        ax1.plot([-2500,2500],[0,0],'k:')
        ax1.plot([-2500,2500],[1,1],'k:')       
        plt.plot([lmts[0],lmts[0]],[1.5,1.5],'r+',markersize=15)        
        plt.plot([lmts[1],lmts[1]],[1.5,1.5],'r+',markersize=15)    
        plt.title(r' $W_{rest}$= ' + np.str('%.3f' % ew_tot) + ' $\pm$ ' + np.str('%.3f' % err_ew_tot) + ' $\AA$')
        ax1.set_xlabel('vel [km/s]')
    
#         ax2=fig.add_subplot(212)
#         ax2.step(vel,n)
#         ax2.set_xlabel('vel [km/s]')
#         ax2.plot([-2500,2500],[0,0],'k:')
        #plt.xlim([lmts[0]-2500,lmts[1]+2500])
        plt.xlim([-600,600])
        plt.show()

    
    return output

from rbvfit import rb_setline as rb
from rbvfit.rb_vfit import rb_veldiff as rb_veldiff
def plot_model(wave_obs,fnorm,enorm,fit,model,outfile= False,xlim=[-600.,600.],verbose=False):
        #This model only works if there are no nuissance paramteres
        

        theta_prime=fit.best_theta
        value1=fit.low_theta
        value2=fit.high_theta
        n_clump=model.nclump 
        n_clump_total=np.int(len(theta_prime)/3)

        ntransition=model.ntransition
        zabs=model.zabs

        samples=fit.samples
        model_mcmc=fit.model

        wave_list=np.zeros( len(model.lambda_rest_original),)
        # Use the input lambda rest list to plot correctly
        for i in range(0,len(wave_list)):
            s=rb.rb_setline(model.lambda_rest_original[i],'closest')
            wave_list[i]=s['wave']


        wave_rest=wave_obs/(1+zabs[0])
        
        best_N = theta_prime[0:n_clump_total]
        best_b = theta_prime[n_clump_total:2 * n_clump_total]
        best_v = theta_prime[2 * n_clump_total:3 * n_clump_total]
        
        low_N = value1[0:n_clump_total]
        low_b = value1[n_clump_total:2 * n_clump_total]
        low_v = value1[2 * n_clump_total:3 * n_clump_total]
        
        high_N = value2[0:n_clump_total]
        high_b = value2[n_clump_total:2 * n_clump_total]
        high_v = value2[2 * n_clump_total:3 * n_clump_total]
            


        #Now extracting individual fitted components
        best_fit, f1 = model.model_fit(theta_prime, wave_obs)

        fig, axs = plt.subplots(ntransition, sharex=True, sharey=False,gridspec_kw={'hspace': 0})
        
        
        BIGGER_SIZE = 18
        plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        index = np.random.randint(0, high=len(samples), size=100)
        
        
        if ntransition == 1:
            #When there are no nuissance parameter
            #Now loop through each transition and plot them in velocity space
            vel=rb_veldiff(wave_list[0],wave_rest)
            axs.step(vel, fnorm, 'k-', linewidth=1.)
            axs.step(vel, enorm, color='r', linewidth=1.)
            # Plotting a random sample of outputs extracted from posterior dis
            for ind in range(len(index)):
                axs.plot(vel, model_mcmc(samples[index[ind], :], wave_obs), color="k", alpha=0.1)
            axs.set_ylim([0, 1.6])
            axs.set_xlim(xlim)
            axs.plot(vel, best_fit, color='b', linewidth=3)
            axs.plot([0., 0.], [-0.2, 2.5], 'k:', lw=0.5)
            # plot individual components
            for dex in range(0,np.shape(f1)[1]):
                axs.plot(vel, f1[:, dex], 'g:', linewidth=3)
    
            for iclump in range(0,n_clump):
                axs.plot([best_v[iclump],best_v[iclump]],[1.05,1.15],'k--',lw=4)
                text1=r'$logN \;= '+ np.str('%.2f' % best_N[iclump]) +'^{ + ' + np.str('%.2f' % (high_N[iclump]-best_N[iclump]))+'}'+ '_{ -' +  np.str('%.2f' % (best_N[iclump]-low_N[iclump]))+'}$'
                axs.text(best_v[iclump],1.2,text1,
                     fontsize=14,rotation=90, rotation_mode='anchor')
                text2=r'$b ='+np.str('%.0f' % best_b[iclump]) +'^{ + ' + np.str('%.0f' % (high_b[iclump]-best_b[iclump]))+'}'+ '_{ -' +  np.str('%.0f' % (best_b[iclump]-low_b[iclump]))+'}$'
    
                axs.text(best_v[iclump]+30,1.2, text2,fontsize=14,rotation=90, rotation_mode='anchor')
         
        
        else:
     
            
            #Now loop through each transition and plot them in velocity space
            for i in range(0,ntransition):
                print(wave_list[i])
                vel=rb_veldiff(wave_list[i],wave_rest)
                axs[i].step(vel, fnorm, 'k-', linewidth=1.)
                axs[i].step(vel, enorm, color='r', linewidth=1.)
                #pdb.set_trace()
                # Plotting a random sample of outputs extracted from posterior distribution
                for ind in range(len(index)):
                    axs[i].plot(vel, model_mcmc(samples[index[ind], :], wave_obs), color="k", alpha=0.1)
                axs[i].set_ylim([0, 1.6])
                axs[i].set_xlim(xlim)
                
                
            
                axs[i].plot(vel, best_fit, color='b', linewidth=3)
                axs[i].plot([0., 0.], [-0.2, 2.5], 'k:', lw=0.5)
    
                # plot individual components
                for dex in range(0,np.shape(f1)[1]):
                    axs[i].plot(vel, f1[:, dex], 'g:', linewidth=3)
                
                for iclump in range(0,n_clump):
                    axs[i].plot([best_v[iclump],best_v[iclump]],[1.05,1.15],'k--',lw=4)
                    if i ==0:
                        text1=r'$logN \;= '+ np.str('%.2f' % best_N[iclump]) +'^{ + ' + np.str('%.2f' % (high_N[iclump]-best_N[iclump]))+'}'+ '_{ -' +  np.str('%.2f' % (best_N[iclump]-low_N[iclump]))+'}$'
                        axs[i].text(best_v[iclump],1.2,text1,
                                 fontsize=14,rotation=90, rotation_mode='anchor')
                        text2=r'$b ='+np.str('%.0f' % best_b[iclump]) +'^{ + ' + np.str('%.0f' % (high_b[iclump]-best_b[iclump]))+'}'+ '_{ -' +  np.str('%.0f' % (best_b[iclump]-low_b[iclump]))+'}$'
                
                        axs[i].text(best_v[iclump]+30,1.2, text2,
                                 fontsize=14,rotation=90, rotation_mode='anchor')
        
        if verbose==True:
            from IPython.display import display, Math
    
            samples = fit.sampler.get_chain(discard=100, thin=15, flat=True)
            nfit = int(fit.ndim / 3)
            N_tile = np.tile("logN", nfit)
            b_tile = np.tile("b", nfit)
            v_tile = np.tile("v", nfit)
            tmp = np.append(N_tile, b_tile)
            text_label = np.append(tmp, v_tile)
            for i in range(len(text_label)):
                mcmc = np.percentile(samples[:, i], [16, 50, 84])
                q = np.diff(mcmc)
                txt = "\mathrm{{{3}}} = {0:.2f}_{{-{1:.2f}}}^{{{2:.2f}}}"
                txt = txt.format(mcmc[1], q[0], q[1], text_label[i])
    
            
                display(Math(txt))

      



        if outfile==False:
            plt.show()
        else:
            outfile_fig =outfile
            fig.savefig(outfile_fig, bbox_inches='tight')

