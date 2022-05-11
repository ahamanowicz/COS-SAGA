import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from astropy.io import ascii
import glob
import os
import subprocess
import sys
sys.path.append("/astro/dust_kg/jduval/METAL")
sys.path.append("/astro/dust_kg/jduval/py_utils")
from numerics import mc_lin_fits_errors, fitexy
#from metal_utils import get_depletions
from compute_gdr import compute_gdr_lv, compute_gdr_integrated
from metal_utils import *
import math
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy
#from scipy.misc import bytescale
from imaging import WCS3DtoWCS2D
#import pymc3 as pm
from scipy.stats import linregress
import corner
import matplotlib
import matplotlib.colors
matplotlib.rc('xtick', labelsize=21)
matplotlib.rc('ytick', labelsize=21)
import matplotlib as mpl
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt
from ci_analysis import cipops, compute_electron_density,compute_density_from_carbon_fine_structure
from make_region_file import make_region_file_from_list
from depletion_utils import *
from catalogs_paperIII import compute_mean_dep_offset_wZ
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.table import vstack, Table


def define_depletions_tenure():

    dep = np.arange(-4, 0.05, 0.1)
    dust_fraction = 1.-10.**(dep)

    fig =plt.figure(figsize = (10,8))
    plt.plot(dep, dust_fraction, 'k-')
    plt.xlabel("Depletion of X", fontsize = 23)
    plt.ylabel("Fraction of X in Dust", fontsize = 23)
    plt.savefig("/Users/duval/Documents/TENURE/STIC_talk/depletion_definition.pdf", format = 'pdf', dpi = 100)
    plt.clf()
    plt.close()

def plot_met_redshift_tenure(type = 'LMC', corr = True, both = True):

    dla_catalogs = np.array(['Quiret2016_DTG_table_', 'DeCia2016_all_data_'])

    fig = plt.figure(figsize = (10,8))

    if both ==False:
        if corr ==True:
            key  = np.array(['tot'])
            label = np.array(['Corrected'])
            color = np.array(['blue'])
        else:
            key = np.array(['gas'])
            label = np.array(['Uncorrected'])
            color = np.array(['red'])
    else:
        key = np.array(['gas', 'tot'])
        label = np.array(['Uncorrected', 'Corrected'])
        color = np.array(['red', 'blue'])


    for j in range(2):
        dla = fits.open(dla_catalogs[j] + type + '.fits')
        dla = dla[1].data

        good = np.where((np.isnan(dla['tot_A_Fe']) == False) & (np.isnan(dla['gas_A_Fe']) == False))

        for k in range(2):
            if j==0:
                plt.errorbar(dla['Z_ABS'][good], np.log10(dla['{}_A_Fe'.format(key[k])])[good] - 7.54 + 12., yerr = dla['err_{}_A_Fe'.format(key[k])][good]/dla['{}_A_Fe'.format(key[k])][good]/np.log(10.),fmt = 'o', alpha = 0.5, label = label[k], color = color[k])
            else:
                plt.errorbar(dla['Z_ABS'][good], np.log10(dla['{}_A_Fe'.format(key[k])])[good] - 7.54 + 12., yerr = dla['err_{}_A_Fe'.format(key[k])][good]/dla['{}_A_Fe'.format(key[k])][good]/np.log(10.), fmt = 'o', alpha = 0.5, color = color[k])

    plt.xlabel("Redshift", fontsize = 25)
    plt.ylabel("[Fe/H]", fontsize = 25)
    plt.xlim([-0.3, 5.3])
    plt.ylim([-3.5, 3])
    plt.legend(loc = 'upper right', fontsize =23)
    plt.tight_layout()
    if both==False:
        plt.savefig("FIGURES_ALL/plot_dla_feoverh_vs_z_tenure_{}_{}.pdf".format(type,label), format = 'pdf', dpi = 100)
    else:
        plt.savefig("FIGURES_ALL/plot_dla_feoverh_vs_z_tenure_both.pdf".format(type,label), format = 'pdf', dpi = 100)
    plt.clf()
    plt.close()

def plot_feldmann_tenure(log_nh0_mw = 21., use_fe_dla=False, use_gal_dla = 'dla'):

    print("COMPUTING MW")
    dgr_mw, e_dgr_mw, dmr_mw,e_dmr_mw,zgr_mw, deps, err_deps, w, a, elt = compute_gdr_lv(galaxy = 'MW',log_nh0 = np.array([log_nh0_mw]))
    #print("COMPUTING LMC")
    dgr_lmc,e_dgr_lmc,dmr_lmc,e_dmr_lmc, zgr_lmc ,deps_lmc, err_deps_lmc, wlmc, almc, elt_lmc = compute_gdr_lv(galaxy = 'LMC',log_nh0 = np.array([log_nh0_mw]))
    #print("COMPUTING SMC")

    print("SMC SMC SMC ")
    dgr_smc,e_dgr_smc, dmr_smc, e_dmr_smc, zgr_smc, deps_smc, err_deps_smc, wsmc, asmc, elt_smc = compute_gdr_lv(galaxy = 'SMC',log_nh0 = np.array([log_nh0_mw]))
    print("SMC ERR DEP ", err_deps_smc)
    #if e_dgr_smc > dgr_smc:
    #    e_dgr_smc = 0.9*dgr_smc

    dgr_smc_int, e_dgr_smc_int, dmr_smc_int, e_dmr_smc_int = compute_gdr_integrated("SMC")
    dgr_lmc_int, e_dgr_lmc_int, dmr_lmc_int, e_dmr_lmc_int = compute_gdr_integrated("LMC")
    #Kalberla+2009 show surface denisty of HI is about constant in inner galaxy at 1.5e21 cm-2. To that, add surface density of molecular gas from RD2016, or 5e20cm-2 in the inner galaxy. That leads to about 2e21 cm-2 through teh disk

    dgr_mw_int, e_dgr_mw_int, dmr_mw_int,e_dmr_mw_int,zgr_mw_int, deps_int, err_deps_int, w, a, elt_mw = compute_gdr_lv(galaxy = 'MW',log_nh0 = np.array([np.log10(2e21)]))


    dgrs= np.array([dgr_mw, dgr_lmc, dgr_smc]).flatten()
    e_dgrs = np.array([e_dgr_mw, e_dgr_lmc, e_dgr_smc]).flatten()

    dgrs_int= np.array([dgr_mw_int, dgr_lmc_int, dgr_smc_int])
    e_dgrs_int = np.array([e_dgr_mw_int, e_dgr_lmc_int, e_dgr_smc_int])

    dmrs = np.array([dmr_mw, dmr_lmc, dmr_smc]).flatten()
    e_dmrs = np.array([e_dmr_mw , e_dmr_lmc, e_dmr_smc])

    dmrs_int = np.array([dmr_mw_int, dmr_lmc_int, dmr_smc_int]).flatten()
    e_dmrs_int = np.array([e_dmr_mw_int , e_dmr_lmc_int, e_dmr_smc_int])



    #ind = np.where(e_dgrs > dgrs)
    #e_dgrs[ind] = 0.9*dgrs[ind]

    dict_fir_lmc = scipy.io.readsav("/astro/dust_kg/jduval/DUST_POOR_COMPONENT/MC_SAVE_FILES/binned_IRAS_PLANCK_surface_brightness_lmc_quad_cirrus_thresh_stray_corr_mbb_a6.400_FITS_0.05_FINAL.save")
    dict_fir_smc = scipy.io.readsav("/astro/dust_kg/jduval/DUST_POOR_COMPONENT/MC_SAVE_FILES/binned_IRAS_PLANCK_surface_brightness_smc_quad_cirrus_thresh_stray_corr_mbb_a21.00_FITS_0.05_FINAL.save")


    fir_gas_lmc = dict_fir_lmc['gas_bins']/0.8e-20
    fir_gd_lmc = dict_fir_lmc['gdr_ratio']
    fir_err_gd_lmc = dict_fir_lmc['percentile50_high'] - dict_fir_lmc['percentile50_low']#dict_fir['err_gdr_ratio']


    ind_lmc = np.argmin(np.abs(fir_gas_lmc - 10.**(log_nh0_mw)))
    fir_gd_lmc = fir_gd_lmc[ind_lmc]
    fir_gas_lmc = fir_gas_lmc[ind_lmc]
    fir_err_gd_lmc = fir_err_gd_lmc[ind_lmc]

    fir_gas_smc = dict_fir_smc['gas_bins']/0.8e-20
    fir_gd_smc = dict_fir_smc['gdr_ratio']
    fir_err_gd_smc = dict_fir_smc['percentile50_high'] - dict_fir_smc['percentile50_low']#dict_fir['err_gdr_ratio']

    ind_smc = np.argmin(np.abs(fir_gas_smc - 10.**(log_nh0_mw)))
    fir_gd_smc = fir_gd_smc[ind_smc]
    fir_gas_smc = fir_gas_smc[ind_smc]
    fir_err_gd_smc = fir_err_gd_smc[ind_smc]

    fir_gd_smc_int = dict_fir_smc['total_gas_mass']/dict_fir_smc['total_dust_mass']
    fir_err_gd_smc_int =  fir_gd_smc_int*np.sqrt((dict_fir_smc['total_dust_mass_error']/dict_fir_smc['total_dust_mass'])**2 + (dict_fir_smc['total_gas_mass_error']/dict_fir_smc['total_gas_mass'])**2)

    fir_gd_lmc_int = dict_fir_lmc['total_gas_mass']/dict_fir_lmc['total_dust_mass']
    fir_err_gd_lmc_int =  fir_gd_lmc_int*np.sqrt((dict_fir_lmc['total_dust_mass_error']/dict_fir_lmc['total_dust_mass'])**2 + (dict_fir_lmc['total_gas_mass_error']/dict_fir_lmc['total_gas_mass'])**2)

    print("DEPLETIONS DGR ")
    print("DGR INT WITH HE    ", dgrs_int)
    print("ERR DGR INT WITH HE", e_dgrs_int)

    print("DGR NH0 WITH HE    ", dgrs)
    print("ERR DGR NH0 WITH HE", e_dgrs)

    print("FIR GDRS")
    print("FIR DGR INT WITH HE     ", 1./fir_gd_lmc_int, 1./fir_gd_smc_int)
    print("ERR FIR DGR INT WITH HE ", fir_err_gd_lmc_int/fir_gd_lmc_int**2,fir_err_gd_smc_int/fir_gd_smc_int**2)

    print("FIR DGR NH0 WITH HE     ", 1./fir_gd_lmc, 1./fir_gd_smc)
    print("ERR FIR DGR NH0 WITH HE ", fir_err_gd_lmc/fir_gd_lmc**2,fir_err_gd_smc/fir_gd_smc**2)

    print("DEPLETIONS DMR ")
    print("DMR INT WITH HE    ", dmrs_int)
    print("ERR DMR INT WITH HE", e_dmrs_int)

    print("DMR NH0 WITH HE    ", dmrs)
    print("ERR DMR NH0 WITH HE", e_dmrs)


    #HERE total_dust_mass; total_dust_mass_error; total_gas_mass; total_gas_mass_error;

    zs = np.array([1., zgr_lmc/zgr_mw, zgr_smc/zgr_mw])
    gals = np.array(['MW', 'LMC', 'SMC'])
    lmc_color = 'cyan'#'dodgerblue'
    mw_color = 'gold' #'gold'#'green'
    smc_color = 'magenta'#'darkorange''
    gal_cols = [mw_color, lmc_color, smc_color]

    #fir_cols = ['gold', 'deepskyblue', 'limegreen']
    fir_cols=[mw_color, lmc_color, smc_color]
    fir_gd = np.array([136., fir_gd_lmc, fir_gd_smc])
    fir_err_gd = np.array([0., fir_err_gd_lmc,fir_err_gd_smc])
    fir_gd_int = np.array([136., fir_gd_lmc_int, fir_gd_smc_int])
    fir_err_gd_int = np.array([0., fir_err_gd_lmc_int,fir_err_gd_smc_int])

    fir_chris = np.array([np.nan, 0.00433/1.36, 0.000414/1.36])


    z0min, gamma0min, dog0min, dm0min, dp0min, alpha0min = feldmann2015_model(gamma = [2e4] )
    z0min = z0min/0.014
    z0max, gamma0max, dog0max, dm0max, dp0max, alpha0max = feldmann2015_model(gamma = [4e4] )
    z0max = z0max/0.014

    zmin, gammamin, dogmin, dmmin, dpmin, alphamin = feldmann2015_model(gamma = [2e3] )
    zmin = zmin/0.014

    zmax, gammamax, dogmax, dmmax, dpmax, alphamax = feldmann2015_model(gamma = [1e6] )
    zmax = zmax/0.014

    z0, gamma0, dog0, dm0, dp0, alpha0 = feldmann2015_model(gamma = [3e4] )
    z0 = z0/0.014


    #dla = ascii.read('decia2016_table6.dat')
    devis =ascii.read('dustpedia_combined_sample.csv')
    remy_ruyer = ascii.read('remy-ruyer2014_dust_gas_masses.dat')


    plt.clf()
    plt.close()

    fig1, ax1 = plt.subplots(nrows = 1, ncols = 1,figsize = (10, 8))
    fig2, ax2 = plt.subplots(nrows = 1, ncols = 1,figsize = (10, 8))
    fig3, ax3 = plt.subplots(nrows = 1, ncols = 1,figsize = (10, 8))

    ax1.fill_between(zmin, dogmin[:,0],dog0min[:,0], color = 'gray', alpha =0.5)
    ax1.fill_between(zmax, dog0max[:,0],dogmax[:,0], color = 'gray', alpha =0.5)
    ax1.plot(z0, dog0, 'k--')
    ax2.fill_between(zmin, dogmin[:,0],dog0min[:,0], color = 'gray', alpha =0.5)
    ax2.fill_between(zmax, dog0max[:,0],dogmax[:,0], color = 'gray', alpha =0.5)
    ax2.plot(z0, dog0, 'k--')
    ax3.fill_between(zmin, dogmin[:,0],dog0min[:,0], color = 'gray', alpha =0.5)
    ax3.fill_between(zmax, dog0max[:,0],dogmax[:,0], color = 'gray', alpha =0.5)
    ax3.plot(z0, dog0, 'k--')

    dumz = np.arange(5e-3, 3, 0.001)
    ind = np.where(z0 == 1)
    ind = ind[0]
    dumdg = dog0.flatten()[ind]*dumz
    ax1.plot(dumz, dumdg, 'k-', linewidth=3)
    ax1.text(6e-3, 8.e-5, 'D/G '  +u"\u221D" + ' Z', color = 'black', fontsize = 15, rotation = 18)
    ax1.text(0.2, 8e-6, 'Model Tracks', fontsize = 13, color = 'gray')
    ax1.text(0.2, 4e-6, '(Feldmann+2015)', fontsize = 13, color = 'gray')
    ax2.plot(dumz, dumdg, 'k-', linewidth=3)
    ax2.text(6e-3, 8.e-5, 'D/G '  +u"\u221D" + ' Z', color = 'black', fontsize = 15, rotation = 18)
    ax2.text(0.2, 8e-6, 'Model Tracks', fontsize = 13, color = 'gray')
    ax2.text(0.2, 4e-6, '(Feldmann+2015)', fontsize = 13, color = 'gray')
    ax3.plot(dumz, dumdg, 'k-', linewidth=3)
    ax3.text(6e-3, 8.e-5, 'D/G '  +u"\u221D" + ' Z', color = 'black', fontsize = 15, rotation = 18)
    ax3.text(0.2, 8e-6, 'Model Tracks', fontsize = 13, color = 'gray')
    ax3.text(0.2, 4e-6, '(Feldmann+2015)', fontsize = 13, color = 'gray')

    #First panel has DLA and DEPLETIONS BASED D/G, AND FIR D/G SCALED TO NH0_MW
    #SECOND PANEL HAS INTEGRATED D/G FROM FIR, LMC/SMC DEPLETIONS, AND DE VIS

    #DE VIS
    ax2.plot(devis['Z'].data, devis['Mdust'].data/devis['Mgas'].data, 'o',  color = 'dodgerblue', label = 'De Vis+2019 (FIR)', alpha = 0.5)
    ax2.plot(10.**(remy_ruyer['12+log(O/H)'].data-8.76), remy_ruyer['Mdust'].data/remy_ruyer['Mgas'].data, 'o', color = 'mediumblue', alpha = 0.5, label = 'Remy-Ruyer+2014 (FIR)')
    ax3.plot(devis['Z'].data, devis['Mdust'].data/devis['Mgas'].data, 'o',  color = 'dodgerblue', label = 'De Vis+2019 (FIR)', alpha = 0.5)
    ax3.plot(10.**(remy_ruyer['12+log(O/H)'].data-8.76), remy_ruyer['Mdust'].data/remy_ruyer['Mgas'].data, 'o', color = 'mediumblue', alpha = 0.5, label = 'Remy-Ruyer+2014 (FIR)')


    #DLAs - need to scale to log_nh0_mw

    if use_fe_dla==True:
        key='_Fe'
    else:
        key=''
    tfit = ascii.read("LMC_fit_DG_NH.dat")
    fslope = tfit['slope'].data[0]
    foffset= tfit['offset'].data[0]
    err_fslope = tfit['err_slope'].data[0]
    err_foffset= tfit['err_offset'].data[0]
    dla = fits.open("DeCia2016_all_data_"+use_gal_dla + key + ".fits")
    dla = dla[1].data

    scaled_dg  = dla['DTG'] + fslope*(log_nh0_mw-dla['LOG_NHI'])
    err_scaled_dg = np.sqrt(dla['err_DTG']**2 + (fslope*(log_nh0_mw-dla['LOG_NHI']))**2*((err_fslope/fslope)**2 + (dla['ERR_LOG_NHI']/dla['LOG_NHI'])**2))

    #print("DG ")
    #print(scaled_dg)
    #print("ERR ")
    #print(err_scaled_dg)

    good_dla = np.where((dla['LOG_NHI']>=19.5) & (dla['DTG'] >0))
    #test = Table()
    #test['tot_A_Fe'] = dla['tot_A_Fe'][good_dla]
    #test['DTG'] = dla['DTG'][good_dla]
    #test['LOG NHI'] = dla['LOG_NHI'][good_dla]
    #test['scaled DTG'] = scaled_dg[good_dla]

    #ascii.write(test, 'test_scaled_dtg_dla_decia.dat', overwrite=True)


    #UNCOMMENT FOR FIG4
    ax3.errorbar(dla['tot_A_Fe'][good_dla]/10.**(7.54-12.), dla['DTG'][good_dla], fmt= 'o',  color ='red', label = 'DLAs (Quiret+2016, De Cia+2016)' , xerr = dla['err_tot_A_Fe'][good_dla]/10.**(8.76-12.), yerr = dla['ERR_DTG'][good_dla], alpha = 0.5)

    dla = fits.open("Quiret2016_DTG_table_"+use_gal_dla + key + ".fits")
    dla = dla[1].data
    scaled_dg  = dla['DTG'] + fslope*(log_nh0_mw-dla['LOG_NHI'])
    err_scaled_dg = np.sqrt(dla['err_DTG']**2 + (fslope*(log_nh0_mw-dla['LOG_NHI']))**2*((err_fslope/fslope)**2 + (dla['ERR_LOG_NHI']/dla['LOG_NHI'])**2))

    #print("DG ")
    #print(scaled_dg)
    #print("ERR ")
    #print(err_scaled_dg)

    good_dla = np.where((dla['LOG_NHI']>=19.5)& (dla['DTG'] >0))

    #UNCOMMENT FO RFIG4
    ax3.errorbar(dla['tot_A_Fe'][good_dla]/10.**(7.54-12.), dla['DTG'][good_dla], fmt= 'o',  color ='red' , xerr = dla['err_tot_A_Fe'][good_dla]/10.**(8.76-12.), yerr = dla['ERR_DTG'][good_dla], alpha = 0.5)

    #test = Table()
    #test['tot_A_Fe'] = dla['tot_A_Fe'][good_dla]
    #test['DTG'] = dla['DTG'][good_dla]
    #test['LOG NHI'] = dla['LOG_NHI'][good_dla]
    #test['scaled DTG'] = scaled_dg[good_dla]

    #ascii.write(test, 'test_scaled_dtg_dla_quiret.dat', overwrite=True)


    for i in range(len(gals)):

        linewidth = 4

        #UNCOMMENT AND CHANGE FILENAME FOR FIG3 TO GET FIG4
        #ax3.errorbar(zs[i], dgrs[i], yerr = e_dgrs[i],fmt='o', color = gal_cols[i], label = gals[i] + ' (Depletions, log N(H)={:3.0f} cm'.format(log_nh0_mw)+r'$^{-2}$' + ')', markersize = 15, alpha = 0.7, markerfacecolor = 'none', linewidth = linewidth)
        ax3.errorbar(zs[i], dgrs_int[i], yerr = e_dgrs_int[i],fmt='o', color = gal_cols[i], label = gals[i] + ' (Depletions, integrated)', markersize = 17, alpha= 0.7)
        #ax3.plot(zs[i], dgrs[i],'o', markersize = 15, markerfacecolor = 'none', linewidth = linewidth, color = gal_cols[i])
        ax3.plot(zs[i], dgrs_int[i], 'o', markersize = 17, color = gal_cols[i])

        #ax1.errorbar(zs[i], [1./fir_gd[i]], yerr = fir_err_gd[i]/fir_gd[i]**2, fmt= '*', color = fir_cols[i], label = gals[i] + ' (FIR, log N(H)={:3.0f} cm'.format(log_nh0_mw)+r'$^{-2}$' + ')', markersize = 23, alpha = 0.7, markerfacecolor = 'none', linewidth = linewidth)
        ax1.errorbar(zs[i], [1./fir_gd_int[i]], yerr = fir_err_gd_int[i]/fir_gd_int[i]**2, fmt= '*', color = fir_cols[i], label = gals[i] + ' (FIR, RD2017)', markersize = 23, alpha  = 0.7)
        #ax2.errorbar(zs[i], [1./fir_gd[i]], yerr = fir_err_gd[i]/fir_gd[i]**2, fmt= '*', color = fir_cols[i], label = gals[i] + ' (FIR, log N(H)={:3.0f} cm'.format(log_nh0_mw)+r'$^{-2}$' + ')', markersize = 23, alpha = 0.7, markerfacecolor = 'none', linewidth = linewidth)
        ax2.errorbar(zs[i], [1./fir_gd_int[i]], yerr = fir_err_gd_int[i]/fir_gd_int[i]**2, fmt= '*', color = fir_cols[i], label = gals[i] + ' (FIR, RD2017)', markersize = 23, alpha  = 0.7)
        #ax3.errorbar(zs[i], [1./fir_gd[i]], yerr = fir_err_gd[i]/fir_gd[i]**2, fmt= '*', color = fir_cols[i], label = gals[i] + ' (FIR, log N(H)={:3.0f} cm'.format(log_nh0_mw)+r'$^{-2}$' + ')', markersize = 23, alpha = 0.7, markerfacecolor = 'none', linewidth = linewidth)
        ax3.errorbar(zs[i], [1./fir_gd_int[i]], yerr = fir_err_gd_int[i]/fir_gd_int[i]**2, fmt= '*', color = fir_cols[i], label = gals[i] + ' (FIR, RD2017)', markersize = 23, alpha  = 0.7)

        ax1.plot(zs[i], fir_chris[i], '*', color = fir_cols[i], markerfacecolor = 'none', markersize = 23, label = gals[i] + ' (FIR, Clark+2021)')
        ax2.plot(zs[i], fir_chris[i], '*', color = fir_cols[i], markerfacecolor = 'none', markersize = 23, label = gals[i] + ' (FIR, Clark+2021)')
        ax3.plot(zs[i], fir_chris[i], '*', color = fir_cols[i], markerfacecolor = 'none', markersize = 23, label = gals[i] + ' (FIR, Clark+2021)')


    ax1.set_xlim(left = 5.e-3,right=3.)
    ax1.set_ylim(bottom = 1.e-7, top = 0.1)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel("Z [" + r'$Z_o$' + "]", fontsize = 22)
    ax1.set_ylabel('D/G', fontsize = 22)
    #ax[1].set_ylabel('D/G', fontsize = 16)
    ax1.legend(fontsize = 12, loc = 'upper left')
    fig1.subplots_adjust(top = 0.95, bottom = 0.12, left = 0.12, right = 0.95)

    ax2.set_xlim(left = 5.e-3,right=3.)
    ax2.set_ylim(bottom = 1.e-7, top = 0.1)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel("Z [" + r'$Z_o$' + "]", fontsize = 22)
    ax2.set_ylabel('D/G', fontsize = 22)
    #ax[1].set_ylabel('D/G', fontsize = 16)
    ax2.legend(fontsize = 12, loc = 'upper left')
    fig2.subplots_adjust(top = 0.95, bottom = 0.12, left = 0.12, right = 0.95)

    ax3.set_xlim(left = 5.e-3,right=3.)
    ax3.set_ylim(bottom = 1.e-7, top = 0.1)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel("Z [" + r'$Z_o$' + "]", fontsize = 22)
    ax3.set_ylabel('D/G', fontsize = 22)
    #ax[1].set_ylabel('D/G', fontsize = 16)
    #ax3.legend(fontsize = 12, loc = 'upper left')
    fig3.subplots_adjust(top = 0.95, bottom = 0.12, left = 0.12, right = 0.95)



    fig1.savefig("FIGURES_ALL/plot_feldmann2015_mw_lmc_smc_"+ use_gal_dla + key+"_FIG1.pdf", format= 'pdf', dpi = 1000)
    fig2.savefig("FIGURES_ALL/plot_feldmann2015_mw_lmc_smc_"+ use_gal_dla + key+"_FIG2.pdf", format= 'pdf', dpi = 1000)
    fig3.savefig("FIGURES_ALL/plot_feldmann2015_mw_lmc_smc_"+ use_gal_dla + key+"_FIG4.pdf", format= 'pdf', dpi = 1000)
    plt.clf()
    plt.close()


def plot_jenkins(galaxy, newc=False):

    if galaxy == 'MW':
        if newc==False:
            t = fits.open("compiled_depletions_jenkins2009_py_adj_zeropt_zn.fits")
        else:
            t = fits.open("compiled_depletions_jenkins2009_py_adj_zeropt_zn_newc.fits")
    if galaxy == 'SMC':
        t = fits.open("compiled_depletions_jenkins2017_py.fits")
    if galaxy == 'LMC':
        t = fits.open("compiled_depletion_table_cos_stis_wti_corr_sii_ebj_oi_strong_zn_fstar_fix1220_pyh.fits")
    t = t[1].data

    #indel = np.where(t['ELEMENT'] == X)
    #indel= indel[0][0]

    indzn = np.where((t['ELEMENT'] == 'Zn') | (t['ELEMENT'] == 'ZnII'))
    indzn = indzn[0][0]

    indfe = np.where((t['ELEMENT'] == 'Fe') | (t['ELEMENT']=='FeII'))
    indfe = indfe[0][0]

    #ax, e_ax, bx, e_bx, zx = get_fstar_coefs(galaxy, X, use_mw=False)
    afe, e_afe, bfe, e_bfe, zfe = get_fstar_coefs(galaxy, 'Fe', use_mw=False)
    azn,e_azn, bzn, e_bzn, zzn = get_fstar_coefs(galaxy, 'Zn', use_mw=False)

    #azn=-0.5737505499972928
    #bzn = -0.40884941085858206

    #afe = -1.16
    #bfe = -1.48

    znfe = t['DEPLETIONS'][:, indzn]-t['DEPLETIONS'][:, indfe]
    err_znfe = np.sqrt(t['ERR_DEPLETIONS'][:, indzn]**2  + t['ERR_DEPLETIONS'][:, indfe]**2)

    if galaxy == 'LMC':
        fstar = t['FSTAR_MW']
        err_fstar = t['ERR_FSTAR_MW']
    else:
        fstar = t['FSTAR']
        err_fstar = t['ERR_FSTAR']


    #fstar = (t['DEPLETIONS'][:, indfe] + 1.513)/-1.285 + 0.437
    #err_fstar = t['ERR_DEPLETIONS'][:, indfe]/1.285

    a2fe_me = 0.7828285925925926
    b2fe_me = -1.9037037037037037

    a2zn_me =0.7828285925925926
    b2zn_me = -0.9037037037037038

    a2fe_decia = -0.01
    b2fe_decia = -1.26
    a2zn_decia = 0.0
    b2zn_decia = -0.27


    dumx = np.arange(-1,2, 0.1)

    fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize = (5,10), sharex = True)
    fig2, ax2= plt.subplots(nrows = 2, ncols = 2, figsize  = (10,8 ), sharex = True, sharey = True)

    #ax[0].errorbar(fstar, t['DEPLETIONS'][:, indzn], xerr = err_fstar, yerr = t['ERR_DEPLETIONS'][:, indzn], fmt = 'ko', alpha=0.5)
    #ax[0].plot(dumx, bzn  + azn*(dumx-zzn), 'r--')

    print(azn,  bzn)

    ax[0].errorbar(fstar, (t['DEPLETIONS'][:, indzn] - bzn - azn*(fstar - zzn)), xerr = err_fstar, yerr = t['ERR_DEPLETIONS'][:, indzn] , fmt = 'ko', alpha=0.5)
    ax[0].plot(dumx, np.zeros_like(dumx), 'r--')

    ax2[0,0].errorbar(znfe, t['DEPLETIONS'][:, indzn] - a2zn_me - b2zn_me*znfe, xerr = err_znfe, yerr= t['ERR_DEPLETIONS'][:, indzn], label = 'JRD', fmt = 'ko', alpha = 0.5)
    ax2[1,0].errorbar(znfe, t['DEPLETIONS'][:, indfe] - a2fe_me - b2fe_me*znfe, xerr = err_znfe, yerr= t['ERR_DEPLETIONS'][:, indfe], label = 'JRD', fmt = 'ko', alpha = 0.5)
    ax2[0,1].errorbar(znfe, t['DEPLETIONS'][:, indzn] - a2zn_decia - b2zn_decia*znfe, xerr = err_znfe, yerr= t['ERR_DEPLETIONS'][:, indzn], label = 'DC', fmt = 'ko', alpha = 0.5)
    ax2[1,1].errorbar(znfe, t['DEPLETIONS'][:, indfe] - a2fe_decia - b2fe_decia*znfe, xerr = err_znfe, yerr= t['ERR_DEPLETIONS'][:, indfe], label = 'DC', fmt = 'ko', alpha = 0.5)
    ax2[0,0].legend()
    ax2[0,1].legend()
    ax2[0,0].plot(dumx, np.zeros_like(dumx), 'r--')
    ax2[1,0].plot(dumx, np.zeros_like(dumx), 'r--')
    ax2[0,1].plot(dumx, np.zeros_like(dumx), 'r--')
    ax2[1,1].plot(dumx, np.zeros_like(dumx), 'r--')
    ax2[1,0].set_xlabel("[Zn/Fe]", fontsize = 20)
    ax2[0,0].set_ylabel(r'$\delta$(Zn)', fontsize = 20)
    ax2[1,0].set_ylabel(r'$\delta$(Fe)', fontsize = 20)
    ax2[1,1].set_xlabel("[Zn/Fe]", fontsize = 20)


    #ax[1].errorbar(fstar, t['DEPLETIONS'][:, indfe], xerr = err_fstar, yerr = t['ERR_DEPLETIONS'][:, indfe], fmt = 'ko', alpha=0.5)
    #ax[1].plot(dumx, bfe  + afe*(dumx-zfe), 'r--')
    ax[1].errorbar(fstar, (t['DEPLETIONS'][:, indfe] - bfe-afe*(fstar-zfe)), xerr = err_fstar, yerr = t['ERR_DEPLETIONS'][:, indfe], fmt = 'ko', alpha=0.5)
    ax[1].plot(dumx, np.zeros_like(dumx) , 'r--')


    #ax[2].errorbar(fstar, znfe, xerr = err_fstar, yerr = err_znfe, fmt = 'ko', alpha=0.5)
    #ax[2].plot(dumx, bzn-bfe + azn*(dumx-zzn) - afe*(dumx-zfe), 'r--')
    ax[2].errorbar(fstar, znfe - (bzn-bfe + azn*(fstar-zzn) - afe*(fstar-zfe)) , xerr = err_fstar, yerr = err_znfe, fmt = 'ko', alpha=0.5)

    ax[2].plot(dumx, np.zeros_like(dumx), 'r--')

    elements = np.array([r'$\delta$(Zn)', r'$\delta$(Fe)', '[Zn/Fe]'])
    labels =np.array( ['Residual on {}'.format(element) for element in elements])
    for i in range(3):
        ax[i].set_xlim(left = -1, right = 1.5)
        ax[i].set_ylabel(labels[i], fontsize = 18)

    ax[2].set_xlabel(r'F$_*$', fontsize = 23)
    fig.subplots_adjust(wspace = 0, hspace = 0, left = 0.25, right = 0.95, top = 0.98, bottom = 0.1)
    fig.savefig("FIGURES_ALL/plot_fstar_zn_fe_{}_JRD_FIT.pdf".format(galaxy), format = 'pdf', dpi = 100)
    fig2.subplots_adjust(wspace = 0, hspace = 0, left = 0.2, right = 0.95, top = 0.98, bottom = 0.1)
    fig2.savefig("plot_residuals_me_decia.pdf", format = 'pdf', dpi = 100)

    plt.clf()
    plt.close()




def plot_test_new_dla_dep_method_on_mw_lmc_smc(galaxy):

    """
    Here, I take the [Zn/Fe] from the MW, LMC, SMC measurements and apply the Decia and my new prescriptions for getting depletions out of [Zn/Fe] and I compare the resulting metallicities to see which ones work best
    """

    elements = np.array(['Mg', 'Si', 'S', 'Cr', 'Fe', 'Zn'])
    nel = len(elements)

    fig, ax = plt.subplots(figsize = (12, 8), nrows= 2, ncols = 3, sharex = True, sharey = True)
    fig2, ax2 = plt.subplots(figsize = (12, 8), nrows= 2, ncols = 3, sharex = True, sharey = True)

    gal_colors = np.array(['darkorange', 'dodgerblue', 'magenta'])
    gal_markers = np.array(['^', '*', 'x'])

    gals = np.array(['MW', 'LMC', 'SMC'])
    #gals = np.array([galaxy])

    for i in range(nel):

        x = i % 3
        y = i//3

        if galaxy =='MW':
            dep_table= fits.open("compiled_depletions_jenkins2009_py_adj_zeropt_zn.fits")
            dep_table = dep_table[1].data
            indzn = np.where(dep_table['Element'] == 'Zn')[0][0]
            indfe = np.where(dep_table['Element'] == 'Fe')[0][0]
            indx = np.where(dep_table['Element'] == elements[i] )[0][0]

            good = np.where(dep_table['LOG_NH']>19.5)
            dep_table = dep_table[good]

        if galaxy == 'SMC':
            dep_table= fits.open("compiled_depletions_jenkins2017_py.fits")
            dep_table = dep_table[1].data
            indzn = np.where(dep_table['Element'] == 'ZnII')[0][0]
            indfe = np.where(dep_table['Element'] == 'FeII')[0][0]
            indx = np.where(dep_table['Element'] == elements[i] + 'II')[0][0]
        if galaxy=='LMC':
            dep_table = fits.open("compiled_depletion_table_cos_stis_wti_corr_sii_ebj_oi_strong_zn_fstar_fix1220_pyh.fits")
            dep_table = dep_table[1].data
            indzn = np.where(dep_table['Element'] == 'ZnII')[0][0]
            indfe = np.where(dep_table['Element'] == 'FeII')[0][0]
            indx = np.where(dep_table['Element'] == elements[i] + 'II')[0][0]
            good = np.where((dep_table['FLAG'][:, indzn] == 'v') & (dep_table['FLAG'][:, indfe] == 'v') & (dep_table['FLAG'][:, indx] == 'v'))
            dep_table = dep_table[good]


        azn_sol = get_ref_abundance('Zn', 'MW')
        afe_sol = get_ref_abundance('Fe', 'MW')
        #azn_gal = get_ref_abundance('Zn', gals[j])
        #zfe_gal = get_ref_abundance('Fe', gals[j])

        znfe = dep_table['LOG_NX'][:, indzn]-dep_table['LOG_NX'][:, indfe] - (azn_sol-afe_sol)
        err_znfe = np.sqrt(dep_table['ERR_LOGNX'][:, indzn]**2 + dep_table['ERR_LOGNX'][:, indfe]**2)
        true_deps = dep_table['DEPLETIONS'][:, indx]
        err_true_deps = dep_table['ERR_DEPLETIONS'][:, indx]

        deps_decia = np.zeros_like(znfe)
        err_deps_decia = np.zeros_like(znfe)

        for it in range(len(znfe)):
            deps_decia[it], err_deps_decia[it] = get_depletions_decia(znfe[it], err_znfe[it], elements[i], use_gal = 'dla')

        bad = np.where(deps_decia >0.)
        if len(bad[0])>0:
            deps_decia[bad] = 0.

        #ax[y,x].errorbar( true_deps,deps_decia, xerr = err_true_deps,  yerr = err_deps_decia, color = 'black', fmt = 'o', label = 'De Cia+2016 [Zn/Fe] - ' +  r'$\delta$(X) relation')

        residual_decia2 = np.sqrt(np.nanmean((true_deps-deps_decia)**2)) #/(err_true_deps + err_deps_decia)**2))
        residual_decia0  = np.nanmean((deps_decia-true_deps))#/np.sqrt(err_deps_decia**2 + err_true_deps**2))
        residual_decia1 = np.nanmean(np.abs(deps_decia-true_deps))

        print("Residual DeCia ", elements[i], residual_decia0, residual_decia1, residual_decia2)

        ax[y,x].errorbar(znfe, deps_decia-true_deps,  yerr = np.sqrt(err_deps_decia**2 + err_true_deps**2), xerr= err_znfe,  color = 'black', fmt = 'o', label = 'De Cia+16 ' + '({:4.2f}; {:4.2f})'.format(residual_decia0, residual_decia2), alpha = 0.5 )

        ax2[y,x].errorbar(true_deps, deps_decia,  yerr = err_deps_decia, xerr= err_true_deps,  color = 'black', fmt = 'o', label = 'De Cia+2016' ,alpha = 0.5 )
        #ax[y,x].errorbar(znfe, true_deps,  yerr = err_true_deps, xerr= err_znfe, fmt = 'o', color = 'green', label = 'True')
        #ax[y,x].errorbar(znfe, deps_decia, xerr = err_znfe, yerr = err_deps_decia, fmt = 'o', color = 'black', label = 'De Cia')



        #looping over the galaxies for the depletion relations
        for k in range(len(gals)):
            deps_gal = np.zeros_like(znfe)
            err_deps_gal = np.zeros_like(znfe)
            for it in range(len(znfe)):
                deps_gal[it], err_deps_gal[it] = get_depletions_decia(znfe[it], err_znfe[it], elements[i], use_gal= gals[k])

            bad = np.where(deps_gal >0.)
            if len(bad[0])>0:
                deps_gal[bad] = 0.
            residual2 = np.sqrt(np.nanmean((true_deps-deps_gal)**2))#/(err_true_deps**2 + err_deps_gal**2))) #relative to error
            residual1 = np.nanmean(np.abs(true_deps-deps_gal))
            residual0 = np.nanmean((deps_gal-true_deps))#/np.sqrt(err_deps_gal**2 + err_true_deps**2)) #previously experessed as fraction of error
            print("Residual ", elements[i], gals[k], residual0, residual1, residual2)

            #ax[y,x].errorbar(true_deps, deps_gal, xerr = err_true_deps, yerr = err_deps_gal, fmt = gal_markers[k], color = gal_colors[k], label =gals[k] + ' [Zn/Fe -' + r'$\delta$(X) relation')
            #ax[y,x].errorbar(znfe, deps_gal, xerr = err_znfe, yerr=err_deps_gal, fmt = 'o', color = gal_colors[k], label = gals[k]+ ' [Zn/Fe -' + r'$\delta$(X) relation')
            ax[y,x].errorbar(znfe, deps_gal-true_deps, yerr = np.sqrt(err_deps_gal**2 + err_true_deps**2), xerr= err_znfe, fmt = gal_markers[k], color = gal_colors[k],label = gals[k] + ' ({:4.2f}; {:4.2f})'.format(residual0, residual2), alpha = 0.5)
            ax2[y,x].errorbar(true_deps, deps_gal, yerr = err_deps_gal, xerr= err_true_deps, fmt = gal_markers[k], color = gal_colors[k],label = gals[k], alpha = 0.5)

        if y==1:
            #ax[y,x].set_xlabel('Measured ' + r'$\delta$(X)', fontsize = 20)
            ax[y,x].set_xlabel("[Zn/Fe]", fontsize = 20)
            ax2[y,x].set_xlabel("Measured Depletion", fontsize = 20)
        if x==0:
            if y == 1:
                ax[y,x].set_ylabel("                                    Estimated - Measured Depletion in the {} ".format(galaxy), fontsize =20)
            ax2[y,x].set_ylabel("Estimated Depletion", fontsize = 20)
        ax[y,x].text(-0.45, 1.5, elements[i], fontsize = 18)
        ax2[y,x].text(-2.9, 0.5, elements[i], fontsize = 18)
        ax[y,x].minorticks_on()
        ax2[y,x].minorticks_on()

        ax[y,x].tick_params(which='major', width=2, length = 6)
        ax[y,x].tick_params(which='minor', width=1, length = 4)
        ax2[y,x].tick_params(which='major', width=2, length = 6)
        ax2[y,x].tick_params(which='minor', width=1, length = 4)


        #ax[y,x].set_xlim(left = -4, right = 1)
        ax[y,x].set_xlim(left = -0.5, right = 2.5)
        ax[y,x].set_ylim(bottom = -2.5, top = 2)
        ax2[y,x].set_xlim(left = -3, right = 1)
        ax2[y,x].set_ylim(bottom = -3.5, top = 1)
        #ax[y,x].set_ylim(bottom = -2, top = 2)
        dumx = np.arange(-0.5,2.5, 0.1)
        dumx2 = np.arange(-3.5, 1, 0.1)
        #ax[y,x].plot(dumx, dumx, 'k-')
        #ax[y,x].plot(dumx, dumx + 0.5, 'k--')
        #ax[y,x].plot(dumx, dumx-0.5, 'k--')
        ax[y,x].plot(dumx, np.zeros_like(dumx), 'r--')
        ax2[y,x].plot(dumx2, dumx2, 'r--')

        ax[y,x].legend(loc = 'lower left', fontsize =12)
        ax2[y,x].legend(loc = 'lower right', fontsize = 12)

    fig.subplots_adjust(left = 0.12, right = 0.95, bottom = 0.12, top = 0.95, wspace = 0., hspace = 0.)
    fig2.subplots_adjust(left = 0.12, right = 0.95, bottom = 0.12, top = 0.95, wspace = 0., hspace = 0.)
    fig.savefig('FIGURES_ALL/plot_test_MW_LMC_SMC_depletions_new_prescriptions_DIF_{}.pdf'.format(galaxy), format = 'pdf', dpi =100)
    fig2.savefig('FIGURES_ALL/plot_test_MW_LMC_SMC_depletions_new_prescriptions_{}.pdf'.format(galaxy), format = 'pdf', dpi =100)
    plt.clf()
    plt.close()

def plot_dust_composition(use_gal_dla = 'dla'):

    title_type = use_gal_dla
    if title_type == 'dla':
        title_type = 'De Cia+2016'

    dla_catalogs = np.array(['Quiret2016_DTG_table_', 'DeCia2016_all_data_'])

    colors = [ 'blue',  'green', 'darkorange', 'magenta', 'red']
    elements = [ 'C', 'O', 'Mg', 'Si', 'Fe']

    dla1 = fits.open('Quiret2016_DTG_table_'+ use_gal_dla + '.fits')
    dla1 = dla1[1].data
    good_dla = np.where((dla1['LOG_NHI']>=19.5)& (dla1['DTG'] >0))
    dla1 = Table(dla1[good_dla])

    dla2 = fits.open('DeCia2016_all_data_'+ use_gal_dla + '.fits')
    dla2 = dla2[1].data
    good_dla = np.where((dla2['LOG_NHI']>=19.5)& (dla2['DTG'] >0))
    dla2 = Table(dla2[good_dla])

    dla = vstack([dla1, dla2])

    metallicity = dla['tot_A_Fe']/10.**(7.54-12.)

    ind_sort = np.argsort(metallicity)
    dla = dla[ind_sort]
    metallicity = metallicity[ind_sort]

    mass_frac_x = np.zeros((len(dla), len(elements)), dtype = 'float32')


    for j in range(len(elements)):

        aref = get_ref_abundance(elements[j], 'MW')
        aref = 10**(aref-12.)
        wref = get_atomic_weight(elements[j])

        measured = np.where((np.isnan(dla['tot_A_{}'.format(elements[j])]) == False) & (dla['lim_{}'.format(elements[j])] =='v'))
        estimated = np.where((np.isnan(dla['tot_A_{}'.format(elements[j])]) == True) |  (dla['lim_{}'.format(elements[j])] !='v'))

        mass_frac_x[measured, j] = (1-10**(dla['dep_{}'.format(elements[j])][measured]))*dla['tot_A_{}'.format(elements[j])][measured]*wref/dla['DTG'][measured]/1.36
        mass_frac_x[estimated, j] = (1-10**(dla['dep_{}'.format(elements[j])][estimated]))*dla['est_tot_A_{}'.format(elements[j])][estimated]*wref/dla['DTG'][estimated]/1.36


    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (10,15), sharex = True, gridspec_kw={'height_ratios': [4,1]})
    #ycoord = np.arange(len(dla))
    #ylabels = ['{:4.2f}'.format(x) for x in np.log10(metallicity)]
    left = len(dla)*[0]
    for idx in range(len(elements)):
        ax[0].barh(np.log10(metallicity), mass_frac_x[:, idx], left = left, color= colors[idx], height = 0.01, alpha= 0.7)#, tick_label = ylabels)
        left= left + mass_frac_x[:, idx]

    #Now the local Group
    gals = np.array(['MW','LMC', 'SMC'])
    mets = np.array([1., 0.5, 0.2])
    frac_x_lg = np.zeros((3, len(elements)), dtype = 'float32')
    for i in range(3):

        #the median log N(H) of DLAs with metallicities > 20% solar is 20.5, or 3.16e20
        dgr, err_dgr, dmr,err_dmr,zgr_mw_int, deps, err_deps, w, a, elt= compute_gdr_lv(galaxy = gals[i],log_nh0 = np.array([np.log10(3.16e20)]))
        deps = deps.flatten()

        for j in range(len(elements)):
            aref = get_ref_abundance(elements[j], gals[i])
            aref = 10**(aref-12.)
            wref = get_atomic_weight(elements[j])

            ind_el = np.where(elt == elements[j] )
            frac_x_lg[i, j] =  (1-10.**(deps[ind_el]))*wref*aref/(dgr[0]*1.36)

    left = 3*[0]
    for idx in range(len(elements)):
        ax[0].barh(np.log10(mets), frac_x_lg[:, idx], left = left, color= colors[idx], height = 0.04, alpha= 0.5, edgecolor = 'black', linewidth = 2)#, tick_label = ylabels)
        left= left + frac_x_lg[:, idx]

    #plot difference dust types in the second PANEL

    dust = np.array(['MgSiO'+r'$_3$', 'Mg'+r'$_2$' + 'SiO'+r'$_4$', 'Mg'+r'$_2$' + 'Fe' + r'$_2$' + 'SiO' + r'$_4$', 'SiC', 'FeC', 'C'])
    ycoord= np.linspace(0 , 1, 6)
    frac_known  = np.zeros((6, len(elements)), dtype = 'float32')
    frac_known[0, 0]= 0.
    frac_known[0, 1]  = 3.*16./(3.*16 + 24. + 28.)
    frac_known[0, 2] = 24./(3.*16 + 24. + 28.)
    frac_known[0,3] = 28/(3.*16 + 24. + 28.)
    frac_known[0, 4] = 0.

    frac_known[1, 0] = 0
    frac_known[1, 1]  = 4*16./(4.*16 + 2*24 + 28.)
    frac_known[1, 2] = 2*24./(4.*16 + 2*24 + 28.)
    frac_known[1, 3] = 28./(4.*16 + 2*24 + 28.)
    frac_known[1, 4] = 0.

    frac_known[2, 0] = 0.
    frac_known[2, 1] = 4.*16./(4.*16 + 2*24. + 2*56  + 28.)
    frac_known[2, 2] = 2*24./(4.*16 + 2*24. + 2*56  + 28.)
    frac_known[2, 3] = 28./(4.*16 + 2*24. + 2*56  + 28.)
    frac_known[2, 4] = 2*56./(4.*16 + 2*24. + 2*56  + 28.)

    frac_known[3, 0] = 12./(12. + 28.)
    frac_known[3, 1] = 0.
    frac_known[3, 2] = 0.
    frac_known[3, 3] = 28./(12.  + 28.)
    frac_known[3, 4] = 0.

    frac_known[4, 0] =12./(12. + 56.)
    frac_known[4, 1] = 0.
    frac_known[4, 2] = 0.
    frac_known[4, 3] = 0.
    frac_known[4, 4] = 56./(12. + 56.)


    frac_known[5, 0] = 1.
    frac_known[5, 1] = 0.
    frac_known[5, 2] = 0.
    frac_known[5, 3] = 0.
    frac_known[5, 4] = 0.

    left = 6*[0]
    for idx in range(len(elements)):
        ax[1].barh(ycoord, frac_known[:, idx], left = left, color= colors[idx], height = 0.1, alpha= 0.7, tick_label = dust)
        left= left + frac_known[:, idx]


    #ax.set_yticklabels(ylabels)
    ax[0].set_title("Dust Composition for the {} calibration".format(title_type) , fontsize = 20)
    ax[1].set_xlabel("Fraction of dust mass contributed by X", fontsize =22)
    ax[0].set_ylabel("log Z/Z" + r'$_0$', fontsize = 22)
    ax[0].set_ylim(bottom = -2.2, top = 1)
    labels = elements
    ax[0].legend(labels, fontsize = 16, ncol=len(elements), frameon=False)
    ax[0].spines['right'].set_visible(True)
    ax[0].spines['left'].set_visible(True)
    ax[0].spines['top'].set_visible(True)
    ax[0].spines['bottom'].set_visible(True)
    ax[0].set_axisbelow(True)
    ax[0].xaxis.grid(color='gray', linestyle='dashed', visible = True)
    ax[0].minorticks_on()
    ax[0].tick_params(which='major', width=2, length = 6)
    ax[0].tick_params(which='minor', width=1, length = 4)
    ax[1].tick_params(axis = 'x', which = 'major', width = 2, length = 6)
    ax[1].tick_params(axis = 'x', which = 'minor', width = 1, length = 4)
    fig.subplots_adjust(left = 0.2, right = 0.98, bottom = 0.1, top = 0.95, wspace = 0, hspace = 0)
    fig.savefig("FIGURES_ALL/plot_dust_composition_bar_{}.pdf".format(use_gal_dla), format = 'pdf', dpi = 100)
    plt.clf()
    plt.close()

def plot_dm_vs_z_int_dla(composition =True):
    """
    Similar to following function, but this time takes integrated values in the MW, LMC, and SMC (instead of at a given log NH) and also plots the DLAs
    """


    dla_catalogs = np.array(['Quiret2016_DTG_table_', 'DeCia2016_all_data_'])

    use_gal_dla = np.array(['dla', 'MW', 'LMC', 'SMC'])

    colors = [ 'blue',  'green', 'darkorange', 'magenta', 'red']
    elements = [ 'C', 'O', 'Mg', 'Si', 'Fe']


    fig, ax = plt.subplots(nrows =2, ncols = 2, figsize = (10,8), sharex = True, sharey = True)
    ms = 5

    for k in range(4):
        ypos = k//2
        xpos = k%2
        for i in range(2):
            dla = fits.open(dla_catalogs[i] + use_gal_dla[k] + '.fits')
            dla = dla[1].data

            good_dla = np.where((dla['LOG_NHI']>=19.5)& (dla['DTG'] >0))

            dla = dla[good_dla]

            if composition ==False:
                if i==0:
                    #ax.errorbar(dla['tot_A_Fe']/10.**(7.54-12.), dla['DTM'], xerr = dla['err_tot_A_Fe'], yerr = dla['err_DTM'], color = 'black', fmt = 'o', alpha = 0.5, label = "D/M")
                    ax[ypos, xpos].plot(dla['tot_A_Fe']/10.**(7.54-12.), dla['DTM'], 'o', color = 'black', alpha = 0.5, label = "D/M")

                else:
                    #ax.errorbar(dla['tot_A_Fe']/10.**(7.54-12.), dla['DTM'], xerr = dla['err_tot_A_Fe'], yerr = dla['err_DTM'], color = 'black', fmt = 'o', alpha = 0.5)
                    ax[ypos, xpos].plot(dla['tot_A_Fe']/10.**(7.54-12.), dla['DTM'], 'o', color = 'black', alpha = 0.5)

                #ax.errorbar(dla['tot_A_Fe']/10.**(7.54-12.), dla['DTM'], xerr = dla['err_tot_A_Fe'], yerr = dla['err_DTM'], color = 'black', fmt = 'o', alpha = 0.5)

            for j in range(len(elements)):

                aref = get_ref_abundance(elements[j], 'MW')
                aref = 10**(aref-12.)
                wref = get_atomic_weight(elements[j])

                measured = np.where((np.isnan(dla['tot_A_{}'.format(elements[j])]) == False) & (dla['lim_{}'.format(elements[j])] =='v'))
                estimated = np.where((np.isnan(dla['tot_A_{}'.format(elements[j])]) == True) |  (dla['lim_{}'.format(elements[j])] !='v'))

                if composition ==False:
                    if i==0:
                        #ax.errorbar(dla['tot_A_Fe'][measured]/10.**(7.54-12.), (1-10**(dla['dep_{}'.format(elements[j])][measured])), xerr = dla['err_tot_A_Fe'][measured], yerr = 10**(dla['dep_{}'.format(elements[j])][measured])*dla['err_dep_{}'.format(elements[j])][measured], color = colors[j], fmt = 'o', alpha = 0.5, label = elements[j])
                        ax[ypos, xpos].plot(dla['tot_A_Fe'][measured]/10.**(7.54-12.), (1-10**(dla['dep_{}'.format(elements[j])][measured])), 'o', color = colors[j], alpha = 0.5, label = elements[j], markersize  =ms)
                    else:
                        #ax.errorbar(dla['tot_A_Fe'][measured]/10.**(7.54-12.), (1-10**(dla['dep_{}'.format(elements[j])][measured])), xerr = dla['err_tot_A_Fe'][measured], yerr = 10**(dla['dep_{}'.format(elements[j])][measured])*dla['err_dep_{}'.format(elements[j])][measured], color = colors[j], fmt = 'o', alpha = 0.5)
                        ax[ypos, xpos].plot(dla['tot_A_Fe'][measured]/10.**(7.54-12.), (1-10**(dla['dep_{}'.format(elements[j])][measured])),  'o', color = colors[j], alpha = 0.5, markersize = ms)
                #ax.errorbar(dla['tot_A_Fe'][estimated]/10.**(7.54-12.), (1-10**(dla['dep_{}'.format(elements[j])][estimated])), xerr = dla['err_tot_A_Fe'][estimated], yerr = 10**(dla['dep_{}'.format(elements[j])][estimated])*dla['err_dep_{}'.format(elements[j])][estimated], color = colors[j], fmt = 'o', alpha = 0.5)
                    ax[ypos, xpos].plot(dla['tot_A_Fe'][estimated]/10.**(7.54-12.), (1-10**(dla['dep_{}'.format(elements[j])][estimated])), 'o', alpha = 0.5, markersize = ms)


                if composition==True:

                    #Previous way to compute errors. They were unreasonably large
                    #yerr1_m = wref*dla['tot_A_{}'.format(elements[j])][measured]/dla['DTG'][measured]*np.sqrt((dla['err_tot_A_{}'.format(elements[j])][measured]/dla['tot_A_{}'.format(elements[j])][measured])**2 + (dla['err_DTG'][measured]/dla['DTG'][measured])**2)
                    #yerr1_est = wref*dla['est_tot_A_{}'.format(elements[j])][estimated]/dla['DTG'][estimated]*np.sqrt((dla['err_est_tot_A_{}'.format(elements[j])][estimated]/dla['est_tot_A_{}'.format(elements[j])][estimated])**2 + (dla['err_DTG'][estimated]/dla['DTG'][estimated])**2)

                    #print(np.sqrt((dla['err_tot_A_{}'.format(elements[j])][measured]/dla['tot_A_{}'.format(elements[j])][measured])**2 + (dla['err_DTG'][measured]/dla['DTG'][measured])**2))


                    #yerr2_m = 10**(dla['dep_{}'.format(elements[j])][measured])*wref*dla['tot_A_{}'.format(elements[j])][measured]/dla['DTG'][measured]*np.sqrt((yerr1_m*dla['DTG'][measured]/wref/dla['tot_A_{}'.format(elements[j])][measured])**2 +   (np.log(10.)*dla['err_dep_{}'.format(elements[j])][measured])**2)
                    #yerr2_est = 10**(dla['dep_{}'.format(elements[j])][estimated])*wref*dla['est_tot_A_{}'.format(elements[j])][estimated]/dla['DTG'][estimated]*np.sqrt((yerr1_est*dla['DTG'][estimated]/wref/dla['est_tot_A_{}'.format(elements[j])][estimated])**2 +   (np.log(10.)*dla['err_dep_{}'.format(elements[j])][estimated])**2)

                    #yerr_m = np.sqrt(yerr1_m**2  + yerr2_m**2)
                    #yerr_est = np.sqrt(yerr1_est**2  + yerr2_est**2)

                    # This is the fraction of dust mass contributed by each element
                    yerr_m = (1-10**(dla['dep_{}'.format(elements[j])][measured]))*dla['tot_A_{}'.format(elements[j])][measured]*wref/dla['DTG'][measured]/1.36*np.sqrt( (np.log(10.)*dla['err_dep_{}'.format(elements[j])][measured])**2 + (dla['err_DTG'][measured]/dla['DTG'][measured])**2 + (dla['err_tot_A_{}'.format(elements[j])][measured]/dla['tot_A_{}'.format(elements[j])][measured])**2)
                    yerr_est = (1-10**(dla['dep_{}'.format(elements[j])][estimated]))*dla['est_tot_A_{}'.format(elements[j])][estimated]*wref/dla['DTG'][estimated]/1.36*np.sqrt( (np.log(10.)*dla['err_dep_{}'.format(elements[j])][estimated])**2 + (dla['err_DTG'][estimated]/dla['DTG'][estimated])**2 + (dla['err_tot_A_{}'.format(elements[j])][estimated]/dla['tot_A_{}'.format(elements[j])][estimated])**2)

                    #ax.errorbar(dla['tot_A_Fe'][measured]/10.**(7.54-12.), (1-10**(dla['dep_{}'.format(elements[j])][measured]))*dla['tot_A_{}'.format(elements[j])][measured]*wref/dla['DTG'][measured]/1.36, xerr = dla['err_tot_A_Fe'][measured], yerr = yerr_m, color = colors[j], fmt = 'o', alpha = 0.5, label = elements[j])
                    #ax.errorbar(dla['tot_A_Fe'][estimated]/10.**(7.54-12.), (1-10**(dla['dep_{}'.format(elements[j])][estimated]))*dla['est_tot_A_{}'.format(elements[j])][estimated]*wref/dla['DTG'][estimated]/1.36, xerr = dla['err_tot_A_Fe'][estimated], yerr = yerr_est, color = colors[j], fmt = 'o', alpha = 0.5)

                    if i == 0:
                        ax[ypos, xpos].plot(dla['tot_A_Fe'][measured]/10.**(7.54-12.), (1-10**(dla['dep_{}'.format(elements[j])][measured]))*dla['tot_A_{}'.format(elements[j])][measured]*wref/dla['DTG'][measured]/1.36, 'o',  color = colors[j], alpha = 0.5, label = elements[j], markersize =ms)
                    else:
                        ax[ypos, xpos].plot(dla['tot_A_Fe'][measured]/10.**(7.54-12.), (1-10**(dla['dep_{}'.format(elements[j])][measured]))*dla['tot_A_{}'.format(elements[j])][measured]*wref/dla['DTG'][measured]/1.36, 'o',  color = colors[j], alpha = 0.5, markersize = ms)
                    ax[ypos, xpos].plot(dla['tot_A_Fe'][estimated]/10.**(7.54-12.), (1-10**(dla['dep_{}'.format(elements[j])][estimated]))*dla['est_tot_A_{}'.format(elements[j])][estimated]*wref/dla['DTG'][estimated]/1.36, 'o', color = colors[j],  alpha = 0.5, markersize = ms)

        gals = np.array(['MW','LMC', 'SMC'])
        mets = np.array([1., 0.5, 0.2])
        markers = np.array(['X', '*', 'P'])
        for i in range(3):

            #if i ==0:
            #   dgr, err_dgr, dmr,err_dmr,zgr_mw_int, deps, err_deps, w, a, elt= compute_gdr_lv(galaxy = 'MW',log_nh0 = np.array([np.log10(2e21)]))
            #    deps = deps.flatten()
            #    dmx=(1.-10.**(deps))
            #    err_dmx = 10.**(deps)*np.log(10.)*err_deps
            #    dx =  (1-10.**(deps))*w*a/dgr[0]
            #    err_dx=dx*np.sqrt((np.log(10.)*err_deps)**2 + (err_dgr[0]/dgr[0])**2 )
            #else:
                #dgr, err_dgr, dmr, err_dmr, dx, err_dx, dmx, err_dmx, elt= compute_gdr_integrated(gals[i], composition = True)
                # HERE I SHOW THE DUST COMPOSITION IN THE MW/LMC/SMC FOR LOG N(H) = 20 SINCE IT IS THE MEDIAN LOG NH OF DLAS WITH Z > 0.2 Z_0
                dgr, err_dgr, dmr,err_dmr,zgr_mw_int, deps, err_deps, w, a, elt= compute_gdr_lv(galaxy = 'MW',log_nh0 = np.array([np.log10(1e20)]))
                deps = deps.flatten()
                dmx=(1.-10.**(deps))
                err_dmx = 10.**(deps)*np.log(10.)*err_deps
                dx =  (1-10.**(deps))*w*a/dgr[0]
                err_dx=dx*np.sqrt((np.log(10.)*err_deps)**2 + (err_dgr[0]/dgr[0])**2 )
                if composition ==False:
                    ax[ypos, xpos].plot([mets[i]], dmr, markers[i], color = 'black', markersize = 20)
                for j in range(len(elements)):
                    ind_el = np.where(elt == elements[j] )
                    if composition==True:
                        ax[ypos, xpos].plot([mets[i]], dx[ind_el], markers[i], markersize = 15, color = colors[j], alpha= 0.6)
                        ax[ypos, xpos].plot([mets[i]], dx[ind_el], markers[i], markersize = 15, color = 'black', markerfacecolor = 'none')
                    else:
                        ax[ypos, xpos].plot([mets[i]], dmx[ind_el], markers[i], markersize = 15, color = colors[j], alpha  =0.6)
                        ax[ypos, xpos].plot([mets[i]], dmx[ind_el], markers[i], markersize = 15, color = 'black', markerfacecolor = 'none')



        ax[ypos, xpos].plot([13], [6e-3], 'ko', markersize = 10)
        ax[ypos, xpos].text(20, 5.5e-3, 'DLAs', fontsize = 15, color = 'black')
        ax[ypos, xpos].plot([13], [4e-3], 'kP', markersize = 10)
        ax[ypos, xpos].text(20, 3.6e-3, 'SMC', fontsize = 15, color = 'black')
        ax[ypos, xpos].plot([13], [2.5e-3], 'k*', markersize = 10)
        ax[ypos, xpos].text(20, 2.3e-3, 'LMC', fontsize = 15, color = 'black')
        ax[ypos, xpos].plot([13], [1.6e-3], 'kX', markersize = 10)
        ax[ypos, xpos].text(20, 1.4e-3, 'MW', fontsize = 15, color = 'black')

        if ypos == 1:
            ax[ypos, xpos].set_xlabel("Z/" + r'$Z_o$' , fontsize = 22)
        if composition==True:
            if xpos == 0 and ypos == 1:
                ax[ypos, xpos].set_ylabel("                                       Fraction of dust mass contributed by X", fontsize = 22)
            ax[ypos, xpos].set_ylim(bottom = 1e-3, top = 0.9)
        else:
            if xpos == 0 and ypos == 1:
                ax[ypos, xpos].set_ylabel("                       Fraction of X in dust", fontsize = 22)
            ax[ypos, xpos].set_ylim(bottom = 1e-3, top = 1.1)

        ax[ypos, xpos].set_xlim(left = 5e-3, right = 99.)
        ax[ypos, xpos].set_xscale('log')
        ax[ypos, xpos].set_yscale('log')
        ax[ypos, xpos].minorticks_on()
        x_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 3)
        ax[ypos, xpos].xaxis.set_major_locator(x_major)
        x_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
        ax[ypos, xpos].xaxis.set_minor_locator(x_minor)
        ax[ypos, xpos].tick_params(which='major', width=2, length = 6)
        ax[ypos, xpos].tick_params(which='minor', width=1, length = 4)
        #nbins = len(ax[ypos, xpos].get_xticklabels())
        #ax[ypos, xpos].xaxis.set_major_locator(MaxNLocator(nbins = nbins, prune='upper'))

        if use_gal_dla[k] =='dla':
            text  = 'DC16'
        else:
            text = use_gal_dla[k]
        ax[ypos, xpos].text(1e-2, 1.1e-3, text  + ' [Zn/Fe] - ' + r'$\delta$(X)' + ' relation' )

    ax[0, 0].legend(fontsize = 15)
    fig.subplots_adjust(bottom = 0.12, left = 0.12, top = 0.98, right = 0.98, wspace = 0, hspace = 0)
    if composition ==True:
        plt.savefig('FIGURES_ALL/plot_dust_composition_dla_LMC_SMC.pdf', format = 'pdf', dpi = 100)
    else:
        plt.savefig('FIGURES_ALL/plot_dust_to_metal_dla_LMC_SMC.pdf', format = 'pdf', dpi = 100)

    plt.clf()
    plt.close()

def plot_dm_vs_z():

    log_NH = np.array([ 21.])

    #the table below is generated from compute_mean_dep_offset_wZ(log_NH = log_NH)

    t = ascii.read('depletions_NH_fits_MW_LMC_SMC.dat')


    gals = np.array(['MW', 'LMC', 'SMC'])
    Zs = np.array([1., 0.5, 0.2])
    marks = np.array(['*' , 'o', 'X'])

    colors = ['black', 'darkviolet', 'blue', 'deepskyblue','cyan', 'green', 'lime', 'gold', 'darkorange', 'gray', 'magenta',  'red', 'darkred']

    fig, ax = plt.subplots(nrows =1, ncols = 1, figsize = (10,8))

    for i in range(len(log_NH)):
        for j in range(len(t)):

            arr = np.zeros(3)
            for k in range(3):

                if t['X'].data[j] != 'D/M' and t['X'].data[j] != 'D/M (w/o C, O)':

                    yerr = t['err dep {} log NH = {:3.1f}'.format(gals[k],log_NH[i])].data[j]*10**(t['dep {} log NH = {:3.1f}'.format(gals[k],log_NH[i])].data[j])*np.log(10.)

                    if j==0:
                        ax.errorbar(Zs[k], (1-10.**(t['dep {} log NH = {:3.1f}'.format(gals[k],log_NH[i])].data[j])), yerr = yerr, fmt = marks[k], color = colors[j], markersize = 10, label = gals[k])
                    else:
                        ax.errorbar(Zs[k], (1-10.**(t['dep {} log NH = {:3.1f}'.format(gals[k],log_NH[i])].data[j])), yerr = yerr, fmt =marks[k], color = colors[j], markersize = 10)
                    arr[k] = (1-10.**(t['dep {} log NH = {:3.1f}'.format(gals[k],log_NH[i])].data[j]))
                else:
                    if t['X'].data[j] == 'D/M':
                        yerr = t['err dep {} log NH = {:3.1f}'.format(gals[k],log_NH[i])].data[j]
                        ax.errorbar(Zs[k], t['dep {} log NH = {:3.1f}'.format(gals[k],log_NH[i])].data[j], yerr = yerr, fmt = marks[k], color = colors[j], markersize = 10)
                        arr[k] = t['dep {} log NH = {:3.1f}'.format(gals[k],log_NH[i])].data[j]

            #if t['X'].data[j] == 'D/M' or t['X'].data[j] == 'D/M (w/o C, O)' or t['X'].data[j] == 'Mg' or t['X'].data[j] == 'Ti' or t['X'].data[j] == 'C' or t['X'].data[j]=='O':
            if t['X'].data[j] == 'D/M' or t['X'].data[j] == 'Mg' or t['X'].data[j] == 'Ti' or t['X'].data[j] == 'C' or t['X'].data[j]=='O':
                ax.text(Zs[0] + 0.03, arr[0],  t['X'].data[j], color  = colors[j], fontsize = 15)
            else:
                if t['X'].data[j] != 'D/M (w/o C, O)':
                    ax.text( Zs[2]-0.1 , arr[2], t['X'].data[j], color  = colors[j], fontsize = 15)

            if j < 2:
                ax.plot(Zs, arr, '--', color = colors[j])
            else:
                if t['X'].data[j] != 'D/M (w/o C, O)':
                    ax.plot(Zs,arr, '-', color  = colors[j])

    ax.set_xlim(left = 0,right = 1.3)
    ax.set_xlabel(r'Z/Z$_0$', fontsize = 23)
    #plt.ylabel(r'(1-10$^{\delta(X)}$)' + ', D/M', fontsize = 23)
    ax.set_ylabel("Fraction of metals in dust", fontsize = 23)
    ax.minorticks_on()
    #plt.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.legend(loc = 'lower right', fontsize = 16)
    plt.savefig('FIGURES_ALL/plot_dust_fractions.pdf', format = 'pdf', dpi = 100)
    plt.clf()
    plt.close()


def plot_met_redshift(element = 'Fe', separate = False):

    dla_catalogs = np.array(['Quiret2016_DTG_table_', 'DeCia2016_all_data_'])
    types = np.array(['dla', 'MW', 'LMC', 'SMC'])

    labels = ['De Cia+2016', 'MW', 'LMC', 'SMC']

    colors = np.array(['black', 'darkorange', 'dodgerblue', 'magenta'])

    solar_abund = get_ref_abundance(element, "MW")

    if separate==False:
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,8))
    else:
        fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (10,8), sharex = True, sharey = True)
    for i in range(len(types)):
        if separate ==False:
            plt_ax = ax
        else:
            xpos = i%2
            ypos = i//2
            plt_ax = ax[ypos, xpos]

        for j in range(2):
            dla = fits.open(dla_catalogs[j] + types[i] + '.fits')
            dla = dla[1].data

            if j==0:
                plt_ax.errorbar(dla['Z_ABS'], np.log10(dla['tot_A_{}'.format(element)]) - solar_abund + 12., yerr = dla['err_tot_A_{}'.format(element)]/dla['tot_A_{}'.format(element)]/np.log(10.),fmt = 'o', color = colors[i], alpha = 0.5, label = labels[i])
            else:
                plt_ax.errorbar(dla['Z_ABS'], np.log10(dla['tot_A_{}'.format(element)]) - solar_abund + 12., yerr = dla['err_tot_A_{}'.format(element)]/dla['tot_A_{}'.format(element)]/np.log(10.), fmt = 'o', color = colors[i], alpha = 0.5)

        if separate==False or (separate ==True and ypos ==1):
            plt_ax.set_xlabel("Redshift", fontsize = 23)
        if separate == False or (separate==True and xpos ==0):
            plt_ax.set_ylabel("[{}/H]".format(element), fontsize = 23)
        plt_ax.legend(loc = 'upper right', fontsize =14)
        plt_ax.minorticks_on()
        plt_ax.tick_params(which='major', width=2, length = 6)
        plt_ax.tick_params(which='minor', width=1, length = 4)
        plt_ax.grid(visible=True, which='major',linestyle = '--')

    fig.subplots_adjust(top = 0.98, bottom = 0.12, left = 0.12, right = 0.98, wspace = 0, hspace = 0)
    plt.savefig("FIGURES_ALL/plot_dla_{}_over_h_vs_z.pdf".format(element), format = 'pdf', dpi = 100)
    plt.clf()
    plt.close()



def compare_dla_dust():

    dla_catalogs = np.array(['Quiret2016_DTG_table_', 'DeCia2016_all_data_'])
    types = np.array(['MW', 'LMC', 'SMC'])

    labels = [ 'MW', 'LMC', 'SMC']

    colors = np.array([ 'darkorange', 'dodgerblue', 'magenta'])
    cmaps  = np.array(['Oranges', 'Blues', 'Greens'])

    elements = np.array(['C', 'O', 'Mg', 'Si', 'Fe', 'Cr'])
    nwide = 3
    nhigh = 2

    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (10,8))
    fig2, ax2 = plt.subplots(ncols = 3, nrows = 2, figsize = (12,9), sharex = True, sharey = True)

    dla_min_AZN = -9.6
    dla_max_AZN = -7.3

    for i in range(len(types)):
        for j in range(2):
            dla = fits.open(dla_catalogs[j] + types[i] + '.fits')
            dla = dla[1].data
            ref = fits.open(dla_catalogs[j] + 'dla.fits')
            ref = ref[1].data

            xerr = ref['err_DTG']
            yerr = dla['err_DTG']/ref['DTG']*np.sqrt((ref['err_DTG']/ref['DTG'])**2 + (dla['err_DTG']/dla['DTG'])**2)


            znfe = np.log10(dla['gas_A_Zn']) - np.log10(dla['gas_A_Fe']) - (4.7-7.54)
            good = np.where(np.isnan(znfe)==False)
            good = good[0]


            #dla_alpha = 1-0.9*(np.log10(dla['tot_A_Fe'])-dla_min_AFE)/(dla_max_AFE-dla_min_AFE)
            dla_alpha = 1-0.8*(np.log10(dla['gas_A_Zn'])-dla_min_AZN)/(dla_max_AZN-dla_min_AZN)
            #alpha = 1-0.9*(np.log10(dla['tot_A_Fe'][good])-np.nanmin(np.log10(dla['tot_A_Fe'][good])))/(np.nanmax(np.log10(dla['tot_A_Fe'][good]))-np.nanmin(np.log10(dla['tot_A_Fe'][good])))
            if j==0:
                ax.errorbar(ref['DTG'], dla['DTG']/ref['DTG'], xerr = xerr, yerr  = yerr, color = colors[i], fmt = 'o', alpha = 0.5, label = labels[i] + " [Zn/Fe]-" + r'$\delta$(X)' + " relation")
            else:
                ax.errorbar(ref['DTG'], dla['DTG']/ref['DTG'], xerr = xerr, yerr  = yerr, color = colors[i], fmt = 'o', alpha = 0.5)


            for k in range(len(elements)):
                xpos = k % nwide
                ypos = k//nwide


                #ax2[ypos, xpos].errorbar(1-10.**(ref['dep_{}'.format(elements[k])]), 1.-10.**(dla['dep_{}'.format(elements[k])]), xerr =ref['err_dep_{}'.format(elements[k])]*10.**(ref['dep_{}'.format(elements[k])]), yerr=10.**(dla['dep_{}'.format(elements[k])])*dla['err_dep_{}'.format(elements[k])] , fmt = 'o', alpha = 0.5, color = colors[i])
                #for idla in range(len(good)):
                ax2[ypos, xpos].scatter(1-10.**(ref['dep_{}'.format(elements[k])][good]), 1.-10.**(dla['dep_{}'.format(elements[k])][good]), marker='o', s = 30, cmap = cmaps[i], c = dla_alpha[good])
                #ax2[ypos, xpos].plot(1-10.**(ref['dep_{}'.format(elements[k])][good]), 1.-10.**(dla['dep_{}'.format(elements[k])][good]), 'o', markerfacecolor = 'none',markersize = 8, color = 'black')


    dumx = 10.**(np.arange(-6., 0, 0.1))
    dumy = np.zeros_like(dumx) + 1.
    ax.plot(dumx, dumy, '-k')
    ax.text(10.**(-6.3), dumy[0], '1:1', fontsize  = 15)
    ax.plot(dumx, dumy*0.5, '--k')
    ax.text(10.**(-6.3), dumy[0]*0.5, '/2', fontsize  = 15)
    ax.plot(dumx, dumy*2., '--k')
    ax.text(10.**(-6.3), dumy[0]*2., 'x2', fontsize  = 15)
    ax.plot(dumx, dumy*5., '--', color = 'gray')
    ax.text(10.**(-6.3), dumy[0]*5, 'x5', fontsize  = 15, color = 'gray')
    ax.plot(dumx, dumy/5., '--', color = 'gray')
    ax.text(10.**(-6.3), dumy[0]/5., '/5', fontsize  = 15, color = 'gray')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(left = 10.**(-6.6), right = 1.)
    ax.set_xlabel("DLA D/G De Cia+2016", fontsize = 22)
    ax.set_ylabel("(D/G) this work / (D/G) De Cia+2016", fontsize = 22)
    ax.minorticks_on()
    y_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 4)
    ax.yaxis.set_major_locator(y_major)
    y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.tick_params(which='major', width=2, length = 6)
    ax.tick_params(which='minor', width=1, length = 4)


    if xpos == 0:
        nbins = len(axp.get_yticklabels())
        print('TEST', nbins)
        axp.yaxis.set_major_locator(MaxNLocator(nbins = nbins, prune='upper'))

    ax.legend(loc = 'upper left', fontsize = 17)
    fig.tight_layout()
    fig.savefig("FIGURES_ALL/plot_comparison_doh_DLAs.pdf", format = 'pdf', dpi  =100)

    dumx = 10.**(np.arange(-3., 0.05, 0.1))
    for k in range(len(elements)):
        xpos = k % nwide
        ypos = k//nwide


        #ymin, ymax = ax2[ypos, xpos].get_ylim()
        #xmin, xmax = ax2[ypos, xpos].get_xlim()
        ax2[ypos, xpos].text(0, 0.88, elements[k], fontsize = 22)
        #ax2[ypos, xpos].set_xscale('log')
        #ax2[ypos, xpos].set_yscale('log')
        ax2[ypos, xpos].plot(dumx, dumx, '--k')
        ax2[ypos, xpos].set_xlim(left= -0.05, right = 1.1)
        ax2[ypos, xpos].set_ylim(bottom = -0.05, top = 1.1)
        ax2[ypos,xpos].minorticks_on()
        ax2[ypos, xpos].tick_params(which='major', width=2, length = 6)
        ax2[ypos, xpos].tick_params(which='minor', width=1, length = 4)
        #ax2[ypos, xpos].text(10.**(-3), dumx[0]*0.8, '1:1', fontsize  = 15)
        #ax2[ypos, xpos].plot(dumx, dumx*0.5, '--k')
        #ax2[ypos, xpos].text(10.**(-3), dumx[0]*0.3, '/2', fontsize  = 15)
        #ax2[ypos, xpos].plot(dumx, dumx*2., '--k')
        #ax2[ypos, xpos].text(10.**(-3), dumx[0]*2., 'x2', fontsize  = 15)
        #ax2[ypos, xpos].plot(dumx, dumx*5., '--', color = 'gray')
        #ax2[ypos, xpos].text(10.**(-3), dumx[0]*5, 'x5', fontsize  = 15, color = 'gray')
        #ax2[ypos, xpos].plot(dumx, dumx/5., '--', color = 'gray')
        #ax2[ypos, xpos].text(10.**(-3), dumx[0]/7., '/5', fontsize  = 15, color = 'gray')
    ax2[1, 1].set_xlabel(" Fraction of X in dust (De Cia+2016)", fontsize =22)
    ax2[1,0].set_ylabel("                              Fraction of X in dust (this work) " , fontsize =22)
    ax2[0,0].plot(0.3, 0.18, 'o', color = 'darkorange')
    ax2[0 ,0].text(0.35, 0.16, 'MW [Zn/Fe]-' +  r'$\delta$(X) relation', fontsize = 12)
    ax2[0,0].plot(0.3, 0.1, 'o', color = 'dodgerblue')
    ax2[0,0].text(0.35, 0.08, 'LMC [Zn/Fe]-'+r'$\delta$(X) relation', fontsize = 12)
    ax2[0,0].plot(0.3, 0.02, 'o', color = 'green')
    ax2[0,0].text(0.35, 0.0, 'SMC [Zn/Fe]-'+r'$\delta$(X) relation', fontsize = 12)


    #vmin = np.nanmin(np.log10(dla['tot_A_Fe'][good])) -(7.54-12)
    #vmax=np.nanmax(np.log10(dla['tot_A_Fe'][good])) -(7.54-12)
    vmin = dla_min_AZN -(4.7-12)
    vmax=dla_max_AZN -(4.7-12)

    for i in range(3):
        #axcb = fig2.add_axes([0.12+0.27*i, 0.93, 0.25, 0.07])
        #cbar = mpl.colorbar.ColorbarBase(axcb, cmap=cmaps[i], norm=norm,spacing='proportional', ticks=cbounds, boundaries=bounds, format='%0.2f')
        #cbar = fig.colorbar(cmap,  orientation = 'vertical')
        #cbar.set_label('[Fe/H]', rotation=90, fontsize = 18)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmaps[i]+'_r', norm=norm)
        sm.set_array([])

        ##divider = make_axes_locatable(ax2[0,i])
        ##cax = divider.new_vertical(size='5%', pad=0.1)
        cax = inset_axes(ax2[0, i], width = '80%', height = '5%', loc = 'upper center')
        fig2.add_axes(cax)
        cbar = fig2.colorbar(sm, cax=cax, orientation='horizontal', ticks=np.linspace(vmin, vmax, 4))
        ##ax2[0,i].colorbar( loc = 't', label = '[Fe/H]', ticks = 5, values = np.linspace(vmin, vmax, 5))
        ##cbar  = fig2.colorbar(sm, ticks=np.linspace(vmin, vmax, 5), ax=ax2[0,i], shrink = 0.95, orientation = 'horizontal', pad = 0.1, location = 'top')

        cbar.set_ticks(np.linspace(vmin, vmax, 4))
        cbar.set_ticklabels(['{:3.1f}'.format(x) for x in np.linspace(vmin, vmax, 4)])
        #cbar.ax.tick_params(axis='x',direction='in',labeltop='on')
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.tick_params(labelsize=15)
        #cbar.set_label('[Fe/H]')
        cax.set_title('DLA gas [Zn/H]', fontsize = 18)

    fig2.subplots_adjust(top = 0.93, right = 0.98, left = 0.12, bottom = 0.12, wspace = 0., hspace = 0.)
    fig2.savefig('FIGURES_ALL/plot_comparison_dmx_DLAs.pdf', format = 'pdf', dpi = 100)

    plt.clf()
    plt.close()


def compare_dla_metallicities(element = 'Fe'):

    dla_catalogs = np.array(['Quiret2016_DTG_table_', 'DeCia2016_all_data_'])
    types = np.array(['MW', 'LMC', 'SMC'])

    labels = [ 'MW', 'LMC', 'SMC']

    colors = np.array([ 'darkorange', 'dodgerblue', 'magenta'])

    solar_abund = get_ref_abundance(element, "MW")

    fig, plt_ax = plt.subplots(ncols = 1, nrows = 1, figsize = (10,8))

    for i in range(len(types)):
        for j in range(2):
            dla = fits.open(dla_catalogs[j] + types[i] + '.fits')
            dla = dla[1].data
            ref = fits.open(dla_catalogs[j] + 'dla.fits')
            ref = ref[1].data

            xerr = ref['err_tot_A_{}'.format(element)]/ref['tot_A_{}'.format(element)]/np.log(10.)
            yerr = dla['err_tot_A_{}'.format(element)]/dla['tot_A_{}'.format(element)]/np.log(10.)
            est_xerr = ref['err_est_tot_A_{}'.format(element)]/ref['est_tot_A_{}'.format(element)]/np.log(10.)
            est_yerr = dla['err_est_tot_A_{}'.format(element)]/dla['est_tot_A_{}'.format(element)]/np.log(10.)


            if j==0:
                #plt_ax.errorbar(np.log10(ref['tot_A_{}'.format(element)])-solar_abund + 12., np.log10(dla['tot_A_{}'.format(element)])-solar_abund + 12., xerr = xerr, yerr  = yerr, color = colors[i], fmt = 'o', alpha = 0.5, label = labels[i] + " [Zn/Fe]-" + r'$\delta$(X)' + " relation")
                plt_ax.errorbar(np.log10(ref['tot_A_{}'.format(element)])-solar_abund + 12., np.log10(dla['tot_A_{}'.format(element)]) - np.log10(ref['tot_A_{}'.format(element)]), xerr = xerr, yerr  = np.sqrt(yerr**2  + xerr**2), color = colors[i], fmt = 'o', alpha = 0.5, label = labels[i] + " [Zn/Fe]-" + r'$\delta$(X)' + " relation")

                #plt_ax.errorbar(np.log10(ref['est_tot_A_{}'.format(element)])-solar_abund + 12., np.log10(dla['est_tot_A_{}'.format(element)])-solar_abund + 12., xerr = est_xerr, yerr  = est_yerr, color = colors[i], fmt = 'o', alpha = 0.5,  markerfacecolor = 'none')
            else:
                #plt_ax.errorbar(np.log10(ref['tot_A_{}'.format(element)])-solar_abund + 12., np.log10(dla['tot_A_{}'.format(element)])-solar_abund + 12., xerr = xerr, yerr  = yerr, color = colors[i], fmt = 'o', alpha = 0.5)
                plt_ax.errorbar(np.log10(ref['tot_A_{}'.format(element)])-solar_abund + 12., np.log10(dla['tot_A_{}'.format(element)])-np.log10(ref['tot_A_{}'.format(element)]), xerr = xerr, yerr  = np.sqrt(yerr**2 + xerr**2), color = colors[i], fmt = 'o', alpha = 0.5)


                #plt_ax.errorbar(np.log10(ref['est_tot_A_{}'.format(element)])-solar_abund + 12., np.log10(dla['est_tot_A_{}'.format(element)])-solar_abund + 12., xerr = est_xerr, yerr  = est_yerr, color = colors[i], fmt = 'o', alpha = 0.5, markerfacecolor = 'none')

    dumx = np.arange(-2.5, 1.5, 0.1)
    dumy = np.zeros_like(dumx)
    plt_ax.plot(dumx, dumy, '-k')
    plt_ax.plot(dumx, dumy-0.5, '--k')
    plt_ax.plot(dumx, dumy+0.5, '--k')

    plt_ax.set_xlabel("DLA " + '[{}/H]'.format(element) + r'$_{tot}$' + " DeCia+2016", fontsize = 22)

    plt_ax.set_ylabel(r'$\Delta$' + '[{}/H]'.format(element) + r'$_{tot}$' + " (this work - De Cia+2016)", fontsize = 22)
    plt_ax.minorticks_on()
    plt_ax.tick_params(which='major', width=2, length = 6)
    plt_ax.tick_params(which='minor', width=1, length = 4)
    plt_ax.legend(loc = 'upper left', fontsize = 17)
    plt.tight_layout()
    plt.savefig("FIGURES_ALL/plot_comparison_metallicities_DLAs_{}.pdf".format(element), format = 'pdf', dpi  =100)
    plt.clf()
    plt.close()



def plot_znfe_deps_dlas():

    dla_catalogs = np.array(['Quiret2016_DTG_table_', 'DeCia2016_all_data_'])
    types = np.array(['dla', 'MW', 'LMC', 'SMC'])

    labels = ['De Cia+2016', 'MW', 'LMC', 'SMC']

    colors = np.array(['black', 'darkorange', 'dodgerblue', 'magenta'])

    elements = np.array(['C', 'O', 'Mg', 'Si', 'S', 'Cr', 'Fe', 'Zn'])

    tab = Table()
    els = []
    cats = []
    arr_types = []
    a_coefs = []
    b_coefs =[]

    plt.clf()
    plt.close()
    fig, ax = plt.subplots(nrows = 2, ncols = 4, sharex = True, sharey = False, figsize = (16, 7))

    for i in range(len(types)):
        for j in range(2):
            dla = fits.open(dla_catalogs[j] + types[i] + '.fits')
            dla = dla[1].data

            for k in range(len(elements)):
                y = k//4
                x = k % 4


                xerr = np.sqrt( (dla['err_gas_A_Zn']/dla['gas_A_Zn']/np.log(10.))**2 + (dla['err_gas_A_Fe']/dla['gas_A_Fe']/np.log(10.))**2 )

                if ((( elements[k] == 'C') and (types[i] == 'dla')) ==False):
                    if j == 0:
                        #ax[y,x].errorbar(np.log10(dla['gas_A_Zn'])  - np.log10(dla['gas_A_Fe'])  + 2.84, dla['dep_{}'.format(elements[k])], xerr = xerr , yerr = dla['err_dep_{}'.format(elements[k])], fmt='o', color = colors[i], alpha = 0.5)
                        ax[y,x].plot(np.log10(dla['gas_A_Zn'])  - np.log10(dla['gas_A_Fe'])  + 2.84, dla['dep_{}'.format(elements[k])], 'o', color = colors[i], alpha = 0.5)
                    else:
                        #ax[y,x].errorbar(np.log10(dla['gas_A_Zn'])  - np.log10(dla['gas_A_Fe'])  + 2.84, dla['dep_{}'.format(elements[k])], xerr = xerr , yerr = dla['err_dep_{}'.format(elements[k])], fmt='o', color = colors[i], label = labels[i], alpha = 0.5 )
                        ax[y,x].plot(np.log10(dla['gas_A_Zn'])  - np.log10(dla['gas_A_Fe'])  + 2.84, dla['dep_{}'.format(elements[k])], 'o', color = colors[i], alpha = 0.5, label = labels[i])

                dumx = np.arange(-1,2, 0.01)

                if types[i] == 'dla':
                    a2, e_a2, b2,e_b2, znfe0, e_znfe0 = get_a2_b2_decia(elements[k])
                    dumy = a2 + b2*dumx
                    dumy[np.where(dumx < znfe0)] = 0.
                else:
                    dumy = np.zeros_like(dumx)
                    for ii in range(len(dumx)):
                        this_dep, err_this_dep = get_dep_znfe_realization(dumx[ii], 0., types[i], elements[k])
                        dumy[ii] = this_dep

                ax[y,x].plot(dumx, dumy, '--', color = colors[i])
                ax[y,x].minorticks_on()
                ax[y,x].tick_params(which='major', width=2, length = 6)
                ax[y,x].tick_params(which='minor', width=1, length = 4)
                #ax.xaxis.set_minor_locator(AutoMinorLocator(4))
                #ax.yaxis.set_minor_locator(AutoMinorLocator(4))
                if y == 1:
                    ax[y, x].set_xlabel("[Zn/Fe]", fontsize = 22)
                if x == 0:
                    ax[y,x].set_ylabel(r'$\delta$' + '(X)', fontsize = 22)


                #good = np.where((np.isnan(dla['dep_{}'.format(elements[k])])==False) & (dla['dep_{}'.format(elements[k])] < 0.)  & (np.isnan(np.log10(dla['gas_A_Zn'])  - np.log10(dla['gas_A_Fe']))==False))
                #coefs = np.polyfit(np.log10(dla['gas_A_Zn'][good])  - np.log10(dla['gas_A_Fe'][good])  + 2.84, dla['dep_{}'.format(elements[k])][good], 1)

                arr_types.append(types[i])
                cats.append(dla_catalogs[j])
                els.append(elements[k])
                #a_coefs.append(coefs[0])
                #b_coefs.append(coefs[1])

    for k in range(len(elements)):
         y = k//4
         x = k % 4
         ax[y,x ].set_xlim(left = -1, right = 2.2)
         #ax[y,x].set_ylim(bottom = -4, top = 1)
         y_min, y_max = ax[y,x].get_ylim()
         ax[y,x].text(-0.7, y_min*0.9, elements[k], fontsize  =22)
    ax[1,3].legend(loc = 'lower right', fontsize = 14)

    fig.subplots_adjust(bottom = 0.12, top = 0.95, left = 0.08, right = 0.98, wspace = 0.35, hspace = 0)

    plt.savefig('FIGURES_ALL/plot_znfe_deps_DLAs.pdf', format = 'pdf', dpi = 100)

    plt.clf()
    plt.close()

    #tab['cats'] = np.array(cats)
    #tab['types'] = np.array(arr_types)
    #tab['element'] = np.array(els)
    #tab['a0'] =np.array(b_coefs)
    #tab['a1'] = np.array(a_coefs)

    #ascii.write(tab, 'znfe_coefs.dat', overwrite=True)




def summary_plot():

    """
    Was plotting log N(X) vs log N(X) for all systems, but it looks crappy
    """

    dla_files = ['Quiret2016_DTG_table_dla.fits', 'DeCia2016_all_data_dla.fits']
    gal_files= ['compiled_depletions_jenkins2009_py_adj_zeropt_zn.fits', 'METAL_depletions_and_ci_results_with_env_fix1220.fits', 'compiled_depletions_jenkins2017_py.fits']

    dla_colors= ['cyan', 'gray']
    gal_colors= ['darkorange', 'dodgerblue', 'green']

    elements = ['Mg', 'Si', 'Fe', 'Zn']

    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 8))

    for i in range(2):
        dla = fits.open(dla_files[i])
        dla = dla[1].data
        good = np.where(dla['DTG']>0)
        good = good[0]

        for j in range(len(elements)):


            y=j//2
            x = j % 2

            ax[y,x].errorbar(dla['LOG_NHI'][good], np.log10(dla['gas_A_{}'.format(elements[j])][good])+ dla['LOG_NHI'][good], xerr = dla['ERR_LOG_NHI'][good], yerr = dla['err_gas_A_{}'.format(elements[j])][good]/dla['gas_A_{}'.format(elements[j])][good]/np.log(10.) ,  fmt = 'o', color = dla_colors[i])


    for i in range(3):
        gal = fits.open(gal_files[i])
        gal = gal[1].data

        els = np.array([x.replace('I', '') for x in gal['ELEMENT']])

        for j in range(len(elements)):

            y=j//2
            x = j % 2


            indel =np.where(els == elements[j])
            indel = indel[0][0]

            if 'ERR_LOG_NH' in gal.columns.names:
                err_log_nh = gal['ERR_LOG_NH']
            else:
                err_log_nh = gal['ERR_LOG_NHI']
            ax[y,x].errorbar(gal['LOG_NHI'], gal['LOG_NX'][:, indel],xerr = err_log_nh, yerr = gal['ERR_LOGNX'][:, indel],  fmt = 'o', color = gal_colors[i])

            if  y==1:
                ax[y,x].set_xlabel("LOG N(H)", fontsize = 22)
            if x == 0:
                ax[y,x].set_ylabel("LOG N(X)", fontsize = 22)
            ax[y,x].text(20, np.nanmax(gal['LOG_NX'][:, indel])*0.98, elements[j], fontsize = 22)
    fig.subplots_adjust(bottom = 0.12, left = 0.12, right = 0.98, top = 0.98, wspace = 0.15, hspace = 0.5)
    plt.show()



def fit_lmc_dg_NH():

    #D/G values for METAL sightlines
    log_nh, err_log_nh, dgrs, err_dgrs, dmrs, err_dmrs, zgrs, out_deps, out_edeps, atm, ref = compute_gdr_sightline()
    #FIt the dgrs wtih log NH

    slope, offset, err_slope, err_offset, cov=fitexy(log_nh, err_log_nh , dgrs,err_dgrs)
    print(slope, offset, err_slope, err_offset)

    plt.errorbar(log_nh, dgrs, xerr = err_log_nh, yerr = err_dgrs, fmt = 'ko')
    dumx = np.arange(19.5, 22.2, 0.1)
    plt.plot(dumx, dumx*slope + offset, 'r-')
    plt.show()



def redo_RD2017_gdr():


    dict_fir_smc = scipy.io.readsav("/astro/dust_kg/jduval/DUST_POOR_COMPONENT/MC_SAVE_FILES/binned_IRAS_PLANCK_surface_brightness_smc_quad_cirrus_thresh_stray_corr_mbb_a21.00_FITS_0.05_FINAL.save")

    dict_fir_lmc = scipy.io.readsav("/astro/dust_kg/jduval/DUST_POOR_COMPONENT/MC_SAVE_FILES/binned_IRAS_PLANCK_surface_brightness_lmc_quad_cirrus_thresh_stray_corr_mbb_a6.400_FITS_0.05_FINAL.save")

    fir_gas_lmc = dict_fir_lmc['gas_bins']/0.8e-20
    fir_gd_lmc = dict_fir_lmc['gdr_ratio']
    fir_err_gd_lmc = dict_fir_lmc['percentile50_high'] - dict_fir_lmc['percentile50_low']#dict_fir['err_gdr_ratio']

    fir_gas_smc = dict_fir_smc['gas_bins']/0.8e-20
    fir_gd_smc = dict_fir_smc['gdr_ratio']
    fir_err_gd_smc = dict_fir_smc['percentile50_high']- dict_fir_smc['percentile50_low']#dict_fir['err_gdr_ratio']

    fir_gd_smc_int = dict_fir_smc['total_gas_mass']/dict_fir_smc['total_dust_mass']
    fir_err_gd_smc_int =  fir_gd_smc_int*np.sqrt((dict_fir_smc['total_dust_mass_error']/dict_fir_smc['total_dust_mass'])**2 + (dict_fir_smc['total_gas_mass_error']/dict_fir_smc['total_gas_mass'])**2)

    fir_gd_lmc_int = dict_fir_lmc['total_gas_mass']/dict_fir_lmc['total_dust_mass']
    fir_err_gd_lmc_int =  fir_gd_lmc_int*np.sqrt((dict_fir_lmc['total_dust_mass_error']/dict_fir_lmc['total_dust_mass'])**2 + (dict_fir_lmc['total_gas_mass_error']/dict_fir_lmc['total_gas_mass'])**2)

    fig = plt.figure(figsize = (10,8))

    plt.errorbar(np.log10(fir_gas_lmc), 1./fir_gd_lmc, fmt = 'ko', yerr = fir_err_gd_lmc/fir_gd_lmc**2, ecolor = 'k', alpha = 0.7, label = 'LMC (resolved)')

    plt.errorbar(np.log10(fir_gas_smc), 1./fir_gd_smc, fmt = 'ro', yerr = fir_err_gd_smc/fir_gd_smc**2, ecolor = 'r', alpha = 0.7, label = 'SMC (resolved)')

    plt.fill_between(np.log10(fir_gas_lmc), np.zeros_like(fir_gas_lmc)  + 1./fir_gd_lmc_int + fir_err_gd_lmc_int/fir_gd_lmc_int**2 , np.zeros_like(fir_gas_lmc)  + 1./fir_gd_lmc_int - fir_err_gd_lmc_int/fir_gd_lmc_int**2, color = 'gray', alpha = 0.3)
    plt.plot(np.log10(fir_gas_lmc), np.zeros_like(fir_gas_lmc)  + 1./fir_gd_lmc_int , 'k-', alpha = 0.7, label = 'LMC (integrated)')

    plt.fill_between(np.log10(fir_gas_smc), np.zeros_like(fir_gas_smc)  + 1./fir_gd_smc_int + fir_err_gd_smc_int/fir_gd_smc_int**2 , np.zeros_like(fir_gas_smc)  + 1./fir_gd_smc_int - fir_err_gd_smc_int/fir_gd_smc_int**2, color = 'red', alpha = 0.3)
    plt.plot(np.log10(fir_gas_smc), np.zeros_like(fir_gas_smc)  + 1./fir_gd_smc_int , 'r-', alpha = 0.7, label = 'SMC (integrated)')

    print(1./fir_gd_smc_int, 1./fir_gd_lmc_int)

    plt.yscale('log')
    plt.xlabel("log N(H)", fontsize = 22)
    plt.ylabel("D/G", fontsize = 22)
    plt.ylim([1.e-5, 1.e-2])
    plt.legend(fontsize= 15, loc = 'lower right')
    plt.tight_layout()

    plt.savefig("plot_RD2017_DG.pdf", format = 'pdf', dpi = 100)
    plt.clf()
    plt.close()



def plot_abundance_ratios(element1, element2, element3, element4=0, lowy  = -2, highy =2,  lowx = -2, highx = 2.,plot_dep = False, plot_other=True, plot_dla =True, plot_clusters=False,dir = './FIGURES_ALL/', use_gal_dla = 'dla', use_fe_dla = False, nmc = 30):

    """
    plots el1/el2 vs el3/H

    """

    #solar abundances:

    ref_abund = ascii.read('reference_abundances.dat')
    rind1 = np.where(ref_abund['Element'].data == element1)
    rind1 = rind1[0][0]
    ra1 = ref_abund['MW_A'].data[rind1]

    rind2 = np.where(ref_abund['Element'].data == element2)
    rind2 = rind2[0][0]
    ra2 = ref_abund['MW_A'].data[rind2]

    rind3 = np.where(ref_abund['Element'].data == element3)
    rind3 = rind3[0][0]
    ra3 = ref_abund['MW_A'].data[rind3]

    if element4 != 0:
        rind4 = np.where(ref_abund['Element'].data == element4)
        rind4 = rind4[0][0]
        ra4 = ref_abund['MW_A'].data[rind4]



    ref_ratio = 10.**(ra1-12)/10.**(ra2-12)
    if element4 ==0:
        ref_a = 10.**(ra3-12)
    else:
        ref_a = 10.**(ra3-12.)/10.**(ra4-12.)
    ref_lr = np.log10(ref_ratio)
    ref_la = np.log10(ref_a)

    print("REFERENCES ", ref_lr, ref_la)

    #METAL LMC measurememts
    table = fits.open("compiled_depletion_table_cos_stis_wti_corr_sii_ebj_oi_strong_zn_fstar_fix1220_pyh.fits")
    table = table[1].data

    log_nh= table['LOG_NH']
    err_log_nh = table['ERR_LOG_NHI']

    nh = 10.**(log_nh)
    err_nh = nh*np.log(10.)*err_log_nh

    elements = np.array([x.replace('I', '') for x in table['ELEMENT']])

    ind1 = np.where(element1 == elements)
    ind1 = ind1[0][0]
    ind2 = np.where(element2 == elements)
    ind2 = ind2[0][0]
    ind3 = np.where(element3 == elements)
    ind3 = ind3[0][0]
    if element4 !=0:
        ind4 = np.where(element4 == elements)
        ind4 = ind4[0][0]

    lmc_gas_col1 = 10.**(table['LOG_NX'][:, ind1])
    lmc_gas_col1_err = lmc_gas_col1 * np.log(10.)*table['ERR_LOGNX'][:, ind1]

    lmc_gas_col2 = 10.**(table['LOG_NX'][:, ind2])
    lmc_gas_col2_err = lmc_gas_col2 * np.log(10.)*table['ERR_LOGNX'][:, ind2]


    lmc_gas_col3 = 10.**(table['LOG_NX'][:, ind3])
    lmc_gas_col3_err = lmc_gas_col3 * np.log(10.)*table['ERR_LOGNX'][:, ind3]
    if element4 !=0:
        lmc_gas_col4 = 10.**(table['LOG_NX'][:, ind4])
        lmc_gas_col4_err = lmc_gas_col4 * np.log(10.)*table['ERR_LOGNX'][:, ind4]

    lmc_gas_met3 = lmc_gas_col3/nh
    lmc_gas_met3_err = lmc_gas_met3*np.sqrt((lmc_gas_col3_err/lmc_gas_col3)**2 + (err_nh/nh)**2)

    lmc_dep3 = table['DEPLETIONS'][:, ind3]
    lmc_err_dep3 = table['ERR_DEPLETIONS'][:, ind3]

    if element4 == 0:
        valid_lmc = np.where((table['FLAG'][:, ind1]=='v') & (table['FLAG'][:, ind2]=='v') & (table['FLAG'][:, ind3]=='v'))
        valid_lmc = valid_lmc[0]
    else:
        valid_lmc = np.where((table['FLAG'][:, ind1]=='v') & (table['FLAG'][:, ind2]=='v') & (table['FLAG'][:, ind3]=='v') & (table['FLAG'][:,  ind4] =='v'))
        valid_lmc = valid_lmc[0]

    log_nh = log_nh[valid_lmc]

    lmc_gas_col1 = lmc_gas_col1[valid_lmc]
    lmc_gas_col1_err = lmc_gas_col1_err[valid_lmc]

    lmc_gas_col2 = lmc_gas_col2[valid_lmc]
    lmc_gas_col2_err = lmc_gas_col2_err[valid_lmc]

    lmc_gas_col3 = lmc_gas_col3[valid_lmc]
    lmc_gas_col3_err = lmc_gas_col3_err[valid_lmc]

    if element4 !=0:
        lmc_gas_col4 = lmc_gas_col4[valid_lmc]
        lmc_gas_col4_err = lmc_gas_col4_err[valid_lmc]


    lmc_gas_met3 = lmc_gas_met3[valid_lmc]
    lmc_gas_met3_err = lmc_gas_met3_err[valid_lmc]

    lmc_dep3 = lmc_dep3[valid_lmc]
    lmc_err_dep3 = lmc_err_dep3[valid_lmc]

    lmc_ratio = lmc_gas_col1/lmc_gas_col2
    lmc_ratio_err = lmc_ratio*np.sqrt((lmc_gas_col1_err/lmc_gas_col1)**2+ (lmc_gas_col2_err/lmc_gas_col2)**2)



    #MW, SMC measurememts
    mw_log_nh1, mw_err_log_nh1, mw_log_nh21, mw_err_log_nh21,  fstar_mw, mw_dep1, mw_err_dep1, mw_gas_met1, mw_gas_met1_err, mw_t1  = get_depletions(element1, "J09", "MW", nofilter = True)
    mw_log_nh2, mw_err_log_nh2, mw_log_nh22, mw_err_log_nh22, fstar_mw2,  mw_dep2, mw_err_dep2, mw_gas_met2, mw_gas_met2_err, mw_t2 = get_depletions(element2, "J09", "MW", nofilter = True)
    mw_log_nh3,  mw_err_log_nh3, mw_log_nh23, mw_err_log_nh23, fstar_mw3, mw_dep3, mw_err_dep3, mw_gas_met3, mw_gas_met3_err, mw_t3 = get_depletions(element3, "J09", "MW", nofilter = True)


    smc_log_nh1,  smc_err_log_nh1, smc_log_nh21, smc_err_log_nh21, fstar_smc1, smc_dep1, smc_err_dep1, smc_gas_met1, smc_gas_met1_err , smc_t1 = get_depletions(element1, "J17", "SMC", nofilter = True)
    smc_log_nh2,  smc_err_log_nh2, smc_log_nh23, smc_err_log_nh22,fstar_smc2, smc_dep2, smc_err_dep2, smc_gas_met2, smc_gas_met2_err, smc_t2  = get_depletions(element2, "J17", "SMC", nofilter = True)
    smc_log_nh3,  smc_err_log_nh3, smc_log_nh23, smc_err_log_nh23,fstar_smc3, smc_dep3, smc_err_dep3, smc_gas_met3, smc_gas_met3_err, smc_t3  = get_depletions(element3, "J17", "SMC", nofilter = True)

    if element4!=0:
        mw_log_nh4,  mw_err_log_nh4, mw_log_nh24, mw_err_log_nh24, fstar_mw4, mw_dep4, mw_err_dep4, mw_gas_met4, mw_gas_met4_err, mw_t4 = get_depletions(element4, "J09", "MW", nofilter = True)
        smc_log_nh4,  smc_err_log_nh4, smc_log_nh24, smc_err_log_nh24,fstar_smc4, smc_dep4, smc_err_dep4, smc_gas_met4, smc_gas_met4_err, smc_t3  = get_depletions(element4, "J17", "SMC", nofilter = True)


    mw_ratio = mw_gas_met1/mw_gas_met2
    mw_ratio_err = mw_ratio*np.sqrt((mw_gas_met1_err/mw_gas_met1)**2 + (mw_gas_met2_err/mw_gas_met2)**2)

    if element1!='O' and element2!='O':
        smc_ratio = smc_gas_met1/smc_gas_met2
        smc_ratio_err = smc_ratio*np.sqrt((smc_gas_met1_err/smc_gas_met1)**2 + (smc_gas_met2_err/smc_gas_met2)**2)
    else:
        smc_ratio = np.zeros(10)
        smc_ratio_err = np.zeros(10)

    if plot_dep==False:
        xx_mw3 = np.log10(mw_gas_met3) - ref_la
        err_xx_mw3 = mw_gas_met3_err/mw_gas_met3/np.log(10.)
        xx_smc3 = np.log10(smc_gas_met3)-ref_la
        err_xx_smc3 = smc_gas_met3_err/smc_gas_met3/np.log(10.)
        xx_lmc3=np.log10(lmc_gas_met3)-ref_la
        err_xx_lmc3=lmc_gas_met3_err/lmc_gas_met3/np.log(10.)
        #lowx = -3
        #highx = 2
        dep_key = ''
        xtitle = '[' + element3 + '/H]'
        xrange= [-2,0]

        if element4 !=0:
            xx_mw3 =np.log10(mw_gas_met3/mw_gas_met4)-ref_la
            err_xx_mw3 =  mw_gas_met3/mw_gas_met4*np.sqrt((mw_gas_met3_err/mw_gas_met3)**2+ (mw_gas_met4_err/mw_gas_met4)**2)/(mw_gas_met3/mw_gas_met4)/np.log(10.)
            xx_smc3 =np.log10(smc_gas_met3/smc_gas_met4)-ref_la
            err_xx_smc3 =  smc_gas_met3/smc_gas_met4*np.sqrt((smc_gas_met3_err/smc_gas_met3)**2+ (smc_gas_met4_err/smc_gas_met4)**2)/(smc_gas_met3/smc_gas_met4)/np.log(10.)

            xx_lmc3 = np.log10(lmc_gas_col3/lmc_gas_col4)-ref_la
            err_xx_lmc3 = lmc_gas_col3/lmc_gas_col4*np.sqrt((lmc_gas_col3_err/lmc_gas_col3)**2+ (lmc_gas_col4_err/lmc_gas_col4)**2)/(lmc_gas_col3/lmc_gas_col4)/np.log(10.)
            xtitle = '[' + element3 + '/' + element4  + ']'
            xrange=[-1,1]

        xfit = np.arange(-2, 2, 0.1)

        ratio_fit_mw, fstar_fits_mw, d1mw, e_d1mw,d2mw,e_d2mw, d3mw,e_d3mw,d4mw,e_d4mw =  get_ratios_from_fits('MW', element1, element2, ratio3o4_arr = xfit,  err_ratio3o4_arr = np.zeros_like(xfit), el3=element3, el4=element4)
        ratio_fit_lmc, fstar_fits_lmc, d1lmc,e_d1lmc, d2lmc, e_d2lmc,d3lmc,e_d3lmc, d4lmc,e_d4lmc =  get_ratios_from_fits('LMC', element1, element2, ratio3o4_arr = xfit,  err_ratio3o4_arr = np.zeros_like(xfit), el3=element3, el4=element4)
        ratio_fit_smc, fstar_fits_smc, d1smc,e_d1smc, d2smc,e_d2smc, d3smc, e_d3smc,d4smc,e_d4smc =  get_ratios_from_fits('SMC', element1, element2, ratio3o4_arr = xfit,  err_ratio3o4_arr = np.zeros_like(xfit), el3=element3, el4=element4)

        rand_ratio_fit_mw = np.zeros((nmc, len(ratio_fit_mw)), dtype = 'float32')
        probs_mw = np.zeros(nmc, dtype = 'float32')
        rand_ratio_fit_lmc = np.zeros((nmc, len(ratio_fit_lmc)), dtype = 'float32')
        probs_lmc = np.zeros(nmc, dtype = 'float32')
        rand_ratio_fit_smc = np.zeros((nmc, len(ratio_fit_smc)), dtype = 'float32')
        probs_smc = np.zeros(nmc, dtype = 'float32')

        for i in range(nmc):
            ratio_fit_mw_i, fstar_fits_mw_i, d1mw_i, e_d1mw_i,d2mw_i,e_d2mw_i, d3mw_i,e_d3mw_i,d4mw_i,e_d4mw_i, prob_mwi =  get_ratios_from_fits('MW', element1, element2, ratio3o4_arr = xfit, err_ratio3o4_arr = np.zeros_like(xfit), el3=element3, el4=element4, out = 'rand')
            ratio_fit_lmc_i, fstar_fits_lmc_i, d1lmc_i,e_d1lmc_i, d2lmc_i, e_d2lmc_i,d3lmc_i,e_d3lmc_i, d4lmc_i,e_d4lmc_i, prob_lmci =  get_ratios_from_fits('LMC', element1, element2, ratio3o4_arr = xfit,err_ratio3o4_arr = np.zeros_like(xfit), el3=element3, el4=element4, out = 'rand')
            ratio_fit_smc_i, fstar_fits_smc_i, d1smc_i,e_d1smc_i, d2smc_i,e_d2smc_i, d3smc_i, e_d3smc_i,d4smc_i,e_d4smc_i, prob_smci =  get_ratios_from_fits('SMC', element1, element2, ratio3o4_arr = xfit, err_ratio3o4_arr = np.zeros_like(xfit),el3=element3, el4=element4, out = 'rand')


            rand_ratio_fit_mw[i, :] = ratio_fit_mw_i
            probs_mw[i] = prob_mwi
            rand_ratio_fit_lmc[i, :] = ratio_fit_lmc_i
            probs_lmc[i] = prob_lmci
            rand_ratio_fit_smc[i, :] = ratio_fit_smc_i
            probs_smc[i] = prob_smci

    else:
        xx_mw3= mw_dep3 #- (mw_log_nh3-20.)
        err_xx_mw3= mw_err_dep3
        xx_smc3= smc_dep3# - (smc_log_nh3-20.)
        err_xx_smc3 = smc_err_dep3
        xx_lmc3 = lmc_dep3 #- (log_nh-20.)
        err_xx_lmc3 = lmc_err_dep3

        #lowx = -3.3
        #highx = 1
        dep_key = '_dep'
        xtitle = r'$\delta$(' + element3 + ')'

        #this is the way I used to get teh fitted relation. Now I just get it from a linear range of znfe
        #THE NEW WAY WILL NOT WORK IF THE RATIO GIVEN IS NOT ZN/FE
        #xfit = np.arange(-4, 0, 0.1)
        #ratio_fit_mw, fstar_fits_mw, d1mw, e_d1mw, d2mw, e_d2mw,d3mw,e_d3mw,d4mw, e_d4mw  =  get_ratios_from_fits('MW', element1, element2, dep3 = xfit, el3=element3 )
        #ratio_fit_lmc, fstar_fits_lmc, d1lmc, e_d1lmc,d2lmc, e_d2lmc,d3lmc, e_d3lmc,d4lmc, e_d4lmc =  get_ratios_from_fits('LMC', element1, element2,dep3 =xfit,  el3=element3 )
        #ratio_fit_smc, fstar_fits_smc, d1smc,e_d1smc, d2smc,e_d2smc, d3smc, e_d3smc,d4smc ,e_d4smc=  get_ratios_from_fits('SMC', element1, element2,dep3 =xfit, el3=element3 )

        # PUT BACK AT STEP OF 0.01 LATER
        lin_znfe = np.arange(-1, 2, 0.1)
        dep_fit_lmc = np.zeros_like(lin_znfe)
        dep_fit_smc = np.zeros_like(lin_znfe)
        dep_fit_mw =  np.zeros_like(lin_znfe)
        for ii in range(len(lin_znfe)):
            dep_fit_lmc[ii], dumlmc = get_dep_znfe_realization(lin_znfe[ii], 0., 'LMC', element3)
            dep_fit_mw[ii], dummw = get_dep_znfe_realization(lin_znfe[ii], 0., 'MW', element3)
            dep_fit_smc[ii], dumsmc = get_dep_znfe_realization(lin_znfe[ii], 0., 'SMC', element3)

        #deps_decia = np.zeros_like(xx_lmc3)
        #deps_lmc = np.zeros_like(xx_lmc3)


        #for it in range(len(xx_lmc3)):
        #    deps_decia_i, err_deps_decia = get_depletions_decia(np.log10(lmc_ratio[it])-ref_lr, 0.05, element3, use_gal = 'dla')
        #    deps_lmc_i, err_deps_lmc = get_depletions_decia(np.log10(lmc_ratio[it])-ref_lr, 0.05, element3, use_gal = 'LMC')
        #    deps_decia[it] = deps_decia_i
        #    deps_lmc[it] = deps_lmc_i

        #print("De Cia residual ", np.sqrt(np.nanmean((deps_decia - xx_lmc3)**2)))
        #print("LMC residual ", np.sqrt(np.nanmean((deps_lmc - xx_lmc3)**2)))


        #No longer needed. Replaced by get_dep_znfe_realization
        #xfit = np.arange(-4, 0, 0.1)

        #rand_ratio_fit_mw = np.zeros((nmc, len(xfit)), dtype = 'float32')
        #probs_mw = np.zeros(nmc, dtype = 'float32')
        #rand_ratio_fit_lmc = np.zeros((nmc, len(xfit)), dtype = 'float32')
        #probs_lmc = np.zeros(nmc, dtype = 'float32')
        #rand_ratio_fit_smc = np.zeros((nmc, len(xfit)), dtype = 'float32')
        #probs_smc = np.zeros(nmc, dtype = 'float32')
        #for i in range(nmc):
        #    ratio_fit_mw_i, fstar_fits_mw_i, d1mw_i, e_d1mw_i,d2mw_i,e_d2mw_i, d3mw_i,e_d3mw_i,d4mw_i,e_d4mw_i, prob_mwi =  get_ratios_from_fits('MW', element1, element2, dep3 = xfit, el3=element3 , out = 'rand')
        #    ratio_fit_lmc_i, fstar_fits_lmc_i, d1lmc_i,e_d1lmc_i, d2lmc_i, e_d2lmc_i,d3lmc_i,e_d3lmc_i, d4lmc_i,e_d4lmc_i, prob_lmci =  get_ratios_from_fits('LMC', element1, element2, dep3 = xfit, el3=element3 , out = 'rand')
        #    ratio_fit_smc_i, fstar_fits_smc_i, d1smc_i,e_d1smc_i, d2smc_i,e_d2smc_i, d3smc_i, e_d3smc_i,d4smc_i,e_d4smc_i, prob_smci =  get_ratios_from_fits('SMC', element1, element2, dep3 = xfit, el3=element3, out = 'rand')
        #    rand_ratio_fit_mw[i, :] = ratio_fit_mw_i
        #    probs_mw[i] = prob_mwi
        #    rand_ratio_fit_lmc[i, :] = ratio_fit_lmc_i
        #    probs_lmc[i] = prob_lmci
        #    rand_ratio_fit_smc[i, :] = ratio_fit_smc_i
        #    probs_smc[i] = prob_smci

        rand_dep_fit_mw = np.zeros((nmc, len(lin_znfe)))
        rand_dep_fit_lmc = np.zeros((nmc, len(lin_znfe)))
        rand_dep_fit_smc = np.zeros((nmc, len(lin_znfe)))


        for i in range(nmc):
            for ii in range(len(lin_znfe)):
                rand_dep_fit_lmc[i,ii], dumlmc = get_dep_znfe_realization(lin_znfe[ii], 0., 'LMC', element3, rand=True)
                rand_dep_fit_mw[i,ii], dummw = get_dep_znfe_realization(lin_znfe[ii], 0., 'MW', element3, rand=True)
                rand_dep_fit_smc[i,ii], dumsmc = get_dep_znfe_realization(lin_znfe[ii], 0., 'SMC', element3, rand=True)

        std_dep_fit_mw = np.nanstd(rand_dep_fit_mw, axis = 0)
        std_dep_fit_lmc = np.nanstd(rand_dep_fit_lmc, axis = 0)
        std_dep_fit_smc = np.nanstd(rand_dep_fit_smc, axis = 0)

        xrange = [-3.5,0.3]

    #probs_mw = 10.**(probs_mw - np.nanmax(probs_mw))
    #probs_lmc = 10.**(probs_lmc - np.nanmax(probs_lmc))
    #probs_smc= 10.**(probs_smc - np.nanmax(probs_smc))


    yy_mw3 = np.log10(mw_ratio)-ref_lr# - (mw_log_nh3-20.)
    yy_lmc3 = np.log10(lmc_ratio) - ref_lr #- (log_nh-20.)
    yy_smc3 = np.log10(smc_ratio)-ref_lr# - (smc_log_nh3-20.)
    err_yy_smc3 = smc_ratio_err/smc_ratio/np.log(10.)
    err_yy_lmc3 = lmc_ratio_err/lmc_ratio/np.log(10.)
    err_yy_mw3 =   mw_ratio_err/mw_ratio/np.log(10.)


    #DLAs

    #Plot the DLA measurements

    #dla = ascii.read("DLA_depletion_sample_quiret2016.dat")
    if plot_dla ==True:
        if use_fe_dla==True:
            key = '_Fe'
        else:
            key=''

        dla_catalogs = ['Quiret2016_DTG_table_{}'.format(use_gal_dla) + key+'.fits', 'DeCia2016_all_data_{}'.format(use_gal_dla) + key +'.fits']
        colors = ['cyan', 'gray']
        dla_labels = ['Quiret+2016', 'De Cia+2016']

    plt.clf()
    plt.close()

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,8))
    lmc_color = 'dodgerblue'
    smc_color = 'magenta' #'green'
    mw_color = 'darkorange'
    msize = 10

    if plot_dla==True:

        dla_min_AFE  = -6.6
        dla_max_AFE = -3.
        dla_min_AZN = -9.6
        dla_max_AZN = -7.3

        for i in range(2):
            dla = fits.open(dla_catalogs[i])
            dla = dla[1].data

            if element4==0:
                valid_dla = np.where((dla['lim_{}'.format(element1)]=='v') & (dla['lim_{}'.format(element2)]=='v') & (dla['lim_{}'.format(element3)]=='v'))
            else:
                valid_dla = np.where((dla['lim_{}'.format(element1)]=='v') & (dla['lim_{}'.format(element2)]=='v') & (dla['lim_{}'.format(element3)]=='v')& (dla['lim_{}'.format(element4)]=='v'))
            dla = dla[valid_dla]

            dla_nhi = 10.**(dla['LOG_NHI'])

            dla_a1 = dla['gas_A_{}'.format(element1)]
            dla_a2 =  dla['gas_A_{}'.format(element2)]
            dla_a3 = dla['gas_A_{}'.format(element3)]
            dla_dep3 = dla['dep_{}'.format(element3)]
            if element4 !=0:
                dla_a4 = dla['gas_A_{}'.format(element4)]


            dla_ratio = dla_a1/dla_a2
            dla_met3 = dla_a3

            if element4 != 0:
                dla_met3 = dla_a3/dla_a4

            dla_a2_1, dla_e_a2_1, dla_b2_1, dla_eb2_1, dla_znfe0_1, dla_err_znfe0_1 = get_a2_b2_decia(element1)
            dla_a2_2, dla_e_a2_2, dla_b2_2, dla_eb2_2, dla_znfe0_2, dla_err_znfe0_2 = get_a2_b2_decia(element2)
            dla_a2_3, dla_e_a2_3, dla_b2_3, dla_eb2_3, dla_znfe0_3, dla_err_znfe0_3 = get_a2_b2_decia(element3)

            dla_dumx = np.arange(-1, 2.6, 0.1) #[Zn/Fe]
            dla_delta1 = dla_a2_1 + dla_b2_1*dla_dumx
            top = np.where(dla_delta1 >0.)
            if len(top[0]) > 0:
                dla_delta1[top] = 0.
            dla_delta2 = dla_a2_2 + dla_b2_2*dla_dumx
            top = np.where(dla_delta2 >0.)
            if len(top[0]) > 0:
                dla_delta2[top] = 0.
            dla_delta3 = dla_a2_3 + dla_b2_3*dla_dumx
            top = np.where(dla_delta3 >0.)
            if len(top[0]) > 0:
                dla_delta3[top] = 0.

            dla_ratio12 = dla_delta1-dla_delta2


            cmap = 'viridis'
            #dla_alpha = 1-0.9*(np.log10(dla['tot_A_Fe'])-dla_min_AFE)/(dla_max_AFE-dla_min_AFE)
            dla_alpha = 1-0.9*(np.log10(dla['gas_A_Zn'])-dla_min_AZN)/(dla_max_AZN-dla_min_AZN)

            if plot_dep==False:

                #dla_a2_4, dla_e_a2_4, dla_b2_4, dla_eb2_4, dla_znfe0_4, dla_err_znfe0_4 = get_a2_b2_mw_lmc_smc('dla', element4)
                #dla_delta4 = dla_a2_4 + dla_b2_4*dla_dumx
                #dla_ratio34 = dla_delta3  - dla_delta4
                #plt.plot(dla_ratio34, dla_ratio12, 'k--')

                if i == 0:
                    #ax.plot(np.log10(dla_met3)-ref_la, np.log10(dla_ratio)-ref_lr,'o', color = 'dimgrey', label = 'DLAs (Q16, DC16)', alpha = 0.5, markersize = msize)
                    ax.scatter(np.log10(dla_met3)-ref_la, np.log10(dla_ratio)-ref_lr,  marker='o', s = 100, cmap = cmap, c = dla_alpha)

                else:
                    #ax.plot(np.log10(dla_met3)-ref_la, np.log10(dla_ratio)-ref_lr,'o', color = 'dimgrey', alpha = 0.5, markersize = msize)
                    ax.scatter(np.log10(dla_met3)-ref_la, np.log10(dla_ratio)-ref_lr, marker='o', s = 100, cmap = cmap, c =dla_alpha)

            else:

                #reversing the axes here
                if use_gal_dla=='dla':
                    if i == 0:
                        ax.plot(dla_ratio12, dla_delta3, 'k--', label = 'DLAs (DC16)')
                    else:
                        ax.plot(dla_ratio12, dla_delta3, 'k--')


                    #plt.plot(dla_delta3, dla_ratio12, 'k--')
                if i == 0:
                    #plt.plot(dla_dep3, np.log10(dla_ratio)-ref_lr, 'o', color  = 'dimgrey', label = 'DLAs (Q16, DC16)', alpha = 0.5, markersize = msize)
                    #ax.plot( np.log10(dla_ratio)-ref_lr, dla_dep3, 'o', color  = 'dimgrey', label = 'DLAs (Q16, DC16)', alpha = 0.5, markersize = msize)
                    ax.scatter( np.log10(dla_ratio)-ref_lr, dla_dep3,  marker='o', s = 100, cmap = cmap, c =dla_alpha)

                else:
                    #plt.plot(dla_dep3, np.log10(dla_ratio)-ref_lr, 'o', color  = 'dimgrey', alpha = 0.5, markersize = msize)
                    #ax.plot( np.log10(dla_ratio)-ref_lr, dla_dep3, 'o', color  = 'dimgrey', alpha = 0.5, markersize = msize)

                    ax.scatter( np.log10(dla_ratio)-ref_lr, dla_dep3, marker='o', s = 100, cmap = cmap, c =dla_alpha)




    if plot_dep==True or element4!=0:

        if plot_dep==False:
            ax.plot(xfit, ratio_fit_mw, '--', color  = mw_color)
            ax.plot(xfit, ratio_fit_lmc, '--', color  = lmc_color)
            ax.plot(xfit, ratio_fit_smc, '--', color  = smc_color)
        else:
            #this was the old method
            #plt.plot(ratio_fit_mw,xfit, 'x', color  = mw_color)
            #plt.plot( ratio_fit_lmc,xfit, 'x', color  = lmc_color)
            #plt.plot(ratio_fit_smc, xfit, 'x', color  = smc_color)
            if element1=='Zn' and element2 =='Fe':
                ax.plot(lin_znfe, dep_fit_mw, '--', color = mw_color)
                ax.plot(lin_znfe, dep_fit_lmc, '--', color = lmc_color)
                ax.plot(lin_znfe, dep_fit_smc, '--', color = smc_color)

                print("TEST")
                print(std_dep_fit_mw)
                print(dep_fit_mw)

                ax.fill_between(lin_znfe, dep_fit_mw-std_dep_fit_mw, dep_fit_mw + std_dep_fit_mw , color = mw_color,  alpha = 0.3)
                ax.fill_between(lin_znfe, dep_fit_lmc-std_dep_fit_lmc, dep_fit_lmc + std_dep_fit_lmc , color = lmc_color,  alpha = 0.3)
                ax.fill_between(lin_znfe, dep_fit_smc-std_dep_fit_smc, dep_fit_smc + std_dep_fit_smc , color = smc_color,  alpha = 0.3)


        #COMMENT THIS DEBUG TEST
        #plt.plot(deps_decia, yy_lmc3, 'ko')
        #plt.plot(deps_lmc, yy_lmc3, 'o', color = 'green')

    if plot_dep==True:
        ax.errorbar( yy_mw3 , xx_mw3,yerr = err_xx_mw3, xerr = err_yy_mw3, fmt ='o',color = mw_color, capsize = 0, alpha = 0.7, label = 'MW', markersize = msize)
        ax.errorbar(yy_lmc3 , xx_lmc3, yerr = err_xx_lmc3, xerr = err_yy_lmc3, fmt ='o', color = lmc_color,capsize = 0, alpha = 0.6, label = 'LMC', markersize = msize)
        ax.errorbar( yy_smc3, xx_smc3, yerr = err_xx_smc3, xerr = err_yy_smc3, fmt='o', color =smc_color, capsize = 0, alpha = 0.6, label = 'SMC', markersize = msize)


    else:

        ax.errorbar(xx_mw3 , yy_mw3 ,xerr = err_xx_mw3, yerr = err_yy_mw3, fmt ='o',color = mw_color, capsize = 0, alpha = 0.7, label = 'MW', markersize = msize)
        ax.errorbar(xx_lmc3, yy_lmc3 , xerr = err_xx_lmc3, yerr = err_yy_lmc3, fmt ='o', color = lmc_color,capsize = 0, alpha = 0.6, label = 'LMC', markersize = msize)
        ax.errorbar(xx_smc3 , yy_smc3, xerr = err_xx_smc3, yerr = err_yy_smc3, fmt='o', color =smc_color, capsize = 0, alpha = 0.6, label = 'SMC', markersize = msize)




    #if ('Mg' not in [element1, element2, element3]) and ('S' not in [element1, element2, element3]) and ('O' not in [element1, element2, element3]):
    #    plt.errorbar(np.log10(tlmc_gas_met3)-ref_la, np.log10(tlmc_ratio)-ref_lr, xerr = tlmc_gas_met3_err/tlmc_gas_met3/np.log(10.), yerr = tlmc_ratio_err/tlmc_ratio/np.log(10.), fmt ='ko', mfc = 'none', capsize = 0)
    #    plt.errorbar(np.log10(tsmc_gas_met3)-ref_la, np.log10(tsmc_ratio)-ref_lr, xerr = tsmc_gas_met3_err/tsmc_gas_met3/np.log(10.), yerr = tsmc_ratio_err/tsmc_ratio/np.log(10.), fmt='o', color = 'deepskyblue', mfc = 'none', capsize = 0)


    if plot_clusters==True and element1 =='S' and element2=='Fe' and element3=='Fe' and plot_dep==True:

            ct = ascii.read('/astro/dust_kg/jduval/CLUSTER_DEPLETIONS/target_abundances.csv', delimiter = ',')

            cluster_colors = ['magenta', 'cyan', 'red', 'darkblue']
            for ic in range(len(ct)):
                print("TEST "  , ct['N(SII)'].data[ic] ,  ct['N(FeII)'].data[ic], ref_ratio )
                ax.plot(ct['N(SII)'].data[ic] - ct['N(FeII)'].data[ic] - ref_lr, ct['delta(Fe)'].data[ic],'*', color = cluster_colors[ic], label = ct['Region'].data[ic], markersize = 20)
                ax.plot(ct['N(SII)'].data[ic] - ct['N(FeII)'].data[ic] - ref_lr, ct['delta(Fe)'].data[ic], 'k*',  markersize = 20, markerfacecolor = 'none')

    if plot_dep==True:
        ax.set_ylabel(xtitle, fontsize = 23)
        ax.set_xlabel('[' + element1+'/'+element2 + ']', fontsize = 23)
    else:
        ax.set_xlabel(xtitle, fontsize = 23)
        ax.set_ylabel('[' + element1+'/'+element2 + ']', fontsize = 23)

    #plt.xlim([lowx, highx])
    dla_key = ''
    if plot_dla==False:
        dla_key = '_nodla'

    if lowx == 0:
        lowx = xrange[0]
    if highx ==0:
        highx = xrange[1]

    if plot_dep==True:
        ax.set_xlim(left= lowy, right = highy)
        ax.set_ylim(bottom = lowx, top = highx)
    else:
        ax.set_xlim(left = lowx, right = highx)
        ax.set_ylim(bottom = lowy, top = highy)

    ax.minorticks_on()
    ax.tick_params(which='major', width=2, length = 6)
    ax.tick_params(which='minor', width=1, length = 4)

    if plot_other ==True:
        ax.legend(fontsize = 16, loc = 'lower left')

    if plot_dla==True:
        vmin = dla_min_AZN -(4.7-12)
        vmax=dla_max_AZN -(4.7-12)

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap+'_r', norm=norm)
        sm.set_array([])

        ##divider = make_axes_locatable(ax2[0,i])
        ##cax = divider.new_vertical(size='5%', pad=0.1)
        cax = inset_axes(ax, width = '80%', height = '5%', loc = 'upper center')
        fig.add_axes(cax)
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=np.linspace(vmin, vmax, 4))
        ##ax2[0,i].colorbar( loc = 't', label = '[Fe/H]', ticks = 5, values = np.linspace(vmin, vmax, 5))
        ##cbar  = fig2.colorbar(sm, ticks=np.linspace(vmin, vmax, 5), ax=ax2[0,i], shrink = 0.95, orientation = 'horizontal', pad = 0.1, location = 'top')

        cbar.set_ticks(np.linspace(vmin, vmax, 4))
        cbar.set_ticklabels(['{:3.1f}'.format(x) for x in np.linspace(vmin, vmax, 4)])
        #cbar.ax.tick_params(axis='x',direction='in',labeltop='on')
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.tick_params(labelsize=15)
        #cbar.set_label('[Fe/H]')
        cax.set_title('DLA gas-phase [Zn/H]', fontsize = 18)

    #fig.tight_layout()
    fig.subplots_adjust(left = 0.15, bottom = 0.1, right = 0.95, top = 0.9)

    cluster_key = ''
    if plot_clusters ==True:
        cluster_key = '_cluster'

    if plot_dla==True:
        if element4 ==0:
            plt.savefig(dir + "plot_abundances_" + element1 + '_' + element2 + '_' + element3 + dep_key + '_{}'.format(use_gal_dla) + key + cluster_key+'.pdf', format= 'pdf', dpi = 1000)
        else:
            plt.savefig(dir + "plot_abundances_" + element1 + '_' + element2 + '_' + element3 + '_' + element4 + dep_key + '_{}'.format(use_gal_dla) + key + cluster_key +'.pdf', format= 'pdf', dpi = 1000)
    else:
        if element4 ==0:
            plt.savefig(dir + "plot_abundances_" + element1 + '_' + element2 + '_' + element3 + dep_key + cluster_key + '_nodla.pdf', format= 'pdf', dpi = 1000)
        else:
            plt.savefig(dir + "plot_abundances_" + element1 + '_' + element2 + '_' + element3 + '_' + element4 + dep_key + cluster_key + '_nodla.pdf', format= 'pdf', dpi = 1000)



    plt.clf()
    plt.close()




def plot_depletions_inter_el(el1, el2, plot_fits = True, plot_points=True, ymin = -4, xmin = -4, ymax = 1, xmax = 1,  dir = './FIGURES_ALL/'):

    """
    PLots depletion of elements 2 (list or array) vs depletion of element 1 (string)
    """

    table = fits.open("compiled_depletion_table_cos_stis_wti_corr_sii_ebj_oi_strong_zn_fstar_fix1220_pyh.fits")
    table = table[1].data
    elements = np.array([x.replace('I', '') for x in table['ELEMENT']])

    ind1 = np.where(el1 == elements)
    ind1 = ind1[0]

    lmc_dep1 = table['DEPLETIONS'][:, ind1]
    lmc_err_dep1 = table['ERR_DEPLETIONS'][:, ind1]
    flags1 = table['FLAG'][:, ind1]
    instr1 = table['GRATING'][:, ind1]



    mw_log_nh1, mw_err_log_nh1, mw_log_nh21, mw_err_log_nh21,  fstar_mw1, mw_dep1, mw_err_dep1, mw_gas_met1, mw_gas_met_err1, mw_t1  = get_depletions(el1, "J09", "MW", nofilter = True)
    smc_log_nh1, smc_err_log_nh1, smc_log_nh21, smc_err_log_nh21,  fstar_smc1, smc_dep1, smc_err_dep1, smc_gas_met1, smc_gas_met_err1, smc_t1  = get_depletions(el1, "J17", "SMC", nofilter = True)


    ax1_lmc, e_ax1_lmc, bx1_lmc, e_bx1_lmc, zx1_lmc = get_fstar_coefs("LMC", el1, use_mw=False)
    ax1_mw, e_ax1_mw, bx1_mw, e_bx1_mw, zx1_mw = get_fstar_coefs("MW", el1, use_mw=False)
    ax1_smc, e_ax1_smc, bx1_smc, e_bx1_smc, zx1_smc = get_fstar_coefs("SMC", el1, use_mw=False)

    plt.clf()
    plt.close()

    nwide = min([len(el2), 2])
    nhigh = (len(el2)-1)//2 +1

    fig,ax = plt.subplots(nhigh, nwide, sharex = True, sharey = True,figsize = (18*nwide//nhigh,10))

    mfc = 'none'
    lmc_color = 'dodgerblue'
    mw_color = 'darkorange'
    smc_color= 'magenta'

    xmin = -3.3
    xmax = 0.3
    ymin = -4.
    ymax = 1.

    nel2 = len(el2)

    for i in range(nel2):
        xpos = i % nwide
        ypos = i // nwide

        if nhigh ==1:
            if nwide ==1:
                plt_ax = ax
            else:
                plt_ax = ax[xpos]
        else:
            plt_ax = ax[ypos, xpos]

        ind2 = np.where(el2[i] == elements)
        ind2 = ind2[0]

        ax2_smc, e_ax2_smc, bx2_smc, e_bx2_smc, zx2_smc = get_fstar_coefs("SMC", el2[i], use_mw=False)
        ax2_lmc, e_ax2_lmc, bx2_lmc, e_bx2_lmc, zx2_lmc = get_fstar_coefs("LMC", el2[i], use_mw=False)
        ax2_mw, e_ax2_mw, bx2_mw, e_bx2_mw, zx2_mw = get_fstar_coefs("MW", el2[i], use_mw=False)


        lmc_dep2 = table['DEPLETIONS'][:, ind2]
        lmc_err_dep2 = table['ERR_DEPLETIONS'][:, ind2]
        flags2 = table['FLAG'][:, ind2]


        instr2 = table['GRATING'][:, ind2]

        values = np.where((flags1 == 'v') & (flags2 == 'v'))
        values = values[0]

        v1l2 =np.where((flags1=='v') & (flags2 == 'l'))
        v1u2 = np.where((flags1 == 'v') & (flags2 == 'u'))

        l1v2 =np.where((flags2=='v') & (flags1 == 'l'))
        u1v2 = np.where((flags2 == 'v') & (flags1 == 'u'))

        l1l2 = np.where((flags1 == 'l') & (flags2 == 'l'))
        u1u2 = np.where((flags1 == 'u') & (flags2 == 'u'))

        u1l2 = np.where((flags1 == 'u') & (flags2 == 'l'))
        l1u2=np.where((flags1 == 'l') & (flags2 == 'u'))

        v1l2 = v1l2[0]
        v1u2 = v1u2[0]
        l1v2 = l1v2[0]
        u1v2 = u1v2[0]
        l1l2 = l1l2[0]
        u1u2 = u1u2[0]
        u1l2 = u1l2[0]
        l1u2= l1u2[0]


        #plt.fill_between(lindep1, lindep2_low, lindep2_up, facecolor = 'darkgreen', edgecolor = 'darkgreen', alpha = 0.3)

        if plot_fits==True:

            lindep1  = np.arange(ymin,ymax, 0.1)

            lindep2_lmc = bx2_lmc + ax2_lmc*((lindep1-bx1_lmc)/ax1_lmc + zx1_lmc-zx2_lmc)
            trends_lmc, probs_lmc = mc_dep_fit_rel_errors(lindep1, ax1_lmc, e_ax1_lmc, bx1_lmc,e_bx1_lmc, zx1_lmc,ax2_lmc, e_ax2_lmc, bx2_lmc,e_bx2_lmc, zx2_lmc,nmc = 10)

            #for imc in range(len(probs_lmc)):
            #    plt_ax.plot(lindep1, trends_lmc[:,imc], '-', color = lmc_color, alpha = np.sqrt(probs_lmc[imc]))
            plt_ax.plot(lindep1, lindep2_lmc, '-', color= lmc_color)#, label = 'LMC')

            lindep2_mw = bx2_mw + ax2_mw*((lindep1-bx1_mw)/ax1_mw + zx1_mw-zx2_mw)
            trends_mw, probs_mw = mc_dep_fit_rel_errors(lindep1, ax1_mw, e_ax1_mw, bx1_mw,e_bx1_mw, zx1_mw,ax2_mw, e_ax2_mw, bx2_mw,e_bx2_mw, zx2_mw, nmc = 10)

            #for imc in range(len(probs_mw)):
            #    plt_ax.plot(lindep1, trends_mw[:,imc], '-', color = mw_color, alpha = np.sqrt(probs_mw[imc]))

            plt_ax.plot(lindep1, lindep2_mw, '--', color = mw_color)#,  label = 'MW')

            lindep2_smc = bx2_smc + ax2_smc*((lindep1-bx1_smc)/ax1_smc + zx1_smc-zx2_smc)

            trends_smc, probs_smc = mc_dep_fit_rel_errors(lindep1, ax1_smc, e_ax1_smc, bx1_smc,e_bx1_smc, zx1_smc,ax2_smc, e_ax2_smc, bx2_smc,e_bx2_smc, zx2_smc,nmc = 10)

            #for imc in range(len(probs_smc)):
            #    plt_ax.plot(lindep1, trends_smc[:,imc], '-', color = smc_color, alpha = np.sqrt(probs_smc[imc]))
                #print("TEST ALPHA", np.sqrt(probs_smc[imc])/10.)
            plt_ax.plot(lindep1, lindep2_smc, '--', color = smc_color)#, label = 'SMC')

        if plot_points==True:

            mw_log_nh2, mw_err_log_nh2, mw_log_nh22, mw_err_log_nh22,  fstar_mw2, mw_dep2, mw_err_dep2, mw_gas_met2, mw_gas_met_err2, mw_t2  = get_depletions(el2[i], "J09", "MW", nofilter = True)

            if len(mw_log_nh2)>0 and len(mw_log_nh1) >0:
                alpha_mw_mg = 0.5
                if el2[i] == 'Mg':
                    alpha_mw_mg = 0.3
                plt_ax.errorbar(mw_dep1, mw_dep2, xerr = mw_err_dep1, yerr =  mw_err_dep2, fmt = 'o', color =mw_color, label = 'MW', mfc = 'none', alpha = alpha_mw_mg)

            plt_ax.errorbar(lmc_dep1[values].flatten(), lmc_dep2[values].flatten(), xerr = lmc_err_dep1[values].flatten(), yerr = lmc_err_dep2[values].flatten(), fmt = 'o', color = lmc_color, label = 'LMC', mfc = 'none', alpha =0.7)

            if len(v1l2)>0:
                plt_ax.errorbar(lmc_dep1[v1l2].flatten(), lmc_dep2[v1l2].flatten(), xerr = lmc_err_dep1[v1l2].flatten(), yerr = 0.5, fmt = '.', color = lmc_color, lolims = True, mfc= mfc, alpha = 0.7)


            if len(v1u2)>0:
                plt_ax.errorbar(lmc_dep1[v1u2].flatten(), lmc_dep2[v1u2].flatten(), xerr = lmc_err_dep1[v1u2].flatten(), yerr = 0.5, fmt = '.', color = lmc_color, uplims = True, mfc= mfc, alpha = 0.7)

            if len(l1v2)>0:
                plt_ax.errorbar(lmc_dep1[l1v2].flatten(), lmc_dep2[v1l2].flatten(), xerr = 0.5, yerr = lmc_dep2[l1v2].flatten(), fmt = '.', color = lmc_color, xlolims = True, mfc= mfc, alpha = 0.7)

            if len(u1v2)>0:
                plt_ax.errorbar(lmc_dep1[u1v2].flatten(), lmc_dep2[u1v2].flatten(), xerr = 0.5, yerr = lmc_dep2[u1v2].flatten(), fmt = '.', color = lmc_color, xuplims = True, mfc= mfc, alpha = 0.7)

            if len(l1l2)>0:
                plt_ax.errorbar(dep1_lmc[l1l2].flatten(), lmc_dep2[l1l2].flatten(), xerr =0.5, yerr = 0.5, fmt = '.', color = lmc_color, xlolims = True, lolims = True, mfc= mfc, alpha = 0.7)

            if len(u1u2)>0:
                plt_ax.errorbar(lmc_dep1[u1u2].flatten(), lmc_dep2[u1u2].flatten(), xerr =0.5, yerr = 0.5, fmt = '.', color = lmc_color, xuplims = True, uplims = True, mfc= mfc, alpha = 0.7)

            if len(u1l2)>0:
                plt_ax.errorbar(lmc_dep1[u1l2].flatten(), lmc_dep2[u1l2].flatten(), xerr =0.5, yerr = 0.5, fmt = '.', color = lmc_color, xuplims = True, lolims = True, mfc= mfc, alpha  = 0.7)

            if len(l1u2)>0:
                plt_ax.errorbar(lmc_dep1[l1u2].flatten(), lmc_dep2[l1u2].flatten(), xerr =0.5, yerr = 0.5, fmt = '.', color = lmc_color, xlolims = True, uplims = True, mfc= mfc, alpha = 0.7)



            smc_log_nh2, smc_err_log_nh2, smc_log_nh22, smc_err_log_nh22,  fstar_smc2, smc_dep2, smc_err_dep2, smc_gas_met2, smc_gas_met_err2, smc_t2 = get_depletions(el2[i], "J17", "SMC", nofilter = True)


            if len(smc_log_nh2)>0 and len(smc_log_nh1) >0:
                plt_ax.errorbar(smc_dep1, smc_dep2, xerr = smc_err_dep1, yerr = smc_err_dep2, fmt = 'o', color = smc_color,label = 'SMC', mfc = 'none', alpha = 0.5)


        plt_ax.set_xlim(left=xmin, right=xmax)
        plt_ax.set_ylim(bottom = ymin,top=ymax)
        dumx = np.arange(xmin, xmax, 0.1)
        plt_ax.plot(dumx,np.zeros_like(dumx), '--', color = 'darkgray', alpha =0.5)
        plt_ax.text(-3., 0.1, el2[i], fontsize = 22)
        #plt_ax.set_xticks(np.arange(xmin, xmax, 1))
        #plt_ax.set_xticklabels(['{}'.format(x) for x in list(np.arange(xmin,xmax,1))])


        #get teh MW fits to Fstar to make the Fstar axis
        mw_fits = ascii.read("jenkins09_table4_dep_fits_adj_zeropt_zn.dat")
        ind_el_mw = np.where(mw_fits['Elements'].data == el1)
        ind_el_mw = ind_el_mw[0]
        if (len(ind_el_mw)>0) and (ypos==0):
            #xaxis = np.arange(xmin, xmax,1)
            Ael = mw_fits['AX'].data[ind_el_mw[0]]
            Bel =  mw_fits['BX'].data[ind_el_mw[0]]
            zel =  mw_fits['zX'].data[ind_el_mw[0]]
            fstar_mw = np.arange(-1.,1.1,1)#(xaxis-Bel)/Ael + zel
            dep_mw= (fstar_mw-zel)*Ael + Bel
            plt_ax2 = plt_ax.twiny()
            plt_ax2.set_xticks(dep_mw)
            plt_ax2.set_xticklabels(['{:2.1f}'.format(x) for x in fstar_mw])
            plt_ax2.set_xlabel(r'$F_*$', fontsize = 23)
            plt_ax2.set_xlim(plt_ax.get_xlim())
            plt_ax2.minorticks_on()
        if ypos ==nhigh-1:
            plt_ax.set_xlabel(r'$\delta$'+'(' + el1 + ')', fontsize = 23)
        if xpos==0:
            #ax[ypos, xpos].set_ylabel(r'$\delta$'+'(' + el2[i] + ')', fontsize = 18)
            plt_ax.set_ylabel(r'$\delta$'+'(X)', fontsize = 23)
        plt_ax.minorticks_on()
        plt_ax.legend( fontsize = 13, loc = 'lower right')

    fig.subplots_adjust(wspace = 0, hspace = 0, bottom = 0.1, top = 0.9, left = 0.15, right = 0.95)
    plt.savefig(dir + 'plot_dep_' + el1 + '_' + '_'.join(el2) + '.pdf', format = 'pdf', dpi = 1000)

    plt.clf()
    plt.close()




def plot_depletions_galaxies( plot_one = '', plot_z=False, dir = './FIGURES_ALL/', plot_fits = False, plot_points=True, plot_dla=False, plot_clusters=False, use_gal_dla='dla', use_fe_dla=False):


    lmc_fits_table="METAL_depletion_fits_coefs_lognh_pyh_20_22.dat"
    mw_fits_table="MWJ09_depletion_fits_coefs_lognh_pyh_20_22_adj_zeropt_zn.dat"
    smc_fits_table ="SMCJ17_depletion_fits_coefs_lognh_pyh_20_22.dat"

    fits_tables = np.array([mw_fits_table, lmc_fits_table, smc_fits_table])

    lmc_program = 'METAL'
    smc_program = 'J17'
    mw_program = 'J09'

    programs = np.array([mw_program, lmc_program, smc_program])

    lmc_color = 'dodgerblue'
    smc_color = 'magenta'
    mw_color = 'darkorange'

    colors= np.array([mw_color, lmc_color, smc_color])

    galaxies = np.array(['MW', 'LMC', 'SMC'])

    nhmin = 19
    nhmax = 22.5
    ymin = -3.5
    ymax = 1

    xrange = [nhmin, nhmax]
    xtext = nhmax -0.55
    ytext = ymax - 0.5

    if plot_z ==True:
        ymin = 1.e-11
        ymax = 5.e-5
        ytext = ymax * 0.1

    if plot_one =='':
        lines= np.array(['MgII', 'SiII', 'SII', 'TiII', 'CrII', 'FeII', 'NiII', 'CuII',  'ZnII'])
        #lines = np.array(['MgII', 'SiII', 'FeII', 'CrII'])
        #lines = np.array(['SiII', 'FeII',  'SII', 'ZnII'])

        nwide = 3
        nhigh = 3
        width = 14
        height =12
    else:
        lines = np.array([plot_one])
        nwide = 1
        nhigh = 1
        width =  10
        height  = 8

    elements  = np.array([x.replace('I', '') for x in lines])

    nel = len(lines)

    if plot_z==True:
        savekey  = 'gasmet'
    else:
        savekey = 'dep'

    plt.clf()
    plt.close()

    #First plot the DLAs
    key = ''
    if use_fe_dla==True:
        key = '_Fe'

    dla_catalogs = ['Quiret2016_DTG_table_{}'.format(use_gal_dla) + key+'.fits', 'DeCia2016_all_data_{}'.format(use_gal_dla) + key +'.fits']
    dla_colors = ['cyan', 'gray']
    dla_labels = ['Quiret+2016', 'De Cia+2016']

    #linfstar = np.arange(xrange[0],xrange[1], 0.1)
    linfstar = np.arange(19.5, 22.5, 0.1)

    #if input_plot==False:
    fig, ax = plt.subplots(nhigh,nwide, figsize = (width,height), sharex=True, sharey=True)

    for i in range(nel):
        xpos = i % nwide
        ypos = i // nwide

        if plot_one=='':
            axp = ax[ypos, xpos]
        else:
            axp = ax
        if plot_dla==True:
            for ic in range(2):
                dla = fits.open(dla_catalogs[ic])
                dla = dla[1].data

                if 'dep_{}'.format(lines[i].replace('I', '')) in dla.columns.names:

                    cmap = matplotlib.colors.ListedColormap(['gray', 'blueviolet','magenta', 'limegreen', 'blue'])#['gold', 'blue','limegreen', 'blueviolet', 'magenta', 'gray'])
                    #boundaries = [ 0.,  -0.5, -1, -1.5, -2]
                    #norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N, clip=True)

                    z = np.log10(dla['tot_A_Fe']) +12. - 7.54# - np.nanmin(np.log10(dla['tot_A_Fe']))
                    good = np.where(np.isnan(z)==False)
                    if plot_z==False:

                        im = axp.scatter(dla['LOG_NHI'][good], dla['dep_{}'.format(lines[i].replace('I', ''))][good], c=z[good] , cmap = cmap, alpha = 0.5)
                    else:
                        im= axp.scatter(dla['LOG_NHI'][good], dla['gas_A_{}'.format(lines[i].replace('I', ''))][good], c = z[good], alpha= 0.5, cmap = cmap)

        for j in range(len(galaxies)):
            galaxy = galaxies[j]

            if galaxy == 'LMC':
                log_nh, err_log_nh, log_nh2, err_log_nh2,  fstar, dep, err_dep, gas_met, gas_met_err, targets, flags = get_metal_dep_sample(elements[i],nofilter = False)

            else:
                log_nh, err_log_nh, log_nh2, err_log_nh2,  fstar, dep, err_dep, gas_met, gas_met_err, targets = get_depletions(elements[i], programs[j], galaxy, nofilter = False)
                flags = np.array(['v',]*len(log_nh))

            yerr = 0.5

            values = np.where(flags == 'v')
            values = values[0]
            upper = np.where(flags=='u') #l/u reversed since upper limit on column density becomes lower limit on depletion
            upper = upper[0]
            lower = np.where(flags=='l')
            lower = lower[0]


            if plot_z ==False:

                dumx = np.arange(nhmin, nhmax,0.1)
                axp.plot(dumx, np.zeros_like(dumx), '--', color = 'darkgray', alpha = 0.5)

                if plot_points ==True:
                    if len(values)>0:
                        axp.errorbar(log_nh[values], dep[values], xerr = err_log_nh[values], yerr= err_dep[values], fmt = 'o', color = colors[j], mfc = 'none', label = galaxy, alpha = 0.5)

                    if len(lower) >0:
                        axp.errorbar(log_nh[lower], dep[lower], xerr = err_log_nh[lower], yerr = 0.5, fmt = '.', mfc = 'none', color = colors[j], lolims = True, alpha  =0.5)
                    if len(upper) > 0:
                        axp.errorbar(log_nh[upper], dep[upper], xerr = err_log_nh[upper], yerr = 0.5, fmt = '.', mfc = 'none', color = colors[j], uplims = True, alpha =0.5)



                if plot_fits==True:


                    print(fits_tables[j])
                    dep_fits_table = ascii.read(fits_tables[j])

                    #print(dep_fits_table.columns)


                    tind = np.where(dep_fits_table['Elements'].data == elements[i])
                    tind = tind[0]

                    if len(tind)>0:

                        tind = tind[0]

                        this_ax = dep_fits_table['Slope'].data[tind]
                        err_ax = dep_fits_table['err_slope'].data[tind]
                        bx = dep_fits_table['Intercept'].data[tind]
                        err_bx = dep_fits_table['err_intercept'].data[tind]
                        zx = dep_fits_table['NH0'].data[tind]
                        rvalue = dep_fits_table['Rvalue'].data[tind]
                        pvalue = dep_fits_table['Pvalue'].data[tind]

                        trend  = bx + this_ax*(linfstar-zx)
                        trends, probs = mc_lin_fits_errors(linfstar, this_ax, err_ax, bx, err_bx, zx,  nmc = 100)

                        for imc in range(len(probs)):
                            axp.plot(linfstar, trends[:,imc], '-', color = colors[j], alpha = np.sqrt(probs[imc])/3.)
                            axp.plot(linfstar, trend, '-', color =colors[j])


            else:

                if len(values)>0:

                    this_err = gas_met_err[values]
                    errb = np.where(this_err >= gas_met[values])
                    if len(errb[0])>0:
                        this_err[errb[0]] = 0.9*gas_met[values[errb]]

                    axp.errorbar(log_nh[values], gas_met[values], xerr = err_log_nh[values], yerr = this_err , fmt = 'o', color = colors[j], mfc = 'none', label = galaxy, alpha = 0.5)
                if len(lower) >0:
                    axp.errorbar(log_nh[lower], gas_met[lower], xerr = err_log_nh[lower], yerr = gas_met[lower], fmt = '.', mfc = 'none', color = colors[j], lolims = True, alpha =0.5)
                if len(upper) > 0:
                    axp.errorbar(log_nh[upper], gas_met[upper], xerr = err_log_nh[upper], yerr = 0.5*gas_met[upper], fmt = '.', mfc = 'none', color = colors[j], uplims = True, alpha  =0.5)



        if plot_clusters==True and (plot_one=='S' or plot_one=='Fe'):

            ct = ascii.read('/astro/dust_kg/jduval/CLUSTER_DEPLETIONS/target_abundances.csv', delimiter = ',')

            cluster_colors = ['magenta', 'cyan', 'red', 'darkblue']
            for ic in range(len(ct)):


                axp.plot(ct['N(HI)'].data[ic], ct['delta({})'.format(plot_one)].data[ic], '*', color = cluster_colors[ic], label = ct['Region'].data[ic], markersize = 20)
                axp.plot(ct['N(HI)'].data[ic], ct['delta({})'.format(plot_one)].data[ic], 'k*', markerfacecolor = 'none', markersize = 20)

        axp.legend(fontsize = 15, loc = 'lower left')
        axp.text(xtext, ytext, elements[i], fontsize = 18)
        axp.minorticks_on()

        if ypos == nhigh-1:
            axp.set_xlabel('log N(H)', fontsize = 23)
        if xpos == 0:
            if plot_z ==False:
                axp.set_ylabel(r'$\delta(X)$', fontsize = 23)
            else:
                axp.set_ylabel("Gas X/H", fontsize = 23)

        axp.set_xlim(left = nhmin, right = nhmax)
        axp.set_ylim(bottom = ymin, top = ymax)

        if plot_z==True:
            axp.set_yscale('log')
            y_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 4)
            axp.yaxis.set_major_locator(y_major)
            y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
            axp.yaxis.set_minor_locator(y_minor)

            if xpos == 0:
                nbins = len(axp.get_yticklabels())
                print('TEST', nbins)
                axp.yaxis.set_major_locator(MaxNLocator(nbins = nbins, prune='upper'))
            #axp.get_yticklabels()[0].set_visible(False)
            #ytickmarks = 10**(np.arange(-11, -5.9, 1))
            #yticklabels = ytickmarks
            #axp.set_yticks(ytickmarks)
            #axp.set_yticklabels(yticklabels)
            #axp.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))


    if plot_dla==True:
        fig.subplots_adjust(hspace = 0, wspace = 0, top = 0.95, bottom = 0.12, left = 0.12, right = 0.8)
        cbar_ax = fig.add_axes([0.82, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)#, label = '[Fe/H]')
        cbar.set_label('[Fe/H]', rotation=270, fontsize = 20, labelpad = 20 )
    else:
        fig.subplots_adjust(hspace = 0, wspace = 0, top = 0.95, bottom = 0.12, left = 0.12, right = 0.95)
    if plot_one != '':
        plt.tight_layout()
        #axp.set_ylim(bottom = 1e-9, top = 5e-5)
    if plot_dla==False:
        plt.savefig(dir + "plot_MW_LMC_SMC_depletions_" + savekey + plot_one + '.pdf', format = 'pdf', dpi =1000)
    else:
        if use_fe_dla==True:
            fe_key = '_Fe'
        else:
            fe_key = ''
        plt.savefig(dir + "plot_MW_LMC_SMC_depletions_" + savekey + plot_one + '_' + use_gal_dla + fe_key + '.pdf', format = 'pdf', dpi =1000)


    plt.clf()
    plt.close()



def debug():
    t= fits.open("DeCia2016_all_data_LMC.fits")
    t = t[1].data

    deps = np.zeros(len(t))

    for i in range(len(t)):
        delta, err_delta = get_depletions_decia(np.log10(t['gas_A_Zn'][i]/t['gas_A_Fe'][i]) - (4.7-7.54),0.02, 'Fe', use_gal = 'LMC')
        deps[i] = delta

    ratio_fit_lmc, fstar_fits_lmc, d1lmc,e_d1lmc, d2lmc, e_d2lmc,d3lmc,e_d3lmc, d4lmc,e_d4lmc =  get_ratios_from_fits('LMC', 'Zn', 'Fe',  dep3 = deps, el3='Fe')

    plt.plot(t['dep_Fe'], deps, 'ko')
    dumx = np.arange(-3, 1, 0.1)
    plt.plot(dumx, dumx, '-', color = 'green')
    plt.show()

    plt.plot(t['dep_Fe'], np.log10(t['gas_A_Zn']/t['gas_A_Fe']) - (4.7-7.54), 'ko', alpha = 0.5)
    plt.plot(deps, np.log10(t['gas_A_Zn']/t['gas_A_Fe']) - (4.7-7.54), 'ro', alpha = 0.5)
    plt.plot(deps, ratio_fit_lmc, 'o', color ='green', alpha = 0.5)
    plt.show()


def plot_feldmann(log_nh0_mw = 21., use_fe_dla=False, use_gal_dla = 'dla', plot_dla=True, newc=False):

    print("COMPUTING MW")
    dgr_mw, e_dgr_mw, dmr_mw,e_dmr_mw,zgr_mw, deps, err_deps, w, a, elt = compute_gdr_lv(galaxy = 'MW',log_nh0 = np.array([log_nh0_mw]), newc = newc)
    #print("COMPUTING LMC")
    dgr_lmc,e_dgr_lmc,dmr_lmc,e_dmr_lmc, zgr_lmc ,deps_lmc, err_deps_lmc, wlmc, almc, elt_lmc = compute_gdr_lv(galaxy = 'LMC',log_nh0 = np.array([log_nh0_mw]), newc = newc)
    #print("COMPUTING SMC")

    print("SMC SMC SMC ")
    dgr_smc,e_dgr_smc, dmr_smc, e_dmr_smc, zgr_smc, deps_smc, err_deps_smc, wsmc, asmc, elt_smc = compute_gdr_lv(galaxy = 'SMC',log_nh0 = np.array([log_nh0_mw]), newc = newc)
    print("SMC ERR DEP ", err_deps_smc)
    #if e_dgr_smc > dgr_smc:
    #    e_dgr_smc = 0.9*dgr_smc

    dgr_smc_int, e_dgr_smc_int, dmr_smc_int, e_dmr_smc_int = compute_gdr_integrated("SMC", newc = newc)
    dgr_lmc_int, e_dgr_lmc_int, dmr_lmc_int, e_dmr_lmc_int = compute_gdr_integrated("LMC", newc = newc)
    #Kalberla+2009 show surface denisty of HI is about constant in inner galaxy at 1.5e21 cm-2. To that, add surface density of molecular gas from RD2016, or 5e20cm-2 in the inner galaxy. That leads to about 2e21 cm-2 through teh disk

    dgr_mw_int, e_dgr_mw_int, dmr_mw_int,e_dmr_mw_int,zgr_mw_int, deps_int, err_deps_int, w, a, elt_mw = compute_gdr_lv(galaxy = 'MW',log_nh0 = np.array([np.log10(2e21)]) ,newc = newc)


    dgrs= np.array([dgr_mw, dgr_lmc, dgr_smc]).flatten()
    e_dgrs = np.array([e_dgr_mw, e_dgr_lmc, e_dgr_smc]).flatten()

    dgrs_int= np.array([dgr_mw_int, dgr_lmc_int, dgr_smc_int])
    e_dgrs_int = np.array([e_dgr_mw_int, e_dgr_lmc_int, e_dgr_smc_int])

    dmrs = np.array([dmr_mw, dmr_lmc, dmr_smc]).flatten()
    e_dmrs = np.array([e_dmr_mw , e_dmr_lmc, e_dmr_smc])

    dmrs_int = np.array([dmr_mw_int, dmr_lmc_int, dmr_smc_int]).flatten()
    e_dmrs_int = np.array([e_dmr_mw_int , e_dmr_lmc_int, e_dmr_smc_int])



    #ind = np.where(e_dgrs > dgrs)
    #e_dgrs[ind] = 0.9*dgrs[ind]

    dict_fir_lmc = scipy.io.readsav("/astro/dust_kg/jduval/DUST_POOR_COMPONENT/MC_SAVE_FILES/binned_IRAS_PLANCK_surface_brightness_lmc_quad_cirrus_thresh_stray_corr_mbb_a6.400_FITS_0.05_FINAL.save")
    dict_fir_smc = scipy.io.readsav("/astro/dust_kg/jduval/DUST_POOR_COMPONENT/MC_SAVE_FILES/binned_IRAS_PLANCK_surface_brightness_smc_quad_cirrus_thresh_stray_corr_mbb_a21.00_FITS_0.05_FINAL.save")


    fir_gas_lmc = dict_fir_lmc['gas_bins']/0.8e-20
    fir_gd_lmc = dict_fir_lmc['gdr_ratio']
    fir_err_gd_lmc = dict_fir_lmc['percentile50_high'] - dict_fir_lmc['percentile50_low']#dict_fir['err_gdr_ratio']


    ind_lmc = np.argmin(np.abs(fir_gas_lmc - 10.**(log_nh0_mw)))
    fir_gd_lmc = fir_gd_lmc[ind_lmc]
    fir_gas_lmc = fir_gas_lmc[ind_lmc]
    fir_err_gd_lmc = fir_err_gd_lmc[ind_lmc]

    fir_gas_smc = dict_fir_smc['gas_bins']/0.8e-20
    fir_gd_smc = dict_fir_smc['gdr_ratio']
    fir_err_gd_smc = dict_fir_smc['percentile50_high'] - dict_fir_smc['percentile50_low']#dict_fir['err_gdr_ratio']

    ind_smc = np.argmin(np.abs(fir_gas_smc - 10.**(log_nh0_mw)))
    fir_gd_smc = fir_gd_smc[ind_smc]
    fir_gas_smc = fir_gas_smc[ind_smc]
    fir_err_gd_smc = fir_err_gd_smc[ind_smc]

    fir_gd_smc_int = dict_fir_smc['total_gas_mass']/dict_fir_smc['total_dust_mass']
    fir_err_gd_smc_int =  fir_gd_smc_int*np.sqrt((dict_fir_smc['total_dust_mass_error']/dict_fir_smc['total_dust_mass'])**2 + (dict_fir_smc['total_gas_mass_error']/dict_fir_smc['total_gas_mass'])**2)

    fir_gd_lmc_int = dict_fir_lmc['total_gas_mass']/dict_fir_lmc['total_dust_mass']
    fir_err_gd_lmc_int =  fir_gd_lmc_int*np.sqrt((dict_fir_lmc['total_dust_mass_error']/dict_fir_lmc['total_dust_mass'])**2 + (dict_fir_lmc['total_gas_mass_error']/dict_fir_lmc['total_gas_mass'])**2)

    print("DEPLETIONS DGR ")
    print("DGR INT WITH HE    ", dgrs_int)
    print("ERR DGR INT WITH HE", e_dgrs_int)

    print("DGR NH0 WITH  HE    ", dgrs)
    print("ERR DGR NH0 WITH HE", e_dgrs)

    print("FIR GDRS")
    print("FIR DGR INT WITH HE     ", 1./fir_gd_lmc_int, 1./fir_gd_smc_int)
    print("ERR FIR DGR INT WITH HE ", fir_err_gd_lmc_int/fir_gd_lmc_int**2,fir_err_gd_smc_int/fir_gd_smc_int**2)

    print("FIR DGR NH0 WITH HE     ", 1./fir_gd_lmc, 1./fir_gd_smc)
    print("ERR FIR DGR NH0 WITH HE ", fir_err_gd_lmc/fir_gd_lmc**2,fir_err_gd_smc/fir_gd_smc**2)

    print("DEPLETIONS DMR ")
    print("DMR INT WITH HE    ", dmrs_int)
    print("ERR DMR INT WITH HE", e_dmrs_int)

    print("DMR NH0 WITH HE    ", dmrs)
    print("ERR DMR NH0 WITH HE", e_dmrs)


    chris_dgrs = np.array([np.nan, 1./647., 1./3637.])



    #HERE total_dust_mass; total_dust_mass_error; total_gas_mass; total_gas_mass_error;

    zs = np.array([1., zgr_lmc/zgr_mw, zgr_smc/zgr_mw])
    gals = np.array(['MW', 'LMC', 'SMC'])
    lmc_color = 'cyan'#'dodgerblue'
    mw_color = 'gold' #'gold'#'green'
    smc_color = 'magenta'#'darkorange''
    gal_cols = [mw_color, lmc_color, smc_color]

    #fir_cols = ['gold', 'deepskyblue', 'limegreen']
    fir_cols=[mw_color, lmc_color, smc_color]
    fir_gd = np.array([np.nan, fir_gd_lmc, fir_gd_smc])
    fir_err_gd = np.array([0., fir_err_gd_lmc,fir_err_gd_smc])
    fir_gd_int = np.array([np.nan, fir_gd_lmc_int, fir_gd_smc_int])
    fir_err_gd_int = np.array([0., fir_err_gd_lmc_int,fir_err_gd_smc_int])


    z0min, gamma0min, dog0min, dm0min, dp0min, alpha0min = feldmann2015_model(gamma = [2e4] )
    z0min = z0min/0.014
    z0max, gamma0max, dog0max, dm0max, dp0max, alpha0max = feldmann2015_model(gamma = [4e4] )
    z0max = z0max/0.014

    zmin, gammamin, dogmin, dmmin, dpmin, alphamin = feldmann2015_model(gamma = [2e3] )
    zmin = zmin/0.014

    zmax, gammamax, dogmax, dmmax, dpmax, alphamax = feldmann2015_model(gamma = [1e6] )
    zmax = zmax/0.014

    z0, gamma0, dog0, dm0, dp0, alpha0 = feldmann2015_model(gamma = [3e4] )
    z0 = z0/0.014


    #dla = ascii.read('decia2016_table6.dat')
    devis =ascii.read('dustpedia_combined_sample.csv')
    remy_ruyer = ascii.read('remy-ruyer2014_dust_gas_masses.dat')


    plt.clf()
    plt.close()

    fig, ax = plt.subplots(nrows = 1, ncols = 1,figsize = (10, 8))


    ax.fill_between(zmin, dogmin[:,0],dog0min[:,0], color = 'gray', alpha =0.5)
    ax.fill_between(zmax, dog0max[:,0],dogmax[:,0], color = 'gray', alpha =0.5)
    ax.plot(z0, dog0, 'k--')

    dumz = np.arange(5e-3, 3, 0.1)
    ind = np.where(z0 == 1)
    ind = ind[0]
    dumdg = dog0.flatten()[ind]*dumz
    ax.plot(dumz, dumdg, 'k-', linewidth=3)
    ax.text(6e-3, 8.e-5, 'D/G '  +u"\u221D" + ' Z', color = 'black', fontsize = 15, rotation = 18)
    ax.text(0.2, 8e-6, 'Model Tracks', fontsize = 13, color = 'gray')
    ax.text(0.2, 4e-6, '(Feldmann+2015)', fontsize = 13, color = 'gray')

    #First panel has DLA and DEPLETIONS BASED D/G, AND FIR D/G SCALED TO NH0_MW
    #SECOND PANEL HAS INTEGRATED D/G FROM FIR, LMC/SMC DEPLETIONS, AND DE VIS

    #DE VIS
    ax.plot(devis['Z'].data, devis['Mdust'].data/devis['Mgas'].data, 'o',  color = 'dodgerblue', label = 'De Vis+2019 (FIR)', alpha = 0.5)
    ax.plot(10.**(remy_ruyer['12+log(O/H)'].data-8.76), remy_ruyer['Mdust'].data/remy_ruyer['Mgas'].data, 'o', color = 'mediumblue', alpha = 0.5, label = 'Remy-Ruyer+2014 (FIR)')


    #DLAs - need to scale to log_nh0_mw

    if plot_dla==True:
        if use_fe_dla==True:
            key='_Fe'
        else:
            key=''
        tfit = ascii.read("LMC_fit_DG_NH.dat")
        fslope = tfit['slope'].data[0]
        foffset= tfit['offset'].data[0]
        err_fslope = tfit['err_slope'].data[0]
        err_foffset= tfit['err_offset'].data[0]
        dla = fits.open("DeCia2016_all_data_"+use_gal_dla + key + ".fits")
        dla = dla[1].data

        scaled_dg  = dla['DTG'] + fslope*(log_nh0_mw-dla['LOG_NHI'])
        err_scaled_dg = np.sqrt(dla['err_DTG']**2 + (fslope*(log_nh0_mw-dla['LOG_NHI']))**2*((err_fslope/fslope)**2 + (dla['ERR_LOG_NHI']/dla['LOG_NHI'])**2))

        #print("DG ")
        #print(scaled_dg)
        #print("ERR ")
        #print(err_scaled_dg)

        good_dla = np.where((dla['LOG_NHI']>=19.5) & (dla['DTG'] >0))
        #test = Table()
        #test['tot_A_Fe'] = dla['tot_A_Fe'][good_dla]
        #test['DTG'] = dla['DTG'][good_dla]
        #test['LOG NHI'] = dla['LOG_NHI'][good_dla]
        #test['scaled DTG'] = scaled_dg[good_dla]

        #ascii.write(test, 'test_scaled_dtg_dla_decia.dat', overwrite=True)

        ax.errorbar(dla['tot_A_Fe'][good_dla]/10.**(7.54-12.), dla['DTG'][good_dla], fmt= 'o',  color ='red', label = 'DLAs (Quiret+2016, De Cia+2016)' , xerr = dla['err_tot_A_Fe'][good_dla]/10.**(8.76-12.), yerr = dla['ERR_DTG'][good_dla], alpha = 0.5)
        #x.errorbar(dla['tot_A_Fe'][good_dla]/10.**(7.54-12.), scaled_dg[good_dla], fmt= 'o',  markerfacecolor = 'none', color ='red', label = 'DLAs (Quiret+2016, De Cia+2016)' , xerr = dla['err_tot_A_Fe'][good_dla]/10.**(8.76-12.), yerr = err_scaled_dg[good_dla], alpha = 0.5)

        dla = fits.open("Quiret2016_DTG_table_"+use_gal_dla + key + ".fits")
        dla = dla[1].data
        scaled_dg  = dla['DTG'] + fslope*(log_nh0_mw-dla['LOG_NHI'])
        err_scaled_dg = np.sqrt(dla['err_DTG']**2 + (fslope*(log_nh0_mw-dla['LOG_NHI']))**2*((err_fslope/fslope)**2 + (dla['ERR_LOG_NHI']/dla['LOG_NHI'])**2))

        #print("DG ")
        #print(scaled_dg)
        #print("ERR ")
        #print(err_scaled_dg)

        good_dla = np.where((dla['LOG_NHI']>=19.5)& (dla['DTG'] >0))

        ax.errorbar(dla['tot_A_Fe'][good_dla]/10.**(7.54-12.), dla['DTG'][good_dla], fmt= 'o',  color ='red' , xerr = dla['err_tot_A_Fe'][good_dla]/10.**(8.76-12.), yerr = dla['ERR_DTG'][good_dla], alpha = 0.5)
        #ax.errorbar(dla['tot_A_Fe'][good_dla]/10.**(7.54-12.), scaled_dg[good_dla], fmt= 'o',  color ='red' , markerfacecolor = 'none' , xerr = dla['err_tot_A_Fe'][good_dla]/10.**(8.76-12.), yerr = err_scaled_dg[good_dla], alpha = 0.5)

    for i in range(len(gals)):

        linewidth = 4

        ##ax.errorbar(zs[i], dgrs[i], yerr = e_dgrs[i],fmt='o', color = gal_cols[i], label = gals[i] + ' (Depletions, log N(H)={:3.0f} cm'.format(log_nh0_mw)+r'$^{-2}$' + ')', markersize = 15, alpha = 0.7, markerfacecolor = 'none', linewidth = linewidth)
        ax.errorbar(zs[i], dgrs_int[i], yerr = e_dgrs_int[i],fmt='o', color = gal_cols[i], label = gals[i] + ' (Depletions, integrated)', markersize = 17, alpha= 0.7, markeredgecolor = 'black')
        ##ax.plot(zs[i], dgrs[i],'o', markersize = 15, markerfacecolor = 'none', linewidth = linewidth, color = gal_cols[i])
        #ax.plot(zs[i], dgrs_int[i], 'o', markersize = 17, color = 'black', markerfacecolor = 'none', linewidth = 12)



        if gals[i] != 'MW':
            #ax.errorbar(zs[i], [1./fir_gd[i]], yerr = fir_err_gd[i]/fir_gd[i]**2, fmt= '*', color = fir_cols[i], label = gals[i] + ' (FIR, log N(H)={:3.0f} cm'.format(log_nh0_mw)+r'$^{-2}$' + ')', markersize = 23, alpha = 0.7, markerfacecolor = 'none', linewidth = linewidth)

            ax.errorbar(zs[i], [1./fir_gd_int[i]], yerr = fir_err_gd_int[i]/fir_gd_int[i]**2, fmt= '*', color = fir_cols[i], label = gals[i] + ' (FIR, integrated, RD2017)', markersize = 23, alpha  = 0.7, markeredgecolor = 'black')
            #ax.plot(zs[i], [1./fir_gd_int[i]], markersize = 17, color = 'black', markerfacecolor = 'none', linewidth = 12)


    ax.set_xlim(left = 5.e-3,right=3.)
    ax.set_ylim(bottom = 4.e-7, top = 0.1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Z/" + r'$Z_o$' , fontsize = 22)
    ax.minorticks_on()
    y_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 10)
    ax.yaxis.set_major_locator(y_major)
    y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.tick_params(which='major', width=2, length = 6)
    ax.tick_params(which='minor', width=1, length = 4)
    ax.set_ylabel('D/G', fontsize = 22)
    #ax[1].set_ylabel('D/G', fontsize = 16)
    ax.legend(fontsize = 12, loc = 'upper left')
    fig.subplots_adjust(top = 0.95, bottom = 0.12, left = 0.12, right = 0.95)


    c_key = ''
    if newc ==True:
        c_key = '_newc'
    if plot_dla==True:
        plt.savefig("FIGURES_ALL/plot_feldmann2015_mw_lmc_smc_"+ c_key + use_gal_dla + key+".pdf", format= 'pdf', dpi = 1000)

    else:
        plt.savefig("FIGURES_ALL/plot_feldmann2015_mw_lmc_smc_nodla" + c_key + ".pdf", format= 'pdf', dpi = 1000)

    plt.clf()
    plt.close()

def plot_feldmann_two_panels(log_nh0_mw = 21., use_fe_dla=False, use_gal_dla = 'dla'):

    print("COMPUTING MW")
    dgr_mw, e_dgr_mw, dmr_mw,e_dmr_mw,zgr_mw, deps, err_deps, w, a = compute_gdr_lv(galaxy = 'MW',elt = ['C', 'O', 'Mg', 'Si', 'Fe',  'Ni', 'Zn', 'Ti'],log_nh0 = np.array([log_nh0_mw]))
    #print("COMPUTING LMC")
    dgr_lmc,e_dgr_lmc,dmr_lmc,e_dmr_lmc, zgr_lmc ,deps_lmc, err_deps_lmc, wlmc, almc = compute_gdr_lv(galaxy = 'LMC',elt = ['C', 'O', 'Mg', 'Si', 'Fe',  'Ni', 'Zn', 'Ti'],log_nh0 = np.array([log_nh0_mw]))
    #print("COMPUTING SMC")

    print("SMC SMC SMC ")
    dgr_smc,e_dgr_smc, dmr_smc, e_dmr_smc, zgr_smc, deps_smc, err_deps_smc, wsmc, asmc = compute_gdr_lv(galaxy = 'SMC',elt = ['C', 'O', 'Mg', 'Si',  'Fe',  'Ni', 'Zn', 'Ti'],log_nh0 = np.array([log_nh0_mw]))
    print("SMC ERR DEP ", err_deps_smc)
    #if e_dgr_smc > dgr_smc:
    #    e_dgr_smc = 0.9*dgr_smc

    dgr_smc_int, e_dgr_smc_int = compute_gdr_integrated("SMC")
    dgr_lmc_int, e_dgr_lmc_int = compute_gdr_integrated("LMC")
    #Kalberla+2009 show surface denisty of HI is about constant in inner galaxy at 1.5e21 cm-2. To that, add surface density of molecular gas from RD2016, or 5e20cm-2 in the inner galaxy. That leads to about 2e21 cm-2 through teh disk

    dgr_mw_int, e_dgr_mw_int, dmr_mw_int,e_dmr_mw_int,zgr_mw_int, deps_int, err_deps_int, w, a = compute_gdr_lv(galaxy = 'MW',elt = ['C', 'O', 'Mg', 'Si', 'Fe',  'Ni', 'Zn', 'Ti'],log_nh0 = np.array([np.log10(2e21)]))


    dgrs= np.array([dgr_mw, dgr_lmc, dgr_smc]).flatten()
    e_dgrs = np.array([e_dgr_mw, e_dgr_lmc, e_dgr_smc]).flatten()

    dgrs_int= np.array([dgr_mw_int, dgr_lmc_int, dgr_smc_int])
    e_dgrs_int = np.array([e_dgr_mw_int, e_dgr_lmc_int, e_dgr_smc_int])

    #ind = np.where(e_dgrs > dgrs)
    #e_dgrs[ind] = 0.9*dgrs[ind]

    dict_fir_lmc = scipy.io.readsav("/astro/dust_kg/jduval/DUST_POOR_COMPONENT/MC_SAVE_FILES/binned_IRAS_PLANCK_surface_brightness_lmc_quad_cirrus_thresh_stray_corr_mbb_a6.400_FITS_0.05_FINAL.save")
    dict_fir_smc = scipy.io.readsav("/astro/dust_kg/jduval/DUST_POOR_COMPONENT/MC_SAVE_FILES/binned_IRAS_PLANCK_surface_brightness_smc_quad_cirrus_thresh_stray_corr_mbb_a21.00_FITS_0.05_FINAL.save")


    fir_gas_lmc = dict_fir_lmc['gas_bins']/0.8e-20
    fir_gd_lmc = dict_fir_lmc['gdr_ratio']
    fir_err_gd_lmc = dict_fir_lmc['percentile50_high'] - dict_fir_lmc['percentile50_low']#dict_fir['err_gdr_ratio']


    ind_lmc = np.argmin(np.abs(fir_gas_lmc - 10.**(log_nh0_mw)))
    fir_gd_lmc = fir_gd_lmc[ind_lmc]
    fir_gas_lmc = fir_gas_lmc[ind_lmc]
    fir_err_gd_lmc = fir_err_gd_lmc[ind_lmc]

    fir_gas_smc = dict_fir_smc['gas_bins']/0.8e-20
    fir_gd_smc = dict_fir_smc['gdr_ratio']
    fir_err_gd_smc = dict_fir_smc['percentile50_high'] - dict_fir_smc['percentile50_low']#dict_fir['err_gdr_ratio']

    ind_smc = np.argmin(np.abs(fir_gas_smc - 10.**(log_nh0_mw)))
    fir_gd_smc = fir_gd_smc[ind_smc]
    fir_gas_smc = fir_gas_smc[ind_smc]
    fir_err_gd_smc = fir_err_gd_smc[ind_smc]

    fir_gd_smc_int = dict_fir_smc['total_gas_mass']/dict_fir_smc['total_dust_mass']
    fir_err_gd_smc_int =  fir_gd_smc_int*np.sqrt((dict_fir_smc['total_dust_mass_error']/dict_fir_smc['total_dust_mass'])**2 + (dict_fir_smc['total_gas_mass_error']/dict_fir_smc['total_gas_mass'])**2)

    fir_gd_lmc_int = dict_fir_lmc['total_gas_mass']/dict_fir_lmc['total_dust_mass']
    fir_err_gd_lmc_int =  fir_gd_lmc_int*np.sqrt((dict_fir_lmc['total_dust_mass_error']/dict_fir_lmc['total_dust_mass'])**2 + (dict_fir_lmc['total_gas_mass_error']/dict_fir_lmc['total_gas_mass'])**2)

    print("DEPLETIONS DGR ")
    print("DGR INT WITH HE    ", dgrs_int)
    print("ERR DGR INT WITH HE", e_dgrs_int)

    print("DGR NH0 WITH HE    ", dgrs)
    print("ERR DGR NH0 WITH HE", e_dgrs)

    print("FIR GDRS")
    print("FIR DGR INT WITH HE     ", 1./fir_gd_lmc_int, 1./fir_gd_smc_int)
    print("ERR FIR DGR INT WITH HE ", fir_err_gd_lmc_int/fir_gd_lmc_int**2,fir_err_gd_smc_int/fir_gd_smc_int**2)

    print("FIR DGR NH0 WITH HE     ", 1./fir_gd_lmc, 1./fir_gd_smc)
    print("ERR FIR DGR NH0 WITH HE ", fir_err_gd_lmc/fir_gd_lmc**2,fir_err_gd_smc/fir_gd_smc**2)

    chris_dgrs = np.array([np.nan, 1./647., 1./3637.])



    #HERE total_dust_mass; total_dust_mass_error; total_gas_mass; total_gas_mass_error;

    zs = np.array([1., zgr_lmc/zgr_mw, zgr_smc/zgr_mw])
    gals = np.array(['MW', 'LMC', 'SMC'])
    lmc_color = 'cyan'#'dodgerblue'
    mw_color = 'gold'#'green'
    smc_color = 'magenta'#'darkorange''
    gal_cols = [mw_color, lmc_color, smc_color]

    #fir_cols = ['gold', 'deepskyblue', 'limegreen']
    fir_cols=[mw_color, lmc_color, smc_color]
    fir_gd = np.array([np.nan, fir_gd_lmc, fir_gd_smc])
    fir_err_gd = np.array([0., fir_err_gd_lmc,fir_err_gd_smc])
    fir_gd_int = np.array([np.nan, fir_gd_lmc_int, fir_gd_smc_int])
    fir_err_gd_int = np.array([0., fir_err_gd_lmc_int,fir_err_gd_smc_int])


    z0min, gamma0min, dog0min, dm0min, dp0min, alpha0min = feldmann2015_model(gamma = [2e4] )
    z0min = z0min/0.014
    z0max, gamma0max, dog0max, dm0max, dp0max, alpha0max = feldmann2015_model(gamma = [4e4] )
    z0max = z0max/0.014

    zmin, gammamin, dogmin, dmmin, dpmin, alphamin = feldmann2015_model(gamma = [2e3] )
    zmin = zmin/0.014

    zmax, gammamax, dogmax, dmmax, dpmax, alphamax = feldmann2015_model(gamma = [1e6] )
    zmax = zmax/0.014

    z0, gamma0, dog0, dm0, dp0, alpha0 = feldmann2015_model(gamma = [3e4] )
    z0 = z0/0.014


    #dla = ascii.read('decia2016_table6.dat')
    devis =ascii.read('dustpedia_combined_sample.csv')


    plt.clf()
    plt.close()

    fig, ax = plt.subplots(nrows = 1, ncols = 2,figsize = (12, 7), sharey= True)


    ax[0].fill_between(zmin, dogmin[:,0],dog0min[:,0], color = 'gray', alpha =0.5)
    ax[1].fill_between(zmin, dogmin[:,0],dog0min[:,0], color = 'gray', alpha =0.5)
    ax[0].fill_between(zmax, dog0max[:,0],dogmax[:,0], color = 'gray', alpha =0.5)
    ax[1].fill_between(zmin, dog0max[:,0],dogmax[:,0], color = 'gray', alpha =0.5)
    ax[0].plot(z0, dog0, 'k--')
    ax[1].plot(z0, dog0, 'k--')

    dumz = np.arange(5e-3, 2, 0.1)
    dumdg = dgrs[0]*dumz
    ax[0].plot(dumz, dumdg, 'g-', linewidth=3)
    ax[1].plot(dumz, dumdg, 'g-', linewidth=3)
    ax[0].text(6e-3, 6.e-5, 'D/G '  +u"\u221D" + ' Z', color = 'green', fontsize = 15, rotation = 28)
    ax[1].text(6e-3, 6.e-5, 'D/G '  +u"\u221D" + ' Z', color = 'green', fontsize = 15, rotation = 28)
    ax[0].text(0.2, 8e-6, 'Model Tracks', fontsize = 13, color = 'gray')
    ax[0].text(0.2, 5e-6, '(Feldmann+2015)', fontsize = 13, color = 'gray')
    ax[1].text(0.2, 8e-6, 'Model Tracks', fontsize = 13, color = 'gray')
    ax[1].text(0.2, 5e-6, '(Feldmann+2015)', fontsize = 13, color = 'gray')

    #First panel has DLA and DEPLETIONS BASED D/G, AND FIR D/G SCALED TO NH0_MW
    #SECOND PANEL HAS INTEGRATED D/G FROM FIR, LMC/SMC DEPLETIONS, AND DE VIS

    #DE VIS
    ax[1].plot(devis['Z'].data, devis['Mdust'].data/devis['Mgas'].data, 'o',  color = 'dodgerblue', label = 'Nearby galaxies (FIR)', alpha = 0.5)
    ax[0].plot(devis['Z'].data, devis['Mdust'].data/devis['Mgas'].data, 'o',  color = 'gray', label = 'Nearby galaxies (FIR)', alpha = 0.5)
    #plt.plot(10**(dla['[M/H]tot'].data), dla['DTM'].data*10.**(dla['[M/H]tot'].data)/150., 'o', markerfacecolor = 'none', color = 'gray', label = 'De Cia et al. (2016)')


    #DLAs - need to scale to log_nh0_mw

    if use_fe_dla==True:
        key='_Fe'
    else:
        key=''
    tfit = ascii.read("LMC_fit_DG_NH.dat")
    fslope = tfit['slope'].data[0]
    foffset= tfit['offset'].data[0]
    err_fslope = tfit['err_slope'].data[0]
    err_foffset= tfit['err_offset'].data[0]
    dla = fits.open("DeCia2016_all_data_"+use_gal_dla + key + ".fits")
    dla = dla[1].data

    scaled_dg  = dla['DTG'] + fslope*(log_nh0_mw-dla['LOG_NHI'])
    err_scaled_dg = np.sqrt(dla['err_DTG']**2 + (fslope*(log_nh0_mw-dla['LOG_NHI']))**2*((err_fslope/fslope)**2 + (dla['ERR_LOG_NHI']/dla['LOG_NHI'])**2))

    #print("DG ")
    #print(scaled_dg)
    #print("ERR ")
    #print(err_scaled_dg)

    good_dla = np.where((dla['LOG_NHI']>=19.5) & (dla['DTG'] >0))
    test = Table()
    test['tot_A_Fe'] = dla['tot_A_Fe'][good_dla]
    test['DTG'] = dla['DTG'][good_dla]
    test['LOG NHI'] = dla['LOG_NHI'][good_dla]
    test['scaled DTG'] = scaled_dg[good_dla]

    ascii.write(test, 'test_scaled_dtg_dla_decia.dat', overwrite=True)



    ax[0].errorbar(dla['tot_A_Fe'][good_dla]/10.**(7.54-12.), dla['DTG'][good_dla], fmt= 'o',  color ='darkorange', label = 'DLAs' , xerr = dla['err_tot_A_Fe'][good_dla]/10.**(8.76-12.), yerr = dla['ERR_DTG'][good_dla], alpha = 0.5)
    #ax[0].errorbar(dla['tot_A_Fe'][good_dla]/10.**(7.54-12.), scaled_dg[good_dla], fmt= 'o',  color ='red', label = 'DLAs scaled to log N(H)={:3.0f}'.format(log_nh0_mw) + ' cm' + r'$^{-2}$' , xerr = dla['err_tot_A_Fe'][good_dla]/10.**(8.76-12.), yerr = err_scaled_dg[good_dla], alpha = 0.5)
    #ax[0].plot(dla['tot_A_Fe'][good_dla]/10.**(7.54-12), dla['DTG'][good_dla], 'o', color= 'gray', alpha = 0.5)
    ax[1].errorbar(dla['tot_A_Fe'][good_dla]/10.**(7.54-12.), dla['DTG'][good_dla], fmt= 'o',  color ='gray', label = 'DLAs' , xerr = dla['err_tot_A_Fe'][good_dla]/10.**(8.76-12.), yerr = dla['ERR_DTG'][good_dla], alpha = 0.5)

    dla = fits.open("Quiret2016_DTG_table_"+use_gal_dla + key + ".fits")
    dla = dla[1].data
    scaled_dg  = dla['DTG'] + fslope*(log_nh0_mw-dla['LOG_NHI'])
    err_scaled_dg = np.sqrt(dla['err_DTG']**2 + (fslope*(log_nh0_mw-dla['LOG_NHI']))**2*((err_fslope/fslope)**2 + (dla['ERR_LOG_NHI']/dla['LOG_NHI'])**2))

    #print("DG ")
    #print(scaled_dg)
    #print("ERR ")
    #print(err_scaled_dg)

    good_dla = np.where((dla['LOG_NHI']>=19.5)& (dla['DTG'] >0))

    ax[0].errorbar(dla['tot_A_Fe'][good_dla]/10.**(7.54-12.), dla['DTG'][good_dla], fmt= 'o',  color ='darkorange' , xerr = dla['err_tot_A_Fe'][good_dla]/10.**(8.76-12.), yerr = dla['ERR_DTG'][good_dla], alpha = 0.5)
    #ax[0].errorbar(dla['tot_A_Fe'][good_dla]/10.**(7.54-12.), scaled_dg[good_dla], fmt= 'o',  color ='red', xerr = dla['err_tot_A_Fe'][good_dla]/10.**(8.76-12.), yerr = err_scaled_dg[good_dla], alpha = 0.5)
    #ax[0].plot(dla['tot_A_Fe'][good_dla]/10.**(7.54-12), dla['DTG'][good_dla], 'o', color= 'cyan', alpha = 0.5)
    ax[1].errorbar(dla['tot_A_Fe'][good_dla]/10.**(7.54-12.), dla['DTG'][good_dla], fmt= 'o',  color ='gray' , xerr = dla['err_tot_A_Fe'][good_dla]/10.**(8.76-12.), yerr = dla['ERR_DTG'][good_dla], alpha = 0.5)

    test = Table()
    test['tot_A_Fe'] = dla['tot_A_Fe'][good_dla]
    test['DTG'] = dla['DTG'][good_dla]
    test['LOG NHI'] = dla['LOG_NHI'][good_dla]
    test['scaled DTG'] = scaled_dg[good_dla]

    ascii.write(test, 'test_scaled_dtg_dla_quiret.dat', overwrite=True)


    for i in range(len(gals)):

        ax[0].errorbar(zs[i], dgrs[i], yerr = e_dgrs[i],fmt='o', color = gal_cols[i], label = gals[i] + ' (UV, log N(H)={:3.0f} cm'.format(log_nh0_mw)+r'$^{-2}$' + ')', markersize = 15, alpha = 0.7)
        ax[1].errorbar(zs[i], dgrs_int[i], yerr = e_dgrs_int[i],fmt='o', color = gal_cols[i], label = gals[i] + ' (UV, integrated)', markersize = 17, alpha= 0.7)
        ax[0].plot(zs[i], dgrs[i],'ko', markersize = 15, markerfacecolor = 'none')
        ax[1].plot(zs[i], dgrs_int[i], 'ko', markersize = 17, markerfacecolor = 'none')



        if gals[i] != 'MW':
            ax[0].errorbar(zs[i], [1./fir_gd[i]], yerr = fir_err_gd[i]/fir_gd[i]**2, fmt= '*', color = fir_cols[i], label = gals[i] + ' (FIR, log N(H)={:3.0f} cm'.format(log_nh0_mw)+r'$^{-2}$' + ')', markersize = 23, alpha = 0.5, markerfacecolor = 'none')

            ax[1].errorbar(zs[i], [1./fir_gd_int[i]], yerr = fir_err_gd_int[i]/fir_gd_int[i]**2, fmt= '*', color = fir_cols[i], label = gals[i] + ' (FIR, integrated, RD2017)', markersize = 23, alpha  = 0.5, markerfacecolor = 'none')

            ax[1].plot(zs[i], [chris_dgrs[i]], '*',  color = fir_cols[i], label = gals[i] + ' (FIR, integrated, Clark+2021)', markersize = 23, alpha  = 0.5)
            ax[1].plot(zs[i], [chris_dgrs[i]], 'k*',  markersize = 23, markerfacecolor = 'none')






    ax[0].set_xlim(left = 5.e-3,right=3.)
    ax[1].set_xlim(left = 5.e-3,right=3.)

    ax[0].set_ylim(bottom = 1.e-7, top = 5)
    ax[1].set_ylim(bottom = 1.e-7, top = 5)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel("Z [" + r'$Z_o$' + "]", fontsize = 16)
    ax[0].set_xlabel("Z [" + r'$Z_o$' + "]", fontsize = 16)


    ax[0].set_ylabel('D/G', fontsize = 16)
    #ax[1].set_ylabel('D/G', fontsize = 16)
    ax[0].legend(fontsize = 10, loc = 'upper left')
    ax[1].legend(fontsize = 10, loc = 'upper left')
    fig.subplots_adjust(top = 0.95, bottom = 0.12, left = 0.12, right = 0.95, wspace = 0)

    plt.savefig("FIGURES_ALL/plot_feldmann2015_mw_lmc_smc_"+ use_gal_dla + key+".pdf", format= 'pdf', dpi = 1000)

    plt.clf()
    plt.close()




def feldmann2015_model(gamma = 10.**(np.arange(3, 6.1, 0.5)) ):

    ng = len(gamma) #7
    nz = 50

    z = 10.**(np.linspace(-3, 0.5, nz))*0.014

    R = 0.46
    epsilon_out = 2.
    y = 6.9e-2
    y_D= 6.9e-4
    r_Z = 0.25
    r_D = 0.
    f_dep = 0.7
    epsilon_SN  = 10.
    #r = 1./(1.-R + epsilon_out)
    #print("CHECK ", r, R)
    r = z*(1.-r_Z)/y/(1.-R)


    alpha = y_D*(1.-R)

    dog = np.zeros([nz, ng])

    dm = y_D/y*(1.-r_Z)/(1.-r_D)*z
    dp = f_dep*z

    for ig in range(len(gamma)):

        beta = gamma[ig]*f_dep*z[:]-(epsilon_SN + R + (1.-r_D)/r)
        dog[:, ig] = beta[:]/(2.*gamma[ig]) + ( (beta/(2.*gamma[ig]))**2 + alpha/gamma[ig])**(0.5)


    return(z, gamma, dog, dm , dp, alpha)

def plot_asano(dm = False):

    #if dm = True, plot the DMR instead of DGR. However, this should not be done since DMR will be driven by C and O, which for now uses MW depletion patterns since we don't have them for LMC and SMC. As a result, DMR does not change with Z.

    dgr_mw, e_dgr_mw,dmr_mw, e_dmr_mw,zgr_mw = compute_gdr_lv(galaxy = 'MW')
    dgr_lmc, e_dgr_lmc, dmr_lmc,e_dmr_lmc,  zgr_lmc = compute_gdr_lv(galaxy = 'LMC')
    dgr_smc, e_dgr_smc, dmr_smc,e_dmr_smc , zgr_smc = compute_gdr_lv(galaxy = 'SMC')

    dgrs= np.array([dgr_mw, dgr_lmc, dgr_smc])
    dmrs = np.array([dmr_mw, dmr_lmc, dmr_smc])
    zs = np.array([1., zgr_lmc/zgr_mw, zgr_smc/zgr_mw])
    gals = np.array(['MW', 'LMC', 'SMC'])
    gal_cols = ['k', 'b', 'r']

    tau_sf = ['0.5', '5', '50']
    sf_colors = ['crimson', 'deeppink', 'fuchsia']


    plt.clf()
    plt.close()

    fig = plt.figure(figsize = (10, 8))

    for i in range(len(tau_sf)):
        t = ascii.read("Result_of_Asanoetal2013/result_sf" + tau_sf[i] + ".dat")

        #plt.plot(t['M_Z']/t['M_ISM']/(zgr_mw), t['M_DUST']/t['M_ISM'], '--', label = r'$\tau_{SF}$' + ' = ' + tau_sf[i] + ' Gyr', color = sf_colors[i])

        if dm ==True:
            plt.plot(t['M_Z']/t['M_ISM']/(zgr_mw), t['M_DUST']/t['M_Z'],  label = r'$\tau_{SF}$' + ' = ' + tau_sf[i] + ' Gyr', color = sf_colors[i])

        else:
            plt.plot(t['M_Z']/t['M_ISM']/(zgr_mw), t['M_DUST']/t['M_ISM']/3.,  label = r'$\tau_{SF}$' + ' = ' + tau_sf[i] + ' Gyr', color = sf_colors[i])


    for i in range(len(gals)):

        if dm ==True:
            plt.plot(zs[i], dmrs[i],'o', color = gal_cols[i], label = gals[i], markersize = 10)
        else:
            plt.plot(zs[i], dgrs[i],'o', color = gal_cols[i], label = gals[i], markersize = 10)

    plt.xlim([0.001, 2])
    if dm==True:
        plt.ylim([1.e-4, 2])
    else:
        plt.ylim([1.e-8, 0.1])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Z [" + r'$Z_o$' + "]", fontsize = 16)

    if dm==True:
        plt.ylabel(r'$M_d/M_Z$', fontsize = 16)
    else:
        plt.ylabel(r'$M_d/M_{ISM}$', fontsize = 16)
    plt.legend(fontsize = 15)

    dmk = ''
    if dm ==True:
        dmk = '_dmr'

    plt.savefig("plot_asano2013_mw_lmc_smc" + dmk + ".eps", format= 'eps', dpi = 1000)

    plt.clf()
    plt.close()


def plot_feldmann_old(log_nh0_mw = 21.):

    print("COMPUTING MW")
    dgr_mw, e_dgr_mw, dmr_mw,e_dmr_mw,zgr_mw, deps, err_deps, w, a = compute_gdr_lv(galaxy = 'MW',elt = ['C', 'O', 'Mg', 'Si', 'Fe',  'Ni', 'Zn', 'Ti'],log_nh0 = np.array([log_nh0_mw]))
    #print("COMPUTING LMC")
    dgr_lmc,e_dgr_lmc,dmr_lmc,e_dmr_lmc, zgr_lmc ,deps_lmc, err_deps_lmc, wlmc, almc = compute_gdr_lv(galaxy = 'LMC',elt = ['C', 'O', 'Mg', 'Si', 'Fe',  'Ni', 'Zn', 'Ti'],log_nh0 = np.array([log_nh0_mw]))
    #print("COMPUTING SMC")
    dgr_smc,e_dgr_smc, dmr_smc, e_dmr_smc, zgr_smc, deps_smc, err_deps_smc, wsmc, asmc = compute_gdr_lv(galaxy = 'SMC',elt = ['C', 'O', 'Mg', 'Si',  'Fe',  'Ni', 'Zn', 'Ti'],log_nh0 = np.array([log_nh0_mw]))
    #if e_dgr_smc > dgr_smc:
    #    e_dgr_smc = 0.9*dgr_smc

    dgr_smc_int, e_dgr_smc_int = compute_gdr_integrated("SMC")
    dgr_lmc_int, e_dgr_lmc_int = compute_gdr_integrated("LMC")
    #Kalberla+2009 show surface denisty of HI is about constant in inner galaxy at 1.5e21 cm-2. To that, add surface density of molecular gas from RD2016, or 5e20cm-2 in the inner galaxy. That leads to about 2e21 cm-2 through teh disk

    dgr_mw_int, e_dgr_mw_int, dmr_mw_int,e_dmr_mw_int,zgr_mw_int, deps_int, err_deps_int, w, a = compute_gdr_lv(galaxy = 'MW',elt = ['C', 'O', 'Mg', 'Si', 'Fe',  'Ni', 'Zn', 'Ti'],log_nh0 = np.array([np.log10(2e21)]))


    dgrs= np.array([dgr_mw, dgr_lmc, dgr_smc])
    e_dgrs = np.array([e_dgr_mw, e_dgr_lmc, e_dgr_smc])

    dgrs_int= np.array([dgr_mw_int, dgr_lmc_int, dgr_smc_int])
    e_dgrs_int = np.array([e_dgr_mw_int, e_dgr_lmc_int, e_dgr_smc_int])

    ind = np.where(e_dgrs > dgrs)
    e_dgrs[ind] = 0.9*dgrs[ind]

    dict_fir_lmc = scipy.io.readsav("/astro/dust_kg/jduval/DUST_POOR_COMPONENT/MC_SAVE_FILES/binned_IRAS_PLANCK_surface_brightness_lmc_quad_cirrus_thresh_stray_corr_mbb_a6.400_FITS_0.05_FINAL.save")
    dict_fir_smc = scipy.io.readsav("/astro/dust_kg/jduval/DUST_POOR_COMPONENT/MC_SAVE_FILES/binned_IRAS_PLANCK_surface_brightness_smc_quad_cirrus_thresh_stray_corr_mbb_a21.00_FITS_0.05_FINAL.save")


    fir_gas_lmc = dict_fir_lmc['gas_bins']/0.8e-20
    fir_gd_lmc = dict_fir_lmc['gdr_ratio']
    fir_err_gd_lmc = dict_fir_lmc['percentile50_high'] - dict_fir_lmc['percentile50_low']#dict_fir['err_gdr_ratio']


    ind_lmc = np.argmin(np.abs(fir_gas_lmc - 10.**(log_nh0_mw)))
    fir_gd_lmc = fir_gd_lmc[ind_lmc]
    fir_gas_lmc = fir_gas_lmc[ind_lmc]
    fir_err_gd_lmc = fir_err_gd_lmc[ind_lmc]

    fir_gas_smc = dict_fir_smc['gas_bins']/0.8e-20
    fir_gd_smc = dict_fir_smc['gdr_ratio']
    fir_err_gd_smc = dict_fir_smc['percentile50_high']- dict_fir_smc['percentile50_low']#dict_fir['err_gdr_ratio']

    ind_smc = np.argmin(np.abs(fir_gas_smc - 10.**(log_nh0_mw)))
    fir_gd_smc = fir_gd_smc[ind_smc]
    fir_gas_smc = fir_gas_smc[ind_smc]
    fir_err_gd_smc = fir_err_gd_smc[ind_smc]

    fir_gd_smc_int = dict_fir_smc['total_gas_mass']/dict_fir_smc['total_dust_mass']
    fir_err_gd_smc_int =  fir_gd_smc_int*np.sqrt((dict_fir_smc['total_dust_mass_error']/dict_fir_smc['total_dust_mass'])**2 + (dict_fir_smc['total_gas_mass_error']/dict_fir_smc['total_gas_mass'])**2)

    fir_gd_lmc_int = dict_fir_lmc['total_gas_mass']/dict_fir_lmc['total_dust_mass']
    fir_err_gd_lmc_int =  fir_gd_lmc_int*np.sqrt((dict_fir_lmc['total_dust_mass_error']/dict_fir_lmc['total_dust_mass'])**2 + (dict_fir_lmc['total_gas_mass_error']/dict_fir_lmc['total_gas_mass'])**2)

    print("DEPLETIONS DGR ")
    print("DGR INT WITH HE", dgrs_int)
    print("DGR NH0 WITH HE", dgrs)
    print(" ERR DGR INT WITH HE", e_dgrs_int)

    print("FIR GDRS")
    print("FIR DGR INT WITH HE ", 1./fir_gd_lmc_int, 1./fir_gd_smc_int)
    print("ERR FIR DGR INT WITH HE ", fir_err_gd_lmc_int/fir_gd_lmc_int**2,fir_err_gd_smc_int/fir_gd_smc_int)



    #HERE total_dust_mass; total_dust_mass_error; total_gas_mass; total_gas_mass_error;

    zs = np.array([1., zgr_lmc/zgr_mw, zgr_smc/zgr_mw])
    gals = np.array(['MW', 'LMC', 'SMC'])
    lmc_color = 'dodgerblue'
    smc_color = 'green'
    mw_color = 'darkorange'
    gal_cols = [mw_color, lmc_color, smc_color]

    fir_cols = ['gold', 'deepskyblue', 'limegreen']
    fir_gd = np.array([np.nan, fir_gd_lmc, fir_gd_smc])
    fir_err_gd = np.array([0., fir_err_gd_lmc,fir_err_gd_smc])
    fir_gd_int = np.array([np.nan, fir_gd_lmc_int, fir_gd_smc_int])
    fir_err_gd_int = np.array([0., fir_err_gd_lmc_int,fir_err_gd_smc_int])


    z, gamma, dog, dm, dp, alpha = feldmann2015_model()
    z = z/0.014

    #dla = ascii.read('decia2016_table6.dat')
    devis =ascii.read('dustpedia_combined_sample.csv')




    plt.clf()
    plt.close()

    fig, ax = plt.subplots(nrows = 2, ncols = 1,figsize = (10, 12), sharex= True)

    #First panel has DLA and DEPLETIONS BASED D/G, AND FIR D/G SCALED TO NH0_MW
    #SECOND PANEL HAS INTEGRATED D/G FROM FIR, LMC/SMC DEPLETIONS, AND DE VIS

    #DE VIS
    ax[1].plot(devis['Z'].data, devis['Mdust'].data/devis['Mgas'].data , 'o', markerfacecolor = 'none', color = 'darkviolet', label = 'De Vis et al. (2019)', alpha = 0.3)
    #plt.plot(10**(dla['[M/H]tot'].data), dla['DTM'].data*10.**(dla['[M/H]tot'].data)/150., 'o', markerfacecolor = 'none', color = 'gray', label = 'De Cia et al. (2016)')


    #DLAs - need to scale to log_nh0_mw
    tfit = ascii.read("LMC_fit_DG_NH.dat")
    fslope = tfit['slope'].data[0]
    foffset= tfit['offset'].data[0]
    err_fslope = tfit['err_slope'].data[0]
    err_foffset= tfit['err_offset'].data[0]
    dla = fits.open("DeCia2016_all_data.fits")
    dla = dla[1].data

    scaled_dg  = dla['DTG'] + fslope*(log_nh0_mw-dla['LOG_NHI'])
    err_scaled_dg = np.sqrt(dla['err_DTG']**2 + (fslope*(log_nh0_mw-dla['LOG_NHI']))**2*((err_fslope/fslope)**2 + (dla['ERR_LOG_NHI']/dla['LOG_NHI'])**2))

    #print("DG ")
    #print(scaled_dg)
    #print("ERR ")
    #print(err_scaled_dg)

    good_dla = np.where((dla['LOG_NHI']>=19.5) & (scaled_dg >0))

    ax[0].errorbar(dla['tot_A_Fe'][good_dla]/10.**(7.54-12.), scaled_dg[good_dla], fmt= 'o',  color ='gray', label = 'De Cia et al. (2016)' , xerr = dla['err_tot_A_Fe'][good_dla]/10.**(8.76-12.), yerr = err_scaled_dg[good_dla])
    #ax[0].plot(dla['tot_A_Fe'][good_dla]/10.**(7.54-12), dla['DTG'][good_dla], 'o', color= 'gray', alpha = 0.5)


    dla = fits.open("Quiret2016_DTG_table.fits")
    dla = dla[1].data
    scaled_dg  = dla['DTG'] + fslope*(log_nh0_mw-dla['LOG_NHI'])
    err_scaled_dg = np.sqrt(dla['err_DTG']**2 + (fslope*(log_nh0_mw-dla['LOG_NHI']))**2*((err_fslope/fslope)**2 + (dla['ERR_LOG_NHI']/dla['LOG_NHI'])**2))

    #print("DG ")
    #print(scaled_dg)
    #print("ERR ")
    #print(err_scaled_dg)

    good_dla = np.where((dla['LOG_NHI']>=19.5)& (scaled_dg >0))

    ax[0].errorbar(dla['tot_A_Fe'][good_dla]/10.**(7.54-12.), scaled_dg[good_dla], fmt= 'o',  color ='cyan', label = 'Quiret et al. (2016)' , xerr = dla['err_tot_A_Fe'][good_dla]/10.**(8.76-12.), yerr = err_scaled_dg[good_dla])
    #ax[0].plot(dla['tot_A_Fe'][good_dla]/10.**(7.54-12), dla['DTG'][good_dla], 'o', color= 'cyan', alpha = 0.5)

    for i in range(len(gals)):

        ax[0].errorbar(zs[i], dgrs[i], yerr = e_dgrs[i],fmt='o', color = gal_cols[i], label = gals[i] + ' (UV, log N(H) = {:5.2f} cm'.format(log_nh0_mw)+r'$^{-2}$' + ')', markersize = 15, alpha = 0.5)
        ax[1].errorbar(zs[i], dgrs_int[i], yerr = e_dgrs_int[i],fmt='o', color = gal_cols[i], label = gals[i] + ' (UV, integrated)', markersize = 17, alpha= 0.5)



        if gals[i] != 'MW':
            ax[0].errorbar(zs[i], [1./fir_gd[i]], yerr = fir_err_gd[i]/fir_gd[i]**2, fmt= '*', color = fir_cols[i], label = gals[i] + '(FIR, log N(H) = {:5.2f} cm'.format(log_nh0_mw)+r'$^{-2}$' + ')', markersize = 15, alpha = 0.5)
            ax[1].errorbar(zs[i], [1./fir_gd_int[i]], yerr = fir_err_gd_int[i]/fir_gd_int[i]**2, fmt= '*', color = fir_cols[i], label = gals[i] + '(FIR)', markersize = 15, alpha  = 0.5)
            #plt.errorbar(zs[i], [1./fir_gd_int[i]], yerr = [0.5*1./fir_gd_int[i]], fmt= '*', color = fir_cols[i], label = gals[i] + ' (FIR)', markersize = 20, alpha  = 0.5)


    for i in range(len(gamma)):

        ax[0].plot(z, dog[:, i], '-', label = r'$\gamma=$' + '{:2.1e}'.format(gamma[i]))
        ax[1].plot(z, dog[:, i], '-', label = r'$\gamma=$' + '{:2.1e}'.format(gamma[i]))

    ax[0].set_xlim(left = 1.e-4,right=2.)
    ax[1].set_xlim(left = 1.e-4,right=2.)

    ax[0].set_ylim(bottom = 1.e-8, top = 1.)
    ax[1].set_ylim(bottom = 1.e-8, top = 1.)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel("Z [" + r'$Z_o$' + "]", fontsize = 16)


    ax[0].set_ylabel('D/G', fontsize = 16)
    ax[1].set_ylabel('D/G', fontsize = 16)
    ax[0].legend(fontsize = 13, loc = 'upper left')
    ax[1].legend(fontsize = 13, loc = 'upper left')
    fig.subplots_adjust(top = 0.95, bottom = 0.12, left = 0.12, right = 0.95, hspace = 0)

    plt.savefig("FIGURES_ALL/plot_feldmann2015_mw_lmc_smc.pdf", format= 'pdf', dpi = 1000)

    plt.clf()
    plt.close()





#############
