
'''
Binning model spectrum to data
'''

import numpy as np
import pandas as pd
from scipy.integrate import trapz
from scipy.ndimage import gaussian_filter1d as gauss_conv
from scipy.interpolate import InterpolatedUnivariateSpline as interp


# Useful for Instruments
def fwhm_ins(inst_dir, wl_data, instrument):   
    
    '''
    Adapted from POSEIDON code.
    
    Evaluate the full width at half maximum (FWHM) for the Point Spread 
    Function (PSF) of a given instrument mode at each bin centre wavelength.
    
    FWHM (μm) = wl (μm) / R_native 
    
    This assumes a Gaussian PSF with FWHM = native instrument spectral resolution.

    Args:
        wl_data (np.array of float): 
            Bin centre wavelengths of data points (μm).
        instrument (str):
            Instrument name corresponding to the dataset
            (e.g. WFC3_G141, JWST_NIRSpec_PRISM, JWST_NIRISS_SOSS_Ord2). 
    
    Returns:
        fwhm (np.array of float):
            Full width at half maximum as a function of wavelength (μm).

    '''
        
    N_bins = len(wl_data)

   
    
    # For the below instrument modes, FWHM assumed constant as function of wavelength
    if   (instrument == 'STIS_G430'):   
        fwhm = 0.0004095 * np.ones(N_bins)  # HST STIS
    elif (instrument == 'STIS_G750'):   
        fwhm = 0.0007380 * np.ones(N_bins)  # HST STIS
    elif (instrument == 'WFC3_G280'):
        fwhm = 0.0057143 * np.ones(N_bins)  # HST WFC3
    elif (instrument == 'WFC3_G102'):
        fwhm = 0.0056350 * np.ones(N_bins)  # HST WFC3
    elif (instrument == 'WFC3_G141'):
        fwhm = 0.0106950 * np.ones(N_bins)  # HST WFC3
    elif (instrument == 'LDSS3_VPH_R'):
        fwhm = 0.0011750 * np.ones(N_bins)  # Magellan LDSS3
    
    # For JWST, we need to be a little more precise
    elif (instrument.startswith('JWST')):    

        # Find precomputed instrument spectral resolution file
        res_file = inst_dir + '/JWST/' + instrument + '_resolution.dat'
        
        # Check that file exists
        if (inst_dir(res_file) == False):
            print("Error! Cannot find resolution file for: " + instrument)
            raise SystemExit
            
        # Read instrument resolution file
        resolution = pd.read_csv(res_file, sep=' ', header=None)
        wl_res = np.array(resolution[0])   # (um)
        R_inst = np.array(resolution[1])   # Spectral resolution (R = wl/d_wl)
        
        # Interpolate resolution to bin centre location of each data point
        R_interp = interp(wl_res, R_inst, ext = 'extrapolate')  
        R_bin = np.array(R_interp(wl_data))
        
        # Evaluate FWHM of PSF for each data point
        fwhm = wl_data / R_bin  # (um)

    elif (instrument == 'IRTF_SpeX'):

        #fwhm_IRTF_SpeX(wl_data)  # Using the external resolution file currently

        # Find precomputed instrument spectral resolution file
        res_file = inst_dir + '/IRTF/' + instrument + '_resolution.dat'
        
        # Check that file exists
        if (inst_dir(res_file) == False):
            print("Error! Cannot find resolution file for: " + instrument)
            raise SystemExit
            
        # Read instrument resolution file
        resolution = pd.read_csv(res_file, sep=' ', header=None)
        wl_res = np.array(resolution[0])   # (um)
        R_inst = np.array(resolution[1])   # Spectral resolution (R = wl/d_wl)
        
        # Interpolate resolution to bin centre location of each data point
        R_interp = interp(wl_res, R_inst, ext = 'extrapolate')  
        R_bin = np.array(R_interp(wl_data))
        
        # Evaluate FWHM of PSF for each data point
        fwhm = wl_data / R_bin  # (um)
    
    # For any other instruments without a known PSF, convolve with a dummy sharp PSF
    else: 
        fwhm = 0.00001 * np.ones(N_bins) 
    
    return fwhm



def initialization(wl_model, wl_data, half_width, inst_dir):
    
    '''
   Initialize required properties for a specific instrument.
   
   Args:
       wl_model (np.ndarray): Model wavelength grid (μm).
       wl_data (np.ndarray): Bin centre wavelengths of data points (μm).
       half_width (np.ndarray): Bin half widths of data points (μm).
       instrument (str): Instrument name corresponding to the dataset.
       inst_dir (str, optional): Directory containing instrument properties.
   
   Returns:
       tuple: Tuple containing sigma, fwhm, sensitivity, bin_left, bin_centre, bin_right, norm.
   '''
   
   
    # Identify instrument sensitivity function for desired instrument mode
  
 
    sens_file = inst_dir + '/dummy_instrument_sensitivity.dat'
     
    # reading instrument sensitivity file
    transmission = pd.read_csv(sens_file, sep=' ',header=None)
    wl_transm = np.array(transmission[0])
    transm = np.array(transmission[1])
    
    # let's get the sensitivity at model wavelengths
    sensitivity = np.zeros(len(wl_model))
    # interpolation
    sens = interp(wl_transm, transm, ext='zeros')
    sensitivity = sens(wl_model)
    sensitivity[sensitivity<0.0] = 0.0
    
    #FWHM of instrument PSF at each data point location
    #fwhm = fwhm_ins(inst_dir, wl_data, instrument)
    fwhm = 0.00001 * np.ones(len(wl_data)) 
    
    # psf sd(um)
    sigma_um = 0.424661*fwhm

    # binning
    bin_lft = np.zeros(len(wl_data)).astype(np.int64)
    bin_centre = np.zeros(len(wl_data)).astype(np.int64)
    bin_rht = np.zeros(len(wl_data)).astype(np.int64)
    sigma = np.zeros((len(wl_data)))
    norm = np.zeros((len(wl_data)))
    
    for n in range(len(wl_data)):
        
           # closest indices on model grid corresponding to bin edges and centre
           bin_lft[n] = np.argmin(np.abs(wl_model - ((wl_data[n] - half_width[n])))) 
           bin_centre[n] = np.argmin(np.abs(wl_model - (wl_data[n]))) 
           bin_rht[n] = np.argmin(np.abs(wl_model - ((wl_data[n] + half_width[n]))))
           
           # standard deviation of instrument PSF in grid spaces at each bin location (approx)
           dwl = 0.5 * (wl_model[bin_centre[n]+1] - wl_model[bin_centre[n]-1])
           sigma[n] = sigma_um[n]/dwl   
           
           # normalisation of sensitivity function for each wl bin      
           norm[n] = trapz(sensitivity[bin_lft[n]:bin_rht[n]], wl_model[bin_lft[n]:bin_rht[n]])  
       
    
    return sigma, fwhm, sensitivity, bin_lft, bin_centre, bin_rht, norm 
    
    

def bin_model_data(model_spectrum, wl_model, wl_data, half_width, inst_dir):
    """
   Bin model spectrum from high resolution to real data points in low resolution.

   Args:
       model_spectrum (np.ndarray): High-resolution model spectrum.
       wl_model (np.ndarray): Model wavelength grid.
       sigma (np.ndarray): Standard deviation of PSF for each data point.
       sensitivity (np.ndarray): Instrument transmission function interpolated to model wavelengths.
       bin_lft (np.ndarray): Closest index on model grid of the left bin edge for each data point.
       bin_centre (np.ndarray): Closest index on model grid of the bin centre for each data point.
       bin_rht (np.ndarray): Closest index on model grid of the right bin edge for each data point.
       norm (np.ndarray): Normalisation constant of the transmission function for each data bin.

   Returns:
       np.ndarray: Binned model data corresponding to real data points.
   """
    
    sigma, fwhm, sensitivity, bin_lft, bin_centre, bin_rht, norm = initialization(wl_model, wl_data, half_width, inst_dir)
    
    N_bins = len(bin_centre)
    data = np.zeros((N_bins))
    spect_binned = np.zeros((N_bins))
    
    for i in range(N_bins):
        
        # Extend convolution beyond bin edge by max(1, 2 PSF std) model grid spaces (std rounded to integer)
        extension = max(1, int(2 * sigma[i]))   
        
        # Convolve spectrum with PSF width appropriate for a given bin 
        spectrum_conv = gauss_conv(model_spectrum[(bin_lft[i]-extension):(bin_rht[i]+extension)], 
                                   sigma=sigma[i], mode='nearest')

        # Catch a (surprisingly common) error
        if (len(spectrum_conv[extension:-extension]) != len(sensitivity[bin_lft[i]:bin_rht[i]])):
            raise Exception("Error: Model wavelength range too small. Please extend model wavelength range to include all data")

        integrand = spectrum_conv[extension:-extension] * sensitivity[bin_lft[i]:bin_rht[i]]
    
        # Integrate convolved spectrum over instrument sensitivity function
        data[i] = trapz(integrand, wl_model[bin_lft[i]:bin_rht[i]])   
        spect_binned[i] = data[i]/norm[i]
        
    return spect_binned
    

    
#------------------------------------------------------------------------------

# Simple binning
def spectral_binning(model_wavelengths, model_transit_depths, real_wavelengths, half_widths):
    binned_transit_depths = []

    for i, wl in enumerate(real_wavelengths):
        # Calculate the wavelength range for binning
        lower_bound = wl - half_widths[i]
        upper_bound = wl + half_widths[i]

        # Find indices of model wavelengths within the range
        indices_within_range = np.where((model_wavelengths >= lower_bound) & (model_wavelengths <= upper_bound))[0]

        if len(indices_within_range) > 0:
            # Calculate the average transit depth within the range
            avg_transit_depth = np.mean(model_transit_depths[indices_within_range])
            binned_transit_depths.append(avg_transit_depth)
        else:
            # No model data within the range, set to NaN or handle as needed
            binned_transit_depths.append(np.nan)

    return np.array(binned_transit_depths)
    
    
    
    
  
