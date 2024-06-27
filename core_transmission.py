'''
Main function which calls all functions and performs all the calculations to calculate the 
transmission spectrum.
'''


from tau_vert_update import read_sigmas, generate_fine_temperature_grid, generate_fine_pressure_grid
from CIA_package import read_CIA
from tau_vert_update import kappa_CIA
import numpy as np
from interpolation_package import extract_aero, extracted_aero
from scipy.ndimage import gaussian_filter1d as gauss_conv
from read_fastchem_grid import read_fastchem_grid, intp_grid


def interpolations(opacity_input_path, chem_species, rayleigh_species, wl_model, interpolation_method, 
                   aerosols = None, aero_sizes = None, path_aero = None):
    """
    Read cross-section data, perform interpolations, and store the results in memory.

    Parameters:
    opacity_input_path (str): Path to opacity input files.
    chem_species (list): List of chemical species.
    T_fine (numpy.ndarray): Fine temperature grid.
    log_P_fine (numpy.ndarray): Fine logarithmic pressure grid.
    wl_model (numpy.ndarray): Wavelength model grid.
    interpolation_method (str): Interpolation method to use.

    Returns:
    dict: Dictionary containing interpolated sigma values for chemical species and CIA.
    """
    
    print("Reading cross-section and performing interpolations")    
    
    T_intp_lo = 400   # lower limit of temperature grid
    T_intp_hi = 3000  # upper limit of temperature grid
    T_intp_step = 10  # adjacent temperature difference
    T_intp = generate_fine_temperature_grid(T_intp_lo, T_intp_hi, T_intp_step)
    
    # Define fine pressure grid (log10(P/bar))
    P_intp_lo = -6.0
    P_intp_hi = 2.0
    P_intp_step = 0.5
    log_P_intp = generate_fine_pressure_grid(P_intp_lo, P_intp_hi, P_intp_step)
    
    chemical_sigma = read_sigmas(opacity_input_path, chem_species, T_intp, log_P_intp, wl_model, interpolation_method) # does both reading and interpolating
    
    print("Performing CIA interpolations")        
    sigma_cia = read_CIA(wl_model, T_intp, opacity_input_path)
      
    
    # Rayleigh sigmas
    #sigma_rayleigh = sigma_H2_values(wl_model) # only H2 included
    sigma_rayleigh = compute_sigma(rayleigh_species, wl_model) # includes both H2 and He (added: April 23)
    
    # Aerosol Sigma
    sigma_aero_stored = None  # Initialize to None
    if path_aero != None:
        
        sigma_aero_stored = extracted_aero(path_aero, aero_sizes, wl_model, aerosols)
     
    stored_sigma = {'chemical_sigma': chemical_sigma, 'sigma_cia': sigma_cia
                    , 'sigma_rayleigh': sigma_rayleigh, 'sigma_aero_stored': sigma_aero_stored, 'T_intp': T_intp,
                    'log_P_intp': log_P_intp}  # storing interpolated sigmas in memory
    
    del chemical_sigma, sigma_cia, sigma_rayleigh
    
    print("Interpolations completed")
    
    return stored_sigma


#----------Seperate Aerosol Interpolation--------------

def intp_aerosol(path_aero, aero_sizes, wl_model, aerosols):
    
    # Aerosol Sigma
    sigma_aero_stored = None  # Initialize to None
    if path_aero != None:
        
        sigma_aero_stored = extracted_aero(path_aero, aero_sizes, wl_model, aerosols)
        
    return sigma_aero_stored

#----------------------------------------------------------------------------------------------------------------------------------

# Initialize free parameters for the chosen atmospheric settings
# Added: May 3

def free_parameters(ref_param, PT_profile, chem_prof, cloud, cloud_type,
                    chem_species, aero_species = None):
    '''
    cloud_type = 'deck' or 'sigmoid'
    '''
    # Written: May 3
    
    params = []      # includes all parameters
    PT_params = []   # pressure-temperature parameters
    X_params = []    # Mixing ratio parameters
    cloud_params =[] # cloud parameters
    phy_params = []  # Physical parameters of the system.
    aero_params = [] # Aerosol parameters
    
    # ---Physical properties------
    if (ref_param == 'R_p_ref'):
        phy_params += ['R_p_ref']
    elif (ref_param == 'P_ref'):
        phy_params += ['P_ref']
    else:
        raise Exception('Notice: R_p_ref or P_ref are the only reference parameters')
    
    params += phy_params
    
    #----P-T profile parameters------
    if (PT_profile == 'isothermal'):
        PT_params += ['T_iso']        
    elif (PT_profile == 'madhu_seager'):
        PT_params += ['alpha1', 'alpha2', 'log_P1', 'log_P2', 'log_P3', 'T_0']
    elif (PT_profile == 'guillot'):
        PT_params += ['kappa_IR', 'gamma_guillot', 'T_int', 'T_equ']
        
    params += PT_params
    
    #----Mixing ratio parameters-----
    for chem in chem_species:
        if (chem_prof == 'isochem'):
            X_params += ['log_'+chem]
        
    if (chem_prof == 'fastchem'):
        X_params += ['c_to_o','metal']
            
    params += X_params
    
    #----Cloud parameters----------- 
    if (cloud != 'off'):
        if (cloud_type=='haze'):
            cloud_params += ['log_a','gamma']
        elif (cloud_type=='deck'):
            cloud_params += ['log_P_cloud_deck']
        elif (cloud_type=='sigmoid'):
            cloud_params += ['w','lambda_sig','log_P_cloud_sigmoid']
            
    params += cloud_params
    
    #---- Aerosol parameters---------
    #aero_params =None
    if aero_species != None:       
        for species in aero_species:
            
                aero_params += ['log_aerosol_'+ species, 'hc']
                
    params += aero_params

    return params, phy_params, PT_params, X_params, cloud_params, aero_params


# Arranging the free parameters in a dictionary format
def get_free_params(chem_species, PT_profile, chem_prof, cloud, cloud_type='deck', ref_param='P_ref', aero_species = None):
    
    params, phy_params, \
    PT_params, X_params, cloud_params, aero_params = free_parameters(ref_param, PT_profile, chem_prof, cloud, cloud_type, chem_species
                                                                     , aero_species)

    param_dict = {'chem_species': chem_species,'params': params, 'phy_params':phy_params, 'PT_params':PT_params,
                            'X_params':X_params, 'cloud_params': cloud_params, 'aero_params': aero_params}
    
    return param_dict

#-----------------------------------------------------------------------------------------------------------------------------------

# setting the prior ranges.
def initialize_prior(param_dict, prior_ranges={}):
        
    param_names = param_dict['params']
    X_params = param_dict['X_params']
    PT_profile = param_dict['PT_params']
    cloud_params = param_dict['cloud_params']
    aero_params = param_dict['aero_params']
    
    default_prior_ranges = { 'T_iso': [400, 3000], 'alpha1': [0.02, 2.00], 'alpha2': [0.02, 2.00], 
                             'log_P1': [-6, 2], 'log_P2': [-6, 2], 'P_ref': [-6, 2],
                             'log_P3': [-2, 2], 'log_X': [-12, -1], 'log_a': [-4, 8], 'gamma': [-20, 2], 
                             'T_0': [400,3000], 'log_P_cloud_deck':[-6,2], 'log_aerosol':[-30,-4]
                            }

    for parameter in param_names:
        
        if (parameter not in prior_ranges):
            if (parameter in X_params):
                if ('log_' in parameter):
                    if ('log_X' in prior_ranges):
                        prior_ranges[parameter] = prior_ranges['log_X']
                    else:
                        prior_ranges[parameter] = default_prior_ranges['log_X']
            
            elif ('T_iso' in parameter):
                if ('T_iso' in prior_ranges):
                    prior_ranges[parameter] = prior_ranges['T_iso']
                else:
                    prior_ranges[parameter] = default_prior_ranges['T_iso']
            
            else:
                prior_ranges[parameter] = default_prior_ranges[parameter]
                
    priors = {'prior_ranges': prior_ranges}

    return priors 
    
#-------------------------------------------------------------------------------------------------

# Calculate the atmospheric profiles here
from path_dist_copy import Temp_profile, Temp_profile_Madhu_Seager, Temp_profile_guillot
from cloud_model import compute_aero_mmm, mix_aerosol

def make_atmospheric_profiles(chem_species, rayleigh_species, chem_prof, PT_profile, P ,
                              g, R_p, P_ref, R_p_ref, log_X = None, aero_species=None, size_aero=None, 
                              c_to_o = None, metal = None, log_aerosol = None,
                              hc = None, T_iso=None, kappa_IR=None, gamma_guillot=None, T_int=None, T_equ=None,
                              log_P3=None, log_P1=None, alpha1=None, T_0=None, alpha2=None, log_P2=None
                              ):
                             
    '''
    log_X = numpy array for the chemical species in case of free chemistry. i.e.,log_X = np.array([-2,-3]) etc,
    chem_eq_species = names of chemical species in case of fastchem usage. i.e.,chem_eq_species = ['H2O1','C1O2'] etc,
    aero_species = names of aerosol species. aero_species = ['ZnS','MgSiO3']
    size_aero = sizes for aerosol species, size_aero = [1.21,2.21] in micron
    c_to_o = carbon to oxygen ratio for fastchem.
    metal = metallicity value for fastchem.
    log_aerosol = aerosol VMR in log.
    hc = parameter controlling the slope of the sigmoid function for sigmoid clouds. range = 0 to 1.
    T_iso = Isothermal temperature.
    P_0 = reference pressure for madhu P-T.
    T_0
    log_P1
    log_P2
    log_P3
    
    '''
    if PT_profile == 'isothermal':
        
        if T_iso is None:
            
            raise Exception("Please specify T_iso in 'make_atmospheric_profiles'.")
            
        # Temperature profile
        T = Temp_profile(P, T_iso , PT_profile='isothermal')
        
    elif PT_profile == 'guillot':
        
        if kappa_IR is None:
            
            raise Exception("Provide all parameters for guillot profile in 'make_atmospheric_profiles'.")

        T = Temp_profile_guillot(P, g, kappa_IR, gamma_guillot, T_int, T_equ)
             
    elif PT_profile == 'madhu_seager':
        
        if ((log_P3 < log_P2) or (log_P3 < log_P1)):
            
            T = 0
        
        else:
            
            if alpha1 is None:
                
                raise Exception("Specify all the Madhu_PT temperature parameters in 'make_atmospheric_profiles'.")
              
            T = Temp_profile_Madhu_Seager(P, alpha1, alpha2, log_P1, log_P2, log_P3, T_0)  
            T = gauss_conv(T, sigma=3, axis=0, mode='nearest')

         
    # Initialize X as an empty 2D array-----------------------------------------
    X = np.array([]).reshape(0, 0)  # An empty 2D array with shape (0, 0)
    
    if  chem_prof=='fastchem':
        # Get VMR profile from FastChem
        fastchem_grid = read_fastchem_grid(chem_species)
        X = intp_grid(P, T, c_to_o, metal, chem_species, fastchem_grid)
        X = np.power(10,X) # converting the log output to linear
    else:
        # Get VMR profile of trace species
        X_const = chem_profile(chem_species, P, log_X)
        X = X_const
    
    # If you want to append the VMR profile of trace species to the FastChem results,
    # you can concatenate them along the second axis (columns)
    #if chem_eq_species is not None and chem_species is not None:
        #X_const = chem_profile(chem_species, P, log_X)
        
        # Ensure X_const and X_equ are 2D arrays and concatenate
        #X = np.hstack((X_equ, X_const))    
        
        
    #VMR profile of bulk species in the atmosphere-----------------------------
    X_rayleigh = chem_profile_bulk(P, X, rayleigh_species)
    
    # Mean molecular mass of trace species
    mu_trace = compute_mmm(P, X, chem_species)  
    
    # Mean molecular mass of rayleigh species
    mu_rayleigh = compute_mmm(P, X_rayleigh, rayleigh_species)  
    
    if aero_species != None:
        
        # Mean molecular mass of aerosol species.
        mu_aero = compute_aero_mmm(P, T, aero_species, size_aero)
        
        mu = mu_trace + mu_rayleigh #+ mu_aero
   
    else:
        # combined mean molecular mass
        mu = mu_trace + mu_rayleigh
       
    # Calculate radial profile-------------------------------------------------
    num_density, r, r_up, r_low, dr = radial_profiles(P, T, g, R_p, P_ref, R_p_ref, mu) 
    
    X_aero = None
    if aero_species != None:
    
        # VMR of aerosol
        X_aero = mix_aerosol(P, T, g, log_aerosol, mu, aero_species, r, hc)
        
    atmospheric_profiles = {'T': T, 'X': X, 'X_rayleigh': X_rayleigh, 'X_aero': X_aero,
                       'num_density': num_density, 'r': r, 'r_up': r_up,
                       'r_low': r_low, 'dr': dr,'mu':mu }
        
    return atmospheric_profiles
        
#-------------------------------------------------------------------------------------------------------------------------------------------

# Calculate the transmission spectrum here.
from path_dist_copy import compute_mmm, radial_profiles,chem_profile, chem_profile_bulk, path_distribution
from tau_vert_update import tau_vert, calc_kappa, transit_depth,calculate_kappa_rayleigh, calc_kappa_aero
from sigma_rayleigh import compute_sigma
from cloud_model import cloud_MacMad17

def build_atmosphere(P, wl_model, stored_sigma, atmospheric_profiles, chem_species, rayleigh_species,
                     cloud, R_p, R_s, aero_species= None,  log_a = None, log_P_cloud_deck = None,
                                        gamma = None, phi_cloud = None, log_P_cloud_sigmoid = None, 
                                        w = None, lambda_sig = None):
    
    '''
    cloud_params = dictionary of cloud parameters.
    '''
    
    N_layers=len(P)    
    N_wl = len(wl_model)
    #--------------------------------------------------------------------------
    
    # Extracting the stored sigma
    interp_sigma = stored_sigma['chemical_sigma']   
    cia = stored_sigma['sigma_cia']   
    sigma_rayleigh = stored_sigma['sigma_rayleigh']
    sigma_aero = stored_sigma['sigma_aero_stored']
    log_P_intp = stored_sigma['log_P_intp']
    T_intp = stored_sigma['T_intp']
                
    #--------------------------------------------------------------------------
    
    # Extracting the atmospheric profile properties
    T = atmospheric_profiles['T']
    X = atmospheric_profiles['X']
    X_rayleigh = atmospheric_profiles['X_rayleigh']
    X_aero = atmospheric_profiles['X_aero']
    num_density = atmospheric_profiles['num_density']
    r_up = atmospheric_profiles['r_up']
    r_low = atmospheric_profiles['r_low']
    dr = atmospheric_profiles['dr']  
    mu = atmospheric_profiles['mu']
    #--------------------------------------------------------------------------    
    
    #b = (r_up+r_low)/2
    b = r_up
    
    # path distribution
    path = path_distribution(b, N_layers, r_up, r_low, dr)
    
    #--------------------------------------------------------------------------
    
    # Calculating kappa
     
    kappa_chemical = calc_kappa(T, P, wl_model, num_density, chem_species, X, interp_sigma
                   , T_intp, log_P_intp)
    
    #kappa_ray = kappa_rayleigh(P, wl_model, sigma_rayleigh, num_density, X_rayleigh)
    kappa_ray = calculate_kappa_rayleigh(P, wl_model, sigma_rayleigh, num_density, X_rayleigh, rayleigh_species)
    kappa_cia = kappa_CIA(X_rayleigh, P, num_density, wl_model, cia, T, T_intp)
    
    if aero_species != None:
        
       kappa_aero = calc_kappa_aero(wl_model, P, aero_species, num_density, X_aero, sigma_aero)
       
    else:
        
       kappa_aero = 0.0
       
    if cloud == 'on':
            
        # clouds from MacDonal & Madhusudhan
        kappa_cloud = cloud_MacMad17(P, num_density, wl_model, log_a, log_P_cloud_deck,
                           gamma, phi_cloud, log_P_cloud_sigmoid, 
                           w, lambda_sig, cloud_dim=1)
        
        # kappa cloud added with total cloud
        kappa = kappa_chemical + kappa_ray + kappa_cia + kappa_cloud + kappa_aero
        
        tau_v =  tau_vert(P, N_wl, dr, kappa)

    else:
        kappa_cloud = np.zeros((len(P), len(wl_model)))
        
        # total kappa without clouds
        kappa = kappa_chemical + kappa_ray + kappa_cia + kappa_aero
        
        tau_v =  tau_vert(P, N_wl, dr, kappa)
        
    
    #spectrum = NIRTRAN(tau_v, path, R_p, r_up, r_low, dr, R_s, N_b, N_wl)
    spectrum = transit_depth(wl_model, R_p, tau_v, path, r_up, r_low, dr, R_s)
    
    # Clear memory by deleting variables that are no longer needed
    del b, path, kappa_chemical, kappa_ray, kappa_cia, kappa_cloud, kappa, tau_v
        
    return spectrum

#------------------------------------------------------------------------------


























