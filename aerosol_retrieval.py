'''Arsl Retrieval
'''

from retrieval_package import ultranest_retrieval

# import required functions
from tau_vert_update import wl_grid, generate_pressure_grid, R_J, R_sun
from core_transmission import build_atmosphere, interpolations, make_atmospheric_profiles, get_free_params,\
                               initialize_prior
from plotting import plot_spectra
import numpy as np

# Provide input opacity path
opacity_input_path = "/home/seps05/Desktop/POSEIDON/inputs/opacity/"
path_aero =  '/home/seps05/Dek_model/NEXOTRANS_mainBranch/condensate data/'

# Provide chemical species
chem_species = ['H2O','CO','CO2','Na','K']
rayleigh_species=['H2','He']

#Provide aerosol species
aerosols = ['ZnS']

# Provide aerosol sizes from the available catalogue of sizes for the aerosols
size_aero = [2.21] #in micron



#----------------------Interpolation-------------------------------------------
interpolation_method = 'cubic'

# Model wavelength
wl_lo = 0.6
wl_hi = 5.5
R = 300
wl_model = wl_grid(wl_lo, wl_hi, R)

# Interpolate and store the interpolated sigmas
stored_sigma = interpolations(opacity_input_path, chem_species, rayleigh_species, wl_model, interpolation_method,
                             aerosols = aerosols, aero_sizes = size_aero, path_aero=path_aero)
#-----------------------Atmosphere-------------------------------------------------------


# Specify the pressure grid of the atmosphere
P_atm_top = 1.0e-7
P_atm_bottom = 100
Num_layers = 100
P = generate_pressure_grid(P_atm_top, P_atm_bottom, Num_layers)

# Define profile details
PT_profile = 'isothermal' # either isothermal/guillot/madhu_seager
chem_prof = 'isochem' # type of chemical profile
cloud = 'off'  # on or 'off'

# Now let us know what we need to provide as free parameters for the chosen profiles
params = get_free_params(chem_species, PT_profile, chem_prof, cloud, cloud_type='sigmoid', aero_species=aerosols)
print("Free parameters to be input: "+ str(params['params']))


#--------

# Planetary properties
g = 4.3058
R_p = 1.27 * R_J
R_p_ref = R_p

# Star radius 
R_s = 0.9* R_sun

# Provide the prior ranges
prior_ranges = {}
prior_ranges['log_H2O'] = [-3.0,-2.0]
prior_ranges['log_CO2'] = [-4.0,-3.0]
prior_ranges['log_CO'] = [-2.5,-1.5]
prior_ranges['log_Na'] = [-6.0,-4.0]
prior_ranges['log_K'] = [-8.0,-6.0]
prior_ranges['P_ref'] = [-4.0,-1.0]
prior_ranges['T_iso'] = [600,700]
prior_ranges['log_aerosol_ZnS'] = [-10,-4]
prior_ranges['hc'] = [0,1]
                
priors = initialize_prior(params, prior_ranges)

data = '/home/seps05/Dek_model/WASP-39b/data_file/wasp39b.csv'

# Run the retrieval.
result = ultranest_retrieval(data, P, stored_sigma, PT_profile, chem_prof, 
                             rayleigh_species, chem_species, g, R_p_ref, wl_model,
                             R_p, R_s, cloud, priors, min_num_live_points=100)