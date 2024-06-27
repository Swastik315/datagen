'''
Equilibrium Retrieval
'''


from retrieval_package import ultranest_retrieval

# import required functions
from tau_vert_update import wl_grid, generate_pressure_grid, R_J, R_sun
from core_transmission import build_atmosphere, interpolations, make_atmospheric_profiles, get_free_params,\
                               initialize_prior
from plotting import plot_spectra
import numpy as np

# Read observation data in csv format
data = '/home/seps05/Dek_model/WASP-39b/data_file/wasp39b.csv'

# Plot the data if you want.
from plotting import plot_data
plot_data(data)
#-----------------------------------------------------------------------------

# Provide input opacity path
opacity_input_path = "/home/seps05/Desktop/POSEIDON/inputs/opacity/"

# Provide chemical species
chem_species = ['H2O','CO2','CO','Na','K','SO2']

chem_eq_species = ['H2O1','C1O2', 'C1O1', 'Na', 'K', 'O2S1']

rayleigh_species=['H2','He'] # This is fixed


#----------------------Interpolation-------------------------------------------
interpolation_method = 'cubic' 

# Model wavelength
wl_lo = 0.3       #Lowest wavelength
wl_hi = 5.5        # highest wavelength
R = 1000         
wl_model = wl_grid(wl_lo, wl_hi, R)

# Interpolate and store the interpolated sigmas
stored_sigma = interpolations(opacity_input_path, chem_species, rayleigh_species, wl_model, interpolation_method)

#-----------------------Atmosphere---------------------------------------------


# Specify the pressure grid of the atmosphere
P_atm_top = 1.0e-7  # Top of the atmosphere pressure
P_atm_bottom = 100  # Bottom of the atmosphere pressure
Num_layers = 100    # Divide the atmosphere in this many layers
P = generate_pressure_grid(P_atm_top, P_atm_bottom, Num_layers)

# Define profile details
PT_profile = 'isothermal' # either isothermal/guillot/madhu_seager

chem_prof = 'isochem'  # Isochem presently

#--------
# Cloud properties

cloud = 'on'  # on or 'off'

cloud_type = 'sigmoid'
# Planetary properties
g = 4.30328                 # gravitational acceleration in m/s^2
R_p = 1.27 * R_J           # planet radius in meters
R_p_ref = R_p              # For making radius of layers, you need to provide this,\
                           # this is where the reference pressure will be retrieved

# Star radius  
R_s = 0.9 * R_sun 

# Let's know the free parameters for the above planet model
params = get_free_params(chem_species, PT_profile, chem_prof, cloud, cloud_type='sigmoid')
print("Free parameters to be input: "+ str(params['params']))

# Provide the prior ranges
prior_ranges = {}
prior_ranges['log_H2O'] = [-3.0,-2.0]
prior_ranges['log_CO2'] = [-4.0,-3.0]
prior_ranges['log_CO'] = [-2.5,-1.0]
prior_ranges['log_Na'] = [-6.0,-4.0]
prior_ranges['log_K'] = [-8.0,-6.0]
prior_ranges['log_SO2'] = [-2.5,-1.5]
prior_ranges['P_ref'] = [-4.0,-1.0]
prior_ranges['w'] = [1,10]
prior_ranges['lambda_sig'] = [0.3,2.0]
prior_ranges['log_P_cloud_sigmoid'] = [-4,2]
prior_ranges['alpha1'] = [0.5,1.5]
prior_ranges['alpha2'] = [0.5,1.5]
prior_ranges['log_P1'] = [-3.0,-1.0]
prior_ranges['log_P2'] = [-3.0,-1.0]
prior_ranges['log_P3'] = [-0.9,-0.5]
prior_ranges['T_0'] = [500,1000]
                
priors = initialize_prior(params, prior_ranges)


# Run the retrieval.
result = ultranest_retrieval(data, P, stored_sigma, PT_profile, chem_prof,
                             rayleigh_species, chem_species, g, R_p_ref, wl_model,
                             R_p, R_s, cloud, priors, min_num_live_points=400)
# Get Corner plots            

from ultranest.plot import cornerplot

cornerplot(result)