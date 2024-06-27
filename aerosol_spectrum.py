
from tau_vert_update import wl_grid, generate_pressure_grid, R_J, R_sun
from core_transmission import build_atmosphere, interpolations, make_atmospheric_profiles, get_free_params
from plotting import plot_spectra
import numpy as np
import time

# Provide input opacity path
opacity_input_path = "/home/seps05/Desktop/POSEIDON/inputs/opacity/"
path_aero =  '/home/seps05/Dek_model/NEXOTRANS_mainBranch/condensate data/'

# Provide chemical species
chem_species = ['H2O','CO','CO2']
rayleigh_species=['H2','He']

#Provide aerosol species
aerosols = ['ZnS']

# Provide aerosol sizes from the available catalogue of sizes for the aerosols
size_aero = [0.01] #in micron



#----------------------Interpolation-------------------------------------------
interpolation_method = 'cubic'

# Model wavelength
wl_lo = 0.3
wl_hi = 30
R = 300
wl_model = wl_grid(wl_lo, wl_hi, R)

# Interpolate and store the interpolated sigmas
stored_sigma = interpolations(opacity_input_path, chem_species, rayleigh_species, wl_model, interpolation_method,
                             aerosols = aerosols, aero_sizes = size_aero, path_aero=path_aero)
#-----------------------Atmosphere-------------------------------------------------------


# Specify the pressure grid of the atmosphere
P_atm_top = 1.0e-6
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


#-----------------------------------------------------------------------------------------
# Let's provide those values

# Isothermal P-T
T_iso = np.array([1200]) # temperature for isothermal profile

# Profile values
log_X = np.array([-3,-4,-5,-6])

# Aerosol VMRs
log_aerosol = [-12.0]

# This factor is necessary for vertical profile of aerosol, 1 means vertically constant
# Between 0 and 1 is for vertically decreasing profile but that is not implemented now.
hc = 1

#--------

# Planetary properties
g = 4.3058
P_ref = -2.0
R_p = 1.27 * R_J
R_p_ref = R_p

# Star radius 
R_s = 0.9* R_sun


# Make the atmospheric profiles
atmospheric_profiles = make_atmospheric_profiles(chem_species, rayleigh_species, chem_prof, log_X,
                                                 PT_profile, P, g, R_p, P_ref, R_p_ref,
                                                 T_iso = T_iso, aero_species = aerosols, size_aero = size_aero,
                                                 log_aerosol = log_aerosol, hc=hc
                                                 )



# calculate the spectrum
spectrum = build_atmosphere(P, wl_model, stored_sigma, atmospheric_profiles,
                            chem_species, rayleigh_species, cloud, R_p, R_s
                            , aero_species=aerosols)

# Plot spectra
plot_spectra(wl_model, spectrum, R, R_bin=80)