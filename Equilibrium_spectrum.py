
from tau_vert_update import wl_grid, generate_pressure_grid, R_J, R_sun
from core_transmission import build_atmosphere, interpolations, make_atmospheric_profiles, get_free_params
from plotting import plot_spectra
import numpy as np
import time

# Provide input opacity path
opacity_input_path = "/home/seps05/Desktop/POSEIDON/inputs/opacity/"

# Provide chemical species
#chem_species = ['H2O']
chem_eq_species = ['H2O1','C1O2', 'C1H4']
species = ['H2O','CO2','CH4']
rayleigh_species=['H2','He']

interpolation_method = 'cubic'

# Model wavelength
wl_lo = 0.3
wl_hi = 6.0
R = 100
wl_model = wl_grid(wl_lo, wl_hi, R)

# Interpolate and store the interpolated sigmas
stored_sigma = interpolations(opacity_input_path, species, rayleigh_species, wl_model, interpolation_method)

# Specify the pressure grid of the atmosphere
P_atm_top = 1.0e-6
P_atm_bottom = 100
Num_layers = 100
P = generate_pressure_grid(P_atm_top, P_atm_bottom, Num_layers)

# Define profile details
PT_profile = 'isothermal' # either isothermal/guillot/madhu_seager
chem_prof = 'equilib' # type of chemical profile
cloud = 'on'  # on or 'off'

# Now let us know what we need to provide as free parameters for the chosen profiles
params = get_free_params(chem_species, PT_profile, chem_prof, cloud, cloud_type='deck')
print("Free parameters to be input: "+ str(params['params']))


#-----------------------------------------------------------------------------------------
# Let's provide those values
cloud_params = {'log_P_cloud_deck':-2}

# Isothermal P-T
T_iso = np.array([1200]) # temperature for isothermal profile


# Profile values
log_X = np.array([-22])


# Planetary properties
g = 4.3058
P_ref = -2.75
R_p = 1.27 * R_J
R_p_ref = R_p

R_s = 0.9* R_sun

# Make the atmospheric profiles
atmospheric_profiles = make_atmospheric_profiles(chem_species, rayleigh_species, chem_prof,
                                                 PT_profile, P, g, R_p, P_ref, R_p_ref, chem_eq_species = chem_eq_species,
                                                 T_iso = T_iso, c_to_o=0.55, metal=10.0
                                                 )

# calculate the spectrum
spectrum = build_atmosphere(P, wl_model, stored_sigma, atmospheric_profiles,
                            chem_species, rayleigh_species, cloud, R_p, R_s, cloud_params=cloud_params
                            )
                            


plot_spectra(wl_model, spectrum, R, R_bin=80)