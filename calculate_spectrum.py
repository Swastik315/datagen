
from tau_vert_update import wl_grid, generate_pressure_grid, R_J, R_sun
from core_transmission import build_atmosphere, interpolations, make_atmospheric_profiles, get_free_params
from plotting import plot_spectra
import numpy as np
import time

t1= time.time()

# Provide input opacity path
opacity_input_path = "/home/seps05/Desktop/POSEIDON/inputs/opacity/"

# Provide chemical species
chem_species = ['H2O','CO2','CH4','CO']
rayleigh_species=['H2','He']


#----------------------Interpolation-------------------------------------------
interpolation_method = 'cubic'

# Model wavelength
wl_lo = 0.6
wl_hi = 5.5
R = 1000
wl_model = wl_grid(wl_lo, wl_hi, R)

# Interpolate and store the interpolated sigmas
stored_sigma = interpolations(opacity_input_path, chem_species, rayleigh_species, wl_model, interpolation_method)
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
params = get_free_params(chem_species, PT_profile, chem_prof, cloud, cloud_type='sigmoid')
print("Free parameters to be input: "+ str(params['params']))


#-----------------------------------------------------------------------------------------
# Let's provide those values

# Isothermal P-T
T_iso = np.array([1357.63]) # temperature for isothermal profile

# guillot P-T
kappa_IR = 0.01
gamma_guillot=0.4
T_int=900
T_equ= 1120.55

# madhu_seager P-T
alpha1 = 0.5
alpha2 = 0.5
log_P1 = -5.52  # Pressure of layer 1-2 boundary
log_P2 = -3.16 # Pressure of inversion
log_P3 = -0.72 # Pressure of layer 2-3 boundary
T_0 = 674.0  # Atmosphere temperature reference value at P = P_0 (K)
P_0 = -2 # Pressure at which T_0 is defined


# Profile values
log_X = np.array([-3.50,-7.00,-7.04,-3.92])

#--------
# Cloud properties

cloud_params =  {
 'w':37, 
 'lambda_sig':1.0,
 'log_P_cloud_sigmoid': -2.0
 
 }

# Planetary properties
g = 7.3978
P_ref = 1.09
R_p = 1.32 * R_J
R_p_ref = R_p

# Star radius 
R_s = 1.23* R_sun


# Make the atmospheric profiles
atmospheric_profiles = make_atmospheric_profiles(chem_species, rayleigh_species,
                                                 chem_prof, PT_profile, P, g, R_p, P_ref, R_p_ref, T_iso=T_iso
                                                 ,log_X=log_X)


# calculate the spectrum
mu = build_atmosphere(P, wl_model, stored_sigma, atmospheric_profiles,
                            chem_species, rayleigh_species, cloud, R_p, R_s
                            )
                            
t2= time.time()

print('time taken to calculate spectrum in seconds:', t2-t1)

# if you have a data file and you want to overlay the forward model on top of the data
import pandas as pd
import matplotlib.pyplot as plt
# Step 2: Read the CSV file into a DataFrame
data= pd.read_csv('/home/seps05/Dek_model/WASP-62b.csv')

spectrum=spectrum
# Plot spectra
plot_spectra(wl_model, mu, R, data=data,R_bin=100)

#--------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Assuming wl_model, spectrum, and spectrum2 are defined
plt.figure(figsize=(12,8),dpi=400)
#plt.plot(wl_model,spectrum3, label='without cloud spectrum')
plt.plot(wl_model, spectrum, label = 'Sigmoid cloud spectrum')
plt.plot(wl_model, spectrum2, label = 'grey cloud deck spectrum')

# Add a horizontal line at the lowest value of spectrum 2
min_spectrum2 = min(spectrum2)
plt.axhline(y=min_spectrum2, color='r', linestyle='--', label=f'Cloud deck level')

plt.legend()
plt.show()


#-----------------------------------------------------------------------------------------


























