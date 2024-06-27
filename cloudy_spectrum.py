#%%
from tau_vert_update import wl_grid, generate_pressure_grid, R_J, R_sun
from core_transmission import build_atmosphere, interpolations, make_atmospheric_profiles, get_free_params
from plotting import plot_spectra
import numpy as np
import time
import csv
from tqdm import tqdm


t1= time.time()

# Provide input opacity path
opacity_input_path = "F:\\NEXOTRANS-main\\"

# Provide chemical species
chem_species = ['H2O','CO','CO2','H2S','K','Na','SO2']
rayleigh_species=['H2','He']

interpolation_method = 'cubic'

# Model wavelength
wl_lo = 0.507394
wl_hi = 5.46871
R = 48
wl_model = wl_grid(wl_lo, wl_hi, R)

# Interpolate and store the interpolated sigmas
stored_sigma = interpolations(opacity_input_path, chem_species, rayleigh_species, wl_model, interpolation_method)

# Specify the pressure grid of the atmosphere
P_atm_top = 1.0e-7
P_atm_bottom = 100
Num_layers = 100
P = generate_pressure_grid(P_atm_top, P_atm_bottom, Num_layers)

# Define profile details
PT_profile = 'madhu_seager' # either isothermal/guillot/madhu_seager
chem_prof = 'isochem' # type of chemical profile
cloud = 'on'  # on or 'off'

# Now let us know what we need to provide as free parameters for the chosen profiles
params = get_free_params(chem_species, PT_profile, chem_prof, cloud, cloud_type='sigmoid')
print("Free parameters to be input: "+ str(params['params']))


#-----------------------------------------------------------------------------------------
# Let's provide those values

# Isothermal P-T
#T_iso = np.array([611.17]) # temperature for isothermal profile. Not needed for madhu_seager profile


# Planetary properties
g = 4.30328
P_ref = -2.09
R_p = 1.27 * R_J
R_p_ref = R_p

# Star radius 
R_s = 0.9* R_sun

# Define the ranges
log_H2O_range = np.linspace(-4, -1, 6)
log_CO_range = np.linspace(-4, -1, 6)
log_CO2_range = np.linspace(-4, -1, 6)
log_H2S_range = np.linspace(-4, -1, 6)
log_K_range = np.linspace(-9, -6, 6)
log_Na_range = np.linspace(-8, -5, 6)
log_SO2_range = np.linspace(-8, -5, 6)

with open('wasp39_final.csv', 'a') as f:
    writer = csv.writer(f)

    # Iterate over the T_iso range
    
    # Iterate over the log_X range for each chemical species
    for log_X_H2O in log_H2O_range:
            for log_X_CO in log_CO_range:
                for log_X_CO2 in log_CO2_range:
                    for log_X_H2S in log_H2S_range:
                            for log_X_K in log_K_range:
                                for log_X_Na in log_Na_range:
                                    for log_X_SO2 in log_SO2_range:
                                         # Create the log_X array
                                        log_X = np.array([log_X_H2O, log_X_CO, log_X_CO2, log_X_H2S, log_X_K, log_X_Na, log_X_SO2])

                                        # Make the atmospheric profiles
                                        atmospheric_profiles = make_atmospheric_profiles(chem_species, rayleigh_species, chem_prof,
                                                                     PT_profile, P, g, R_p, P_ref, R_p_ref,
                                                                     log_X=log_X,
                                                                     alpha1=1.31, alpha2=1.21, log_P1=-1.41, log_P2=-2.23, log_P3=-0.90, T_0=998.62)

                                         # Calculate the spectrum
                                        spectrum = build_atmosphere(P, wl_model, stored_sigma, atmospheric_profiles,
                                                chem_species, rayleigh_species, cloud, R_p, R_s,
                                                w=6.37, lambda_sig=0.79, log_P_cloud_sigmoid=-2.82)

                                        # Store the spectrum result and the corresponding T_iso and log_X values in a list
                                        result = [spectrum, log_X_H2O, log_X_CO, log_X_CO2, log_X_H2S, log_X_K, log_X_Na, log_X_SO2]

                                        # Append the list to the CSV file
                                        writer.writerow(result)

# Make the atmospheric profiles

# These are sigmoid cloud parameters, w = range[1,10],lambda_sig=range[0.4,5.0],log_P_cloud_sigmoid=range[-6,-1]
    
#import pandas as pd
 
#data = pd.read_csv('/home/seps05/Dek_model/WASP-39b/data_file/wasp39b.csv')                 
#plot_spectra(wl_model, spectrum, R, R_bin=200,data=data)

'''
# Plotting PT profile. Jut if you want to see how the profile looks.
from scipy.ndimage import gaussian_filter1d as gauss_conv
from path_dist_copy import Temp_profile_Madhu_Seager
T_mad = Temp_profile_Madhu_Seager(P, alpha1=1.31,alpha2=1.21,log_P1=-1.41,log_P2=-2.23,log_P3=-0.90,T_0=998.62)
T = gauss_conv(T_mad, sigma=3, axis=0, mode='nearest')
import matplotlib.pyplot as plt
plt.yscale('log')
plt.xlim(900,1100)
plt.gca().invert_yaxis()
plt.plot(T,P)
'''
