
from core_transmission import make_atmospheric_profiles, build_atmosphere, get_free_params
from binning import spectral_binning
import ultranest
import pandas as pd
import numpy as np
import ultranest.stepsampler

# To run with MPI - more cores use, mpiexec -n 4 python script.py
# ---------------------------------------------------------------------------------------------------------

# Calling the forward model


def fwd_mod(chem_species, data, rayleigh_species, wl_model, chem_prof, log_X, PT_profile, P, stored_sigma,
            g, R_p, R_s, R_p_ref, cloud, param_values, chem_eq_species=None):
    
    #data = pd.read_csv(data)
    wl_data = data['wl']
    half_width = data['half_width']

    # below lines extract the required variables from param_values dictionary
    P_ref = param_values.get('P_ref')
    T_iso = param_values.get('T_iso')
    kappa_IR = param_values.get('kappa_IR')
    gamma_guillot = param_values.get('gamma_guillot')
    T_int = param_values.get('T_int')
    T_equ = param_values.get('T_equ')
    log_P3 = param_values.get('log_P3')
    log_P1 = param_values.get('log_P1')
    alpha1 = param_values.get('alpha1')
    T_0 = param_values.get('T_0')
    alpha2 = param_values.get('alpha2')
    log_P2 = param_values.get('log_P2')
    cloud_params = param_values.get('cloud_params')
    c_to_o = param_values.get('c_to_o')
    metal = param_values.get('metal')
    log_aerosol = param_values.get('log_aerosol')
    hc = param_values.get('hc')
    aero_species = param_values.get('aero_species')
    size_aero = param_values.get('size_aero')
    log_a = param_values.get('log_a')
    log_P_cloud_deck = param_values.get('log_P_cloud_deck')            
    gamma = param_values.get('gamma')
    phi_cloud = param_values.get('phi_cloud')
    log_P_cloud_sigmoid   =   param_values.get('log_P_cloud_sigmoid')               
    w = param_values.get('w')
    lambda_sig = param_values.get('lambda_sig')
    

    # make the atmosphere
    atmospheric_profiles = make_atmospheric_profiles(chem_species, rayleigh_species, chem_prof, PT_profile, P,
                                                     g, R_p, P_ref, R_p_ref, log_X, chem_eq_species, aero_species, size_aero,
                                                     c_to_o, metal, log_aerosol,
                                                     hc, T_iso, kappa_IR, gamma_guillot, T_int, T_equ,
                                                     log_P3, log_P1, alpha1, T_0, alpha2, log_P2)

    # calculate spectrum
    spectrum = build_atmosphere(P, wl_model, stored_sigma, atmospheric_profiles, chem_species, rayleigh_species,
                         cloud, R_p, R_s, aero_species, log_a, log_P_cloud_deck,
                                            gamma, phi_cloud, log_P_cloud_sigmoid, 
                                            w, lambda_sig)
    spectrum = spectrum[:, 0]

    if np.isnan(spectrum).any():
        # Discard spectrum with NaN values
        return 0, spectrum
    
    y_model = spectral_binning(wl_model, spectrum, wl_data, half_width)
 
    return y_model, spectrum


# ---------------------------------------------------------------------------------


# the retrieval function
def ultranest_retrieval(data, P, stored_sigma, PT_profile, chem_prof,
                        rayleigh_species, chem_species, g, R_p_ref,
                        wl_model, R_p, R_s, cloud, priors, min_num_live_points
                        ,chem_eq_species=None):

    # extract the free parameters to be retrieved
    param_dict = get_free_params(chem_species, PT_profile, chem_prof, cloud, cloud_type='sigmoid')

    param_names = param_dict['params']
    X_params = param_dict['X_params']
    chem_species = param_dict['chem_species']

    prior_ranges = priors['prior_ranges']

    read_data = pd.read_csv(data)
    trdata = read_data['tr_depth']  # 2nd column of data file
    # 3rd column of data file....The 1st column are the wavelength points
    yerror = read_data['error']

    # print(param_names)

    def prior_transform(cube):

        params = cube.copy()

        for i, parameter in enumerate(param_names):

            if (parameter not in X_params):

                lo = prior_ranges[parameter][0]
                hi = prior_ranges[parameter][1]
                params[i] = ((cube[i]*(hi - lo)) + lo)

            elif (parameter in X_params):

                for spec in chem_species:
                    term = '_' + spec

                    if ((term + '_' in parameter) or (parameter[-len(term):] == term)):

                        species = spec

                    lo = prior_ranges[parameter][0]
                    hi = prior_ranges[parameter][1]
                    params[i] = ((cube[i]*(hi - lo)) + lo)

        return params

    def likelihood(params):

        # dictionary mapping names to their values
        param_values = dict(zip(param_names, params))

        # initiate empty array
        log_X = []

        # append all parameters starting with 'log_' to log_X
        for name, value in param_values.items():
            if name.startswith('log_'):
                log_X.append(value)

         # access the parameters by their names
        for name in param_names:
            globals()[name] = param_values[name]

        y_model, spectrum = fwd_mod(chem_species, read_data, rayleigh_species, wl_model,
                                    chem_prof, log_X, PT_profile, P, stored_sigma, g,
                                    R_p, R_s, R_p_ref, cloud, param_values,chem_eq_species)
        
        # since spectrum has size (281,1), this makes (281,)
        #spectrum = spectrum[:, 0]
        
        if np.isnan(spectrum).any():
            
            # Assign penalty to likelihood => point ignored in retrieval
            loglikelihood = -1.0e100
            
            # Quit if given parameter combination is unphysical
            return loglikelihood

        #y_model = spectral_binning(wl_model, spectrum, wl_data, half_width)

        loglikelihood = -0.5*(((y_model - trdata)/yerror)**2).sum()

        return loglikelihood

    sampler = ultranest.ReactiveNestedSampler(param_names, likelihood, prior_transform)

    nsteps = len(param_names)
    
    sampler.stepsampler = ultranest.stepsampler.SliceSampler(nsteps=1,
                                                             generate_direction=ultranest.stepsampler.generate_mixture_random_direction)
    
    result = sampler.run(update_interval_volume_fraction=0.5, Lepsilon=0.3,
                                                                   frac_remain=0.1, dlogz=0.5,min_num_live_points=min_num_live_points) #update_interval_volume_fraction=0.5, Lepsilon=0.3,
                                                                   # frac_remain=0.1, dlogz=0.5,update_interval_volume_fraction=0.2, Lepsilon=0.3,

    sampler.print_results()

    return result


# -------------------------------------------------------------------------------

