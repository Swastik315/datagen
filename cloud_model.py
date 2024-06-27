'''
CLOUD MODELS
'''

import scipy.constants as sc
import numpy as np
# Calculating the extinction due to clouds
from chemical_species import mass_dens


'''
def cloud_MacMad17(P, n, wl_model, cloud_params, cloud_dim=1):
    
    
    cloud_dim =1|2
    cloud_params = [log_a, log_P_cloud, gamma, phi_cloud]
    
    
    kappa_cloud = np.zeros((len(P), len(wl_model)))

    
    if ('log_a' in cloud_params):        
        haze_enabled = 1    
    else:       
        haze_enabled = 0
        
    if ('log_P_cloud_deck' in cloud_params):
        deck_enabled = 1
    else:
        deck_enabled = 0
    
     # Sigmoid activation   
    if ('w' in cloud_params):
       sigmoid_enabled = 1
    else: 
       sigmoid_enabled = 0 
       
    if (haze_enabled==1):
        a = np.power(10.0, cloud_params['log_a'])
        gamma = cloud_params['gamma']
    else:
        a, gamma = 1.0, -4.0
     
    # if a cloud deck is present
    if (deck_enabled==1):
        P_cloud = np.power(10.0, cloud_params['log_P_cloud_deck'])
    else:
        P_cloud = 100.0
     
        # if patchy clouds are present
    if (cloud_dim != 1):
        f_cloud = cloud_params['phi_cloud']
        theta_0 = -90
    else:  
       if (deck_enabled == 1):
           # uniform cloud
                f_cloud, phi_0, theta_0 = 1.0, -90.0, -90.0  
       else:
                f_cloud, phi_0, theta_0 = 0.0, -90.0, 90.0   # Dummy values, not used when cloud-free
     
            
     
    if (haze_enabled==1):
        
        slope = np.power((wl_model/0.35), gamma)
        
        i_bot = np.argmin(np.abs(P - 1000))

        for i in range(i_bot, len(P)):
            
            haze_amplitude = (n[i]*a*5.31e-31)
            
            for l in range(len(wl_model)):
                
                kappa_cloud[i,l] += haze_amplitude * slope[l]
                
                
    if (deck_enabled==1):
        
        kappa_cloud_0 = 1.0e250

        kappa_cloud[(P>P_cloud),:] += kappa_cloud_0
       
        
    # enabling sigmoid cloud
    if (sigmoid_enabled==1):
        
        P_cloud = np.power(10.0, cloud_params['log_P_cloud_sigmoid'])
        
        w = cloud_params['w']
        
        lambda_sig = cloud_params['lambda_sig']
        
        kappa_cloud[(P>P_cloud),:] = (100 / (1 + np.exp(w * (wl_model - lambda_sig))))
        
    return kappa_cloud   
'''




import numpy as np

def cloud_MacMad17(P, n, wl_model, log_a=None, log_P_cloud_deck=None,
                   gamma=None, phi_cloud=None, log_P_cloud_sigmoid=None, 
                   w=None, lambda_sig=None, cloud_dim=1):
    
    '''
    cloud_dim = 1|2
    cloud_params = [log_a, log_P_cloud_deck, gamma, phi_cloud, log_P_cloud_sigmoid, w, lambda_sig]
    '''
    
    kappa_cloud = np.zeros((len(P), len(wl_model)))
    
    haze_enabled = log_a is not None
    deck_enabled = log_P_cloud_deck is not None
    sigmoid_enabled = w is not None
    
    if haze_enabled:
        a = np.power(10.0, log_a)
        gamma = gamma if gamma is not None else -4.0
    else:
        a, gamma = 1.0, -4.0
     
    if deck_enabled:
        P_cloud = np.power(10.0, log_P_cloud_deck)
    else:
        P_cloud = 100.0
     
    if cloud_dim != 1:
        f_cloud = phi_cloud if phi_cloud is not None else 0.0
        theta_0 = -90
    else:  
       if deck_enabled:
           f_cloud, phi_0, theta_0 = 1.0, -90.0, -90.0  
       else:
           f_cloud, phi_0, theta_0 = 0.0, -90.0, 90.0  # Dummy values, not used when cloud-free
     
    if haze_enabled:
        slope = np.power((wl_model/0.35), gamma)
        i_bot = np.argmin(np.abs(P - 1000))
        for i in range(i_bot, len(P)):
            haze_amplitude = (n[i] * a * 5.31e-31)
            for l in range(len(wl_model)):
                kappa_cloud[i, l] += haze_amplitude * slope[l]
                
    if deck_enabled:
        kappa_cloud_0 = 1.0e250
        kappa_cloud[(P > P_cloud), :] += kappa_cloud_0
       
    if sigmoid_enabled:
        P_cloud = np.power(10.0, log_P_cloud_sigmoid)
        w = w
        lambda_sig = lambda_sig
        kappa_cloud[(P > P_cloud), :] = (100 / (1 + np.exp(w * (wl_model - lambda_sig))))
        
    return kappa_cloud


'''
# Test
import numpy as np
# Example usage
P = np.linspace(1e-6, 100, 100)  # Example pressure grid
n = np.ones_like(0.5*P)  # Example number density array
wl_model = np.linspace(0.3, 2.0, 100)  # Example wavelength grid

cloud_params = {'log_a': 2, 'log_P_cloud': -1.0, 'gamma': -4, 'phi_cloud': 0.5}
kappa_cloud = cloud_MacMad17(P, n, wl_model, cloud_params=cloud_params)
'''
#------------------------------------------------------------------------------


'''
AEROSOL MIXING RATIO MODEL
'''


def compute_aero_mmm(P, T, aero_species, size_aero):
    """
    Compute the mean molecular mass for multiple aerosol species.
    
    Parameters:
    P : array_like
        Pressure array (bar).
    T : array_like
        Temperature array (K).
    aero_species : list
        List of aerosol species.
    size_aero : list
        List of aerosol sizes corresponding to each species (micron).
    
    Returns:
    mu : float
        Mean molecular mass of aerosols (kg).
    """
    
    # Initialize number density array
        
    n = np.zeros(len(P))
    
    mu = np.zeros(len(P))
    
    V = mj = np.zeros(len(aero_species))
    
    # Calculate number density (particles per m^3)
    n[:] = (P * 1.0e5) / (sc.k * T) 
    
    # Initialize total mass and total number density
    total_mass = 0.0
    total_number_density = 0.0
    
    # Loop through each aerosol species
    for q, species in enumerate(aero_species):
        
        species = aero_species[q]
        
        # Volume of the particle (m^3)
        V = (4/3) * np.pi * ((size_aero[q]**3) * (1e-6)**3) / 3
        
        # Mass of a single aerosol particle (kg)
        mj = mass_dens[species] * V
        
        for l in range(len(P)): # len(P) is the number of atmospheric layers
            
            # Total mass contribution for this species
            total_mass += (mj * n[l])
            
            # Total number density contribution for this species
            total_number_density += n[l]
        
            # Calculate mean molecular mass (kg)
            mu[l] = total_mass / total_number_density
            
    return mu
    


def mix_aerosol(P, T, g, X0, mu, aero_species, r, hc):
    
    '''

    Parameters
    ----------
    T : TYPE
        Temperature array, equal to number of pressure points.
    hc : TYPE
        free parameter, value ranges from 0 to 1. 1 means constant mixing ratio
    g : TYPE
        planet gravity in m/s^-2.
    r : TYPE
        layer radial distance.
    X0 : TYPE
        Mixing ratio at the lowest layer.
    mu : TYPE
        mean molecular mass of atmosphere
    aero_species : TYPE
        name of aerosol species.

    Returns
    -------
    X : TYPE
        DESCRIPTION.

    '''
    
    X0 = np.power(10, X0)
    N_species = len(aero_species)
    N_layers = len(P)
    
    X = np.zeros((N_layers, N_species))
    
    for q in range(len(aero_species)):
        
        n = 1 / hc # hc less than 1 is making the X zero, so using 1 for now.
        
        for l in range(N_layers):
            
            z = r[l]
            
            H = sc.k * T[l] / (mu[l] * g)  # Scale height
            
            term = (-1 * (n - 1) * z) 
            
            X[l, q] = X0[q] * np.exp(term/ H)
    
    return X
    






























  
            
