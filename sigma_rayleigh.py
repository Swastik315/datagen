import numpy as np
import scipy.constants as sc



def calc_ref_idx(wl):
    """
    Calculate the refractive index using Polarisability-Hohm,1993 and 
    Lorentz-Lorenz relation.

    Parameters:
        wl (numpy.ndarray): Array of wavelengths in micrometers.

    Returns:
        numpy.ndarray: Array of refractive indices corresponding to the input wavelengths.
    """
    
    nu = 1.0e4/wl
    
    f_par = 1.62632           # Constants for Hohm, 1993.
    w_par_sq = 0.23940245 
    f_perp = 1.40105 
    w_perp_sq = 0.29486069 
    
    n_ref = (101325.0/(sc.k * 273.15)) * 1.0e-6
    
    alpha = ((1.0/3.0)*((f_par/(w_par_sq - (nu/219474.6305)**2)) +                  
                      2.0*(f_perp/(w_perp_sq - (nu/219474.6305)**2))))*0.148184e-24    # Convert to cm^3

    eta = np.sqrt((1.0 + (8.0*np.pi*n_ref*alpha/3.0))/(1.0 - (4.0*np.pi*n_ref*alpha/3.0)))  # Lorentz-Lorenz relation 
       
    return eta
  
    
def sigma_H2_values(wl):
    """
   Calculate the absorption cross-section of H2 molecules.

   Parameters:
       wl (numpy.ndarray): Array of wavelengths in micrometers.

   Returns:
       numpy.ndarray: Array of H2 absorption cross-section values corresponding to the input wavelengths.
   """
    
    eta = calc_ref_idx(wl)
    n = 350*((eta-1)**2)
    d = 2.867e19/(wl**4)
    
    sigma_h2 = n/d   
    sigma_h2 = sigma_h2*1.0e-4    
    sigma_h2 = np.flipud(sigma_h2)  # Invert the array
    
    return sigma_h2*0.000009
    
  
#--------------- Alternative Formalism (H2 + He)---------------------------

def compute_sigma(rayleigh_species, wl_model):
    '''
    Returns cross section in cgs units (cm2/g)
    -------
    '''
    
    sigma_ray = np.zeros((len(rayleigh_species), len(wl_model)))
    
    nu = 1e4/wl_model
    n_ref = (101325.0/(sc.k * 273.15)) * 1.0e-6  # Number density (cm^-3) at 0 C and 1 atm (1.01325 bar)
    
    if 'H2' in rayleigh_species:
        eta_H2, F_H2 = H2(wl_model)
        sigma_ray[rayleigh_species.index('H2')] = (((24.0 * np.pi**3 * nu**4)/(n_ref**2)) * 
                                                  (((eta_H2**2 - 1.0)/(eta_H2**2 + 2.0))**2) * F_H2)*1e-4 # 1e-4 converts to m^2
    
    if 'He' in rayleigh_species:
        eta_He, F_He = He(wl_model)
        sigma_ray[rayleigh_species.index('He')] = (((24.0 * np.pi**3 * nu**4)/(n_ref**2)) * 
                                                  (((eta_He**2 - 1.0)/(eta_He**2 + 2.0))**2) * F_He)*1e-4

    sigma_ray = np.transpose(sigma_ray) # converting to two columns
    
    return sigma_ray 



def H2(wl_model):
    '''
    Returns polarisability, king correction factor for Hydrogen.
    '''
    f_par = 1.62632           
    w_par_sq = 0.23940245 
    f_perp = 1.40105 
    w_perp_sq = 0.29486069 
    alpha = compute_polarisability(wl_model, f_par, w_par_sq, f_perp, w_perp_sq)
    
    # To cm^3 
    eta = Lorentz_Lorenz(alpha*0.148184e-24 )    
    gamma = anisotropy(wl_model, f_par, w_par_sq, f_perp, w_perp_sq)
    F = king_correction(alpha, gamma)
    
    return  eta, F


def He(wl_model):
    """Returns polarisability, King correction factor for Helium
    """
    
    eta = 1.0 + ((0.014755297/(426.29740 - 1.0/(wl_model**2)))*1.0018141444038913)  
    eta[wl_model < 0.2753] = 1.00003578
    eta[wl_model > 0.4801] = 1.0 + (0.01470091/(423.98 - 1.0/(wl_model[wl_model > 0.4801]**2)))   
    eta[wl_model > 2.0586] = 1.00003469
    F = 1.000000 * wl_model**0   
    
    return eta , F



def compute_polarisability(wl_model, f_par, w_par_sq,f_perp,w_perp_sq):
        """Calculate polarisability from Hohm equation 

        Notes
        -----
        Hohm, U. 1993. Mol.Phys.,78:929
        """
        nu = 1e4/wl_model
        # Now calculate polarisability using formula from Hohm, 1993
        alpha = ((1.0/3.0)*((f_par/(w_par_sq - (nu/219474.6305)**2)) +                 
                       2.0*(f_perp/(w_perp_sq - (nu/219474.6305)**2))))     
        return alpha


def anisotropy(wl_model, f_par, w_par_sq,f_perp,w_perp_sq):
    
        """get polarisability anisotropy from Hohm 1993"""
        
        nu = 1e4/wl_model  
        gamma = ((f_par/(w_par_sq - (nu/219474.6305)**2)) - 
                 (f_perp/(w_perp_sq - (nu/219474.6305)**2)))    
        return gamma
    

def Lorentz_Lorenz(alpha):
    
        """Lorentz-Lorenz relation""" 
        
        n_ref = (101325.0/(sc.k * 273.15)) * 1.0e-6 

        return np.sqrt((1.0 + (8.0*np.pi*n_ref*alpha/3.0))/(1.0 - (4.0*np.pi*n_ref*alpha/3.0)))  


def king_correction(alpha,gamma):
        
        return 1.0 + 2.0 * (gamma/(3.0*alpha))**2 




'''
# TEST
import numpy as np
import scipy.constants as sc
from tau_vert_update import wl_grid
rayleigh_species=['H2','He']
wl_min = 0.5
wl_max = 5.0
R =200
wl_model = wl_grid(wl_min, wl_max, R)

sigma_rayl = compute_sigma(rayleigh_species, wl_model)

import matplotlib.pyplot as plt

sigm_H2 = sigma_rayl[:,0]
sigma_He = sigma_rayl[:,1]
plt.plot(wl_model, sigm_H2)
plt.plot(wl_model, sigma_He)

'''




























    
  
    
