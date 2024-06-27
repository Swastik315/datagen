
# Formulating the equation for Path Difference

import numpy as np
import scipy.constants as sc
from chemical_species import masses
from numba import njit



def Temp_profile(P, T_iso , PT_profile='isothermal'):
   
    '''
    Creates an 1D isothermal temperature profile
    
    P = atmospheric pressure layer array
    T_deep = single temperature for
    
    Output: Isothermal temperature array
            
    '''    
    
    if PT_profile == 'isothermal':
        
        T = np.full_like(P, fill_value=T_iso)
        
    else:
        raise ValueError("Unsupported PT_profile. Currently supports 'isothermal' only.")
    
    return T



def Temp_profile_guillot(P, g, kappa_IR, gamma_guillot, T_int, T_equ):
    '''
    Creates a 1D temperature profile based on the Guillot temperature profile model.

    Args:
        P (numpy array): Atmospheric pressure layer array (bar).
        kappa_IR (float): Infrared opacity (cm^2/g).
        gamma_guillot (float): Ratio between visual and infrared opacity.
        g (float): Planetary surface gravity (m/s^2).
        T_int (float): Planetary internal temperature (K).
        T_equ (float): Planetary equilibrium temperature (K).

    Returns:
        numpy array: Temperature profile array (K).
    '''
    
    # Define Guillot temperature profile function
    
    T = np.zeros((len(P)))
    
    # Define Guillot temperature profile function
    
    # Precompute constants
    tau = P * 1e6 * kappa_IR / g
    T_irr = T_equ * np.sqrt(2.)
    term1 = 0.75 * T_int**4.
    term2 = 0.75 * T_irr**4. / 4.

    # Calculate temperature profile
    exp_term = np.exp(-gamma_guillot * tau * 3.**0.5)
    T = (term1 * (2. / 3. + tau) +
         term2 * (2. / 3. + 1. / gamma_guillot / 3.**0.5 +
                  (gamma_guillot / 3.**0.5 - 1. / 3.**0.5 / gamma_guillot) *
                  exp_term))**0.25
 
    return T



def Temp_profile_Madhu_Seager(P, alpha1, alpha2, log_P1, log_P2, log_P3, T_0, P_0=1.0e-2):
    
   
   T = np.zeros((len(P))) 
   
   # Find index of pressure closest to the set pressure
   i_set = np.argmin(np.abs(P - P_0))
   P_set_i = P[i_set]
   
   log_P = np.log10(P)
   log_P_min = np.log10(np.min(P))
   log_P_set_i = np.log10(P_set_i)

   if (log_P_set_i >= log_P3):
       
       T3 = T_0  
       T2 = T3 - ((1.0/alpha2)*(log_P3 - log_P2))**2    
       T1 = T2 + ((1.0/alpha2)*(log_P1 - log_P2))**2    
       T0 = T1 - ((1.0/alpha1)*(log_P1 - log_P_min))**2   
       
   elif (log_P_set_i >= log_P1):   
       
       T2 = T_0 - ((1.0/alpha2)*(log_P_set_i - log_P2))**2  
       T1 = T2 + ((1.0/alpha2)*(log_P1 - log_P2))**2   
       T3 = T2 + ((1.0/alpha2)*(log_P3 - log_P2))**2
       T0 = T1 - ((1.0/alpha1)*(log_P1 - log_P_min))**2   
       
   elif (log_P_set_i < log_P1):  
   
       T0 = T_0 - ((1.0/alpha1)*(log_P_set_i - log_P_min))**2
       T1 = T0 + ((1.0/alpha1)*(log_P1 - log_P_min))**2   
       T2 = T1 - ((1.0/alpha2)*(log_P1 - log_P2))**2  
       T3 = T2 + ((1.0/alpha2)*(log_P3 - log_P2))**2
       
   for i in range(len(P)):
       
       if (log_P[i] >= log_P3):
           T[i] = T3
       elif ((log_P[i] < log_P3) and (log_P[i] > log_P1)):
           T[i] = T2 + np.power(((1.0/alpha2)*(log_P[i] - log_P2)), 2.0)
       elif (log_P[i] <= log_P1):
           T[i] = T0 + np.power(((1.0/alpha1)*(log_P[i] - log_P_min)), 2.0)
   
   #T = np.flipud(T)
   return T
 



def chem_profile(chem_species, P, log_X):
    
    '''
    Creates a 1D vertically constant chemical profile based on volume mixing ratio values.

    Args:
        chem_species (list): List of chemical species names.
        P (numpy array): Atmospheric pressure layer array (bar).
        log_X (numpy array): Volume mixing ratio values in logarithmic scale.

    Returns:
        numpy array: 2D array representing the chemical profile for each species across atmospheric layers.
                     Rows correspond to layers, columns correspond to species.
    '''
    
    
    X_profile = np.zeros((len(P), len(chem_species)))
    
    X = np.power(10.0, log_X) # converting log values to linear (added: Feb 17, 2024)
    
    for q in range(len(chem_species)):
        
           X_profile[:,q] = X[q]

            
    return X_profile




def chem_profile_bulk(P, X_profile, rayleigh_species):
    
    '''
   Calculates the H2 and He mixing ratios from the minor species.

   Args:
       P (numpy array): Atmospheric pressure layer array (bar).
       X_profile (numpy array): 2D array of volume mixing ratios for all chemical species across atmospheric layers.
       rayleigh_species (list): List of Rayleigh scattering species, typically ['H2', 'He'].

   Returns:
       numpy array: 2D array representing the H2 and He mixing ratios across atmospheric layers.
                    Rows correspond to layers, columns correspond to H2 and He.
                    
   Note: This relation is accurate only for H2/He dominated atmospheres.
   '''
    
    
    X_prof = np.transpose(X_profile)
    
    X_rayleigh = np.zeros((len(rayleigh_species), len(P)))

    H2_He_fraction = 0.17647
    
    if ('H2' and 'He' in rayleigh_species):
           
            X_H2 = (1.0 - np.sum((X_prof), axis=0))/(1.0 + H2_He_fraction)            
            X_He = X_H2*H2_He_fraction
            
            X_rayleigh[0,:] = X_H2            
            X_rayleigh[1,:] = X_He
            
    X_rayleigh = np.transpose(X_rayleigh)
            
    return X_rayleigh
    
       

#-----------------------------------------------
def compute_mmm(P, X_profile, chem_species):
    
    '''
   Computes the mean molecular mass of the atmosphere.

   Args:
       P (numpy array): Atmospheric pressure layer array (bar).
       X_profile (numpy array): 2D array of volume mixing ratios for all chemical species across atmospheric layers.
       chem_species (list): List of chemical species in the atmosphere.

   Returns:
       numpy array: Array containing the mean molecular mass for each atmospheric layer (kg).
   '''
    
    N_layers = len(P)    
    N_species = len(chem_species)
    
    mu = np.zeros(shape=(N_layers))
     
    for l in range(N_layers):
        
        for q in range(N_species):
            
            species = chem_species[q]
            
            mu[l] += X_profile[l,q] * masses[species]
            
    mu = mu * sc.u # amu to kg
    
    return mu




def number_density(P, T):
    
    ''' 
    Computing the number density(n) for each atmospheric layer.
    
    T = isothermal temperature profile
    
    Returns a numpy array for n.
    '''
    N_layers = len(P)

    n = np.zeros(shape=(N_layers))
    

    # number density
    n[:] = (P * 1.0e5)/(sc.k * T[:])  # k = Boltzman's constant
        
    return n
 
  
def radial_profiles(P, T, g, R_p, P_ref, R_p_ref, mu):
    ''' 
    Compute radial profiles of number density, radius, upper boundary, lower boundary, and layer thickness.

    Args:
        P (numpy array): Atmospheric pressure layer array (bar).
        T (numpy array): Temperature profile array (K).
        g (float): Surface gravity of the planet (m/s^2).
        R_p (float): Planetary radius (m).
        P_ref (float): Reference pressure for radius calculation (bar).
        R_p_ref (float): Reference radius corresponding to reference pressure (m).
        mu (numpy array): Mean molecular mass array for each layer (kg).

    Returns:
        tuple: Tuple containing arrays for number density (n), radius (r), upper boundary (r_up),
               lower boundary (r_low), and layer thickness (dr).
    '''

    # Store number of layers for convenience
    N_layers = len(P)

    # Initialise 3D radial profile arrays    
    r = np.zeros((N_layers))
    r_up = np.zeros((N_layers))
    r_low = np.zeros((N_layers))
    dr = np.zeros((N_layers))
    n = np.zeros((N_layers))

    log_P = np.log(P)
    
    P_ref = np.power(10, P_ref)  
    
    # Compute number density in each atmospheric layer (ideal gas law)
    n[:] = (P*1.0e5)/((sc.k)*T[:])   # 1.0e5 to convert bar to Pa

    # Set reference pressure and reference radius (r(P_ref) = R_p_ref)
    P_0 = P_ref      # 10 bar default value
    r_0 = R_p_ref    # Radius at reference pressure

    # Find index of pressure closest to reference pressure (10 bar)
    ref_index = np.argmin(np.abs(P - P_0))

    # Set reference radius
    r[ref_index] = r_0

    # Compute integrand for hydrostatic calculation
    integ = (sc.k * T[:])/(R_p**2 * g * mu[:])

    # Initialise stored values of integral for outwards and inwards sums
    integral_out = 0.0
    integral_in = 0.0

    # Working outwards from reference pressure
    for i in range(ref_index+1, N_layers, 1):
    
        integral_out += 0.5 * (integ[i] + integ[i-1]) * (log_P[i] - log_P[i-1])  # Trapezium rule integration
    
        r[i] = 1.0/((1.0/r_0) + integral_out)
    
    # Working inwards from reference pressure
    for i in range((ref_index-1), -1, -1):   
    
        integral_in += 0.5 * (integ[i] + integ[i+1]) * (log_P[i] - log_P[i+1])   # Trapezium rule integration
    
        r[i] = 1.0/((1.0/r_0) + integral_in)

    # Use radial profile to compute thickness and boundaries of each layer
    for i in range(1, N_layers-1): 
    
        r_up[i] = 0.5*(r[(i+1)] + r[i])
        r_low[i] = 0.5*(r[i] + r[(i-1)])
        dr[i] = 0.5 * (r[(i+1)] - r[(i-1)])
    
    # Edge cases for bottom layer and top layer    
    r_up[0] = 0.5*(r[1] + r[0])
    r_up[(N_layers-1)] = r[(N_layers-1)] + 0.5*(r[(N_layers-1)] - r[(N_layers-2)])

    r_low[0] = r[0] - 0.5*(r[1] - r[0])
    r_low[(N_layers-1)] = 0.5*(r[(N_layers-1)] + r[(N_layers-2)])

    dr[0] = (r[1] - r[0])
    dr[(N_layers-1)] = (r[(N_layers-1)] - r[(N_layers-2)])
        
    return  n, r, r_up, r_low, dr





@njit
def path_distribution(b, N_layers, r_up, r_low, dr):
   '''
    Compute the path distribution through the atmosphere for each impact parameter.

    Args:
        b (numpy array): Impact parameter array.
        N_layers (int): Number of atmospheric layers.
        r_up (numpy array): Upper boundary array for each layer (m).
        r_low (numpy array): Lower boundary array for each layer (m).
        dr (numpy array): Layer thickness array (m).

    Returns:
        numpy array: Path distribution array for each impact parameter and atmospheric layer.
   ''' 
   
   N_b = b.shape[0]   
   path = np.zeros((N_b, N_layers))
   
   
   r_up2 = r_up*r_up   
   r_low2 = r_low*r_low  
   b_2 = b*b
   
   for i in range(N_b):
       
       for l in range( N_layers):
           
           if (b[i] < r_up[l]):
               
               x1 = np.sqrt(r_up2[l] - b_2[i])
               
               if (b[i] > r_low[l]):
                   
                   x2 = 0.0
                    
               else:
                   
                   x2 = np.sqrt(r_low2[l] - b_2[i])
                   
               path[i,l] = 2 * (x1 -x2)/dr[l]
               
           else:
               
               path[i,l] = 0.0
   
   
   return path

    
'''    
# case where the ray passes between r_up and r_low
def path_distribution(b, N_layers, r_up, r_low, dr):

 
   
   N_b = b.shape[0]
   
   path = np.zeros((N_b, N_layers))
   
   i_deep = 0
   
   r_up2 = r_up*r_up
      
   b_2 = b*b
   
   for i in range(N_b):
       
       for l in range(i_deep, N_layers):
            
           if (r_up[l] > b[i] > r_low[l]): 
               
              path[i,l] = np.sqrt(r_up2[l] - b_2[i])/dr[l]

   
   return path
'''      

