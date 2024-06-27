
import numpy as np
import h5py
from chemical_species import chem_species
from closest_index import closest_index, prior_index_V2
from interpolation_package import interp_P, interp_T, intp_sigma, \
                                  interpolate_temperature_sigma, weight_factors, T_interpolate_sigma,\
                                      T_interpolation_init



def read_sigmas( input_path, chem_species, T_fine, log_P_fine, wl_model, interp):
     
    '''
   Function to read the opacity database file for chemical species.
   
   Reads the HDF file. Outputs temperature, pressure, wavelength, and cross-section
   grids related to the chemical species.
   
   Args:
       chem_species (list): List of chemical species to read from the database.
       T_fine (numpy array): Fine grid of temperatures.
       log_P_fine (numpy array): Fine grid of log pressures.
       wl_model (numpy array): Wavelength array for the model.
       interp (str): Interpolation method ('cubic' or 'opacity_sampling').
       
   Returns:
       numpy array: 4D array containing cross-section data for each species.
   '''
    nu_model = 1.0e4/wl_model
    
    N_T_fine, N_P_fine, N_species, N_wl = len(T_fine), len(log_P_fine), len(chem_species), len(wl_model)
    N_nu = len(nu_model)
    
    if input_path == None:
    
        raise Exception(" Insert correct path for opacity in function 'read_sigmas'")
    
    sigma_file  = h5py.File(input_path + 'opacity_for_WASP-39b.hdf5', 'r')
    
    sigma = np.zeros((N_species, N_P_fine, N_T_fine, N_wl))
    
 
    for q, spec_q in enumerate(chem_species):

        # Read the grids in the opacity datset of the species q
        
        T_q = np.array(sigma_file[spec_q + '/T'])
        
        log_P_q = np.array(sigma_file[spec_q + '/log(P)'])
        
        nu_q = np.array(sigma_file[spec_q + '/nu'])
        
        log_sigma_q = np.array(sigma_file[spec_q + '/log(sigma)'], dtype=np.float32)
        
        N_T = len(T_q)
    
        N_P = len(log_P_q)
    
        
        if interp == 'cubic':
        
            sigma_interpolat_P =  interp_P(wl_model, log_P_fine, nu_q, log_sigma_q)
        
            del log_sigma_q
            
            sigma[q,:,:,:] = interp_T(sigma_interpolat_P, T_fine, T_q)
            
            del sigma_interpolat_P, nu_q
            
        elif interp == 'opacity_sampling':
            
            y = np.zeros(N_T_fine, dtype=np.int64)
            
            w_T = T_interpolation_init(N_T_fine, T_q, T_fine, y)
            
            x,b1,b2 = weight_factors(log_P_fine)
                  
            nu_model = nu_model[::-1] # from small to big conversion.
  
            sigma_pre_inp = intp_sigma(N_P_fine, N_T, N_P, N_wl, log_sigma_q,
                        x, nu_model, b1, b2, nu_q, N_nu, T_q)
            
            del log_sigma_q
              
            sigma[q,:,:,:] = T_interpolate_sigma(N_P_fine, N_T_fine, N_T, N_wl, sigma_pre_inp, T_q, T_fine, y, w_T)

    sigma_file.close()    
       
    return sigma




def wl_grid(wl_min, wl_max, R):
    
    '''
    Model wavelength grid.
    
    Args:
        wl_min (float): Minimum wavelength for the spectrum calculation.
        wl_max (float): Maximum wavelength for the spectrum calculation.
        R (float): Wavelength resolution (lambda/delta.lambda).
        
    Returns:
        numpy array: Wavelength grid.
    '''

    # Constant R -> uniform in log(wl)
    
    delta_log_wl = 1.0/R
    N_wl = int(np.round((np.log(wl_max) - np.log(wl_min)) / delta_log_wl))  # Ensure N_wl is an integer
    log_wl = np.linspace(np.log(wl_min), np.log(wl_max), N_wl)    

    wl = np.exp(log_wl)
    
    return wl   


def generate_fine_temperature_grid(T_fine_min, T_fine_max, T_fine_step):
    return np.arange(T_fine_min, T_fine_max + T_fine_step, T_fine_step)

def generate_fine_pressure_grid(log_P_fine_min, log_P_fine_max, log_P_fine_step):
    return np.arange(log_P_fine_min, log_P_fine_max + log_P_fine_step, log_P_fine_step)


def generate_pressure_grid(P_min, P_max, N_layers):
    """
    Generate a pressure grid using logarithmic spacing.

    Parameters:
    - P_min (float): Minimum pressure in bar.
    - P_max (float): Maximum pressure in bar.
    - N_layers (int): Number of layers.

    Returns:
    - P (ndarray): Pressure grid.
    """
    P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)
    return P



def calc_kappa(T, P, wl, n, chem_species, X, sigma, T_fine, log_P_fine):
    
    '''
   Calculating extinction coefficient 'kappa' for molecules and atoms.

   Args:
       T (numpy array): Temperature profile array.
       P (numpy array): Atmospheric pressure layer array.
       wl (numpy array): Wavelength grid.
       n (numpy array): Number density array.
       chem_species (list): List of chemical species names.
       X (numpy array): Mixing ratio profile array.
       sigma (numpy array): Cross-section array.
       T_fine (numpy array): Fine temperature array for interpolation.
       log_P_fine (numpy array): Fine logarithmic pressure array for interpolation.
       P_deep (float): Deep pressure below which the atmosphere is opaque. Default is 1000.0.

   Returns:
       numpy array: Extinction coefficient kappa array.
   '''
        
    kappa = np.zeros((len(P), len(wl)))       
    N_T_fine = len(T_fine)    
    N_P_fine = len(log_P_fine)
    
    for l in range(len(P)):
        
        n_tot = n[l]
        
        # Find closest index in fine temperature array to given layer temperature
        idx_T_fine = closest_index(T[l], T_fine[0], T_fine[-1], N_T_fine)
        
        idx_P_fine = closest_index(np.log10(P[l]), log_P_fine[0], log_P_fine[-1], N_P_fine)
        
        #---------------- Absorption coeffecient for Chemical Species-------------------------
        for q in range(len(chem_species)):
            
            # number density of a particular species
            n_q = n_tot * X[l,q]
            
            for i in range(len(wl)):
                
                kappa[l,i] += n_q * sigma [q, idx_P_fine, idx_T_fine, i]
    
     
    return kappa
 

def calc_kappa_aero(wl_model, P, aero_species, n, X_aero, sigma_aero):
    '''
    

    Parameters
    ----------
    wl_model : TYPE
        model wavelength array.
    P : TYPE
        Atmospheric layer pressure array.
    aero_species : TYPE
        aersol species array(str).
    n : TYPE
        number density array.
    X_aero : TYPE
        Aerosol mixing ratio array.
    sigma_aero : TYPE
        aerosol extinction cross-section.

    Returns
    -------
    kappa_aero : TYPE
        Aerosol extinction coeffecient.

    '''
    
    kappa_aero = np.zeros((len(P),len(wl_model)))
    
    for l in range(len(P)):
        
        n_l = n[l]
    
        for q in range(len(aero_species)):
            
            n_q = n_l * X_aero[l,q]
        
            for i in range(len(wl_model)):
                
                kappa_aero[l,i] += n_q * sigma_aero[i,q]
                
    return kappa_aero
                  
                
# calculating extinction coeffecients for H2 only rayleigh scattering
def kappa_rayleigh(P, wl_model, sigma_rayleigh, n, X_rayleigh):
    '''
  Calculate Rayleigh scattering extinction coefficient kappa_ray.

  Args:
      P (numpy array): Atmospheric pressure layer array.
      wl_model (numpy array): Wavelength grid.
      sigma_rayleigh (numpy array): Rayleigh scattering cross-section array.
      n (numpy array): Number density array.
      X_rayleigh (numpy array): Mixing ratio profile array for Rayleigh species.

  Returns:
      numpy array: Extinction coefficient kappa_ray for Rayleigh scattering.
  '''
    
    
    kappa_ray = np.zeros((len(P), len(wl_model)))
    
    for l in range(len(P)):
        
        n_l = n[l]
        
        for i in range(len(wl_model)):
            
            n_q = n_l * X_rayleigh[l,0]
            
            kappa_ray[l,i] = n_q * sigma_rayleigh[i]
            
    return kappa_ray



# calculating extinction coeffecients for both H2 and He rayleigh scattering
def calculate_kappa_rayleigh(P, wl_model, sigma_rayleigh, n, X_rayleigh, rayleigh_species):
    
    '''
    Calculate Rayleigh scattering extinction coefficient kappa_rayleigh.

    Args:
        P (numpy array): Atmospheric pressure layer array (bar).
        wl_model (numpy array): Wavelength grid.
        sigma_rayleigh (numpy array): Rayleigh scattering cross-section array.
        n (numpy array): Number density array.
        X_rayleigh (numpy array): Mixing ratio profile array for Rayleigh species.
        rayleigh_species (list): List of Rayleigh species names ('H2' and/or 'He').

    Returns:
        numpy array: Extinction coefficient kappa_rayleigh for Rayleigh scattering.
    '''
    
    kappa_H2 = np.zeros((len(P), len(wl_model)))
    kappa_He = np.zeros((len(P), len(wl_model)))
    kappa_ray = np.zeros((len(P), len(wl_model)))

    
    for l in range(len(P)):
        
        n_l = n[l]
           
       # for q in range(len(rayleigh_species)):
        if 'H2' in rayleigh_species:
            
            n_q = n_l * X_rayleigh[l,0]
            # H2 is the first species.
            
            for i in range(len(wl_model)):
                kappa_H2[l,i] += n_q * sigma_rayleigh[i,0]
                
        if 'He' in rayleigh_species:
            
            n_q = n_l * X_rayleigh[l,1]
            # He is the second species.
            
            for i in range(len(wl_model)):
                kappa_H2[l,i] += n_q * sigma_rayleigh[i,1]
                
    kappa_ray = kappa_H2 + kappa_He
                 
            
    return kappa_ray
        
    
      

# calculating extinction coeffecients for collisionally induced absorption

def kappa_CIA(X_rayleigh, P, n, wl_model, sigma_cia, T, T_fine, cia_species = ['H2-H2','H2-He']):
    
    '''
   Calculate CIA scattering extinction coefficient kappa_cia.

   Args:
       X_rayleigh (numpy array): Mixing ratio profile array for Rayleigh species.
       P (numpy array): Atmospheric pressure layer array.
       n (numpy array): Number density array.
       wl_model (numpy array): Wavelength grid.
       sigma_cia (numpy array): CIA scattering cross-section array.
       T (numpy array): Temperature array.
       T_fine (numpy array): Fine temperature grid array.
       P_deep (float): Deep pressure below which atmosphere is opaque (default: 1000.0).

   Returns:
       numpy array: Extinction coefficient kappa_cia for CIA scattering.
   '''
    
    
    N_T_fine = len(T_fine)
    
    kappa_cia = np.zeros((len(P), len(wl_model)))
     
    for l in range(len(P)):   
        
        n_tot = n[l]
        
        idx_T_fine = closest_index(T[l], T_fine[0], T_fine[-1], N_T_fine)
 
        for species in cia_species:
            
            if species=='H2-H2':
                
                                    
                n_1 = n_tot*X_rayleigh[l,0]                                  
                n_2 = n_tot*X_rayleigh[l,0]
                
                n_n1 = n_1*n_2

                 
            elif species=='H2-He':
                
                n_1 = n_tot*X_rayleigh[l,0]   
                n_2 = n_tot*X_rayleigh[l,1]
                
                n_n2 = n_1*n_2

                
             # For each wavelength
        for q in range(len(cia_species)):
            
            for i in range(len(wl_model)):
                
                n_n = n_n1 + n_n2
                                            
                kappa_cia[l,i] += n_n * sigma_cia[q, idx_T_fine, i]    
                 
              
    return kappa_cia                                               


    
     
def tau_vert(P, N_wl, dr, kappa):
    
    '''
   Calculate vertical optical depth.

   Args:
       P (numpy array): Atmospheric pressure layer array.
       N_wl (int): Number of wavelengths.
       dr (numpy array): Thickness of each layer.
       kappa (numpy array): Extinction coefficient array.
       kappa_cloud (numpy array): Extinction coefficient due to clouds array (optional).
       clouds (bool): Flag indicating if clouds are present (default: False).

   Returns:
       numpy array: Vertical optical depth array.
   '''
    tau_vert = np.zeros(shape=(len(P), N_wl))
    
   # if cloud=='on':
             
     #   for i in range(N_wl):
                      
     #       tau_vert[:,i] = ((kappa[:,i] + kappa_cloud[i,:])) * dr[:]
    
   # else:
        
    for i in range(N_wl):
            
        tau_vert[:,i] = kappa[:,i] * dr[:]
            
    return tau_vert
 



def transit_depth(wl, R_p, tau_vert, path, r_up, r_low, dr, R_s):
    
    '''
    Main function to calculate transit depth by solving the equation 
    of radiative transfer.
    '''
    
    # factor to include both hemispheres edges.
    phi_edges = np.array([-np.pi/2.0, np.pi/2.0])
    North_phi_edge = np.pi/2.0 + phi_edges[:-1]
    South_phi_edge = (-1.0*North_phi_edge)[::-1] + 2.0*np.pi
    
    phi_appended = np.append(North_phi_edge, South_phi_edge)
    phi_sorted = np.sort(phi_appended)
    dphi = np.diff(phi_sorted)
    
    
    b = r_up[:]  
    db = dr [:] 
    
    N_b = b.shape[0]
        
    transit_depth = np.zeros((len(wl)))   
    Transmission=np.zeros((N_b, len(wl)))
    
    R_max = r_up[-1] # maximal radial extent
    
    R_ptop = R_max * R_max   
    A_top = np.pi * R_ptop
    
    R_s2 = R_s * R_s
        
    Transmission= np.exp(-1.0*np.tensordot(path,tau_vert,axes=([1],[0])))      
    Area_atm = np.outer((b*db), dphi)
    atmos = np.tensordot(Transmission, Area_atm, axes=([0], [0]))       
    transit_depth = (A_top - atmos)/(np.pi * R_s2)
    
    return transit_depth
    
    
def NIRTRAN(tau_vert, path, R_p, r_up, r_low, dr, R_s, N_b, N_wl):
    """
   Calculate transit depths as a function of wavelength.

   Parameters:
       tau_vert : array
           Vertical optical depth.
       b : array
           Impact parameter array.
       path : array
           Path distribution array.
       R_p : float
           Planetary radius.
       r_up : array
           Radius of the topmost layer.
       dr : array
           Thickness of an atmospheric shell.
       R_s : float
           Stellar radius.
       N_b : int
           Number of impact parameter points.
       N_wl : int
           Number of wavelength points.

   Returns:
       array
           Transit depths as a function of wavelength.
   """
    R_s_squared = R_s*R_s    
    Trans = np.zeros(shape=(N_b, N_wl))   
    tr = np.zeros((N_wl))  
    b = (r_up[:]+r_low[:])/2
 
    db = dr[:]
 
    Area_atm = (b*db)
    
    Trans[:,:] = np.exp(-1.0*np.tensordot(path, tau_vert, axes=([1],[0])))
    
    x = 2*(1-Trans)   
    y= np.tensordot(x, Area_atm, axes=([0],[0]))
    tr =   (R_p**2 + y)/R_s_squared
     
    return tr


# Some Constant Values
R_J = 7.1492e7
R_sun = 6.957e8

























