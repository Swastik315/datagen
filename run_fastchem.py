'''
Function to call Fastchem
'''
import numpy as np
import pyfastchem
from astropy import constants as const
import os
from save_output import saveChemistryOutputPandas
import pickle


def run_fastchem(temperature, pressure, c_to_o, metallicity, fastchem_species):
    
    '''
    The pressure and temperatures are numpy arrays and should be of same size. One to One Corres-
    pondence between temperature and pressure.
    
    Code Author: Tonmoy Deka
    
    '''
    
    #mixing_ratios = np.zeros(len(pressure), len(fastchem_species))
    
    c_to_o = np.full(len(pressure), c_to_o)
    metallicity = np.full(len(pressure), metallicity)
    
    fastchem = pyfastchem.FastChem(
        '/home/seps05/fastchem/input/element_abundances/asplund_2020.dat',
        '/home/seps05/fastchem/input/logK/logK.dat',
        1)
    
    output_dir = '/home/seps05/fastchem/output_deka'
    
    solar_abundances = np.array(fastchem.getElementAbundances())

    nb_points = metallicity.size
    number_densities = np.zeros((nb_points, fastchem.getGasSpeciesNumber()))
    total_element_density = np.zeros(nb_points)
    mean_molecular_weight = np.zeros(nb_points)
    element_conserved = np.zeros((nb_points, fastchem.getElementNumber()), dtype=int)
    fastchem_flags = np.zeros(nb_points, dtype=int)
    nb_iterations = np.zeros(nb_points, dtype=int)
    nb_chemistry_iterations = np.zeros(nb_points, dtype=int)
    nb_cond_iterations = np.zeros(nb_points, dtype=int)

    # Get indices for O and C from FastChem
    index_C = fastchem.getElementIndex('C')
    index_O = fastchem.getElementIndex('O')

    for i in range(0, nb_points):
        
        element_abundances = np.copy(solar_abundances)
        
        #scale the element abundances, except those of H and He
        for j in range(0, fastchem.getElementNumber()):
            
           if fastchem.getElementSymbol(j) != 'H' and fastchem.getElementSymbol(j) != 'He':
               
              element_abundances[j] *= metallicity[i]
              
              element_abundances[index_C] = element_abundances[index_O] * c_to_o[i]

              fastchem.setElementAbundances(element_abundances)
              
              #create the input and output structures for FastChem
              input_data = pyfastchem.FastChemInput()
              output_data = pyfastchem.FastChemOutput()
               
              # Loop over temperature and pressure combinations
        
              temp = temperature[i]
              pres = pressure[i]
          
              input_data.temperature = [temp]
              input_data.pressure = [pres]
    
              fastchem_flag = fastchem.calcDensities(input_data, output_data)
    
              # Copy FastChem input and output into pre-allocated arrays
              number_densities[i, :] = np.array(output_data.number_densities[0])
              total_element_density[i] = output_data.total_element_density[0]
              mean_molecular_weight[i] = output_data.mean_molecular_weight[0]
              element_conserved[i, :] = output_data.element_conserved[0]
              fastchem_flags[i] = output_data.fastchem_flag[0]
              nb_iterations[i] = output_data.nb_iterations[0]
              nb_chemistry_iterations[i] = output_data.nb_chemistry_iterations[0]
              nb_cond_iterations[i] = output_data.nb_cond_iterations[0]
            
    #convergence summary report
    print("FastChem reports:")
    print("  -", pyfastchem.FASTCHEM_MSG[np.max(fastchem_flag)])

    if np.amin(output_data.element_conserved) == 1:
      print("  - element conservation: ok")
    else:
      print("  - element conservation: fail")

    #check if output directory exists
    #create it if it doesn't
    os.makedirs(output_dir, exist_ok=True)
    
    gas_number_density = pressure*1e6 / (const.k_B.cgs * temperature)
    
    gas_number_density = gas_number_density.reshape(-1,1)

    #this would save the output of all species
    #here, the data is saved as a pandas DataFrame inside a pickle file
    saveChemistryOutputPandas(output_dir + '/chemistry.pkl', 
                         temperature, pressure,
                         total_element_density, 
                         mean_molecular_weight, 
                         number_densities,
                         fastchem, 
                         None, 
                         c_to_o, 'C/O')
    
    with open('/home/seps05/fastchem/output_deka/chemistry.pkl', 'rb') as f:
        data =pickle.load(f)
        
     # Extracting the columns for the species specified in fastchem_species
    extracted_densities = np.zeros((nb_points, len(fastchem_species)))
    for i, species in enumerate(fastchem_species):
        if species in data.columns:
            extracted_densities[:, i] = data[species].to_numpy()
        else:
            print(f"Species {species} not found in saved data")
            
    
    mixing_ratio = extracted_densities
    
    return mixing_ratio


'''
# Test EXAMPLE.

temperature = np.array([1.63E+03,
1.64E+03,
1.65E+03,
1.66E+03,
1.67E+03,
1.68E+03,
1.69E+03,
1.70E+03,
1.72E+03,
1.72E+03,
1.73E+03,
1.74E+03,
1.75E+03,
1.76E+03,
1.77E+03,
1.78E+03,
1.79E+03,
1.80E+03,
1.81E+03,
1.82E+03,
1.84E+03,
1.85E+03,
1.87E+03,
1.89E+03,
1.90E+03,
1.92E+03,
1.93E+03,
1.95E+03,
1.96E+03,
1.97E+03,
2.00E+03,
2.02E+03,
2.05E+03,
2.08E+03,
2.12E+03,
2.14E+03,
2.17E+03,
2.22E+03,
2.26E+03,
2.29E+03,
2.32E+03,
2.34E+03,
2.37E+03,
2.39E+03,
2.41E+03,
2.44E+03,
2.45E+03,
2.47E+03,
2.49E+03,
2.50E+03,
2.52E+03,
2.54E+03,
2.57E+03,
2.60E+03,
2.65E+03,
2.67E+03,
2.70E+03,
2.72E+03,
2.76E+03,
2.79E+03,
2.82E+03,
2.84E+03,
2.89E+03,
2.91E+03,
2.96E+03,
2.98E+03,
3.02E+03,
3.06E+03,
3.11E+03,
3.14E+03,
3.17E+03,
3.20E+03,
3.25E+03,
3.30E+03,
3.32E+03,
3.35E+03,
3.41E+03,
3.44E+03,
3.49E+03,
3.54E+03,
3.59E+03,
3.65E+03,
3.70E+03,
3.75E+03,
3.79E+03,
3.85E+03,
3.89E+03,
3.93E+03,
3.97E+03,
3.98E+03,
])


pressure = np.array([1.08E-03,
1.79E-03,
2.00E-03,
2.26E-03,
2.73E-03,
3.04E-03,
3.25E-03,
3.55E-03,
4.10E-03,
4.57E-03,
4.99E-03,
5.51E-03,
5.96E-03,
6.65E-03,
7.26E-03,
8.20E-03,
9.05E-03,
9.88E-03,
1.08E-02,
1.25E-02,
1.41E-02,
1.62E-02,
1.87E-02,
2.18E-02,
2.44E-02,
2.66E-02,
2.91E-02,
3.43E-02,
3.79E-02,
4.27E-02,
4.83E-02,
5.45E-02,
6.08E-02,
6.87E-02,
7.75E-02,
8.37E-02,
9.14E-02,
1.06E-01,
1.17E-01,
1.29E-01,
1.41E-01,
1.50E-01,
1.64E-01,
1.73E-01,
1.81E-01,
2.00E-01,
2.11E-01,
2.21E-01,
2.33E-01,
2.44E-01,
2.58E-01,
2.75E-01,
2.97E-01,
3.32E-01,
3.83E-01,
4.09E-01,
4.47E-01,
4.77E-01,
5.27E-01,
5.76E-01,
6.36E-01,
6.87E-01,
7.84E-01,
8.56E-01,
9.77E-01,
1.04E+00,
1.19E+00,
1.33E+00,
1.54E+00,
1.66E+00,
1.81E+00,
1.98E+00,
2.31E+00,
2.61E+00,
2.82E+00,
3.01E+00,
3.55E+00,
3.88E+00,
4.42E+00,
4.99E+00,
5.83E+00,
6.72E+00,
7.59E+00,
8.57E+00,
9.46E+00,
1.08E+01,
1.18E+01,
1.32E+01,
1.44E+01,
1.47E+01])

# or

temperature = np.full(100,1357.63)
pressure = np.linspace(1.0e-7, 100,100)


c_to_o =  0.55
metallicity =  1.023
fastchem_species = [ 'H2O1','C1O2','C1O1','C1H4']
mixo = run_fastchem(temperature, pressure, c_to_o, metallicity, fastchem_species)


import matplotlib.pyplot as plt
plt.xscale('log')
plt.yscale('log')
plt.plot(mixo[:,0],pressure, label = 'H2O')
plt.plot(mixo[:,1], pressure, label='CO2')
plt.plot(mixo[:,2],pressure, label='CO')
plt.plot(mixo[:,3],pressure, label='CH4')
plt.gca().invert_yaxis()  # Flip the y-axis
plt.legend()
plt.plot()
'''
















