import matplotlib.pyplot as plt 
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd


# Binning high resolution spectrum to low resolution
def bin_transit_depths(df, original_resolution, new_resolution):
    # Calculate the bin size
    bin_size = original_resolution / new_resolution

    # Create lists to store binned data
    binned_wl = []
    binned_depth = []

    # Iterate over each bin
    for bin_start in range(0, len(df), int(bin_size)):
        # Calculate the bin end
        bin_end = min(bin_start + int(bin_size), len(df))

        # Extract the transit depths within the current bin
        depths_in_bin = df['tr_depth'].iloc[bin_start:bin_end]

        # Calculate the average depth within the bin
        average_depth = depths_in_bin.mean()

        # Take the wavelength of the start of the bin
        bin_wavelength = df['wl'].iloc[bin_start]

        # Append binned depth and wavelength to the lists
        binned_wl.append(bin_wavelength)
        binned_depth.append(average_depth)

    # Create a DataFrame from the lists
    binned_df = pd.DataFrame({'wl': binned_wl, 'binned_depth': binned_depth})

    return binned_df



def plot_spectra(wl_model, spectrum, R=None, R_bin=None, data=None):
    
    '''
    Plotting function with which you can plot the high resolution data in original R value or 
    also plot a binned spectrum at R_bin on top of the high resolution data
    
    plot_spectra(wl_model,spectrum,R) will plot the high res spectrum only
    plot_spectra(wl_model, spectrum,R,R_bin=100) will plot a binned spectrum of res 100 on top of 
    the high res spectrum.
    
    '''
    
    spectrum_high_res = spectrum[:, 0]  # Original high-resolution spectrum
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)  # Create figure and axis objects

    
    if R is not None and R_bin is not None:
        # Bin the transit depths if both original and new resolutions are provided
        original_transit_depths_df = pd.DataFrame({'wl': wl_model, 'tr_depth': spectrum_high_res})
        binned_transit_depths_df = bin_transit_depths(original_transit_depths_df, R, R_bin)
        wl_model_binned = binned_transit_depths_df['wl']
        spectrum_binned = binned_transit_depths_df['binned_depth']

        ax.plot(wl_model, spectrum_high_res, color='violet', alpha=0.5, label='R='+str(R))
        ax.plot(wl_model_binned, spectrum_binned, color='cornflowerblue', label='Binned (R=' + str(R_bin) + ')')
        
    elif R is not None:
        # Plot only the high-resolution spectra
        ax.plot(wl_model, spectrum_high_res, color='cornflowerblue')

    elif R_bin is not None:
        # Binning resolution is provided but original resolution is not
        print("Please provide the original resolution (R) for binning.")
        return

    else:
        # No resolution provided
        print("Please provide either the original resolution (R) or the binning resolution (R_bin).")
        return
    if data is not None:
        # Plot observational data if provided
        ax.errorbar(data['wl'], data['tr_depth'], xerr=data['half_width'], yerr=data['error'], fmt='s', markersize=3, capsize=4, color='black',label='Observations')


    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.set_ylabel('Transit depth $(R_p/R_*)^2$', fontsize=15)
    ax.set_xlabel('Wavelength($\mu m$)', fontsize=15)
    #plt.xscale('log')
    ax.legend()
    ax.set_xlim(wl_model[0], wl_model[-1])
    plt.show()



# Example usage:
# original_resolution = 10000
# new_resolution = 100
# plot_spectra(wl_model, spectrum, original_resolution=original_resolution, new_resolution=new_resolution)


# For plotting the data
def plot_data(path_to_data, xlim=None, ylim=None):
    '''
    

    Parameters
    ----------
    path_to_data :
        provides path to data directory.
    xlim : tuple, optional
        x-axis limits. The default is None.
    ylim : 
        y-axis limits. The default is None.

    Returns
    -------
    Figure for the data

    '''
    data = pd.read_csv(path_to_data)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.errorbar(data['wl'], data['tr_depth'], xerr=data['half_width'], yerr=data['error'], fmt='s', markersize=5, capsize=4, color='gray',label='Observations')
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
        
    ax.set_ylabel('Transit depth $(R_p/R_*)^2$', fontsize=15)
    ax.set_xlabel('Wavelength($\mu m$)', fontsize=15)
    
    plt.show()


# simple plotting function
def plot_spectra_alt(wl_model, spectrum):

   # y_padding = 0.07 * (np.max(spectrum) - np.min(spectrum)) 
        
    plt.figure(figsize=(12,8), dpi=300)

    plt.plot(wl_model, spectrum)

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.ylabel('Transit depth $(R_p/R_*)^2$', fontsize=15)

    plt.xlabel('Wavelength($\mu m$)', fontsize=15)
      
    plt.xlim(wl_model[0], wl_model[-1])   

    #plt.ylim(np.min(spectrum) - y_padding, np.max(spectrum) + y_padding)
    
    #plt.xscale('log')
    
    plt.plot(wl_model, spectrum, color = 'dodgerblue')

    plt.plot()

