from numba import njit

@njit
def closest_index(value, grid_start, grid_end, N_grid):
    '''
    Same as 'prior_index_V2', but for the closest index (i.e. can also round up).

    Args:
        val (float): 
            The value for which closest index is desired.
        grid_start (float):
            The value at the left edge of the uniform grid (array[0]).
        grid_start (float):
            The value at the right edge of the uniform grid (array[-1]).
        N_grid (int):
            The number of points on the uniform grid.

    Returns:
        (int):
            The index of the uniform grid closest to 'value'.

    '''

    # Single value grids only have one element, so return 0
    if (N_grid == 1):
        return 0

    # Set to lower boundary
    if (value < grid_start): 
        return 0
    
    # Set to upper boundary
    elif (value > grid_end):
        return N_grid-1
    
    # Use the equation of a straight line, then round to nearest integer.
    else:
        i = (N_grid-1) * ((value - grid_start) / (grid_end - grid_start))
        if ((i%1) <= 0.5):
            return int(i)     # Round down
        else:
            return int(i)+1   # Round up
        


@njit
def prior_index_V2(value, grid_start, grid_end, N_grid):
    ''' 
    Find the previous index of a *uniformly spaced* grid closest to a specified 
    value. When a uniform grid can be assumed, this function is much faster 
    than 'prior_index' due to there being no need for a loop. However, 
    for non-uniform grids one should still default to 'prior_index'.
    This function assumes the input grid monotonically increases.

    Args:
        value (float):
            The value for which the prior grid index is desired.
        grid_start (float):
            The value at the left edge of the uniform grid (array[0]).
        grid_start (float):
            The value at the right edge of the uniform grid (array[-1]).
        N_grid (int):
            The number of points on the uniform grid.

    Returns:
        (int):
            Prior index of the grid corresponding to the value.

    '''
    
    # Set to lower boundary
    if (value < grid_start):
        return 0
    
    # Set to upper boundary
    elif (value > grid_end):
        return N_grid-1
    
    # Use the equation of a straight line, then round down to integer.
    else:
        i = (N_grid-1) * ((value - grid_start) / (grid_end - grid_start))
        return int(i)



            
            
            
@njit           
def prior_index(value, grid, start = 0):
    ''' 
    Search a grid to find the previous index closest to a specified value (i.e. 
    the index of the grid where the grid value is last less than the value). 
    This function assumes the input grid monotonically increases.

    Args:
        value (float):
            Value for which the prior grid index is desired.
        grid (np.array of float):
            Input grid.
        start (int):
            Optional start index when existing knowledge is available.

    Returns:
        index (int):
            Prior index of the grid corresponding to the value.

    '''
        
    if (value > grid[-1]):
        return (len(grid) - 1)
    
    # Check if value out of bounds, if so set to edge value
    if (value < grid[0]): value = grid[0]
    if (value > grid[-2]): value = grid[-2]
    
    index = start
    
    for i in range(len(grid)-start):
        if (grid[i+start] > value): 
            index = (i+start) - 1
            break
            
    return index

@njit        
def closest_index2(value, array_min, array_max, array_size):
    """
    Find the index of the closest value in an array.

    Parameters:
    - value: The value to find the closest index for.
    - array_min: The minimum value of the array.
    - array_max: The maximum value of the array.
    - array_size: The size of the array.

    Returns:
    - closest_idx: The index of the closest value in the array.
    """
    # Calculate the step size
    step_size = (array_max - array_min) / (array_size - 1)

    # Find the index of the closest value
    closest_idx = int((value - array_min) / step_size + 0.5)

    # Ensure the index is within valid bounds
    closest_idx = max(0, min(array_size - 1, closest_idx))

    return closest_idx
