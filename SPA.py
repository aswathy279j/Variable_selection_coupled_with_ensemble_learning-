import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def successive_projections_algorithm(X, n_select):
    """
    Successive Projections Algorithm for variable selection to minimize collinearity.
    
    Parameters:
        X : ndarray (samples x variables)
            Feature matrix
    
        n_select : int
            Number of variables to select
    
    Returns:
        selected_vars : list of int
            Indices of selected variables
    """
    n_samples, n_vars = X.shape
    selected_vars = []
    
    # Choose initial variable as the one with largest norm
    norms = np.linalg.norm(X, axis=0)
    first_var = np.argmax(norms)
    selected_vars.append(first_var)
    
    # Remaining variables which are not selected 
    remaining_vars = list(set(range(n_vars)) - set(selected_vars))
    
    for _ in range(1, n_select):
        proj_norms = []
        for var in remaining_vars:
            # Project variable vector onto subspace orthogonal to selected variables
            x_var = X[:, var]
            selected_matrix = X[:, selected_vars]
            
            # Projection matrix of selected variables
            P = selected_matrix @ np.linalg.pinv(selected_matrix)
            proj = x_var - P @ x_var
            
            proj_norms.append(np.linalg.norm(proj))
        
        # Select variable with maximum projection norm
        max_proj_index = np.argmax(proj_norms)
        next_var = remaining_vars[max_proj_index]
        selected_vars.append(next_var)
        remaining_vars.remove(next_var)
        
    return selected_vars

def spa_workflow(csv_file, n_select=10):
    """Load data, run SPA variable selection, and save outputs"""
    data = pd.read_csv(csv_file)
    X = data.iloc[:, 2:].values
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]
    
    print(f"Running SPA to select {n_select} variables...")
    selected_vars = successive_projections_algorithm(X, n_select)
    print(f"Selected variables: {selected_vars}")
    
    # Save selected variable indices and column names
    var_names = data.columns[2:]
    selected_names = var_names[selected_vars]
    df_vars = pd.DataFrame({'selected_index': selected_vars, 'wavenumber': selected_names})
    vars_filename = f"SPA_SelectedVars_{base_filename}.xlsx"
    df_vars.to_excel(vars_filename, index=False)
    print(f"Selected variables saved to '{vars_filename}'")
    
    # Save filtered dataset with selected variables + identifiers
    filtered_data = pd.concat([data.iloc[:, :2], data.iloc[:, 2:].iloc[:, selected_vars]], axis=1)
    filtered_filename = f"SPA_FilteredData_{base_filename}.xlsx"
    filtered_data.to_excel(filtered_filename, index=False)
    print(f"Filtered dataset saved to '{filtered_filename}'")

# load data # replace with your filename # choose number of variables to select
spa_workflow('myfilename.csv', n_select=100)