import os
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt

def random_frog_plsda(X, Y_dummy, n_iter=10000, n_init=5, n_components=10, cv_folds=5, random_state=42):
    """
    Random Frog feature selection adapted for PLS-DA (for classification data).

    Parameters: 

    X : ndarray
        Predictor matrix (samples x variables)
    Y_dummy : ndarray
        Dummy-encoded target matrix (samples x classes)
    n_iter : int
        Number of iterations in MCMC loop
    n_init : int
        Initial number of randomly selected variables
    n_components : int
        Number of PLS components
    cv_folds : int
        Cross-validation folds
    random_state : int
        Random seed

    Returns:
    
    selected_vars : ndarray
        Selected variable indices
    selection_frequencies : ndarray
        Selection frequencies of each variable
    mse_list : list
        Cross-validation errors over iterations
    """
    np.random.seed(random_state)
    n_samples, n_vars = X.shape
    frequencies = np.zeros(n_vars)
    mse_list = []

    # Initialize with a random subset
    current_subset = set(np.random.choice(n_vars, size=min(n_init, n_vars), replace=False))

    def cv_score(subset):
        if len(subset) == 0:
            return np.inf
        X_sub = X[:, list(subset)]
        pls = PLSRegression(n_components=min(n_components, len(subset), X_sub.shape[0]-1))
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(pls, X_sub, Y_dummy, cv=cv, scoring='neg_mean_squared_error')
        return -scores.mean()

    current_score = cv_score(current_subset)

    for iteration in range(n_iter):
        # Propose a new subset by randomly adding/removing one variable
        if np.random.rand() > 0.5 and len(current_subset) < n_vars:
            addition = np.random.choice(list(set(range(n_vars)) - current_subset))
            proposed_subset = current_subset | {addition}
        elif len(current_subset) > 1:
            removal = np.random.choice(list(current_subset))
            proposed_subset = current_subset - {removal}
        else:
            proposed_subset = set(current_subset)

        proposed_score = cv_score(proposed_subset)

        # Metropolis-like acceptance rule
        if (proposed_score <= current_score) or (np.random.rand() < 0.1):
            current_subset = proposed_subset
            current_score = proposed_score

        for var in current_subset:
            frequencies[var] += 1

        mse_list.append(current_score)

    selection_frequencies = frequencies / n_iter

    threshold = np.median(selection_frequencies)
    selected_vars = np.where(selection_frequencies >= threshold)[0]

    return selected_vars, selection_frequencies, mse_list


def random_frog_plsda_workflow(csv_file):
    """
    Random Frog workflow for classification (PLS-DA).
    """
    data = pd.read_csv(csv_file) if csv_file.endswith('.csv') else pd.read_excel(csv_file)
    X = data.iloc[:, 2:].values #extract features
    y_labels = data.iloc[:, 1].values  # extract class labels 
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]

    # Convert categorical y to dummy (numeric) form
    lb = LabelBinarizer()
    Y_dummy = lb.fit_transform(y_labels)
    if Y_dummy.ndim == 1:
        Y_dummy = Y_dummy.reshape(-1, 1)

    print(f"Classes encoded as: {list(lb.classes_)}")
    print(f"Y_dummy shape: {Y_dummy.shape}")

    print("Running Random Frog (PLS-DA) variable selection...")
    selected_vars, frequencies, mse_list = random_frog_plsda(X, Y_dummy)

    print(f"Selected {len(selected_vars)} out of {X.shape[1]} variables")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(mse_list)+1), mse_list, color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Cross-Validation Error (MSE)')
    plt.title('Random Frog (PLS-DA) Performance Over Iterations')
    plt.grid(True)
    plot_file = f"RandomFrog_PLSDA_MSE_{base_filename}.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Saved MSE plot to '{plot_file}'")

    # Save selection frequencies
    freq_df = pd.DataFrame({'wavenumber': data.columns[2:], 'selection_frequency': frequencies})
    freq_df = freq_df.sort_values(by='selection_frequency', ascending=False)
    freq_filename = f"RandomFrog_PLSDA_Frequencies_{base_filename}.xlsx"
    freq_df.to_excel(freq_filename, index=False)
    print(f"Saved selection frequencies to '{freq_filename}'")

    # Save filtered dataset with selected variables
    filtered_data = pd.concat([data.iloc[:, :2], data.iloc[:, 2:].iloc[:, selected_vars]], axis=1)
    filtered_filename = f"RandomFrog_PLSDA_Filtered_{base_filename}.xlsx"
    filtered_data.to_excel(filtered_filename, index=False)
    print(f"Saved filtered dataset to '{filtered_filename}'")

    return selected_vars, frequencies, mse_list


# usage # replace with your filename
selected_vars, frequencies, mse_list = random_frog_plsda_workflow('myfilename.csv')