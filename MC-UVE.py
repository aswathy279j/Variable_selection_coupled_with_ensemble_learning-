import os
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt

def mcuve_plsda(X, Y_dummy, n_iter=100, n_components=10, cv_folds=5,
                calibration_ratio=0.75, random_state=42):
    np.random.seed(random_state)
    n_samples, n_vars = X.shape
    n_cal = int(calibration_ratio * n_samples)

    # Store coefficients: shape (iterations, variables, classes)
    coef_matrix = np.zeros((n_iter, n_vars, Y_dummy.shape[1]))
    mse_list = []

    for i in range(n_iter):
        cal_idx = np.random.choice(n_samples, size=n_cal, replace=False)
        X_cal, Y_cal = X[cal_idx, :], Y_dummy[cal_idx, :]

        pls = PLSRegression(n_components=min(n_components, n_cal - 1))
        pls.fit(X_cal, Y_cal)

        # Store coefficients (variables x classes)
        coef_matrix[i, :, :] = pls.coef_.T

        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state + i)
        scores = cross_val_score(pls, X_cal, Y_cal, cv=cv, scoring='neg_mean_squared_error')
        mse_list.append(-scores.mean())

    # Compute mean and std across iterations per variable and class
    mean_coef = np.mean(coef_matrix, axis=0)  # shape (variables, classes)
    std_coef = np.std(coef_matrix, axis=0, ddof=1)

    # Compute reliability index per variable by RMS over classes
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_per_class = np.divide(mean_coef, std_coef + 1e-10)
        rel_per_class[np.isnan(rel_per_class)] = 0  # Replace NaN with 0

    reliability_index = np.sqrt(np.mean(rel_per_class ** 2, axis=1))  # shape (variables,)

    # Selection threshold: median of absolute reliability 
    threshold = np.median(np.abs(reliability_index))
    selected_vars = np.where(np.abs(reliability_index) >= threshold)[0]

    return selected_vars, reliability_index, mse_list

def mcuve_plsda_workflow(csv_file):
    data = pd.read_csv(csv_file) if csv_file.endswith('.csv') else pd.read_excel(csv_file)
    X = data.iloc[:, 2:].values #extracting features
    y_labels = data.iloc[:, 1].values  #extracting class labels
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]

    # PLS-DA based class encoding for classification data
    lb = LabelBinarizer()
    Y_dummy = lb.fit_transform(y_labels)
    if Y_dummy.ndim == 1:
        Y_dummy = Y_dummy.reshape(-1, 1)
    print("Classes encoded:", list(lb.classes_))
    print(f"Dummy Y shape: {Y_dummy.shape}")

    print("Running MC-UVE variable selection...")
    selected_vars, reliability_index, mse_list = mcuve_plsda(X, Y_dummy,
                                                             n_iter=100,
                                                             n_components=10)

    print(f"Selected {len(selected_vars)} variables out of {X.shape[1]}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(mse_list)+1), mse_list, 'o-', color='b')
    plt.xlabel('Monte Carlo iteration')
    plt.ylabel('Cross-validation MSE')
    plt.title('MC-UVE Stability Progression')
    plt.grid(True)
    plt.tight_layout()
    plot_file = f"MCUVE_MSE_{base_filename}.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Plot saved: {plot_file}")

    var_names = data.columns[2:]
    reliability_df = pd.DataFrame({'wavenumber': var_names,
                                  'reliability_index': reliability_index})
    reliability_df = reliability_df.sort_values(by='reliability_index', ascending=False)

    reliab_file = f"MCUVE_ReliabilityIndex_{base_filename}.xlsx"
    reliability_df.to_excel(reliab_file, index=False)
    print(f"Reliability indices saved: {reliab_file}")

    filtered_data = pd.concat([data.iloc[:, :2], data.iloc[:, 2:].iloc[:, selected_vars]], axis=1)
    filtered_file = f"MCUVE_FilteredData_{base_filename}.xlsx"
    filtered_data.to_excel(filtered_file, index=False)
    print(f"Filtered data saved: {filtered_file}")

    print("\nSummary:")
    print(f"Total variables: {X.shape[1]}")
    print(f"Selected variables: {len(selected_vars)}")
    print(f"Top 5 variables:\n{reliability_df.head()}")

    return selected_vars, reliability_index, mse_list

# Usage #replace with your filename
selected_vars, reliability_index, mse_list = mcuve_plsda_workflow('myfilename.csv')