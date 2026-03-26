import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from load_dataset import load_data
import pandas as pd
import sys
from scipy.stats import norm
from scipy.interpolate import griddata
import seaborn as sns
import os

## Load the data
def info_all(modelType):
    database = {'logistic': 'Breast Cancer', 'logistic2' : 'QSAR Oral Toxicity', 'probit': 'Breast Cancer', 'linear': 'Communities and Crime', 'Poisson': 'Seoul Bike Sharing Demand'}
    dataset = database[modelType]
    #modelType = 'linear' # 'logistic', 'probit', 'linear' or 'Poisson'
    if modelType == 'Poisson':
        rangoY=3556
    else:
        rangoY=1
    #database = {'logistic': 'breast_cancer', 'probit': 'breast_cancer', 'linear': 'communities_and_crime', 'Poisson': 'Seoul_bike_sharing_demand'}
    #dataset = database[modelType]

    save_folder = 'new_figures2'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    isnotLogistic2 = modelType != 'logistic2'
    file_data_sparse = f't_{modelType}_sparse_batch_'
    file_data = f't_{modelType}_batch_'

 
    # Load the CSV file

    rnum = 5

    ## Define the gradient, hessian, and inverse link function for the selected model
    if modelType == 'logistic' or modelType == 'logistic2':
        def g(t): # logit
            return np.log(t/(1-t))

        def sigmoid(t):
            t = np.clip(t, -700,700) # Avoid overflow
            return 1 / (1 + np.exp(-t))

        def g_inv(t):
            return sigmoid(t)
        
    elif modelType == 'linear':
        def g(t):
            return t
        
        def g_inv(t):
            return t
        
    elif modelType == 'Poisson':
        def g(t):
            return np.log(t)
        
        def g_inv(t):
            return np.exp(t)

    elif modelType == 'probit':
        def g(t):
            return norm.ppf(t)
        
        def g_inv(t):
            return norm.cdf(t)

    else:
        print('Invalid model type')
        sys.exit()

    if modelType == 'logistic2':
        modelTypeTitle = 'logistic'
    else:
        modelTypeTitle = modelType

    def load_results_from_csv(filename):
        all_dfs = []  
    
        for file in os.listdir("."):
            if file.startswith(filename) and file.endswith(".csv"):
                print(f"Loading file: {file}")
                current_df = pd.read_csv(file)
                all_dfs.append(current_df)
        
        if not all_dfs:
            print(f"No files found with prefix '{filename}'")
            return {}

        df = pd.concat(all_dfs, ignore_index=True)
        
        # Reconstruct the vectors (x0, x_max, a0, a1) from the column names
        results_dict = {}
        
        for _, row in df.iterrows():
            # Reconstruct the vectors and other values
            x0 = [row[f'x0_{i}'] for i in range(len([col for col in df.columns if col.startswith('x0_')]))]
            x_opt = [row[f'xopt_{i}'] for i in range(len([col for col in df.columns if col.startswith('xopt_')]))]
            alpha = row['alpha']
            tau = row['tau']
            y0 = row['y0']
            objective_value = row['objective_value']
            times = row['time_seconds']
            val_x0 = g_inv(row['val_x0'])
            val_xopt = g_inv(row['val_xopt'])

            # Create the result tuple
            result_tuple = (x0, y0, val_x0, x_opt, val_xopt, objective_value, times)
            alpha_ = round(float(alpha),rnum)
            tau_ = round(float(tau),rnum)
            if str((alpha_,tau_)) not in results_dict:
                results_dict[str((alpha_,tau_))] = []
            results_dict[str((alpha_,tau_))].append(result_tuple)
        return results_dict

    def get_results(data, params):
        alpha, tau = params
        alpha_ = round(float(alpha),rnum)
        tau_ = round(float(tau),rnum)
        results = data[str((alpha_,tau_))]
        x0s = np.array([a[0][1:] for a in results])
        xopts = np.array([a[3][1:] for a in results])
        y0s = np.array([a[1] for a in results])
        objective_values = np.array([a[5] for a in results])
        val_x0s = np.array([a[2] for a in results])
        val_xopts = np.array([a[4] for a in results])
        times = np.array([a[6] for a in results])
        return x0s, y0s, val_x0s, xopts, val_xopts, objective_values, times


    data = load_results_from_csv(file_data)
    data_sparse = load_results_from_csv(file_data_sparse)

    aux = np.round(np.arange(0.5, 1.0, 0.05), 2) 
    aux = np.concatenate((aux, np.round(np.array([0.99,0.999]),3)))
    alphas = np.array([1-a for a in aux]) 
    tau05 = g(rangoY*0.5)                  
    fmt = ".5f"

    diff = []
    diff_sparse = []
    times_all = []
    times_all_sparse = []
    _, _, _, _, _, objective_values0, _ = get_results(data, (alphas[0], tau05))
    if isnotLogistic2:
        _, _, _, _, _, objective_values0_sparse, _ = get_results(data_sparse, (alphas[0], tau05))
    for a in alphas:
        params = (a, tau05)
        _, _, _, _, _, objective_values, times = get_results(data, params)
        non_trivial_times = times[times>0]
        times_all.append(non_trivial_times)
        dif = objective_values - objective_values0
        dif = dif[objective_values > 1e-10]
        diff.append(dif)
        if isnotLogistic2:
            _, _, _, _, _, objective_values_sparse, times_sparse = get_results(data_sparse, params)
            non_trivial_times_sparse = times_sparse[times_sparse>0]
            times_all_sparse.append(non_trivial_times_sparse)
            dif_sparse = objective_values_sparse - objective_values0_sparse
            dif_sparse = dif_sparse[objective_values_sparse > 1e-10]  
            diff_sparse.append(dif_sparse)
        #print(f'alpha: {a}, avg time: {np.mean(non_trivial_times):{fmt}}s, std time: {np.std(non_trivial_times):{fmt}}s, n_samples: {len(non_trivial_times)}, median time: {np.median(non_trivial_times):{fmt}}s, min time: {np.min(non_trivial_times):{fmt}}s, max time: {np.max(non_trivial_times):{fmt}}s')

    plt.figure(figsize=(10, 6))
    plt.plot(alphas, [np.median(t) for t in times_all], marker='o', label=r'Median Time, $d=\ell_1$', color='blue')
    plt.fill_between(alphas, [np.percentile(t, 25) for t in times_all], [np.percentile(t, 75) for t in times_all], color='blue', alpha=0.2, label=r'IQR, $d=\ell_1$')
    if isnotLogistic2:
        plt.plot(alphas, [np.median(t) for t in times_all_sparse], marker='s', label=r'Median Time, $d=\ell_1+\ell_0$', color='red')
        plt.fill_between(alphas, [np.percentile(t, 25) for t in times_all_sparse], [np.percentile(t, 75) for t in times_all_sparse], color='red', alpha=0.2, label=r'IQR, $d=\ell_1+\ell_0$')
    plt.xlabel(r"$\alpha$", fontsize=15)
    plt.ylabel("Running Time (seconds)", fontsize=15)
    #plt.title(f"{dataset} dataset with {modelTypeTitle} model", fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.savefig(f"{save_folder}/running_times_{modelType}.pdf")

    plt.figure(figsize=(10, 6))
    plt.plot(alphas, [np.mean(v) for v in diff], marker='o', label='Mean difference in optimal values', color='green')
    plt.fill_between(alphas, [np.mean(v) - np.std(v) for v in diff], [np.mean(v) + np.std(v) for v in diff], color='green', alpha=0.2, label=r'Mean $\pm$ SD')
    plt.xlabel(r"$\alpha$",fontsize=15)
    plt.ylabel("Cost increase of statistical guarantee", fontsize=15)
    #plt.title(f"{dataset} dataset with {modelTypeTitle} model, $d=\ell_1$",fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(f"{save_folder}/difference_objective_values_{modelType}.pdf")

    if isnotLogistic2:
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, [np.mean(v) for v in diff_sparse], marker='s', label='Mean difference in optimal values', color='green')
        plt.fill_between(alphas, [np.mean(v) - np.std(v) for v in diff_sparse], [np.mean(v) + np.std(v) for v in diff_sparse], color='green', alpha=0.2, label='Mean ± SD')
        plt.xlabel(r"$\alpha$",fontsize=15)
        plt.ylabel("Cost increase of statistical guarantee", fontsize=15)
        #plt.title(f"{dataset} dataset with {modelTypeTitle} model, $d=\ell_1+\ell_0$",fontsize=15)
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.savefig(f"{save_folder}/difference_objective_values_{modelType}_sparse.pdf")



    # plt.boxplot(times_all, labels=[f"{a:.3f}" for a in alphas])
    # # Add the mean values as dots
    # mean_val = [np.mean(t) for t in times_all]
    # std_val = [np.std(t) for t in times_all]
    # plt.errorbar(range(1,len(times_all)+1), mean_val, yerr=std_val, fmt='o', color='red', label='Mean ± SD')
    # plt.xlabel("Alpha Values")
    # plt.ylabel("Running Time (seconds)")
    # plt.title(f"Running Times for {modelType} Model (sparse={sparse})")
    # plt.legend(fontsize=12)
    # plt.grid(axis='y', linestyle='--', alpha=0.5)
    # plt.show()
        
    # diff = np.array(diff)
    # df = pd.DataFrame(diff.T, columns=[f"alpha={alphas[i]:.3f}" for i in range(diff.shape[0])])

    # plt.figure(figsize=(10, 6))
    # sns.violinplot(data=df, palette="Set2", inner="quart")
    # ## Add a dot as the mean value and a line for the median
    # for i in range(diff.shape[0]):
    #     plt.scatter(i, np.mean(diff[i]), color='black', marker='D', label='Mean' if i == 0 else "")
    #     plt.plot([i-0.2, i+0.2], [np.median(diff[i]), np.median(diff[i])], color='red', label='Median' if i == 0 else "")
    # plt.xticks(ticks=range(len(alphas)), labels=[f"alpha={alpha:.3f}" for alpha in alphas])
    # plt.title(f"Distribution of Objective Value Differences for {modelType} Model (sparse={sparse})")
    # plt.xlabel("Alpha Values")
    # plt.ylabel("Difference in Objective Values")
    # plt.legend(fontsize=12)
    # plt.grid(axis='y', linestyle='--', alpha=0.5)
    # plt.show()




for modelType in ['logistic', 'logistic2', 'probit', 'linear', 'Poisson']:
    print(f"Printing info for {modelType}")
    info_all(modelType)