import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from load_dataset import load_data
import pandas as pd
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
import sys
from scipy.stats import norm
from scipy.interpolate import griddata
import os

## Load the data
def plot_all(modelType,heatmaps,sparse):
    #modelType = 'linear' # 'logistic', 'probit', 'linear' or 'Poisson'
    if modelType == 'Poisson':
        rangoY=3556
    else:
        rangoY=1
    database = {'logistic': 'breast_cancer', 'probit': 'breast_cancer', 'linear': 'communities_and_crime', 'Poisson': 'Seoul_bike_sharing_demand'}
    dataset = database[modelType]
    folder = 'figures/'
    folder += modelType + '/'

    if heatmaps:
        folder += 'heatmaps/'
    else:
        folder += 'pareto/'

    # Create the folder if it doesn't exist
    if sparse:
        folder += 'sparse/'
        file_data = f'paper2_{modelType}_sparse.csv'
        addname = '_sparse'
    else:
        folder += 'no_sparse/'
        file_data = f'paper2_{modelType}.csv'
        addname = ''

    if not os.path.exists(folder):
        os.makedirs(folder)

    num_indiv = 50
    num_plot = 20
    # Load the CSV file

    rnum = 5

    ## Define the gradient, hessian, and inverse link function for the selected model
    if modelType == 'logistic':
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

    # Function to plot a heatmap of the difference between two matrices
    def plot_heatmap(A, B, row_labels, col_labels, alpha, tau):
        """
        Plots a heatmap of B - A by columns, with row and column labels.

        Parameters:
        - A: 2D list or array, each row is a reference vector a_i to compare with the corresponding row in B.
        - B: 2D matrix where each row is a vector b_i.
        - row_labels: List of labels for the rows (each b_i).
        - col_labels: List of labels for the columns.
        """
        # Convert A and B to NumPy arrays
        num = min(num_indiv,len(A))
        A = np.array(A)[:num]
        B = np.array(B)[:num]
        row_labels = np.array(row_labels)[:num]
        # Check that dimensions match
        if B.shape != A.shape:
            print("B shape:", B.shape, "A shape:", A.shape)
            raise ValueError("Dimensions of B and A must match exactly (row-wise).")

        # Check that the lengths of the labels are correct
        if len(row_labels) != B.shape[0]:
            raise ValueError("Number of row labels does not match number of rows in B.")
        if len(col_labels) != B.shape[1]:
            raise ValueError("Number of column labels does not match number of columns in B.")

        # Compute B - A (subtract row-wise)
        diff_matrix = B - A
        if all(isinstance(label, (int, float)) for label in row_labels):
            rounded_row_labels = [round(label, 1) for label in row_labels]
        else:
            rounded_row_labels = row_labels

        # Create the heatmap
        k, n = A.shape
        cell_height = 0.5
        cell_width = 1 * cell_height
        figsize = (n * cell_width, k * cell_height)
        #plt.figure(figsize=figsize)#11.69,8.27
        plt.figure(figsize=(7,10))#11.69,8.27

        ax = sns.heatmap(
            diff_matrix,
            vmin=-1, vmax=1,
            cmap="seismic",
            annot=False,
            fmt=".2f",
            linewidths=0,
            xticklabels=col_labels
        )#, yticklabels=rounded_row_labels)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        plt.xlabel("Features",fontsize=20)
        plt.ylabel(r"$\mathbf{x_0}$",fontsize=20)

        # Rotate column labels
        plt.xticks(rotation=90, fontsize=5)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(folder + f'heatmap_{modelType}_{alpha:.2f}_{tau}' + addname + '.pdf')
        plt.close()
        #plt.show()

    def plot_heatmaps(A, B, row_labels, col_labels, alphas, tau):
        """
        Plots a sequence of heatmaps of B - A for each alpha in alphas, arranged side by side.

        Parameters:
        - B: 3D array or list of shape (len(alphas), k, n)
        - A: 2D array of shape (k, n), fixed for all alphas
        - row_labels: List of k row labels
        - col_labels: List of n column labels
        - alphas: List of alpha values
        - tau: A parameter used in filename
        - folder: Path to save the figure
        - modelType: Model name used in filename
        - addname: Optional string to append to filename
        """
        A = np.array(A)
        B = np.array(B)
        row_labels = np.array(row_labels)
        num_alphas = len(alphas)

        # Consistency checks
        if B.shape[0] != num_alphas:
            raise ValueError("B must have one slice per alpha.")
        if A.shape != B.shape[1:]:
            raise ValueError("Each B[alpha] must have same shape as B.")
        if len(row_labels) != A.shape[0]:
            raise ValueError("Mismatch between row_labels and number of rows in A.")
        if len(col_labels) != A.shape[1]:
            raise ValueError("Mismatch between col_labels and number of columns in A.")

        #k, n = A.shape
        #cell_height = 0.5
        #cell_width = 0.7
        #figsize = (num_alphas * n * cell_width, k * cell_height)

        fig, axes = plt.subplots(1, num_alphas, figsize=(20,10), sharey=True)
        if num_alphas == 1:
            axes = [axes]

        vmin, vmax = -1, 1  # consistent color range for all heatmaps
        num = min(num_indiv,len(B[0]))
        A = np.array(A)[:num]

        for i, alpha in enumerate(alphas):
            BB = np.array(B[i])[:num]
            diff_matrix = BB - A
            ax = axes[i]
            sns.heatmap(
                diff_matrix,
                ax=ax,
                vmin=vmin, vmax=vmax,
                cmap="seismic",
                annot=False,
                fmt=".2f",
                cbar=False,
                cbar_kws={'shrink': 0.5} if i == num_alphas - 1 else None,
                xticklabels=col_labels,
                yticklabels=row_labels if i == 0 else False,
                linewidths=0
            )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('black')  
                spine.set_linewidth(2)        
            ax.set_title(f"$\\alpha = {alpha:.2f}$", fontsize=28)
            #ax.set_xlabel("Features", fontsize=10)
            if i == 0:
                ax.set_yticklabels(row_labels, rotation=0, fontsize=8)
                ax.set_ylabel(r"$\mathbf{x_0}$", fontsize=28)
            ax.tick_params(axis='x', rotation=90, labelsize=6)
            ax.tick_params(axis='y', labelsize=8)

        fig.supxlabel("Features", fontsize=28)

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # Create a colorbar for the last heatmap
        divider = make_axes_locatable(axes[-1])
        cax = divider.append_axes("right", size="5%", pad=0.1)

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap="seismic")
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize=8)



        plt.tight_layout()
        plt.savefig(folder + f'heatmaps_{modelType}_{tau}' + addname + '.pdf')
        plt.close()
        # plt.show()



    def plot_3d(x,y,z,j):

        xi = np.linspace(min(x), max(x), 50)
        yi = np.linspace(min(y), max(y), 50)
        xi, yi = np.meshgrid(xi, yi)


        zi = griddata((x, y), z, (xi, yi), method='linear')


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none', alpha=0.8)


        ax.scatter(x, y, z, color='r', s=50)
        ax.set_xlabel(r'$\alpha$',fontsize=20)
        ax.set_ylabel(r'$\tau$',fontsize=20)
        ax.set_zlabel(r'$d(\mathbf{x},\mathbf{x_0})$',fontsize=20)
        plt.savefig(folder + f'pareto_{modelType}_3D_{j:.2f}' + addname + '.pdf')
        plt.close()


    def load_results_from_csv(filename):
        # Load the data from the CSV file
        df = pd.read_csv(filename)
        
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
            val_x0 = g_inv(row['val_x0'])
            val_xopt = g_inv(row['val_xopt'])

            # Create the result tuple
            result_tuple = (x0, y0, val_x0, x_opt, val_xopt, objective_value)
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
        return x0s, y0s, val_x0s, xopts, val_xopts, objective_values


    data = load_results_from_csv(file_data)

    aux = np.round(np.arange(0.5, 1.0, 0.05), 2) 
    alphas = np.array([1-a for a in aux]) 
    taus = np.array([g(t*rangoY) for t in aux])                  

    def generate_3d_plot():
        xx = alphas.copy()
        yy = taus.copy()

        j = 3
        x = []
        y = []
        z = []
        for xi in xx:
            for yi in yy:
                params = (xi, yi)
                x0s, y0s, val_x0s, xopts, val_xopts, objective_values = get_results(data, params)
                x.append(xi)
                y.append(yi)
                if len(objective_values) > 1:
                    obj = objective_values[j]
                else:
                    obj = objective_values[0]
                z.append(obj)
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        plot_3d(x, y, z, j)

    def generate_2d_plot():
        t=taus[0]
        plt.figure(figsize=(11.69,8.27))
        plt.xlabel(r'$\alpha$',fontsize=20)
        plt.ylabel(r'$d(\mathbf{x},\mathbf{x_0})$',fontsize=20)
        for j in range(num_plot):
            z=[]
            for a in alphas:
                params = (a, t)
                x0s, y0s, val_x0s, xopts, val_xopts, objective_values = get_results(data, params)
                z.append(objective_values[j])
            plt.plot(alphas, z, '-o')
        plt.savefig(folder + f'pareto_{modelType}_tau_{t}_{j:.2f}' + addname + '.pdf')
        plt.close()

    def generate_2d_plot2():
        a=0.05
        plt.figure(figsize=(11.69,8.27))
        plt.xlabel(r'$\tau$',fontsize=20)
        plt.ylabel(r'$d(\mathbf{x},\mathbf{x_0})$',fontsize=20)
        j=3
        for a in alphas:
            z=[]
            for t in taus:
                #print(a)
                params = (a, t)
                x0s, y0s, val_x0s, xopts, val_xopts, objective_values = get_results(data, params)
                if len(objective_values) > 1:
                    obj = objective_values[j]
                else:
                    obj = objective_values[0]
                z.append(obj)
            plt.plot(taus, z, '-o', label=f'$\\alpha = {a:.2f}$')
        plt.legend()
        plt.savefig(folder + f'pareto_{modelType}_vstau_{j:.2f}' + addname + '.pdf')


    def generate_all_heatmaps():
        columns = load_data(dataset)[-2].columns[1:]
        tau = taus[0]
        for alpha in alphas:
            params = (alpha, tau)
            x0s, y0s, val_x0s, xopts, val_xopts, objective_values = get_results(data, params)
            plot_heatmap(x0s, xopts, range(len(x0s)), columns, alpha, tau)

    def generate_all_heatmaps2():
        columns = load_data(dataset)[-2].columns[1:]
        xoptss = []
        tau = taus[0]
        for alpha in alphas:
            params = (alpha, tau)
            x0s, y0s, val_x0s, xopts, val_xopts, objective_values = get_results(data, params)
            xoptss.append(xopts)
        xoptss = np.array(xoptss)
        plot_heatmaps(x0s, xoptss, range(len(x0s)), columns, alphas, tau)

    if heatmaps:
        generate_all_heatmaps()
        generate_all_heatmaps2()
    else:
        generate_2d_plot()
        generate_2d_plot2()
        generate_3d_plot()

for modelType in ['logistic', 'probit', 'linear', 'Poisson']:
    for heatmaps in [False, True]:
        for sparse in [False, True]:
            print(f"Generating plots for {modelType}, heatmaps={heatmaps}, sparse={sparse}")
            plot_all(modelType, heatmaps, sparse)