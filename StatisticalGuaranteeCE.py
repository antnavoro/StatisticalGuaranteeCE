import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.stats as stats
from load_dataset import load_data
import sys
from scipy.stats import norm
import scipy.optimize
import csv
from concurrent.futures import ProcessPoolExecutor
from itertools import product

modelType = 'Poisson' # 'logistic', 'probit', 'linear' or 'Poisson'
database = {'logistic': 'breast_cancer', 'probit': 'breast_cancer', 'linear': 'communities_and_crime', 'Poisson': 'Seoul_bike_sharing_demand'}
dataset = database[modelType]
np.random.seed(42)
# Load the dataset
X, Y, X_original, omega0 = load_data(dataset)
k, n = X.shape
omega0 = omega0*k
sparse = False

def initial_solution(S, alpha, tau, beta, x0):
    n = len(beta)
    Z = stats.norm.ppf(1-alpha)
    
    # Create model
    model = gp.Model("Optimization")
    model.setParam('OutputFlag', 0)
    x = [] # List of all (normalized) variables
    y = [] # List of (not normalized) integer variables
    categorical_groups = {}
    for i, col in enumerate(X_original.columns):
        if "_categorical_" in col: # If it is a categorical variable (this string is added in load_dataset.py)
            j = int(col.split("_categorical_")[-1])
            if j not in categorical_groups:
                categorical_groups[j] = []
            categorical_groups[j].append(i)

        unique_values = np.unique(X_original[col])
        
        if np.all(np.isin(unique_values, [0, 1])):  # If it is binary
            x.append(model.addVar(vtype=GRB.BINARY, name=f"x_{i}"))
        
        elif np.issubdtype(X_original[col].dtype, np.integer):  # If it is integer
            yi = model.addVar(vtype=GRB.INTEGER, lb=X_original[col].min(), ub=X_original[col].max(), name=f"y_{i}")
            y.append(yi)
            xi = model.addVar(lb=-GRB.INFINITY, name=f"x_{i}")
            x.append(xi)
            model.addConstr(xi==(yi-X_original[col].min())/(X_original[col].max()-X_original[col].min()), name = f'integrality_{i}')
        
        else:  # If it is continuous
            x.append(model.addVar(lb=0, ub=1, name=f"x_{i}")) # Normalized to [0,1]

    for j, indices in enumerate([categorical_groups[j] for j in sorted(categorical_groups.keys())]):
        # Restrictions for the dummy variables associated with the same categorical variable
        max_dummy_sum = X_original.iloc[:, indices].sum(axis=1).max()  
        min_dummy_sum = X_original.iloc[:, indices].sum(axis=1).min()
        model.addConstr(gp.quicksum(x[i] for i in indices) <= max_dummy_sum, f"categorical_{j}_max")
        model.addConstr(gp.quicksum(x[i] for i in indices) >= min_dummy_sum, f"categorical_{j}_min")

    model.addConstr(x[0] == 1, name = 'intercept') # Intercept is fixed to 1
    
    # Define constraint: beta^T x + tau - Z sqrt(z) >= 0
    
    t = model.addVar(name='t')
    model.addConstr(t**2 >= gp.quicksum(x[i] * S[i, j] * x[j] for i in range(n) for j in range(n)), name="t_definition")

    model.setObjective(-beta @ x + Z**2 * t, GRB.MINIMIZE)
    model.write("paper2_solinit.lp")  

    model.optimize()
    #model.printQuality()
    if model.status == GRB.OPTIMAL:
        xsol = [a.X for a in x]
        print(xsol,model.ObjVal)
        return xsol, model.ObjVal  # Return the optimal solution
    else:
        print("No optimal solution found.")
        print("x0:", x0)
        print("tau:", tau)
        print("alpha:", alpha)
        return x0, np.inf  # No optimal solution found



def solve_optimization(S, alpha, tau, beta, x0):
    print('Solving for ', alpha, tau)
    n = len(beta)
    Z = stats.norm.ppf(1-alpha)

    initsol = initial_solution(S, alpha, tau, beta, x0)
    if initsol[1] == np.inf:
        initsol = (x0, np.inf)
        print('Infeasible')
        return x0, np.inf 
    
    # Create model
    model = gp.Model("Optimization")
    model.setParam('OutputFlag', 0) 
    model.setParam('Presolve', 2)
    model.setParam('MIPFocus', 3)
    model.setParam('MIPGap', 0.01)
    #model.setParam('NoRelHeurTime', 30)
    #model.setParam('TimeLimit', 60*60)
    x = [] # List of all (normalized) variables
    y = [] # List of (not normalized) integer variables
    categorical_groups = {}
    for i, col in enumerate(X_original.columns):
        if "_categorical_" in col: # If it is a categorical variable (this string is added in load_dataset.py)
            j = int(col.split("_categorical_")[-1])
            if j not in categorical_groups:
                categorical_groups[j] = []
            categorical_groups[j].append(i)

        unique_values = np.unique(X_original[col])
        
        if np.all(np.isin(unique_values, [0, 1])):  # If it is binary
            x.append(model.addVar(vtype=GRB.BINARY, name=f"x_{i}"))
        
        elif np.issubdtype(X_original[col].dtype, np.integer):  # If it is integer
            yi = model.addVar(vtype=GRB.INTEGER, lb=X_original[col].min(), ub=X_original[col].max(), name=f"y_{i}")
            y.append(yi)
            xi = model.addVar(lb=-GRB.INFINITY, name=f"x_{i}")
            x.append(xi)
            model.addConstr(xi==(yi-X_original[col].min())/(X_original[col].max()-X_original[col].min()), name = f'integrality_{i}')
        
        else:  # If it is continuous
            x.append(model.addVar(lb=0, ub=1, name=f"x_{i}")) # Normalized to [0,1]

    for j, indices in enumerate([categorical_groups[j] for j in sorted(categorical_groups.keys())]):
        # Restrictions for the dummy variables associated with the same categorical variable
        max_dummy_sum = X_original.iloc[:, indices].sum(axis=1).max()  
        min_dummy_sum = X_original.iloc[:, indices].sum(axis=1).min()
        model.addConstr(gp.quicksum(x[i] for i in indices) <= max_dummy_sum, f"categorical_{j}_max")
        model.addConstr(gp.quicksum(x[i] for i in indices) >= min_dummy_sum, f"categorical_{j}_min")

    model.addConstr(x[0] == 1, name = 'intercept') # Intercept is fixed to 1
    for i in range(n):
        x[i].start = initsol[0][i]
    # Activation variables for the l0 norm
    z = model.addVars(n, vtype=GRB.BINARY, name="z")

    # Absolute value variables for the l1 norm
    abs_diff = model.addVars(n, lb=0, name="abs_diff")

    # Define absolute value constraints
    for i in range(n):
        model.addConstr(abs_diff[i] >= x[i] - x0[i], name = f'abs1_{i}')
        model.addConstr(abs_diff[i] >= -(x[i] - x0[i]), name = f'abs2_{i}')

    # Activation constraints
    big_M = np.ones(n) # The dataset is normalised, so the maximum difference is 1
    model.addConstrs((abs_diff[i]<=z[i]*big_M[i] for i in range(n)), name="active")

    # Define objective: minimize squared Euclidean distance to x0
    if sparse:
        model.setObjective(gp.quicksum(abs_diff[i] for i in range(n)) + gp.quicksum(z[i] for i in range(n)) , GRB.MINIMIZE)
    else:
        #model.setObjective(gp.quicksum((x[i]-x0[i])*(x[i]-x0[i]) for i in range(n)) , GRB.MINIMIZE)
        model.setObjective(gp.quicksum(abs_diff[i] for i in range(n)) , GRB.MINIMIZE)
    
    # Define constraint: beta^T x + tau - Z sqrt(z) >= 0
    
    t = model.addVar(name='t')
    model.addConstr(t == beta @ x - tau, name="t_definition")

    L = scipy.linalg.cholesky(S, lower=True)  # A = L L^T


    y = model.addVars(n,lb=-GRB.INFINITY, name="y")

    model.addConstrs((y[i] == gp.quicksum(L.T[i,j]*x[j] for j in range(n)) for i in range(n)), name="y_definition")
    model.addConstr(t*t - Z**2 * gp.quicksum(y[i] * y[i] for i in range(n)) >= 0, name="constraint2")

    #model.addConstr(t*t - Z**2 * gp.quicksum(x[i] * S[i, j] * x[j] for i in range(n) for j in range(n)) >= 0, name="constraint2") 
    model.write("paper2.lp")

    # Optimize model
    model.optimize()
    #model.printQuality()
    if model.status == GRB.OPTIMAL:
        print(model.ObjVal)
        xsol = [a.X for a in x]
        return xsol, model.ObjVal  # Return the optimal solution
    else:
        if model.SolCount > 0:
            print("Mejor solución encontrada con valor objetivo:", model.ObjVal)
            xsol = [a.X for a in x]
            return xsol, np.inf
        print("No optimal solution found.")
        print("x0:", x0)
        print("tau:", tau)
        print("alpha:", alpha)
        # Uncomment the following lines to compute the Irreducible Inconsistent Subsystem (IIS) if the model is infeasible
        # if model.status == gp.GRB.INFEASIBLE:
        #     print("Model is infeasible. Computing IIS...")
        #     model.computeIIS()  # Calcula el IIS

        #     print("\nThe following constraints are making the model infeasible:")
        #     for c in model.getConstrs():
        #         if c.IISConstr:  # Si la restricción está en el IIS
        #             print(f"- {c.ConstrName}")
            
        #     print("\nThe following variable bounds are making the model infeasible:")
        #     for v in model.getVars():
        #         if v.IISLB:  # Si el límite inferior está en el IIS
        #             print(f"- Variable {v.VarName} has an infeasible lower bound.")
        #         if v.IISUB:  # Si el límite superior está en el IIS
        #             print(f"- Variable {v.VarName} has an infeasible upper bound.")
        return x0, np.inf  # No optimal solution found

## Define the gradient, hessian, and inverse link function for the selected model
if modelType == 'logistic':
    def sigmoid(t):
        t = np.clip(t, -20,20) # Avoid overflow
        return 1 / (1 + np.exp(-t))
    
    def g(t): # logit
        return np.log(t/(1-t))

    def g_inv(t):
        return sigmoid(t)

    def gradient_log_likelihood(beta, omega):
        return np.dot(X.T, omega*(Y - sigmoid(np.dot(X, beta))))

    def hessian_log_likelihood(beta, omega):
        p = sigmoid(np.dot(X, beta))
        return -X.T @ np.diag(omega * p * (1 - p)) @ X
    
elif modelType == 'linear':
    def g(t):
        return t

    def g_inv(t):
        return t
    
    def gradient_log_likelihood(beta, omega):
        return np.dot(X.T, omega*(Y-np.dot(X, beta)))
    
    def hessian_log_likelihood(beta, omega):
        return -X.T @ np.diag(omega) @ X
    
elif modelType == 'Poisson':
    def g(t):
        return np.log(t)

    def g_inv(t):
        return np.exp(t)
       
    def gradient_log_likelihood(beta, omega):
        t = np.dot(X, beta)
        t = np.clip(t, -20, 20) # Avoid overflow
        t = np.exp(t)
        return np.dot(X.T, omega*(Y-t))
    
    def hessian_log_likelihood(beta, omega):
        t = np.dot(X, beta)
        t = np.clip(t, -20, 20) # Avoid overflow
        t = np.exp(t)
        return - X.T @ np.diag(omega * t) @ X
    
elif modelType == 'probit':
    def g(t):
        return norm.ppf(t)

    def g_inv(t):
        return norm.cdf(t)
    
    def gradient_log_likelihood(beta, omega):
        t = np.dot(X, beta)
        pdf = norm.pdf(t)
        cdf = norm.cdf(t)
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6) # Avoid division by zero
        return np.dot(X.T, omega * pdf/(cdf*(1-cdf))* (Y-cdf))
    
    def hessian_log_likelihood(beta, omega):
        t = np.dot(X, beta)
        pdf = norm.pdf(t)
        cdf = norm.cdf(t)
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6) # Avoid division by zero
        return - X.T @ np.diag(omega * pdf**2 / (cdf*(1-cdf))) @ X
    
else:
    print('Invalid model type')
    sys.exit()

err = 0
def beta_given_omega(omega,beta0):
    global err
    err_loc = 0
    # Solve grad = 0
    result = scipy.optimize.root(gradient_log_likelihood, beta0, args=(omega), tol=1e-12, jac=hessian_log_likelihood)
    # Check if the optimization converged
    while not result.success:
        err_loc += 1
        if err_loc > 10000:
            print('Convergence error in the grad(L) = 0 equation')
            sys.exit()
        beta0 = np.random.normal(0,np.random.uniform(0, 1),n) # Try again with random initialization
        result = scipy.optimize.root(gradient_log_likelihood, beta0, args=(omega), tol=1e-12, jac=hessian_log_likelihood,method='lm')
    err += err_loc

    return result.x  # Return the optimized beta

def get_x0(l, per):
    ## Return the x0s for the largest y_pred <= q_per, percentile of the predictions
    beta = beta_given_omega(omega0, np.full(n, 0))
    Y_pred = g_inv(np.dot(X, beta))
    
    # Compute the 'per' percentile of the predictions
    q_per = np.percentile(Y_pred, per)
    
    # Filter the indices where the prediction is below or equal to the percentile
    indices_below = np.where(Y_pred <= q_per)[0]
    
    # Order the indices by the predicted value
    indices_sorted = indices_below[np.argsort(Y_pred[indices_below])][-l:]
    
    # Select the corresponding rows of X, Y, and Y_pred
    X_subset = X[indices_sorted, :]
    Y_subset = Y[indices_sorted] # Only for debugging purposes
    Y_pred_subset = Y_pred[indices_sorted] # Only for debugging purposes
    
    return X_subset, Y_subset, Y_pred_subset

beta = beta_given_omega(omega0, np.zeros(n))
# Compute the covariance matrix S
S = np.linalg.inv(-hessian_log_likelihood(beta, omega0))

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
def is_pos_def(x):
    print("Eigenvalues:", np.linalg.eigvals(x))
    return np.all(np.linalg.eigvals(x) > 0)

def solve_and_pack(params):
    alpha, tau, x0y0 = params
    x0, y0 = x0y0
   
    solution = solve_optimization(S, alpha, tau, beta, x0)
    xsol, objective_value = solution
    return [alpha, tau] + list(x0) + [y0] + [np.dot(x0,beta)] + list(xsol) + [np.dot(xsol,beta)] + [objective_value]

if __name__ == "__main__":
    aux = np.round(np.arange(0.5, 1.00, 0.05), 2)  # alphas from 0.5 to 0.95
    taus = np.array([g(t*(np.max(Y)-np.min(Y))) for t in aux])                      
    alphas = np.array([1-a for a in aux]) # Invert the values of alpha
    x0s = [X[i, :] for i in range(X.shape[0])]
    x0sy0s = [(x0s[i], Y[i]) for i in range(len(x0s))]
    combinations = list(product(alphas, taus, x0sy0s))
    if sparse:
        output_file = "paper2_" + modelType + "_sparse.csv"
    else: 
        output_file = "paper2_" + modelType + ".csv"
    header = ['alpha', 'tau'] + [f'x0_{i}' for i in range(X.shape[1])] + ['y0'] + ['val_x0'] + [f'xopt_{i}' for i in range(X.shape[1])] + ['val_xopt'] + ['objective_value']

    tau05 = g(0.5 * (np.max(Y) - np.min(Y)))

    combinations = [(a, tau05, xy) for a in alphas for xy in x0sy0s[:50]] + [(a, tau, x0sy0s[3]) for a in alphas for tau in taus]

    print(f"Total combinations to process: {len(combinations)}")

    results = []
    parallel = True
    if parallel:
        with ProcessPoolExecutor() as executor:
            for result in executor.map(solve_and_pack, combinations):
                results.append(result)
    else:
        for params in combinations:
            result = solve_and_pack(params)
            results.append(result)


    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(results)


    print("Experiment completed and saved to", output_file)
