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
import time

modelType = 'Poisson' # 'logistic', 'logistic2', 'probit', 'linear' or 'Poisson'
database = {'logistic': 'breast_cancer', 'logistic2' : 'qsar_oral_toxicity', 'probit': 'breast_cancer', 'linear': 'communities_and_crime', 'Poisson': 'Seoul_bike_sharing_demand'}
dataset = database[modelType]
np.random.seed(42)
# Load the dataset
X, Y, X_original, omega0 = load_data(dataset)
k, n = X.shape
omega0 = omega0*k
sparse = False


## Define the gradient, hessian, and inverse link function for the selected model
if modelType == 'logistic' or modelType == 'logistic2':
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

beta = beta_given_omega(omega0, np.zeros(n))

# Compute the covariance matrix S
if modelType == 'linear':
    sigma2 = np.sum([omega0[i]*(Y[i]-np.dot(X[i,:],beta))**2 for i in range(k)])/np.sum(omega0)*k/(k-n)
    print("Estimated sigma^2:", sigma2)
    S = np.linalg.inv(-hessian_log_likelihood(beta, omega0)/sigma2)
else:
    S = np.linalg.inv(-hessian_log_likelihood(beta, omega0))

L = scipy.linalg.cholesky(S, lower=True)  # S = L L^T

def initial_solution(alpha, tau, beta, x0):
    n = len(beta)
    Z = stats.norm.ppf(1-alpha)
    print('Calculating initial solution for alpha =', alpha, 'and tau =', tau)
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
        
    t = model.addVar(name='t')
    #model.addConstr(t**2 >= gp.quicksum(x[i] * S[i, j] * x[j] for i in range(n) for j in range(n)), name="t_definition")

    y = model.addVars(n,lb=-GRB.INFINITY, name="y")

    model.addConstrs((y[i] == gp.quicksum(L.T[i,j]*x[j] for j in range(n)) for i in range(n)), name="y_definition")
    model.addConstr(t**2 >= gp.quicksum(y[i] * y[i] for i in range(n)), name="constraint2")


    model.setObjective(-beta @ x + Z * t, GRB.MINIMIZE)
    #model.write("model_solinit.lp")  

    model.optimize()
    #model.printQuality()
    if model.status == GRB.OPTIMAL:
        xsol = [a.X for a in x]
        #print(xsol,model.ObjVal)
        return xsol, model.ObjVal  # Return the optimal solution
    else:
        print("No optimal solution found.")
        print("x0:", x0)
        print("tau:", tau)
        print("alpha:", alpha)
        return x0, np.inf  # No optimal solution found

aux = np.round(np.arange(0.5, 1.00, 0.05), 3)  # alphas from 0.5 to 0.95
aux = np.concatenate((aux, np.round(np.array([0.99,0.999]),3)))
#taus = np.array([g(t*(np.max(Y)-np.min(Y))) for t in [0.5]])                      
taus = np.array([g(t*(np.max(Y)-np.min(Y))) for t in aux])                      
alphas = np.array([1-a for a in aux]) # Invert the values of alpha

initsols = {(a,t) : initial_solution(a, t, beta, np.zeros(n)) for a in alphas for t in taus}


def solve_optimization(alpha, tau, beta, x0):
    print('Solving for ', alpha, tau)
    n = len(beta)
    Z = stats.norm.ppf(1-alpha)

    # Check if x0 satisfy conic equation 
    if beta.T @ x0 - tau >= Z * np.sqrt(x0.T @ L @ L.T @ x0):
        print("x0 is already feasible")
        return x0, 0.0, 0.0  # x0 is already feasible

    initsol = initsols[(alpha,tau)]
    if initsol[1] == np.inf:
        print('Infeasible')
        return x0, np.inf , np.inf
    
    # Create model
    model = gp.Model("Optimization")
    #model.setParam('OutputFlag', 0) 
    model.setParam('Presolve', 2)
    model.setParam('MIPFocus', 3)
    #model.setParam('NumericFocus', 3)
    model.setParam('MIPGap', 0.01)
    #model.setParam('NoRelHeurTime', 30)
    #model.setParam('TimeLimit', 5)
    x = [] # List of all (normalized) variables
    x_int = [] # List of (not normalized) integer variables
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
            x_int.append(yi)
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

    y = model.addVars(n,lb=-GRB.INFINITY, name="y")

    model.addConstrs((y[i] == gp.quicksum(L.T[i,j]*x[j] for j in range(n)) for i in range(n)), name="y_definition")
    model.addConstr(t*t - Z**2 * gp.quicksum(y[i] * y[i] for i in range(n)) >= 0, name="constraint2")

    #model.addConstr(t*t - Z**2 * gp.quicksum(x[i] * S[i, j] * x[j] for i in range(n) for j in range(n)) >= 0, name="constraint2") 
    #model.write("model.lp")

    # Optimize model
    # We measure the time taken to solve the optimization problem
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    total_time = end_time - start_time
    print("Time taken (seconds):", total_time)
    #model.printQuality()
    if model.status == GRB.OPTIMAL:
        print(model.ObjVal)
        xsol = [a.X for a in x]
        val = model.ObjVal
        model.dispose()
        return xsol, val, total_time  # Return the optimal solution
    else:
        if model.SolCount > 0:
            print("Best solution (objective value) found:", model.ObjVal)
            xsol = [a.X for a in x]
            model.dispose()
            return xsol, np.inf, total_time
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
        return x0, np.inf, total_time  # No optimal solution found


def solve_and_pack(params):
    alpha, tau, x0y0 = params
    x0, y0 = x0y0
   
    solution = solve_optimization(alpha, tau, beta, x0)
    xsol, objective_value, total_time = solution
    return [alpha, tau] + list(x0) + [y0] + [np.dot(x0,beta)] + list(xsol) + [np.dot(xsol,beta)] + [objective_value] + [total_time]

if __name__ == "__main__":
    x0s = [X[i, :] for i in range(X.shape[0])]
    x0sy0s = [(x0s[i], Y[i]) for i in range(len(x0s))]
    if sparse:
        base_name = "f_" + modelType + "_sparse" # or t
    else: 
        base_name = "f_" + modelType
    header = ['alpha', 'tau'] + [f'x0_{i}' for i in range(X.shape[1])] + ['y0'] + ['val_x0'] + [f'xopt_{i}' for i in range(X.shape[1])] + ['val_xopt'] + ['objective_value'] + ['time_seconds']

    tau05 = g(0.5 * (np.max(Y) - np.min(Y)))
    #combinations = list(product(alphas, taus, x0sy0s))

    #print(f"Total combinations to process: {len(combinations)}")

    parallel = False

    num = 10000
    x0sy0s_sliced = [x0sy0s[i:i + num] for i in range(0, len(x0sy0s), num)]
    for i, x0sy0s_slice in enumerate(x0sy0s_sliced):
        output_file = f"{base_name}_batch_{i}.csv"
        results = []
        #combinations_slice = [(a, tau05, xy) for a in alphas for xy in x0sy0s_slice]
        combinations_slice = [(a, tau05, xy) for a in alphas for xy in x0sy0s_slice[:50]] + [(a, tau, x0sy0s_slice[3]) for a in alphas for tau in taus]
        print(f"Processing slice {i+1} of {len(x0sy0s_sliced)} with {len(combinations_slice)} combinations")
        if parallel:
            with ProcessPoolExecutor() as executor:
                for result in executor.map(solve_and_pack, combinations_slice):
                    results.append(result)
        else:
            for j, params in enumerate(combinations_slice):
                print(f"Processing combination {j+1} of {len(combinations_slice)}")
                result = solve_and_pack(params)
                results.append(result)

        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(results)

    print("--- Process complete ---")
