import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import sys
import scipy.optimize
import scipy.stats as stats
import os

# Code to do the synthetic data representations of the paper.

np.random.seed(42)
folder = 'toy_example'
if not os.path.exists(folder):
    os.makedirs(folder)
k=500
n=2
linf=-3
lsup=4
def sigmoid(t):
    t = np.clip(t, -20,20) # Avoid overflow
    return 1 / (1 + np.exp(-t))
def logit(t):
    t = np.clip(t, 1e-20, 1-1e-20) # Avoid log(0)
    return np.log(t/(1-t))
x1 = np.random.uniform(linf, lsup, k)
#x1 = np.random.normal(0, 0.01, k)
x2 = np.random.uniform(linf, lsup, k)
beta = np.array([-1.0, 2.0])
alpha=0.05
x0 = np.array([3.0, 1.0])
print(f"Probability of class 1 at x0: {sigmoid(beta[0]*x0[0] + beta[1]*x0[1])}")
tau = logit(0.5)
#tau = logit(0.9)
rad0=0.5
Z=stats.norm.ppf(1-alpha)
y = np.array([np.random.binomial(1, sigmoid(beta[0]*x1[i] + beta[1]*x2[i])) for i in range(k)])

omega = np.ones(k)

# Now we fit the logistic regression model to get the coefficients (beta) without penalization, and the Fisher information matrix
X = np.column_stack((x1, x2))


def g(t): # logit
    return np.log(t/(1-t))

def g_inv(t):
    return sigmoid(t)

def gradient_log_likelihood(beta, omega):
    return np.dot(X.T, omega*(y - sigmoid(np.dot(X, beta))))

def hessian_log_likelihood(beta, omega):
    p = sigmoid(np.dot(X, beta))
    return -X.T @ np.diag(omega * p * (1 - p)) @ X

err = 0
def beta_given_omega(omega,beta0):
    scipy.optimize.minimize
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
    return result.x

beta_hat = beta_given_omega(omega, np.zeros(n))
print(f"Estimated coefficients (beta_hat): {beta_hat}")
print(f"Empirical probability of class 1 at x0: {sigmoid(beta_hat[0]*x0[0] + beta_hat[1]*x0[1])}")
Fisher_info = -hessian_log_likelihood(beta_hat, omega)
print(f"Fisher Information Matrix:\n{Fisher_info}")
S = np.linalg.inv(Fisher_info)
print(f"Inverse of the Fisher Information Matrix (S):\n{S}")
# Eigenvalues of Fisher_info:
eig = np.linalg.eigvals(Fisher_info)
print(f"Eigenvalues of the Fisher Information Matrix: {eig}")
# Eigenvectors:
eigvec = np.linalg.eig(Fisher_info)[1]
print(f"Eigenvectors of the Fisher Information Matrix:\n{eigvec}")
r1 = Z / np.sqrt(np.max(eig))
r2 = np.linalg.norm(beta_hat - beta)
r3 = Z / np.sqrt(np.min(eig))
r4 = np.linalg.norm(beta_hat)
print(r1,r2,r3,r4)

# Plotting the 'ellipsoid of confidence'
theta = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])
L = np.linalg.cholesky(S)
ellipse_points = beta_hat[:, np.newaxis] + Z * (L @ circle)
plt.figure(figsize=(8, 8))
plt.plot(ellipse_points[0, :], ellipse_points[1, :], 'g-', linewidth=2, label='Ellipsoidal uncertainty set')
plt.fill(ellipse_points[0, :], ellipse_points[1, :], color='green', alpha=0.1)
# Also plot a circle
circle_points1 = beta_hat[:, np.newaxis] + r1 * circle
plt.plot(circle_points1[0, :], circle_points1[1, :], 'k--', linewidth=1, label=r'Circle of radius $Z_{1-\alpha}/\sqrt{\lambda_{\max}}$')
circle_points3 = beta_hat[:, np.newaxis] + r3 * circle
plt.plot(circle_points3[0, :], circle_points3[1, :], 'k-.', linewidth=1, label=r'Circle of radius $Z_{1-\alpha}/\sqrt{\lambda_{\min}}$')
plt.plot(beta_hat[0], beta_hat[1], 'bo', label=r'Estimated $\hat{\boldsymbol{\beta}}$')
plt.plot(beta[0], beta[1], 'ro', label=r'True $\boldsymbol{\beta}$')
plt.xlabel(r'$\tilde\beta_1$',fontsize=15)
plt.ylabel(r'$\tilde\beta_2$',fontsize=15)
#plt.title('')
plt.axis('equal')
plt.legend(loc='upper right',fontsize=12)
plt.grid()
plt.savefig(f"{folder}/confidence_ellipsoid.pdf") # in pdf
plt.show()



pinf = linf
psup = lsup

def implicit_plot(a, b, c, d, A, color='red', lab=''):
    # ax+by=c+ d*sqrt(x'*A*x)
    s = np.linspace(pinf, psup, 500)
    xx, yy = np.meshgrid(s, s)
    Q = A[0,0]*xx**2 + (A[0,1] + A[1,0])*xx*yy + A[1,1]*yy**2
    print(min(Q.ravel()), max(Q.ravel()))
    zz = (a*xx + b*yy) - (c + d * np.sqrt(Q))
    plt.rcParams['hatch.color'] = color
    plt.contour(xx, yy, zz, levels=[0], colors=color, linestyle='-', linewidths=2)
    plt.contourf(xx, yy, zz, levels=[0, np.inf], colors=color, alpha=0.1)
    if lab:
        plt.plot([], [], color=color, linestyle='-', linewidth=2, label=lab)

# plt.figure(figsize=(8, 8))
# plt.plot(x1[y==0], x2[y==0], 'ro', alpha=0.1, label='Class 0')
# plt.plot(x1[y==1], x2[y==1], 'bo', alpha=0.1, label='Class 1')
# plt.plot(x0[0], x0[1], 'kx', markersize=10, label='x0')

# implicit_plot(beta_hat[0], beta_hat[1], tau, 0, S, color='blue', lab='Empirical Boundary')
# implicit_plot(beta[0], beta[1], tau, 0, S, color='red', lab='True Boundary')
# implicit_plot(beta_hat[0], beta_hat[1], tau, Z, S, color='green', lab='Confidence Boundary')
# implicit_plot(beta_hat[0], beta_hat[1], tau, 0.12, np.eye(n), color='black', lab='Parameter-robust Boundary')
# #plt.axis('equal')
# plt.xlim(pinf, psup)
# plt.ylim(pinf, psup)
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.title('Synthetic Data with Logistic Boundary')
# plt.legend(loc='best')
# plt.grid()#    plt.grid(True, alpha=0.3)
# plt.show()


def solve_with_gurobi(A, r, b=beta_hat):
    # Create model
    m = gp.Model("projection")
    m.setParam('OutputFlag', 0) 
    
    x = m.addMVar(shape=2, lb=-GRB.INFINITY, name="x")
    
    obj = (x[0] - x0[0])**2 + (x[1] - x0[1])**2
    m.setObjective(obj, GRB.MINIMIZE)

    t = m.addVar(lb=0.0, name="t")
    m.addConstr(t == b @ x - tau, name="def_t")
    
    m.addConstr(t*t - r**2 * (x @ A @ x) >= 0, name="socp_cone")

    # Resolver
    m.optimize()
    
    if m.status == GRB.OPTIMAL:
        return x.X, m.ObjVal
    else:
        return None, None

rplot = np.linspace(0,2.5,500)
c_param_robust = np.array([solve_with_gurobi(np.eye(n), r)[1] for r in rplot])
t=[]
sols = []
for r in rplot:
    xsol = solve_with_gurobi(np.eye(n), r)[0]
    if xsol is not None:
        sols.append(xsol)

c_confidence = solve_with_gurobi(S, Z)[1]
c_empirical = solve_with_gurobi(S, 0)[1]
print(f"Empirical solution: {solve_with_gurobi(S, 0)[0]}, confidence solution: {solve_with_gurobi(S, Z)[0]}")
print(f"Empirical prediction at solution: {sigmoid(beta_hat @ solve_with_gurobi(S, 0)[0])}, real prediction at solution: {sigmoid(beta @ solve_with_gurobi(S, 0)[0])}", f"Confidence prediction at solution: {sigmoid(beta_hat @ solve_with_gurobi(S, Z)[0])}, real prediction at confident solution: {sigmoid(beta @ solve_with_gurobi(S, Z)[0])}")
real_solution, real_sol_value = solve_with_gurobi(S, 0, b=beta)
print('Real solution with true beta:', real_solution, 'with objective value:', real_sol_value)
plt.figure(figsize=(8, 6))
plt.plot(rplot, c_param_robust, label='Parameter-robust CE objective', color='black')
#plt.plot(rplot, c_param_robust, label='Parameter-robust Boundary', color='black')
plt.axhline(c_empirical, color='blue', linestyle='--', label='Ordinary CE objective: ' + f"{c_empirical:.3f}")
plt.axhline(c_confidence, color='green', linestyle='--', label='Robust CE objective: ' + f"{c_confidence:.3f}")
plt.axhline(real_sol_value, color='red', linestyle='--', label='Distance to ground-truth boundary: ' + f"{real_sol_value:.3f}") 
# Vertical lines at r1, r2, r3, r4
plt.axvline(r1, color='yellow', linestyle='-.', label=r'$Z_{1-\alpha}/\sqrt{\lambda_{\max}}=$' + f"{r1:.3f}")
plt.axvline(r2, color='orange', linestyle='-.', label=r'$\|\hat{\boldsymbol{\beta}} - \boldsymbol{\beta}\|$' + f"={r2:.3f}")
plt.axvline(r3, color='purple', linestyle='-.', label=r'$Z_{1-\alpha}/\sqrt{\lambda_{\min}}$' + f"={r3:.3f}")
plt.axvline(r4, color='brown', linestyle='-.', label=r'$\|\hat{\boldsymbol{\beta}}\|$' + f"={r4:.3f}")
plt.xlabel(r'$r$',fontsize=15)
plt.ylabel('Optimal objective value',fontsize=15)
#plt.title('Distance from x0 to Boundary as a Function of Radius')
plt.legend(fontsize=10)
plt.grid()
plt.savefig(f"{folder}/objective_vs_radius.pdf") # in pdf
plt.show()




plt.figure(figsize=(8, 8))
plt.plot(x1[y==0], x2[y==0], 'ro', alpha=0.1, label='Negative Class')
plt.plot(x1[y==1], x2[y==1], 'bo', alpha=0.1, label='Positive Class')
plt.plot(x0[0], x0[1], 'kx', markersize=10, label=r'$x_0$')
#plt.plot(real_solution[0], real_solution[1], 'm.', markersize=20, label='Ground-truth CE')
implicit_plot(beta_hat[0], beta_hat[1], tau, 0, S, color='blue', lab='Empirical region')
implicit_plot(beta[0], beta[1], tau, 0, S, color='red', lab='Ground-truth region')
confidence_solution = solve_with_gurobi(S, Z)[0]
#plt.plot(confidence_solution[0], confidence_solution[1], 'g.', markersize=20, label='Robust CE')
empirical_solution = solve_with_gurobi(S, 0)[0]
print(f"Confidence solution: {confidence_solution}\nEmpirical solution: {empirical_solution}")
plt.plot(empirical_solution[0], empirical_solution[1], 'b.', markersize=15, label='Empirical CE')
#for x in sols:
#    plt.plot(x[0], x[1], 'kx', markersize=5, alpha=0.5)
# plt.plot([], [], 'kx', markersize=5, alpha=0.5, label='Parameter-robust CEs')
#plt.axis('equal')
plt.xlim(pinf, psup)
plt.ylim(pinf, psup)
plt.xlabel(r'$x_1$',fontsize=15)
plt.ylabel(r'$x_2$',fontsize=15)
#plt.title('Synthetic Data with Logistic Boundary')
plt.legend(loc='best', fontsize=12)
plt.grid()#    plt.grid(True, alpha=0.3)
plt.savefig(f"{folder}/synthetic_data_logistic_boundary.pdf") # in pdf
plt.show()

print('Value at r=', r3, ':', solve_with_gurobi(np.eye(n), r3)[1])


plt.figure(figsize=(8, 8))
plt.plot(x1[y==0], x2[y==0], 'ro', alpha=0.1, label='Negative Class')
plt.plot(x1[y==1], x2[y==1], 'bo', alpha=0.1, label='Positive Class')
plt.plot(x0[0], x0[1], 'kx', markersize=10, label=r'$x_0$')

implicit_plot(beta_hat[0], beta_hat[1], tau, 0, S, color='blue', lab='Empirical region')
implicit_plot(beta[0], beta[1], tau, 0, S, color='red', lab='Ground-truth region')
implicit_plot(beta_hat[0], beta_hat[1], tau, Z, S, color='green', lab=r'Robust region $\Omega_\alpha(\hat{\boldsymbol{\beta}})$')
implicit_plot(beta_hat[0], beta_hat[1], tau, rad0, np.eye(n), color='black', lab='Parameter-robust region')
confidence_solution = solve_with_gurobi(S, Z)[0]
empirical_solution = solve_with_gurobi(S, 0)[0]
print(f"Confidence solution: {confidence_solution}\nEmpirical solution: {empirical_solution}")
for x in sols:
    plt.plot(x[0], x[1], 'kx', markersize=5, alpha=0.5)
plt.plot([], [], 'kx', markersize=5, alpha=0.5, label='Parameter-robust CEs')
plt.plot(empirical_solution[0], empirical_solution[1], 'b.', markersize=15, label='Empirical CE')
plt.plot(confidence_solution[0], confidence_solution[1], 'g.', markersize=15, label='Robust CE')
#plt.axis('equal')
plt.xlim(pinf, psup)
plt.ylim(pinf, psup)
plt.xlabel(r'$x_1$',fontsize=15)
plt.ylabel(r'$x_2$',fontsize=15)
#plt.title('Synthetic Data with Logistic Boundary')
plt.legend(loc='upper left', fontsize=12)
plt.grid()#    plt.grid(True, alpha=0.3)
plt.savefig(f"{folder}/synthetic_data_logistic_boundary2.pdf") # in pdf
plt.show()