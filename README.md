# StatisticalGuaranteeCE
Code to generate counterfactual explanations based on the paper "Building Robust Counterfactual Explanations with Statistical Guarantees" by Emilio Carrizosa, Antonio Navas-Orozco. The paper can be found here: https://www.researchgate.net/publication/396529217_Building_Robust_Counterfactual_Explanations_with_Statistical_Guarantees

For the results of the experiments contained in this repository, Python 3.12.8 and Gurobi 12.0.1 were used. These experiments were conducted on a MacBook Pro equipped with an Apple M4 Pro chip (12 cores: 8 performance and 4 efficiency), 48 GB of RAM, and running macOS Sequoia 15.1 (64-bit).

Description of the files:
- File StatisticalGuarantee.py contains the code to compute robust counterfactual explanations. The code starts with the selection of GLM (and with it, its associated dataset). After loading the dataset, a sparsity boolean is defined. Later, several necessary functions are defined. At the end, the code is set to either parallelly or sequentially (as chosen by the user) compute robust CEs to different individual in a one-for-one fashion. Setting sparse to True or False, and modelType to logistic, probit, linear or Poisson, allows for the reproduction of the experiments from the paper. The results are set to be saved in a csv format. Files Poisson.csv, Poisson_sparse.csv, linear.csv, linear_sparse.csv, logistic.csv, logistic_sparse.csv, probit.csv and probit_sparse.csv are the results of running robustCE.py.
- File plot_figures.py contains the code to generate the figures that are included in the paper. These can be consulted in the folder figures in pdf format.
- File load_dataset.py contains the code to load the data sets, which is needed in both StatisticalGuarantee.py and plot_figures.py. The raw data it loads is contained in files SeoulBikeData.csv, breast-cancer-wisconsin.data, communities.data and communities.names.

To run the experiments on the same conditions:
# check python version
python --version

# install requirements
pip install -r REQUIREMENTS.txt

# install Gurobi
from https://www.gurobi.com/products/gurobi-optimizer/
