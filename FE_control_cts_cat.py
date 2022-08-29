import pandas as pd
import bipartitepandas as bpd
import pytwoway as tw
import numpy as np

# FE
fecontrol_params = tw.fecontrol_params(
    {
        'he': True,
        'categorical_controls': 'cat_control',
        'continuous_controls': 'cts_control',
        'Q_var': [
            tw.Q.VarCovariate('psi'),
            tw.Q.VarCovariate('alpha'),
            tw.Q.VarCovariate('cat_control'),
            tw.Q.VarCovariate('cts_control'),
            tw.Q.VarCovariate(['psi', 'alpha']),
            tw.Q.VarCovariate(['cat_control', 'cts_control'])
                 ],
        'Q_cov': [
            tw.Q.CovCovariate('psi', 'alpha'),
            tw.Q.CovCovariate('cat_control', 'cts_control'),
            tw.Q.CovCovariate(['psi', 'alpha'], ['cat_control', 'cts_control'])
        ],
        'ncore': 1
    }
)
# Cleaning
clean_params = bpd.clean_params(
    {
        'connectedness': 'leave_out_spell',
        'collapse_at_connectedness_measure': True,
        'drop_single_stayers': True,
        'drop_returns': 'returners',
        'copy': False
    }
)

# Load data into Pandas DataFrame
df = pd.read_csv("/Users/mazhihao/Desktop/Project/Firm Relocation/pytwoway/sim_data.csv")

# Convert into BipartitePandas DataFrame
bdf_b = bpd.BipartiteDataFrame(df)

bdf_b2 = bdf_b.clean(clean_params)

# Initialize FE estimator
fe_estimator = tw.FEControlEstimator(bdf_b2, fecontrol_params)
# Fit FE estimator
fe_estimator.fit()

print('fe_estimator.summary')