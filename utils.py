import numpy as np


# helper function to concatenate formulas from previous formula and feature
def concat_formula(fm_prev, feat, i):
    if i == 0:
        return fm_prev + feat
    else:
        return fm_prev + ' + ' + feat


# helper function to concatenate formulas from a list of features
def concat_formula_from_empty(feats):
    start_str = 'PM ~ '
    for i in range(len(feats)):
        start_str = concat_formula(start_str, feats[i], i)
    return start_str


# helper function to concatenate formulas from previous formula and a list of features
def concat_formula_from_another(formula, feats):
    for feat in feats:
        formula = concat_formula(formula, feat, 1)
    return formula


# helper function to calculate adjusted R-suqared value and mse for dataset, the linear regression model contains the Intercept term
def calculate_rsquared_and_mse(model, data):
    n = len(data)
    k = len(model.params)
    # print("#params: {}".format(k))
    # print(model.params)

    # df residual equals df total minus df model
    df_resid = (n - 1) - (k - 1)

    predicted_vals = model.predict(data).values
    real_vals = data['PM'].values
    real_mean = np.mean(real_vals)
    RSS = np.sum(np.power(predicted_vals - real_vals, 2))
    TSS = np.sum(np.power(real_vals - real_mean, 2))

    rsquared = 1 - RSS / TSS
    rsquared_adj = 1 - ((1 - rsquared) * (n - 1) / df_resid)

    # mse of residuals (divided by the residual degrees of freedom)
    mse_resid = RSS / df_resid
    mse = RSS / n

    return rsquared, rsquared_adj, mse_resid, mse


# helper function to calculate the mse for dataset, AdaBoost
def calculate_rsquared_and_mse_adaboost(model, x, y):
  n = len(y)

  y_pred = model.predict(x)
  real_mean = np.mean(y)
  RSS = np.sum(np.power(y_pred - y, 2))
  TSS = np.sum(np.power(y - real_mean, 2))

  rsquared = 1 - RSS / TSS
  mse = RSS / n

  return rsquared, mse
