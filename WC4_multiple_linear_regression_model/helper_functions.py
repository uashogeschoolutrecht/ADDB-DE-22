# !pip install scipy==1.7.3
# !pip install pandas==1.5.3
# !pip install statsmodels==0.14.0

import statsmodels.api as sm
import pandas as pd
import scipy 


def linear_reg(y, X):
    """
    Runs a linear regression using statsmodels, extracts results from the
    model. Returns a list of results and the OLS results object.

    :param y: outcome variable
    :param X: predictor variable(s)
    :return results: list of extracted stats/results from statsmodels OLS object
    :return model: OLS results object
    """
    # run regression - add column of 1s to X to serve as intercept
    model = sm.OLS(y, sm.add_constant(X)).fit()
    
    names = list(X)
    # extract results from statsmodel OLS object
    results = [names, model.nobs, model.df_resid, model.df_model,
               model.rsquared, model.fvalue, model.f_pvalue, model.ssr,
               model.centered_tss, model.mse_model, model.mse_resid,
               model.mse_total]

    # create dicts with name of each parameter in model (i.e. predictor
    # variables) and the beta coefficient and p-value
    coeffs = {}
    p_values = {}
    for ix, coeff in enumerate(model.params):
        if ix == 0:
            coeffs['constant'] = coeff
            p_values['constant'] = model.pvalues[ix]
        else:
            coeffs[names[ix-1]] = coeff
            p_values[names[ix-1]] = model.pvalues[ix]

    results.append(coeffs)
    results.append(p_values)

    return results, model

def calculate_change_stats(model_stats):
    """
    Calculates r-squared change, f change, and p-value of f change for
    hierarchical regression results.
    
    f change is calculated using the formula:
    (r_squared change from Step 1 to Step 2 / no. predictors added in Step 2) / 
    (1 - step 2 r_squared) / (no. observations - no. predictors - 1)
    https://www.researchgate.net/post/What_is_a_significant_f_change_value_in_a_hierarchical_multiple_regression
        
    p-value of f change calculated using the formula:
    f with (num predictors added, n - k - 1) ==> n-k-1 = Residual df for Step 2
    https://stackoverflow.com/questions/39813470/f-test-with-python-finding-the-critical-value
    
    :param model_stats: description of parameter x
    :return: list containing r-squared change value, f change value, and
             p-value for f change
    
    """
    # get number of steps 
    num_steps = model_stats['step'].max()

    # calculate r-square change (r-sq of current step minus r-sq of previous step)
    r_sq_change = [model_stats.iloc[step + 1]['r-sq'] -
                   model_stats.iloc[step]['r-sq'] for step in
                   range(0, num_steps - 1)]

    # calculate f change 
    f_change = []
    for step in range(0, num_steps - 1):
        # (r_sq change / number of predictors added)
        f_change_numerator = r_sq_change[step] / (len(model_stats.iloc[step + 1]['predictors'])
                                                  - len(model_stats.iloc[step]['predictors']))
        # (1 - step2 r_sq) / (num obs - number of predictors - 1)
        f_change_denominator = ((1 - model_stats.iloc[step + 1]['r-sq']) /
                                model_stats.iloc[step + 1]['df_resid'])
        # compute f change
        f_change.append(f_change_numerator / f_change_denominator)

    # calculate pvalue of f change
    f_change_pval = [scipy.stats.f.sf(f_change[step], 1,
                                      model_stats.iloc[step + 1]['df_resid'])
                     for step in range(0, num_steps - 1)]

    return [r_sq_change, f_change, f_change_pval]

def add_standardized_betas(results, X, y):
    standardized_betas = []
    for betas in results['beta_coeff']:
        betas_st = {}
        for key in betas.keys():
            if key == 'constant':
                continue
            else:
                betas_st[key] = betas[key]*X[key].std()/y.std()
        standardized_betas.append(betas_st)
    results['standardized_betas'] = standardized_betas
    
    return results


def hierarchical_regression(y, X, variables):
    """
    Runs hierarchical linear regressions predicting y from X. Uses statsmodels
    OLS to run linear regression for each step. Returns results of regression
    in each step as well as r-squared change, f change, and p-value of f change
    for the change from step 1 to step 2, step 2 to step 3, and so on.
    
    The number of lists contained within variables_to_add specifies the number of steps of
    hierarchical regressions. If variables contains two  lists of strings,
    e.g. if variables = [[variable 1, variable 2], [variable 3, variable 4]], then a two-step
    hierarchical regression will be conducted.
    
    :param y: outcome variable (1d array/series)
    :param X: dataframe with at least all the columns with predictor variables. 
            Column names of dataframe must match the variable-names in param variables_to_add
    :param variables: nested lists with each list containing names of predictor
              variables for each step. 

    :return: model_stats - a df (rows = number of steps * cols = 18)
    with following info for each step:
        step = step number
        x = predictor names
        num_obs = number of observations in model
        df_resid = df of residuals
        df_mod = df of model
        r-sq = r-squared
        f = f-value
        f_pval = p-value
        sse = sum of squares of errors
        ssto = total sum of squares
        mse_mod = mean squared error of model
        mse_resid =  mean square error of residuals
        mse_total = total mean square error
        beta_coeff = coefficient values for intercept and predictors
        p_values = p-values for intercept and predictors
        
        #TODO: deze naar boven verplaatsen
        r-sq_change = r-squared change for model (Step 2 r-sq - Step 1 r-sq)
        f_change = f change for model (Step 2 f - Step 1 f)
        f_change_pval = p-value of f-change of model
    :return reg_models: - a nested list containing the step name of each model
    and the OLS model object 
    """
    
    # Loop through steps and run regressions for each step
    results = []
    reg_models = []
    variables_total = []
    for ix, variables_in_step in enumerate(variables): 
        variables_total +=variables_in_step
        currentStepResults, currentStepModel = linear_reg(y, X[variables_total])
        currentStepResults.insert(0, ix + 1)  # add step number to results
        
        results.append(currentStepResults)
        # add model to list of models along with step number
        reg_models.append(['Step ' + str(ix + 1), currentStepModel])
    
    # add results to model_stats dataframe
    model_stats = pd.DataFrame(results)
    
    model_stats.columns = ['step', 'predictors', 'num_obs', 'df_resid',
                           'df_mod', 'r-sq', 'f', 'f_pval', 'sse', 'ssto',
                           'mse_mod', 'mse_resid', 'mse_total', 'beta_coeff',
                           'p_values']

    # calculate r-sq change, f change, p-value of f change
    change_results = calculate_change_stats(model_stats)

    # append step number to change results
    step_nums = [x + 1 for x in [*range(1, len(change_results[0]) + 1)]]
    change_results.insert(0, step_nums)

    # add change results to change_stats dataframe
    change_stats = pd.DataFrame(change_results).transpose()
    change_stats.columns = ['step', 'r-sq_change', 'f_change', 'f_change_pval']

    # merge model_stats and change_stats
    model_stats = pd.merge(model_stats, change_stats, on='step', how='outer')
    
    model_stats = add_standardized_betas(model_stats, X, y)
    
    return model_stats, reg_models


