import pandas as pd
from typing import *
import numpy as np
from scipy import stats
import math

EPSILON = 1e-10

def train_duration_model(training_df: pd.DataFrame, epsilon=EPSILON, **kwargs):
    """
    Train a duration model using the closed form expression for the gamma parameters
    See https://en.wikipedia.org/wiki/Gamma_distribution#Closed-form_estimators,
    where alpha = k, and beta = 1 / theta
    :param model: class, must implement fit with sample_weight and oob_score
    :param training_df:
    :param kwargs: parms to be passed to model.__init__()
    :return: trained model
    """
    training_decay_rates = training_df.groupby('patient_oriented_program_code').apply(calculate_decay_rate)
    training_decay_rates = training_decay_rates + epsilon
    sum_rates = training_decay_rates.sum()
    sum_log_rates = np.log(training_decay_rates).sum()
    sum_rate_log_rates = (training_decay_rates * np.log(training_decay_rates)).sum()
    num_rates = len(training_decay_rates)
    prior_loc = epsilon
    prior_alpha = (num_rates * sum_rates) / ((num_rates * sum_rate_log_rates) - (sum_log_rates * sum_rates))
    prior_scale = ((num_rates * sum_rate_log_rates) - (sum_log_rates * sum_rates)) / (num_rates * num_rates)
    prior_beta = 1.0 / prior_scale

    return prior_loc, prior_scale, prior_alpha, prior_beta

def predict_duration_model(df: pd.DataFrame, prior_alpha, prior_beta, current_date: Union[str, pd.Timestamp], **kwargs):
    ongoing_decay_rate_posteriors = df.groupby('patient_oriented_program_code').apply(
        calculate_posterior_decay_rate, prior_alpha=prior_alpha, prior_beta=prior_beta)

    quantile_95 = ongoing_decay_rate_posteriors.evaluated_cdf.quantile(q=0.95)
    ongoing_decay_rate_posteriors['predicted_underreporting'] = ongoing_decay_rate_posteriors.evaluated_cdf >= quantile_95
    ongoing_decay_rate_posteriors['timestamp'] = current_date
    return ongoing_decay_rate_posteriors

def split_duration_data(pop_df: pd.DataFrame):
    """
    Splits dataset into completed POPs and ongoing POPs
    :param pop_df:
    :return:
    """
    pop_df = pop_df.copy()
    completed_df = pop_df[pop_df.program_status == 'Closed']
    ongoing_df = pop_df[pop_df.program_status == 'Ongoing']
    completed_df = completed_df[completed_df.days_since_previous_case > 0]
    ongoing_df = ongoing_df[ongoing_df.days_since_previous_case > 0]

    return completed_df, ongoing_df

def calculate_decay_rate(pop_cases, epsilon=EPSILON):
  loc, scale = stats.expon.fit(pop_cases.days_since_previous_case, loc=epsilon)
  return scale

def calculate_posterior_decay_rate(pop_cases, prior_alpha, prior_beta):
    days_since_previous_case = pop_cases.days_since_previous_case
    days_since_previous_case = days_since_previous_case.dropna()
    n = len(days_since_previous_case)
    xbar = days_since_previous_case.mean()
    if n > 0:
        days_since_pop_last_case = pop_cases.days_since_pop_last_case.iloc[0]
        patient_oriented_program_id = pop_cases.patient_oriented_program_id.iloc[0]
        days_since_pop_skipped_60_last_case = pop_cases.days_since_pop_skipped_60_last_case.iloc[0]
    else:
        days_since_pop_last_case = 0
        patient_oriented_program_id = pop_cases.patient_oriented_program_id.iloc[0]
        days_since_pop_skipped_60_last_case = 0

    post_alpha = prior_alpha + n
    post_beta = prior_beta + (n * xbar)

    # Analytically calculate probability that duration should be less than the days_since_pop_last_case.
    # We flag cases where this is greater than 0.95
    evaluated_cdf = (1.0 - np.power(post_beta / (post_beta + days_since_pop_last_case), post_alpha))
    evaluated_skipped_60_last_case_cdf = (
                1.0 - np.power(post_beta / (post_beta + days_since_pop_skipped_60_last_case), post_alpha))

    fixed_evaluated_cdf = evaluated_cdf
    if math.isnan(evaluated_cdf):
        # We can get some float errors resulting in NaNs if there are too many prior observations...
        # In that case, we just ignore the evidence.
        fixed_evaluated_cdf = (1.0 - (np.power(prior_beta, prior_alpha) / (
            np.power(prior_beta + days_since_pop_last_case, prior_alpha))))
    return pd.Series({
        'post_alpha': post_alpha,
        'post_beta': post_beta,
        'num_cases': n,
        'xbar': xbar,
        'right_int_days': days_since_pop_last_case,
        'evaluated_cdf': evaluated_cdf,
        'evaluated_skipped_60_last_case_cdf': evaluated_skipped_60_last_case_cdf,
        'fixed_evaluated_cdf': fixed_evaluated_cdf,
        'patient_oriented_program_id': patient_oriented_program_id
    })
