import numpy as np
import pandas as pd
import copy
import itertools
import math
from scipy import stats
from sklearn.utils import resample

import pcwg03_initialize as p_init

import pcwg03_config as pc
import pcwg03_energy_fraction as pef

meta_df, error_df, extra_error_df = p_init.meta_df, p_init.error_df, p_init.extra_error_df

base_bin_e_df = error_df['base_bin_e']
base_total_e_df = error_df['base_total_e']


def get_base_e_df(error_cat):
    """Load Baseline data frames."""

    nme_bin_e = base_bin_e_df.loc[(((base_bin_e_df['error_cat'] == 'by_range')
                                    & (base_bin_e_df['bin_name'] == 'Outer'))
                                   | (base_bin_e_df['error_cat'] == error_cat))
                                  & (base_bin_e_df['error_name'] == 'nme')]

    dc_total_e = base_total_e_df.loc[((base_total_e_df['error_cat'] == 'overall')
                                      | ((base_total_e_df['error_cat'] == 'by_range')
                                         & (base_total_e_df['bin_name'] == 'Outer'))
                                      | (base_total_e_df['error_cat'] == error_cat))
                                     & (base_total_e_df['error_name'] == 'data_count')]

    nme_total_e = base_total_e_df.loc[(((base_total_e_df['error_cat'] == 'by_range')
                                        & (base_total_e_df['bin_name'] == 'Outer'))
                                       | (base_total_e_df['error_cat'] == error_cat))
                                      & (base_total_e_df['error_name'] == 'nme')]

    return nme_bin_e, dc_total_e, nme_total_e


def get_base_total_e(error_cat):
    """Load Baseline total error data frame."""

    base_total_e = base_total_e_df.loc[((base_total_e_df['error_cat'] == error_cat)
                                        | (base_total_e_df['error_cat'] == 'by_range'))
                                       & (base_total_e_df['error_name'] == 'nme')]

    return base_total_e


def get_error_in_bin(df, sheet, by_bin, error_name):

    return df[sheet].loc[(df[sheet]['error_cat'] == by_bin) & (df[sheet]['error_name'] == error_name)]


def get_outer_range_nme(df):

    return df.loc[(df['error_cat'] == 'by_range') & (df['bin_name'] == 'Outer') & (df['error_name'] == 'nme')]


def get_wsti_outer_nme(sheet, error_cat):

    out_df = error_df[sheet + 'total_e'].loc[((error_df[sheet + 'total_e']['error_cat'] == error_cat)
                                              | (error_df[sheet + 'total_e']['bin_name'] == 'Outer'))
                                             & (error_df[sheet + 'total_e']['error_name'] == 'nme')]

    return out_df


def get_sheet_wsti_range_all_total_e(sheet):
    """Load NME data frame of WS-TI, Inner-Outer Range, and Overall bins."""

    sheet_i = sheet + 'total_e'

    error_cat = 'by_ws_ti'

    nme_df = error_df[sheet_i].loc[((error_df[sheet_i]['error_cat'] == error_cat)
                                    | (error_df[sheet_i]['error_cat'] == 'by_range')
                                    | (error_df[sheet_i]['error_cat'] == 'overall'))
                                   & (error_df[sheet_i]['error_name'] == 'nme')]

    ef_df = pef.cal_wsti_ef(error_cat)

    problem_file = pef.check_problematic_file(ef_df, error_cat)

    out_df = pef.remove_problematic_files(nme_df, problem_file, error_cat, pc.wsti_new_bin)

    return out_df


def cal_average_spread(df, u_bin, average_df, spread_df, sheet, rr_choice=pc.robust_resistant_choice):
    """Calculate average and spread statistics for data frame."""

    average = np.empty(len(u_bin))
    spread = np.empty(len(u_bin))

    if rr_choice is None:

        for idx, val in enumerate(u_bin):
            average[idx] = df.loc[df['bin_name'] == val]['error_value'].mean() * 100.
            spread[idx] = df.loc[df['bin_name'] == val]['error_value'].std() * 100.

    else:

        for idx, val in enumerate(u_bin):
            average[idx] = df.loc[df['bin_name'] == val]['error_value'].median() * 100.

            q1 = df.loc[df['bin_name'] == val]['error_value'].quantile(0.25)
            q3 = df.loc[df['bin_name'] == val]['error_value'].quantile(0.75)
            spread[idx] = (q3 - q1) * 100.

    average_df[sheet] = average
    spread_df[sheet] = spread


def strip_df_underscore(df, sheet):
    """Remove underscore in sheet (method) name."""

    df.rename(columns={sheet: sheet.rstrip('_')}, inplace=True)


def strip_df_add_file_count(sheet, df, u_bin):
    """Remove underscore in sheet (method) name and add total submission number of that sheet."""

    df.rename(columns={sheet: sheet.rstrip('_') + ': ' + str(round(len(df) / len(u_bin)))}, inplace=True)


def find_unique_bin_create_dum(series):
    """Create empty array."""

    u_bin = series.unique()

    average = np.empty(len(u_bin))
    spread = np.empty(len(u_bin))

    return u_bin, average, spread


def get_wsti_nme_stat():
    """Get average and spread statistics for WS-TI bins."""

    all_wsti_nme_df = pd.DataFrame()

    for idx, sheet in enumerate(pc.matrix_sheet_name_short):

        wsti_nme_df = get_sheet_wsti_range_all_total_e(sheet)

        u_bin = wsti_nme_df['bin_name'].unique()

        if idx == 0:
            average_df = pd.DataFrame(index=u_bin, columns=[pc.matrix_sheet_name_short])
            spread_df = pd.DataFrame(index=u_bin, columns=[pc.matrix_sheet_name_short])

        cal_average_spread(wsti_nme_df, u_bin, average_df, spread_df, sheet)

        strip_df_underscore(average_df, sheet)
        strip_df_underscore(spread_df, sheet)

        ws_ti_df_toadd = copy.copy(wsti_nme_df)
        ws_ti_df_toadd['method'] = sheet.rstrip('_')
        all_wsti_nme_df = pd.concat([all_wsti_nme_df, ws_ti_df_toadd], axis=0)
        all_wsti_nme_df.reset_index(inplace=True, drop=True)

    return average_df, spread_df, all_wsti_nme_df


def sort_plot_wsti_df_index(df):
    """Sort WS-TI bin order for plotting."""

    df.rename(index={'ALL': 'Overall'}, inplace=True)
    df = df.reindex(index=pc.sort_wsti_index)

    return df


def group_meta_element_in_range(key, value):
    """Combine meta data into groups for plotting grouped histograms."""

    series_to_edit = meta_df[key]

    if key == 'turbi_spower':  # change units for specific power

        series_to_edit = series_to_edit * 1e3

    tickmark_lim = np.linspace(series_to_edit.min(), series_to_edit.max(), 10)

    if key == 'turbi_rated_power':
        round_d = 3
    elif key == 'turbi_d_hh_ratio':
        round_d = 2
    else:
        round_d = 0

    series_edited = ['' for x in range(len(series_to_edit))]

    for i in range(len(series_to_edit)):
        for j in range(len(tickmark_lim) - 1):
            if series_to_edit[i] >= tickmark_lim[j] and series_to_edit[i] <= tickmark_lim[j + 1]:

                if key == 'turbi_d_hh_ratio':

                    tickmark_start = str(np.round(tickmark_lim[j], round_d))
                    tickmark_end = str(np.round(tickmark_lim[j + 1], round_d))

                else:

                    tickmark_start = str(np.round(tickmark_lim[j], round_d))[:-2]
                    tickmark_end = str(np.round(tickmark_lim[j + 1], round_d))[:-2]

                if (series_to_edit.max() >= 100 and tickmark_lim[j] < 100):
                    tickmark_start = '0' + tickmark_start  # add 0 for sorting

                series_edited[i] = tickmark_start + ' - ' + tickmark_end

            if np.isnan(series_to_edit[i]):
                series_edited[i] = str(np.nan)

    meta_df[value] = series_edited


def remove_0_in_label(df):
    """Remove 0 in the beginning of a string for plotting."""

    x_sorted = df['index'].sort_values()
    x_nozero = copy.copy(x_sorted)
    x_nozero = x_nozero.reset_index()

    if isinstance(x_sorted[0], str) and x_sorted[0][0] == '0':

        for idx, val in enumerate(x_sorted):

            if val[0] == '0':
                with pd.option_context('mode.chained_assignment', None):
                    x_nozero['index'][idx] = x_nozero['index'][idx][1:]  # remove 0 in string

    return x_sorted, x_nozero


def get_outer_meta(error, meta_var, bt_c, y_var):
    """Find error for each method, calculate difference from Baseline.
    Correlate error with meta data variables, if they are numerically represented.
    """

    lump_df = pd.DataFrame()
    lump_corr = np.zeros(0)

    for i, sheet in enumerate(pc.matrix_sheet_name_short):

        outer = error_df[sheet + bt_c].loc[(error_df[sheet + bt_c]['error_cat'] == 'by_range')
                                           & (error_df[sheet + bt_c]['bin_name'] == 'Outer')
                                           & (error_df[sheet + bt_c]['error_name'] == error)]

        base = error_df['base_' + bt_c].loc[(error_df['base_' + bt_c]['error_cat'] == 'by_range')
                                            & (error_df['base_' + bt_c]['bin_name'] == 'Outer')
                                            & (error_df['base_' + bt_c]['error_name'] == error)]

        with pd.option_context('mode.chained_assignment', None):

            if sheet == 'base_':

                outer['diff'] = np.NaN

            # calculate difference between correction methods and Baseline
            else:

                outer['diff'] = (abs(outer['error_value']) - abs(base['error_value'])) * 100

            outer['sheet'] = str(sheet)[:-1]

        outer_all = pd.merge(outer, meta_df, on='file_name')

        if all(isinstance(x, (float, int)) for x in meta_df[meta_var]):  # if meta x-axis is numeric

            corr = np.corrcoef(list(outer_all[y_var].values), list(outer_all[meta_var].values))

            if not math.isnan(corr[0][1]):
                lump_corr = np.append(lump_corr, round(corr[0][1], 2))

        lump_df = pd.concat([lump_df, outer_all], sort=True)

    return lump_df, lump_corr


def get_nme_diff_range():
    """Calculate statistical range (max - min) for |NME| differences for each submission."""

    outer_base_te_df = get_outer_range_nme(base_total_e_df)
    outer_dt_te_df = get_outer_range_nme(error_df['den_turb_total_e'])
    outer_d2_te_df = get_outer_range_nme(error_df['den_2dpdm_total_e'])
    outer_dat_te_df = get_outer_range_nme(error_df['den_augturb_total_e'])
    outer_d3_te_df = get_outer_range_nme(error_df['den_3dpdm_total_e'])

    for idx, file in enumerate(outer_base_te_df['file_name'].unique()):

        base_nme = outer_base_te_df.loc[outer_base_te_df['file_name'] == file]['error_value'].values[0] * 100.
        dt_nme = outer_dt_te_df.loc[outer_dt_te_df['file_name'] == file]['error_value'].values[0] * 100.
        d2_nme = outer_d2_te_df.loc[outer_d2_te_df['file_name'] == file]['error_value'].values[0] * 100.
        dat_nme = outer_dat_te_df.loc[outer_dat_te_df['file_name'] == file]['error_value'].values[0] * 100.
        d3_nme = outer_d3_te_df.loc[outer_d3_te_df['file_name'] == file]['error_value'].values[0] * 100.

        method_nme_list = np.array([abs(dt_nme), abs(d2_nme), abs(dat_nme), abs(d3_nme)])

        nme_diff = method_nme_list - abs(base_nme)

        if idx == 0:
            nme_diff_list = nme_diff

        else:
            nme_diff_list = np.vstack((nme_diff_list, nme_diff))

    nme_diff_df = pd.DataFrame(nme_diff_list.T)

    nme_range = nme_diff_df.max() - nme_diff_df.min()

    improve_outer_list = ['Mixed'] * nme_diff_df.shape[1]

    for col in range(nme_diff_df.shape[1]):

        if all(item > 0 for item in nme_diff_df[col]):  # all methods' NMEs worse than Baseline
            improve_outer_list[col] = 'Worse'

        elif all(item < 0 for item in nme_diff_df[col]):
            improve_outer_list[col] = 'Improved'

    nme_range_p_df = pd.DataFrame({'nme': nme_range, 'all': improve_outer_list})
    nme_range_p_df.reset_index(inplace=True)

    return nme_diff_df, nme_range_p_df


def get_methods_nme(error_cat):
    """Get average and spread statistics for Outer Range WS bins."""

    all_outws_nme_df = pd.DataFrame()

    for method in pc.matrix_sheet_name_short:
        df = error_df[method + 'total_e']
        df_s = df.loc[(df['error_cat'] == error_cat) & (df['error_name'] == 'nme')]
        df_s_toadd = copy.copy(df_s)
        df_s_toadd['method'] = method.rstrip('_')
        all_outws_nme_df = pd.concat([all_outws_nme_df, df_s_toadd], axis=0)

    all_outws_nme_df.reset_index(inplace=True, drop=True)

    return all_outws_nme_df


def extract_base_df_s(df):

    base_df = df.loc[df['method'] == 'base']

    return base_df.reset_index()


def extract_method_df_s(sheet, df):

    method_df = df.loc[df['method'] == sheet.rstrip('_')]

    return method_df.reset_index()


def get_bin_array(df_s, bin_name):
    """Get error in error bin."""

    return df_s.loc[(df_s['bin_name'] == bin_name)]['error_value']


def drop_array_na(arr):
    """Locate index in series with NA values."""

    return arr.loc[(pd.isna(arr) == True)].index


def remove_quantile_in_array(arr):
    """Remove x quantile in array.
    To exclude submissions with "extreme improvements".
    """

    bottom = arr.quantile(pc.quantile_cut)  # bottom x%

    return arr.drop(arr[arr.values < bottom].index)


def perform_stat_test(wsti=False, error_cat=None,
                      remove_outlier_choice=False, remove_quantile=False, bonferroni=None, percent_thres=None):
    """Perform statistical tests for |NME| differences between methods and Baseline.
    Calculate percentage of submissions of a method in an error bin that improves from Baseline.
    Perform one-sample, one-sided t-test on |NME| differences.
    Perform two-sided Levene's test on |NME| differences.
    Offer 2 options of outlier removal: either remove by quantile (use pc.quantile_cut),
    or remove by a certain percentage (use pc.percent_thres_choice).
    Offer adjustments to alpha via Bonferroni Correction.
    """

    plot_choice = False

    if wsti is True:
        dum1, dum2, nme_df = get_wsti_nme_stat()
    else:
        if error_cat is None:
            print('missing ')
        else:
            nme_df = get_methods_nme(error_cat)

    base_df_s = extract_base_df_s(nme_df)

    for method_num, method_sheet in enumerate(pc.matrix_sheet_name_short[1:]):

        method_df_s = extract_method_df_s(method_sheet, nme_df)

        if (base_df_s['file_name'].values != method_df_s['file_name'].values).all():
            print('file names in baseline and method df do not match!')

        u_bin = base_df_s['bin_name'].unique()

        pc_improve = np.zeros(len(u_bin))
        diff_ttest = np.zeros(len(u_bin))
        ltest = np.zeros(len(u_bin))

        if method_num == 0:
            pc_df = pd.DataFrame(index=u_bin, columns=[pc.matrix_sheet_name_short[1:]])
            diff_ttest_df = pd.DataFrame(index=u_bin, columns=[pc.matrix_sheet_name_short[1:]])
            ltest_df = pd.DataFrame(index=u_bin, columns=[pc.matrix_sheet_name_short[1:]])
            diff_removal_num = np.zeros(0)

        for idx, val in enumerate(u_bin):

            base_array = get_bin_array(base_df_s, val)
            method_array = get_bin_array(method_df_s, val)

            base_na = drop_array_na(base_array)
            method_na = drop_array_na(method_array)

            # Bonferroni Correction, aka make alpha smaller dependent on the number of stat tests
            if bonferroni == 1:  # looser, because each method is independent
                if wsti is True:
                    alpha_thres = pc.alpha_choice / pc.alpha_thres_wsti_list[idx]
                else:
                    alpha_thres = pc.alpha_choice / (len(u_bin))
            elif bonferroni == 2:  # stricter
                alpha_thres = pc.alpha_choice / (len(u_bin) * len(pc.matrix_sheet_name_short[1:]))
            else:
                alpha_thres = pc.alpha_choice

            if all(base_na) == all(method_na):  # ensure the nan's are at the same indices

                base_data = base_array.dropna()
                method_data = method_array.dropna()

                base_data_dum = copy.deepcopy(base_data)
                method_data_dum = copy.deepcopy(method_data)

                # need 2 samples to do stat tests
                if (len(base_data) > 1) and (len(method_data) > 1):

                    # individual improvement, negative means improved
                    # compare absolute value of NME
                    diff_array = (abs(method_data) - abs(base_data)) * 100.

                    diff_array_dum = copy.deepcopy(diff_array)

                    # make t-test more rigorous by removing data points of "extreme" improvement
                    if remove_outlier_choice is True:

                        if remove_quantile is True:  # remove x percent of "extreme" improvement
                            diff_data_no_outlier = remove_quantile_in_array(diff_array)

                        else:
                            # remove "extreme" improvements above 1 percent of absolute magnitude
                            diff_data_no_outlier = diff_array.drop((diff_array[diff_array.values
                                                                               < -percent_thres].index))

                        # number of removed submissions
                        diff_removal = len(diff_array) - len(diff_data_no_outlier)
                        diff_removal_num = np.append(diff_removal_num, diff_removal)

                        if diff_removal > 0:

                            # if choose to remove outliers, only plot when outliers are successfully removed
                            plot_choice = True
                            diff_array_dum = diff_data_no_outlier

                            if remove_quantile is False:
                                print('remove ' + str(
                                    diff_removal) + ' submissions at: ' + error_cat + ' ' + val + ' ' + error)

                            # remove BOTH "extreme" improvements and deterioration for Levene's test
                            base_data_dum = base_data.drop((diff_array[diff_array.values < -percent_thres].index)
                                                           | (diff_array[diff_array.values > percent_thres].index))
                            method_data_dum = method_data.drop((diff_array[diff_array.values < -percent_thres].index)
                                                               | (diff_array[diff_array.values > percent_thres].index))

                    else:
                        plot_choice = True

                    loc_improve = np.where(diff_array_dum < pc.diff_benchmark)
                    len_improve = np.shape(loc_improve)[1]
                    pc_improve[idx] = 100 * len_improve / len(diff_array_dum)

                    # mean diff of individual error < diff_benchmark
                    if diff_array_dum.mean() < pc.diff_benchmark:
                        diff_ttest[idx] += 1

                        # some error categories do not have enough data
                        # hence t-test may fail after outlier removal
                        try:
                            diff_t_stat = stats.ttest_1samp(diff_array_dum, pc.diff_benchmark)
                        except ZeroDivisionError:
                            class diff_t_stat:
                                statistic = np.nan
                                pvalue = np.nan

                        # one-sample, two-sided t-test
                        # if diff_t_stat.pvalue <= alpha_thres: # reject H0: no diff, or diff = 0

                        # one-sample, one-sided t-test
                        # reject H0: no diff, or diff = 0
                        # Ha: mean diff of individual error < diff_benchmark
                        if ((diff_t_stat.statistic < 0)  # t-statistic < 0 (differ from diff_benchmark)
                                & (diff_t_stat.pvalue / 2 <= alpha_thres)):  # one-sided, half of p-value

                            # mean diff of individual error < diff_benchmark *significantly*
                            diff_ttest[idx] += 1

                            # do KS test when outliers are removed
                            if ((remove_outlier_choice is True) & (plot_choice is True)):
                                ks_stat = stats.kstest(list(diff_data_no_outlier.values), 'norm')
                                if ks_stat.pvalue <= alpha_thres:
                                    # print(method_sheet+' is statistically significant from Gaussian')
                                    pass
                                else:
                                    print(error_cat + ' ' + val + ' ' + b_or_t + ' ' + error + ':')
                                    print(method_sheet + ' is NOT statistically significant from Gaussian')

                    # if np.abs(base_data_dum.std()) > np.abs(method_data_dum.std()): # sd < baseline
                    if np.abs(base_data_dum.var()) > np.abs(method_data_dum.var()):  # variance < baseline
                        ltest[idx] += 1

                        # Levene's test is better than F-test, valid for non-Gaussian distributions
                        f_stat = stats.levene(base_data_dum, method_data_dum)

                        # Levene's test seems to only be 2-sided...
                        # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35a.htm
                        # vs
                        # https://www.itl.nist.gov/div898/handbook/eda/section3/eda359.htm
                        if f_stat.pvalue <= alpha_thres:  # reject H0: same variance
                            ltest[idx] += 1  # sd or variance < baseline *significantly*

        pc_df[method_sheet] = pc_improve
        diff_ttest_df[method_sheet] = diff_ttest
        ltest_df[method_sheet] = ltest

        strip_df_underscore(pc_df, method_sheet)
        strip_df_underscore(diff_ttest_df, method_sheet)
        strip_df_underscore(ltest_df, method_sheet)

    if wsti is True:

        pc_df = sort_plot_wsti_df_index(pc_df)
        diff_ttest_df = sort_plot_wsti_df_index(diff_ttest_df)
        ltest_df = sort_plot_wsti_df_index(ltest_df)

    pc_df.rename(columns=pc.method_dict, inplace=True)
    diff_ttest_df.rename(columns=pc.method_dict, inplace=True)
    ltest_df.rename(columns=pc.method_dict, inplace=True)

    return plot_choice, pc_df, diff_ttest_df, ltest_df, diff_removal_num


def perform_kstest(arr):
    """Perform KS test: array distribution is different from Gaussian or not."""

    ks_stat = stats.kstest(list(arr.values), 'norm')

    if ks_stat.pvalue <= pc.alpha_choice:
        # reject KS test null hypothesis; array differs from Gaussian with statistical significance
        out = True
    else:
        out = False

    return out


def run_bootstrap_append_mean(data, n):
    """Resample data and append its mean in each iteration."""

    out = np.zeros(0)
    boot_count = 0

    while boot_count < pc.boot_loop_num:
        sample = resample(data, replace=True, n_samples=n)

        # get list of means
        out = np.append(out, sample.mean())

        boot_count += 1

    return out


def cal_bootstrap_means(df_boot, remove_outlier=None, hypo_test=None):
    """Get means of bootstrapped samples.
    Each sample contains |NME| differences between a method and Baseline, for each error bin.
    Options of choosing outlier removal and doing bootstrap hypothesis test.
    """

    ref_df = df_boot.loc[df_boot['method'] == 'base']
    u_bin = ref_df['bin_name'].unique()

    if hypo_test is None:
        diff_boot_mean_mat = np.empty((len(pc.matrix_sheet_name_short[1:]), len(u_bin), int(pc.boot_loop_num)))
    else:
        diff_boot_mean_mat = np.empty((len(pc.matrix_sheet_name_short[1:]), len(u_bin), int(pc.boot_loop_num), 2))

    base_df_s = extract_base_df_s(df_boot)

    for method_num, method_sheet in enumerate(pc.matrix_sheet_name_short[1:]):

        method_df_s = extract_method_df_s(method_sheet, df_boot)

        if (base_df_s['file_name'].values != method_df_s['file_name'].values).all():
            print('file names in baseline and method df do not match!')

        diff_removal_num = np.zeros(0)

        for idx, val in enumerate(u_bin):

            base_array = get_bin_array(base_df_s, val)
            method_array = get_bin_array(method_df_s, val)

            base_na = drop_array_na(base_array)
            method_na = drop_array_na(method_array)

            if all(base_na) == all(method_na):  # ensure the nan's are at the same indices

                base_data = (base_array.dropna()) * 100.
                method_data = (method_array.dropna()) * 100.

                base_data_dum = copy.deepcopy(base_data)
                method_data_dum = copy.deepcopy(method_data)

                # need 2 samples to do stat tests
                if (len(base_data) > 1) and (len(method_data) > 1):

                    # individual improvement, negative means improved
                    # compare absolute value of NME
                    diff_array = (abs(method_data_dum) - abs(base_data_dum))

                    diff_data_dum = copy.deepcopy(diff_array)

                    # make t-test more rigorous by removing data points of "extreme" improvement
                    if remove_outlier is not None:

                        diff_data_no_outlier = remove_quantile_in_array(diff_array)

                        # number of removed submissions
                        diff_removal = len(diff_array) - len(diff_data_no_outlier)
                        diff_removal_num = np.append(diff_removal_num, diff_removal)

                        if diff_removal > 0:
                            diff_data_dum = diff_data_no_outlier

                    if hypo_test is None:
                        # bootstrap means from original sample
                        diff_boot_mean_mat[method_num, idx, :] = run_bootstrap_append_mean(diff_data_dum,
                                                                                           len(diff_data_dum))

                    else:
                        # bootstrap means from edited sample
                        new_diff = diff_data_dum - diff_data_dum.mean()

                        diff_boot_mean_mat[method_num, idx, :, 0] = run_bootstrap_append_mean(new_diff,
                                                                                              len(diff_data_dum))
                        diff_boot_mean_mat[method_num, idx, :, 1] = diff_data_dum.mean()

                else:
                    print('warning!!! less than 2 samples in bin ' + val)

            else:
                print('warning!!! baseline & method arrays do not match!!!')

    return diff_boot_mean_mat


def do_ttest_boot(df_boot, diff_boot_array, wsti=False, wilcoxon=None, hypo_test=None):
    """Perform one-sample, one-sided t-test for bootstrapped |NME| differences.
    Option of choosing Wilcoxon test instead of t-test and doing bootstrap hypothesis test.
    Output df for plotting heatmap.
    """

    ref_df = df_boot.loc[df_boot['method'] == 'base']
    u_bin = ref_df['bin_name'].unique()

    for method_num, method_sheet in enumerate(pc.matrix_sheet_name_short[1:]):

        diff_test = np.zeros(len(u_bin))

        if method_num == 0:
            diff_test_df = pd.DataFrame(index=u_bin, columns=[pc.matrix_sheet_name_short[1:]])

        for idx, val in enumerate(u_bin):

            if hypo_test is None:

                # t-test
                # mean diff of individual error < diff_benchmark
                if diff_boot_array[method_num, idx, :].mean() < pc.diff_benchmark:
                    diff_test[idx] += 1

                    # some error categories do not have enough data
                    # hence t-test may fail after outlier removal
                    if wilcoxon is None:  # do t-test

                        try:
                            diff_t_stat = stats.ttest_1samp(diff_boot_array[method_num, idx, :], pc.diff_benchmark)

                        except ZeroDivisionError:
                            class diff_t_stat:
                                statistic = np.nan
                                pvalue = np.nan

                        # one-sample, one-sided t-test
                        # reject H0: no diff, or diff = 0
                        # Ha: mean diff of individual error < diff_benchmark
                        if ((diff_t_stat.statistic < 0)  # t-statistic < 0 (differ from diff_benchmark)
                                & (diff_t_stat.pvalue / 2 <= pc.alpha_choice)):  # one-sided, half of p-value
                            # use alpha_choice instead of alpha_thres here -- no need to use Bonferroni

                            # mean diff of individual error < diff_benchmark *significantly*
                            diff_test[idx] += 1
                    else:

                        try:
                            diff_w_stat = stats.wilcoxon(diff_boot_array[method_num, idx, :])

                        except ZeroDivisionError:
                            class diff_w_stat:
                                statistic = np.nan
                                pvalue = np.nan

                        if (diff_w_stat.pvalue / 2 <= pc.alpha_choice):
                            diff_test[idx] += 1

            else:

                mean_to_compare = np.unique(diff_boot_array[method_num, idx, :, 1])[0]

                outside_prob = ((np.sum(diff_boot_array[method_num, idx, :, 0] < -abs(mean_to_compare))
                                 + np.sum(diff_boot_array[method_num, idx, :, 0] > abs(mean_to_compare)))
                                / pc.boot_loop_num)

                # bootstrap hypothesis
                if outside_prob < pc.alpha_choice:
                    diff_test[idx] += 2

        diff_test_df[method_sheet] = diff_test

        strip_df_underscore(diff_test_df, method_sheet)

    diff_test_df.rename(columns=pc.method_dict, inplace=True)

    if wsti is True:
        diff_test_df = sort_plot_wsti_df_index(diff_test_df)

    return diff_test_df


def cal_bootstrap_ltest_pct(nme_df):
    """Calculate two percentage values on variances of bootstrapped samples.
    For each bootstrap sample:
    Whether the NME distribution of a method has smaller variance than Baseline's
    Whether the lowered variance is statistically sigificant (reject the null hypothesis of Levene's test)
    """

    base_df_s = extract_base_df_s(nme_df)

    for method_num, method_sheet in enumerate(pc.matrix_sheet_name_short[1:]):

        method_df_s = extract_method_df_s(method_sheet, nme_df)

        if (base_df_s['file_name'].values != method_df_s['file_name'].values).all():
            print('file names in baseline and method df do not match!')

        u_bin = base_df_s['bin_name'].unique()

        count_var, count_ltest = (np.zeros(len(u_bin)), np.zeros(len(u_bin)))

        if method_num == 0:
            count_var_df = pd.DataFrame(index=u_bin, columns=[pc.matrix_sheet_name_short[1:]])
            count_ltest_df = pd.DataFrame(index=u_bin, columns=[pc.matrix_sheet_name_short[1:]])

        for idx, val in enumerate(u_bin):

            base_array = get_bin_array(base_df_s, val)
            method_array = get_bin_array(method_df_s, val)

            base_na = drop_array_na(base_array)
            method_na = drop_array_na(method_array)

            if all(base_na) == all(method_na):  # ensure the nan's are at the same indices

                base_data = (base_array.dropna()) * 100.
                method_data = (method_array.dropna()) * 100

                # need 2 samples to do stat tests
                if (len(base_data) > 1) and (len(method_data) > 1):

                    # individual improvement, negative means improved
                    # compare absolute value of NME
                    diff_array = (abs(method_data) - abs(base_data))

                    diff_boot_mean_list = np.zeros(0)
                    boot_count = 0

                    # bootstrap
                    while boot_count < pc.boot_loop_num:
                        diff_boot, method_boot, base_boot = resample(diff_array, method_data, base_data,
                                                                     replace=True, n_samples=len(diff_array))

                        # Levene's test
                        if np.abs(base_boot.var()) > np.abs(method_boot.var()):  # variance < baseline
                            count_var[idx] += 1

                            f_stat = stats.levene(base_boot, method_boot)
                            if f_stat.pvalue <= pc.alpha_choice:  # reject H0: same variance
                                count_ltest[idx] += 1  # sd or variance < baseline *significantly*

                        # get list of means
                        diff_boot_mean_list = np.append(diff_boot_mean_list, diff_boot.mean())

                        boot_count += 1

                    sample_in_boot_pct = stats.percentileofscore(diff_boot_mean_list, diff_array.mean())

                    if ((sample_in_boot_pct < (50 - pc.boot_mean_bound))
                            | (sample_in_boot_pct > (50 + pc.boot_mean_bound))):
                        print((val + ' ' + method_sheet
                               + ' is beyond percentile bounds: ' + str(sample_in_boot_pct)))

        count_var_df[method_sheet] = count_var * 100 / pc.boot_loop_num
        count_ltest_df[method_sheet] = count_ltest * 100 / pc.boot_loop_num

        for df in [count_var_df, count_ltest_df]:

            strip_df_underscore(df, method_sheet)

    return count_var_df, count_ltest_df


def get_summary_df(e_df, sheet_name):
    """Generate data frame of summary statistics of errors.
    Only for Overall, Inner Range, and Outer Range data.
    Contain mean and standard deviation values.
    """

    summary_df = pd.DataFrame(columns=['sheet', 'bin_or_total', 'category', 'error',
                                       'n', 'mean', 'sd'])
    count = 0

    for msns_i in sheet_name:

        sheet = str(msns_i)[:-1]

        for btc_i in pc.bt_choice:

            matrix_df = e_df[msns_i + btc_i]

            for idx, (ecs_i, ens_i) in enumerate(itertools.product(['overall', 'by_range'], pc.error_name[1:])):

                df = matrix_df.loc[(matrix_df['error_cat'] == ecs_i) & (matrix_df['error_name'] == ens_i)]

                if ecs_i == 'overall':

                    summary_df.loc[count] = sheet, btc_i, ecs_i, ens_i, len(df), (df['error_value'] * 100).mean(), \
                                            (df['error_value'] * 100).std()

                    count += 1

                else:

                    for bn_i in ['Inner', 'Outer']:
                        df = df.loc[(df['bin_name'] == bn_i)]

                        summary_df.loc[count] = sheet, btc_i, bn_i, ens_i, len(df), \
                                                (df['error_value'] * 100).mean(), (df['error_value'] * 100).std()

                        df = matrix_df.loc[(matrix_df['error_cat'] == ecs_i) & (matrix_df['error_name'] == ens_i)]

                        count += 1

    return summary_df


def do_summary_diff_stat(summary_df, cat):
    """Perform statistical tests on |NME| differences from summary data frame.
    Perform one-sample, one-sided t-test and two-sided Levene's test.
    Write text of statistical test results into data frame.
    """

    df = summary_df.loc[(summary_df['category'] == cat)
                        & (summary_df['bin_or_total'] == 'total_e')]
    df.reset_index(inplace=True)

    def select_cat_error(one_df, cat):
        if cat == 'overall':
            df_s = one_df.loc[(one_df['error_cat'] == cat) & (one_df['error_name'] == error)]
        else:
            df_s = one_df.loc[(one_df['error_cat'] == 'by_range') & (one_df['bin_name'] == cat)
                              & (one_df['error_name'] == error)]
        return df_s

    pc_stat_col = ['' for x in range(len(df))]
    t_stat_col = ['' for x in range(len(df))]
    f_stat_col = ['' for x in range(len(df))]

    for i in range(len(df)):

        method = df['sheet'][i]
        sheet = method + '_' + df['bin_or_total'][i]
        error = df['error'][i]
        sample_n = df['n'][i]

        base_df = p_init.error_df['base_total_e']
        base_df_s = select_cat_error(base_df, cat)
        base_array = np.array(base_df_s['error_value'] * 100)

        if sheet in p_init.error_df:
            method_df = p_init.error_df[sheet]
        else:
            method_df = p_init.extra_error_df[sheet]

        method_df_s = select_cat_error(method_df, cat)
        method_array = np.array(method_df_s['error_value'] * 100)

        diff_array = abs(method_df_s['error_value'].dropna()) - abs(base_df_s['error_value'].dropna())

        loc_improve = np.where(diff_array < 0)
        len_improve = np.shape(loc_improve)[1]
        pc_improve = 100 * len_improve / len(diff_array.dropna())

        statement_start, statement_mid = '', ''

        if sample_n > 1:

            if diff_array.mean() < 0:  # mean diff of individual error < 0
                statement_start = 'mean error < 0, '

                diff_t_stat = stats.ttest_1samp(diff_array, 0)  # reject H0: no diff, or diff = 0
                if ((diff_t_stat.statistic < 0)  # t-statistic < 0 (differ from diff_benchmark)
                        & (diff_t_stat.pvalue / 2 <= pc.alpha_choice)):  # one-sided, half of p-value
                    statement_mid = '*significantly*'
                else:
                    statement_mid = 'not significant; '

            if np.abs(base_array.var()) > np.abs(method_array.var()):
                f_stat = stats.levene(base_array, method_array)

                if f_stat.pvalue <= pc.alpha_choice:  # reject H0: same variance
                    statement_end = 'variance < baseline *significantly*'
                else:
                    statement_end = 'variance < baseline, not significant'
            else:
                statement_end = ''

        pc_stat_col[i] = pc_improve
        t_stat_col[i] = statement_start + statement_mid
        f_stat_col[i] = statement_end

    with pd.option_context('mode.chained_assignment', None):
        df['improve %'] = pd.Series(pc_stat_col, index=df.index)
        df['paired t test'] = pd.Series(t_stat_col, index=df.index)
        df["Levene's test"] = pd.Series(f_stat_col, index=df.index)

    out_df = df.drop(columns=['index'])

    return out_df


def get_summary_table(input_cat):
    """Produce summary table in Python Notebook."""

    summary_basic_df = get_summary_df(p_init.error_df, pc.matrix_sheet_name_short)
    summary_df = summary_basic_df.append(get_summary_df(p_init.extra_error_df, pc.extra_matrix_sheet_name_short))

    # 12 counts per sheet
    summary_sheet_length = len(summary_df[summary_df['sheet'] == 'base'])
    diff_base_mean = np.zeros(len(summary_df))
    loop_summary_range = len(summary_df) / summary_sheet_length

    count_start, count_end = summary_sheet_length, summary_sheet_length * 2

    for i in range(int(loop_summary_range) - 1):
        diff_base_mean[count_start:count_end] = summary_df.iloc[count_start:count_end]['mean'].values - \
                                                summary_df.iloc[0:12]['mean'].values

        count_start += summary_sheet_length
        count_end += summary_sheet_length

    summary_df['mean_diff'] = diff_base_mean

    return do_summary_diff_stat(summary_df, input_cat)
