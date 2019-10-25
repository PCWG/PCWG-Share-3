import numpy as np
import pandas as pd

import pcwg03_config as pc
import pcwg03_slice_df as psd

# a number very close to 0 for error check
check_energy_cal_thres = 1e-15


def get_energy_df_section(lwslti, lwshti, hwslti, hwshti, itos, outer, inner, error_name, file_name):
    """Return WS-TI, ITI-OS, Inner-Outer Range data frame."""

    df = pd.DataFrame(data={'bin_name': ['LWS-LTI', 'LWS-HTI', 'HWS-LTI', 'HWS-HTI', pc.wsti_new_bin, 'Outer', 'Inner'],
                            'value': [lwslti, lwshti, hwslti, hwshti, itos, outer, inner],
                            'error_name': error_name, 'file_name': file_name})

    return df


def remove_problematic_files(nme_df, problem_file, error_cat, new_bin):
    """Remove submissions from NME data frame according to problem_file."""

    # filter out problematic files
    good_nme_df = nme_df.loc[~nme_df['file_name'].isin(problem_file)]

    good_file = good_nme_df['file_name'].unique()

    for file in good_file:

        one_file = good_nme_df.loc[good_nme_df['file_name'] == file]

        # leftover/residual NME between Outer Range and the sum of all target bins
        out_lo = (one_file.loc[one_file['bin_name'] == 'Outer'].error_value
                  - one_file.loc[one_file['error_cat'] == error_cat].error_value.sum())

        good_nme_df = good_nme_df.append(pd.DataFrame([[new_bin, out_lo.values[0], 'nme', error_cat, file]],
                                                      columns=good_nme_df.columns))

    good_nme_df.reset_index(inplace=True, drop=True)

    return good_nme_df


def get_raw_energy_pct(ef_df):

    return ef_df.loc[ef_df['error_name'] == 'energy_%']


def check_problematic_file(ef_df, error_cat, print_state=None):
    """Check for problematic submissions with duplicating data in WS-TI bins.
    Those files yield negative ITI-OS energy count and data count.
    """

    energy_pct_raw = get_raw_energy_pct(ef_df)

    problem_file = energy_pct_raw.loc[energy_pct_raw['value'] < 0]['file_name']  # problematic files

    if print_state is not None:

        if len(problem_file) > 0:

            print(str(len(problem_file)) + ' ' + error_cat + ' files are problematic')

        else:

            print('none of the files are problematic')

    return problem_file


def get_energy_pct(ef_df, error_cat):

    problem_file = check_problematic_file(ef_df, error_cat)

    return get_raw_energy_pct(ef_df).loc[~get_raw_energy_pct(ef_df)['file_name'].isin(problem_file)]


def cal_wsti_ef(error_cat):
    """Calculate energy fraction for WS-TI bins.
    Similar procedure as in `cal_outws_ef` for Outer Range WS bins.
    """

    nme_bin_e, dc_total_e, nme_total_e = psd.get_base_e_df(error_cat)

    file_all = dc_total_e['file_name'].unique()

    ef_df = pd.DataFrame()

    for file in file_all:

        # data count (in %) -- same for each method

        dc_1file = dc_total_e.loc[(dc_total_e['file_name'] == file)]

        lwslti_dc = (100 * dc_1file.loc[dc_1file['bin_name'] == 'LWS-LTI'].error_value.values
                     / dc_1file.loc[dc_1file['bin_name'] == 'ALL'].error_value.values)[0]

        lwshti_dc = (100 * dc_1file.loc[dc_1file['bin_name'] == 'LWS-HTI'].error_value.values
                     / dc_1file.loc[dc_1file['bin_name'] == 'ALL'].error_value.values)[0]

        hwslti_dc = (100 * dc_1file.loc[dc_1file['bin_name'] == 'HWS-LTI'].error_value.values
                     / dc_1file.loc[dc_1file['bin_name'] == 'ALL'].error_value.values)[0]

        hwshti_dc = (100 * dc_1file.loc[dc_1file['bin_name'] == 'HWS-HTI'].error_value.values
                     / dc_1file.loc[dc_1file['bin_name'] == 'ALL'].error_value.values)[0]

        itos_dc = (100 * (dc_1file.loc[dc_1file['bin_name'] == 'Outer'].error_value.values
                          - np.sum(dc_1file.loc[dc_1file['error_cat'] == error_cat]
                                   .error_value.values))
                   / dc_1file.loc[dc_1file['bin_name'] == 'ALL'].error_value.values)[0]

        outer_dc = (100 * dc_1file.loc[dc_1file['bin_name'] == 'Outer'].error_value.values
                    / dc_1file.loc[dc_1file['bin_name'] == 'ALL'].error_value.values)[0]

        inner_dc = 100 - outer_dc

        # energy fraction (in %) -- same for each method

        nme_bin_e_1file = nme_bin_e.loc[(nme_bin_e['file_name'] == file)]
        nme_total_e_1file = nme_total_e.loc[(nme_total_e['file_name'] == file)]

        lwslti_energy = (nme_total_e_1file.loc[nme_total_e_1file['bin_name'] == 'LWS-LTI'].error_value.values
                         / nme_bin_e_1file.loc[nme_bin_e_1file['bin_name'] == 'LWS-LTI'].error_value.values)[0]

        lwshti_energy = (nme_total_e_1file.loc[nme_total_e_1file['bin_name'] == 'LWS-HTI'].error_value.values
                         / nme_bin_e_1file.loc[nme_bin_e_1file['bin_name'] == 'LWS-HTI'].error_value.values)[0]

        hwslti_energy = (nme_total_e_1file.loc[nme_total_e_1file['bin_name'] == 'HWS-LTI'].error_value.values
                         / nme_bin_e_1file.loc[nme_bin_e_1file['bin_name'] == 'HWS-LTI'].error_value.values)[0]

        hwshti_energy = (nme_total_e_1file.loc[nme_total_e_1file['bin_name'] == 'HWS-HTI'].error_value.values
                         / nme_bin_e_1file.loc[nme_bin_e_1file['bin_name'] == 'HWS-HTI'].error_value.values)[0]

        outer_energy = (nme_total_e_1file.loc[nme_total_e_1file['bin_name'] == 'Outer'].error_value.values
                        / nme_bin_e_1file.loc[nme_bin_e_1file['bin_name'] == 'Outer'].error_value.values)[0]

        itos_energy = outer_energy - np.sum([lwslti_energy, lwshti_energy, hwslti_energy, hwshti_energy])

        inner_energy = 1 - outer_energy

        # group into data frames

        dc_df = get_energy_df_section(lwslti_dc, lwshti_dc, hwslti_dc, hwshti_dc, itos_dc, outer_dc, inner_dc,
                                      'data_%', file)

        energy_df = get_energy_df_section(lwslti_energy * 100, lwshti_energy * 100, hwslti_energy * 100,
                                          hwshti_energy * 100,
                                          itos_energy * 100, outer_energy * 100, inner_energy * 100, 'energy_%', file)

        ef_df = pd.concat([ef_df, dc_df, energy_df], axis=0)
        ef_df.reset_index(inplace=True, drop=True)

        # check to see if the sum of products between energy fraction and bin error is equal to the Outer Range error

        lwslti_bin_e = nme_bin_e_1file.loc[nme_bin_e['bin_name'] == 'LWS-LTI'].error_value.values[0]
        lwshti_bin_e = nme_bin_e_1file.loc[nme_bin_e['bin_name'] == 'LWS-HTI'].error_value.values[0]
        hwslti_bin_e = nme_bin_e_1file.loc[nme_bin_e['bin_name'] == 'HWS-LTI'].error_value.values[0]
        hwshti_bin_e = nme_bin_e_1file.loc[nme_bin_e['bin_name'] == 'HWS-HTI'].error_value.values[0]

        itos_bin_e = ((nme_total_e_1file.loc[nme_bin_e['bin_name'] == 'Outer'].error_value.values
                       - np.sum(nme_total_e_1file.loc[nme_total_e_1file['error_cat'] == error_cat]
                                .error_value.values))
                      / itos_energy)[0]

        check_energy_cal = (
                    np.sum([lwslti_energy * lwslti_bin_e, lwshti_energy * lwshti_bin_e, hwslti_energy * hwslti_bin_e,
                            hwshti_energy * hwshti_bin_e, itos_energy * itos_bin_e])
                    - nme_total_e_1file.loc[nme_total_e_1file['bin_name'] == 'Outer'].error_value.values)

        if check_energy_cal > check_energy_cal_thres or check_energy_cal < -check_energy_cal_thres:
            print('ERROR')
            print(check_energy_cal)

    return ef_df


def cal_check_remove_ef(error_cat, new_bin):
    """Calculate energy fraction and remove problematic submissions."""

    if error_cat == 'by_ws_ti':
        ef_df = cal_wsti_ef(error_cat)

    elif error_cat == 'by_ws_bin_outer':
        ef_df = cal_outws_ef(error_cat)

    problem_file = check_problematic_file(ef_df, error_cat, print_state=True)

    ef_df = ef_df.loc[~ef_df['file_name'].isin(problem_file)]

    base_nme_df = psd.get_base_total_e(error_cat)

    nme_df = remove_problematic_files(base_nme_df, problem_file, error_cat, new_bin)

    nme_df['error_value'] = nme_df['error_value'].astype(float) * 100

    return ef_df, nme_df


def get_wsti_ef_nme():
    """Separate energy fraction and NME data frame into two pairs of data frames for plotting.
    Similar procedure as in `get_outws_ef_nme` for Outer Range WS bins.
    """

    error_cat = 'by_ws_ti'

    ef_filter_df, wsti_io_nme_df = cal_check_remove_ef(error_cat, pc.wsti_new_bin)

    ef_filter_df1 = pd.DataFrame()
    wsti_nme_df1 = pd.DataFrame()

    for bin_i in list(ef_filter_df['bin_name'].unique()[0:5]):

        ef_filter_df1 = pd.concat([ef_filter_df1,
                                   ef_filter_df.loc[ef_filter_df['bin_name'] == bin_i].reset_index()], axis=0)

        wsti_nme_df1 = pd.concat([wsti_nme_df1,
                                  wsti_io_nme_df.loc[wsti_io_nme_df['bin_name'] == bin_i].reset_index()], axis=0)

    ef_filter_df2 = pd.DataFrame()
    wsti_nme_df2 = pd.DataFrame()

    for bin_i in list(ef_filter_df['bin_name'].unique()[5:]):

        ef_filter_df2 = pd.concat([ef_filter_df2,
                                   ef_filter_df.loc[ef_filter_df['bin_name'] == bin_i].reset_index()], axis=0)

        wsti_nme_df2 = pd.concat([wsti_nme_df2,
                                  wsti_io_nme_df.loc[wsti_io_nme_df['bin_name'] == bin_i].reset_index()], axis=0)

    return ef_filter_df1, ef_filter_df2, wsti_nme_df1, wsti_nme_df2


def cal_outws_ef(error_cat):
    """Calculate energy fraction for Outer Range WS bins.
    Similar procedure as in `cal_wsti_ef` for WS-TI bins.
    """

    nme_bin_e, dc_total_e, nme_total_e = psd.get_base_e_df(error_cat)

    outer_ws_bin_list = nme_bin_e.loc[nme_bin_e['error_cat'] == error_cat]['bin_name'].unique()

    dc_all_df = pd.DataFrame()
    ef_all_df = pd.DataFrame()
    dc_ef_all_df = pd.DataFrame()

    file_all = dc_total_e['file_name'].unique()

    for file in file_all:

        dc_1file = dc_total_e.loc[(dc_total_e['file_name'] == file)]

        def find_outer_ws_dc(outer_ws_bin):

            return ((100 * dc_1file.loc[dc_1file['bin_name'] == outer_ws_bin].error_value.values
                     / dc_1file.loc[dc_1file['bin_name'] == 'ALL'].error_value.values)[0])

        ws_dc_list = []

        for bin_i in outer_ws_bin_list:
            ws_dc_list.append(find_outer_ws_dc(bin_i))

        outer_dc = 100 * (dc_1file.loc[dc_1file['bin_name'] == 'Outer'].error_value.values
                          / dc_1file.loc[dc_1file['bin_name'] == 'ALL'].error_value.values)[0]

        outer_dc_leftover = (outer_dc - np.sum(ws_dc_list))

        inner_dc = 100 - outer_dc

        dc_all_list = ws_dc_list + [outer_dc_leftover, outer_dc, inner_dc]

        name_dc_list = list(outer_ws_bin_list) + [pc.outws_new_bin, 'Outer', 'Inner']

        dc_dict = {'bin_name': name_dc_list, 'value': dc_all_list}

        dc_df = pd.DataFrame(dc_dict)
        dc_df['file_name'] = dc_1file['file_name'].unique()[0]
        dc_df['error_name'] = 'data_%'

        dc_all_df = pd.concat([dc_all_df, dc_df], axis=0)
        dc_df.reset_index(inplace=True, drop=True)

        nme_bin_e_1file = nme_bin_e.loc[(nme_bin_e['file_name'] == file)]
        nme_total_e_1file = nme_total_e.loc[(nme_total_e['file_name'] == file)]

        def find_outer_ws_ef(outer_ws_bin):

            return ((100 * nme_total_e_1file.loc[nme_total_e_1file['bin_name'] == outer_ws_bin].error_value.values
                     / nme_bin_e_1file.loc[nme_bin_e_1file['bin_name'] == outer_ws_bin].error_value.values)[0])

        ws_ef_list = []

        for bin_i in outer_ws_bin_list:

            ws_ef_list.append(find_outer_ws_ef(bin_i))

        outer_ef = 100 * (nme_total_e_1file.loc[nme_total_e_1file['bin_name'] == 'Outer'].error_value.values
                          / nme_bin_e_1file.loc[nme_bin_e_1file['bin_name'] == 'Outer'].error_value.values)[0]

        outer_ef_leftover = (outer_ef - np.sum(ws_ef_list))

        inner_ef = 100 - outer_ef

        ef_all_list = ws_ef_list + [outer_ef_leftover, outer_ef, inner_ef]

        ef_dict = {'bin_name': name_dc_list, 'value': ef_all_list}

        ef_df = pd.DataFrame(ef_dict)
        ef_df['file_name'] = dc_1file['file_name'].unique()[0]
        ef_df['error_name'] = 'energy_%'

        ef_all_df = pd.concat([ef_all_df, ef_df], axis=0)
        ef_df.reset_index(inplace=True, drop=True)

        dc_ef_all_df = pd.concat([dc_ef_all_df, dc_df, ef_df], axis=0)
        dc_ef_all_df.reset_index(inplace=True, drop=True)

        def find_outer_ws_bin_e(outer_ws_bin):

            return (nme_bin_e_1file.loc[nme_bin_e['bin_name'] == outer_ws_bin].error_value.values[0])

        ws_bin_e_list = []

        for bin_i in outer_ws_bin_list:

            ws_bin_e_list.append(find_outer_ws_bin_e(bin_i))

        outer_bin_e_leftover = ((nme_total_e_1file.loc[nme_bin_e['bin_name'] == 'Outer'].error_value.values
                                 - np.sum(nme_total_e_1file.loc[nme_total_e_1file['error_cat']
                                                                == error_cat].error_value.values))
                                / (outer_ef_leftover / 100))[0]

        check_energy_cal = (np.sum(np.append((np.array(ws_bin_e_list) * np.array(ws_ef_list) / 100),
                                             outer_ef_leftover * outer_bin_e_leftover / 100))
                            - nme_total_e_1file.loc[nme_total_e_1file['bin_name'] == 'Outer'].error_value.values)

        if check_energy_cal > check_energy_cal_thres or check_energy_cal < -check_energy_cal_thres:
            print('ERROR')
            print(check_energy_cal)

    return dc_ef_all_df


def get_outws_ef_nme():
    """Prepare energy fraction and NME data frames for plotting.
    Similar procedure as in `get_wsti_nme` for WS-TI bins.
    """

    error_cat = 'by_ws_bin_outer'

    dc_ef_all_df, outws_io_nme_df = cal_check_remove_ef(error_cat, pc.outws_new_bin)

    outws_nme_df = pd.DataFrame()

    for bin_i in list(dc_ef_all_df['bin_name'].unique()):

        outws_nme_df = pd.concat([outws_nme_df,
                                  outws_io_nme_df.loc[outws_io_nme_df['bin_name'] == bin_i].reset_index()],
                                 axis=0)

    outws_nme_df['abs_error_value'] = abs(outws_nme_df['error_value'])

    dc_ef_all_df_s = pd.DataFrame()
    outws_nme_df_s = pd.DataFrame()

    for bin_i in list(dc_ef_all_df['bin_name'].unique()[:-2]):

        dc_ef_all_df_s = pd.concat([dc_ef_all_df_s,
                                    dc_ef_all_df.loc[dc_ef_all_df['bin_name'] == bin_i].reset_index()],
                                   axis=0)
        outws_nme_df_s = pd.concat([outws_nme_df_s, outws_nme_df.loc[outws_nme_df['bin_name'] == bin_i].reset_index()],
                                   axis=0)

    return dc_ef_all_df_s, outws_nme_df_s
