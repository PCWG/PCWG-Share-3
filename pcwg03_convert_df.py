import numpy as np
import pandas as pd

import itertools

from pathlib import Path
import pickle

import pcwg03_initialize as p_init

import pcwg03_config as pc
import pcwg03_read_data as prd


def turn_submission_to_series(file):

    submission_series = prd.load_PCWG03(file, 'Submission').read_xls_submission()

    return submission_series


def turn_meta_to_series(file):

    meta_series = prd.load_PCWG03(file, 'Meta Data').read_xls_metadata()

    return meta_series


def turn_error_matrices_to_df(file, sheet, bt_choice):

    error_df = prd.load_PCWG03(file, sheet, bin_or_total=bt_choice).read_xls_matrix()

    return error_df


def loop_matrix_sheet(bt_list, count, error_dict, file_list, sheet_name, sheet_name_short):
    """Put error data frames into dictionary."""

    for bt_count in range(len(bt_list)):

        all_error = [turn_error_matrices_to_df(file, sheet_name[count], bt_list[bt_count])
                     for file in file_list]
        all_error_df = pd.concat(all_error, axis=0)
        all_error_df.reset_index(inplace=True, drop=True)

        error_dict[sheet_name_short[count] + bt_list[bt_count]] = all_error_df

    return error_dict


def search_pkl_existence(pkl_name):

    pkl_file = p_init.py_file_path + '/data_pcwg03_' + pkl_name + '.pkl'
    pkl_path = Path(pkl_file)

    return pkl_path.exists()


def load_existing_pkl(pkl_name):

    pkl_file = p_init.py_file_path + '/data_pcwg03_' + pkl_name + '.pkl'

    print(pkl_name + ' pkl file exists; loading pkl file directly!')

    with open(pkl_file, 'rb') as f:
        out_df = pickle.load(f)

    return out_df


def read_xls_write_pkl(pkl_name, turn_function, data_file):

    print(pkl_name + ' pkl file does not exist; reading in excel files...')

    data = [turn_function(file) for file in data_file]
    out_df = pd.concat(data, axis=1).T

    pkl_file = p_init.py_file_path + '/data_pcwg03_' + pkl_name + '.pkl'

    with open(pkl_file, 'wb') as f:
        pickle.dump(out_df, f)

    return out_df


def get_submission_df(data_file):

    if search_pkl_existence('submission') is True:

        submission_df = load_existing_pkl('submission')

    else:

        submission_df = read_xls_write_pkl('submission', turn_submission_to_series, data_file)

    submission_df = submission_df.rename_axis('file_name').reset_index()

    return submission_df


def get_metadata_df(data_file):

    if search_pkl_existence('metadata') is True:

        meta_df = load_existing_pkl('metadata')

    else:

        meta_df = read_xls_write_pkl('metadata', turn_meta_to_series, data_file)

    meta_df = meta_df.rename_axis('file_name').reset_index()

    # calculate rated power
    meta_df['turbi_rated_power'] = meta_df['turbi_spower']*(np.pi*((meta_df['turbi_dia']/2)**2))/1e3

    # calculate rotor-diameter-to-hub-height ratio
    meta_df['turbi_d_hh_ratio'] = meta_df['turbi_dia']/meta_df['turbi_hh']

    return meta_df


def get_error_df_dict(data_file):
    """Load dictionary of error data frames from pickle file.
    If pickle file does not exist, make one.
    """

    if search_pkl_existence('error') is True:

        error_df_dict = load_existing_pkl('error')

    else:

        print('error pkl file does not exist; reading in excel files...')

        error_df_dict = {}

        for ms_count in range(len(pc.matrix_sheet_name)):
            print('working on ' + pc.matrix_sheet_name[ms_count])

            error_df_dict = loop_matrix_sheet(bt_list=pc.bt_choice, count=ms_count, error_dict=error_df_dict,
                                              file_list=data_file, sheet_name=pc.matrix_sheet_name,
                                              sheet_name_short=pc.matrix_sheet_name_short)

        error_pkl_file = p_init.py_file_path + '/data_pcwg03_error.pkl'

        with open(error_pkl_file, 'wb') as f:
            pickle.dump(error_df_dict, f)

    return error_df_dict


def get_extra_error_df_dict(data_file):

    data_file_path = str(Path(data_file[0]).parent)

    if search_pkl_existence('extra_error') is True:

        extra_error_df_dict = load_existing_pkl('extra_error')

    else:

        print('extra error pkl file does not exist; reading in excel files...')

        extra_error_df_dict = {}

        for correct_i in range(len(pc.correction_list)):

            all_list = []

            for file in data_file:
                all_list.append(prd.load_PCWG03(file,
                                                'Submission').read_xls_extra_matrix(pc.correction_list[correct_i]))

            select_list = [x for x in all_list if x is not None]
            extra_sheet_list = [data_file_path + '/' + x + '.xls' for x in select_list]

            print('working on ' + pc.extra_matrix_sheet_name[correct_i])

            extra_error_df_dict = loop_matrix_sheet(bt_list=pc.bt_choice, count=correct_i,
                                                    error_dict=extra_error_df_dict, file_list=extra_sheet_list,
                                                    sheet_name=pc.extra_matrix_sheet_name,
                                                    sheet_name_short=pc.extra_matrix_sheet_name_short)

        extra_error_pkl_file = p_init.py_file_path + '/data_pcwg03_extra_error.pkl'

        with open(extra_error_pkl_file, 'wb') as f:
            pickle.dump(extra_error_df_dict, f)

    return extra_error_df_dict


def remove_error_entry(sheet_i, bt, df, df_filter, nme_filter_files):
    """Remove submissions that fail the NME filter."""

    sheet = sheet_i + bt
    df_filter[sheet] = df[sheet][(~df[sheet]['file_name'].isin(nme_filter_files))]

    file_out = (len(df[sheet]) - len(df_filter[sheet])) / pc.error_entry
    print('removing ' + str(round(file_out)) + ' files in ' + sheet)


def filter_base_bin_nme(error_df, extra_error_df):
    """Filter out a fraction of submissions based on Baseline NME.
    Submissions with bin NME per bin energy (bin_e) not close to 0 are labelled as bad data.
    """

    error_df_filter = dict.fromkeys(error_df)  # copy keys
    extra_error_df_filter = dict.fromkeys(extra_error_df)

    nme_only = error_df['base_bin_e'].loc[(error_df['base_bin_e']['error_cat'] == 'by_range')
                                          & (error_df['base_bin_e']['bin_name'] == 'Inner')
                                          & (error_df['base_bin_e']['error_name'] == 'nme')]

    nme_filter_files = nme_only.loc[(nme_only['error_value'].values * 100 < -pc.nme_filter_thres)
                                    | (nme_only['error_value'].values * 100 > pc.nme_filter_thres)]['file_name']

    print('data files with baseline bin error per bin energy exceed ' + str(pc.nme_filter_thres) + '% are:')
    print(nme_filter_files)
    print('a total of ' + str(len(nme_filter_files)) + ' files contain bad data')

    for idx, (bt, sheet_i) in enumerate(itertools.product(pc.bt_choice, pc.matrix_sheet_name_short)):

        remove_error_entry(sheet_i, bt, error_df, error_df_filter, nme_filter_files)

    for idx, (bt, sheet_i) in enumerate(itertools.product(pc.bt_choice, pc.extra_matrix_sheet_name_short)):

        remove_error_entry(sheet_i, bt, extra_error_df, extra_error_df_filter, nme_filter_files)

    return error_df_filter, extra_error_df_filter
