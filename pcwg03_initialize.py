import os
import glob
import numpy as np

import pcwg03_config as pc
import pcwg03_convert_df as pcd

# code directory
py_file_path = os.getcwd()

# put generated plots near code
out_plot_path = py_file_path+'/plots/'

data_file = glob.glob(pc.data_file_path+'*.xls')


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


meta_df = pcd.get_metadata_df(data_file)

# group each meta data category into groups, for plotting histograms
for key, value in pc.meta_var_grouped.items():
    group_meta_element_in_range(key, value)

error_df, extra_error_df = pcd.filter_base_bin_nme(pcd.get_error_df_dict(data_file),
                                                   pcd.get_extra_error_df_dict(data_file))
