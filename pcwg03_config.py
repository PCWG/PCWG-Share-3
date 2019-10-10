# Configuration file for PCWG Share-03 analysis
# need to be modified:
# - if the Excel submission format is changed
# - if data with more error sheets are submitted

import copy

##########################
### Require user input ###
##########################

### Data path
### Point to the directory containing the Excel files

data_file_path = 'DATA_PATH'

### Save figure in pdf/png file or not

save_fig = False

########################################################################################################

### Statistics and filtering

# significance level
alpha_choice = 0.05

# difference smaller than diff_benchmark %, then method is better than baseline
# difference larger than diff_benchmark %, then method is worse than baseline
diff_benchmark = 0

# use robust and resistant statistics or not
robust_resistant_choice = None

# the number of bootstrap iterations
boot_loop_num = 1e4

# print message if bootstrapped result is beyond +/- x% of observed result
boot_mean_bound = 10

# inner range nme filtering threshold: x%
nme_filter_thres = 1  # in percent

# remove x percent of outlier improvement
quantile_cut = 0.1

# identify x percent of "extreme" improvement
percent_thres_choice = 1  # in percent

# based on the number of bins in Overall, Inner-Outer Range, and WS-TI & ITI-OS bins
# for Bonferroni Correction
alpha_thres_wsti_list = [1, 2, 2, 5, 5, 5, 5, 5]

### Submission

sub_1st_col_name = 'PCWG-Share-01 Submission Instructions'

data_ref_col_name = 'Dataset Reference'

sub_mat_start_col, sub_mat_end_col = 3, 7

sub_mat_assert_corner_col = 'Active Corrections'
sub_mat_assert_left_col = 'Density Correction Active?'
sub_mat_assert_right_col = 'Production by Height Active?'

sub_basic_col_length = 20

sub_var_names_top = ['sub_ref', 'cal_datetime', 'software_version', 'rep_complete',
                     'random_id', 'data_ref', 'times_ref']

sub_xls_names_top = ['Submission Reference', 'Calculation DateTime',
                     'PCWG Analysis Tool Software Version',
                     'Comfirmation of report export completion',
                     'Dataset Invariant Random ID', 'Dataset Reference',
                     'Timeseries Reference']

sub_xls_names_bot = ['Baseline', 'Density & Turbulence',
                     'Density & 2D Power Deviation Matrix',
                     'Density & Augmented Turbulence (Relaxed)',
                     'Density & 3D Power Deviation Matrix']

sub_var_head_bot = ['_base', '_denturb', '_den2dpdm', '_denaugturb', '_den3dpdm']

sub_var_end_bot = ['_den_corr', '_rews', '_ti_renorm', '_pdm', '_prod_by_height']

def_choice_list = ['A', 'B', 'C']  # inner range definitions

error_entry = 372  # number of entries in each submission in each error sheet

### Meta data

meta_1st_col_name = 'Key'

meta_var_names = ['data_type', 'rews_def_num_h', 'rews_def_veer', 'inner_def',
                  'site_class', 'forest_class', 'iec_class', 'geog_lat', 'geog_continent',
                  'geog_country', 'geog_elev', 'iec_12_1', 'anem_type', 'anem_heat',
                  'turbu_measuremt', 'power_measuremt', 'turbi_dia', 'turbi_hh',
                  'turbi_spower', 'turbi_control', 'year_measuremt', 'year_operatn',
                  'time_zone', 'interpolate_mode']

meta_var_names_turb = copy.copy(meta_var_names)
meta_var_names_turb.extend(['turbi_rated_power', 'turbi_d_hh_ratio'])

meta_xls_names = ['Data Type [Mast, LiDAR, SoDAR, Mast & LiDAR, Mast & SoDAR]',
                  'REWS Definition - Number of Heights', 'REWS Definition - Includes Veer',
                  'Inner Range Definition [A, B or C]',
                  'Outline Site Classification [Flat, Complex or Offshore]',
                  'Outline Forestry Classification [Forested or Non-forested]',
                  'IEC Site Classification [Flat, Complex or Offshore]',
                  'Geography - Approximate Latitude [to no decimal places e.g. 53°]',
                  'Geography - Continent', 'Geography - Country',
                  'Geography - Approximate Elevation Above Sea Level [to nearest 100m] (m)',
                  'Consistency of measurements with IEC 61400-12-1 (2006) [Yes, No or Unknown]',
                  'Anemometry Type [Sonic or Cups]', 'Anemometry Heating [Heated or Unheated]',
                  'Turbulence Measurement Type [LiDAR, SoDAR, Cups or Sonic]',
                  'Power Measurement Type [Transducer, SCADA, Unknown]',
                  'Turbine Geometry - Approximate Diameter (m)',
                  'Turbine Geometry - Approximate Hub Height (m)',
                  'Turbine Geometry - Specific Power (rated power divided by swept area) (kW/m^2)',
                  'Turbine Control Type [Pitch, Stall or Active Stall]',
                  'Vintage - Year of Measurement',
                  'Vintage - Year of First Operation of Turbine', 'Timezone [Local or UTC]',
                  'Interpolation Mode']

meta_xls_names_turb = copy.copy(meta_xls_names)
meta_xls_names_turb.extend(['Turbine Rated Power (MW)', 'Ratio: Rotor Diameter/Turbine Hub Height'])

meta_var_grouped = {'geog_lat': 'geog_lat_grouped', 'turbi_dia': 'turbi_dia_grouped',
                    'turbi_hh': 'turbi_hh_grouped', 'turbi_spower': 'turbi_spower_grouped',
                    'turbi_rated_power': 'turbi_rated_power_grouped',
                    'turbi_d_hh_ratio': 'turbi_d_hh_ratio_grouped'}

meta_xls_grouped_names = ['Geography - Approximate Latitude [to no decimal places e.g. 53°]',
                          'Turbine Geometry - Approximate Diameter (m)',
                          'Turbine Geometry - Approximate Hub Height (m)',
                          'Turbine Geometry - Specific Power (rated power divided by swept area) (W/m^2)',
                          'Turbine rated power (MW)', 'Turbine Geometry - Rotor Diameter/Hub Height']

### Error sheets

error_bin_e_1st_col_name = 'Normalised Mean Error (Bin Error/Bin Energy)'

error_total_e_1st_col_name = 'Normalised Mean Error (Bin Error/Total Energy)'

error_cat_short = ['overall', 'by_ws_bin', 'by_ws_bin_inner', 'by_ws_bin_outer',
                   'by_timeofday', 'by_month', 'by_direction', 'by_range', 'by_ws_ti']

error_cat = ['Overall', 'By (normalised) wind speed bin',
             'By (normalised) wind speed bin for Inner Range data only',
             'By (normalised) wind speed bin for Outer Range data only', 'By Time of Day',
             'By Calendar Month', 'By Direction', 'By Range',
             'Four Cell Matrix\n- High Wind Speed (HSW): Normalised Wind Speed ≥ 0.5\n- Low '
             + 'Wind Speed (LSW): Normalised Wind Speed < 0.5\n- High Turbulence (HTI) : Inner '
             + 'Range Upper TI Bound\n- Low Wind Speed (LSW) : Inner Range Lower TI Bound']

error_bin_len = [0, 14, 14, 14, 23, 11, 35, 1, 3]

error_bin_start_name = ['ALL', '0.0-0.1', '0.0-0.1', '0.0-0.1', '0-1', 'Jan', '355-5', 'Inner',
                        'LWS-LTI']

error_bin_end_name = ['ALL', '1.4-1.5', '1.4-1.5', '1.4-1.5', '23-24', 'Dec', '345-355', 'Outer',
                      'HWS-HTI']

error_name = ['data_count', 'nme', 'nmae']

matrix_sheet_name = ['Baseline', 'Den & Turb', 'Den & 2D PDM',
                     'Den & Aug Turb (Relaxed)', 'Den & 3D PDM']

# Baseline has to start first (for summary_df calculation in python notebook)
matrix_sheet_name_short = ['base_', 'den_turb_', 'den_2dpdm_', 'den_augturb_', 'den_3dpdm_']

bt_choice = ['bin_e', 'total_e']

### Extra error sheets

correction_list = ['Density, REWS (Speed+Veer) & Turbulence', 'Density & REWS (Speed+Veer)',
                   'Density & RAWS (Speed+Veer)', 'Density & RAWS (Speed)', 'Density, REWS (Speed) & Turbulence',
                   'Density & Production by Height', 'Density & REWS (Speed)']

extra_matrix_sheet_name = ['Den, REWS (S+V) & Turb', 'Den & REWS (S+V)', 'Den & RAWS (S+V)',
                           'Den & RAWS (S)', 'Den, REWS (S) & Turb', 'Den & P by H', 'Den & REWS (S)']

extra_matrix_sheet_name_short = ['den_rews_svturb_', 'den_rews_sv_', 'den_raws_sv_', 'den_raws_s_',
                                 'den_rews_sturb_', 'den_pbh_', 'den_rews_s_']

### Plotting

wsti_new_bin = 'ITI-OS'

outws_new_bin = 'Residual'

sort_wsti_index = ['LWS-LTI', 'LWS-HTI', 'HWS-LTI', 'HWS-HTI', wsti_new_bin, 'Outer', 'Inner', 'Overall']

method_dict = {'base': 'Baseline', 'den_turb': 'Den-Turb', 'den_2dpdm': 'Den-2DPDM',
               'den_augturb': 'Den-Augturb', 'den_3dpdm': 'Den-3DPDM'}
