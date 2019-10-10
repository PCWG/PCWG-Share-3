import numpy as np
import pandas as pd

import pcwg03_config as pc

class read_submission:

    def __init__(self, file):

        self.df = pd.read_excel(file, sheet_name='Submission', skiprows=1)

        self.first_col_name = pc.sub_1st_col_name

        load_PCWG03.find_series_col(self)

    def find_data_ref(self):

        data_ref = load_PCWG03.locate_1to1_entry(self, pc.data_ref_col_name)

        return data_ref

class load_PCWG03:

    def __init__(self, file, sheet, bin_or_total=None):

        self.file = file

        self.df = pd.read_excel(file, sheet_name=sheet, skiprows=1)

        if sheet == 'Submission':

            self.first_col_name = pc.sub_1st_col_name

            self.find_series_col()

            self.mat_col_num = self.target_col_num + 1

            self.mat_start_col, self.mat_end_col = pc.sub_mat_start_col, pc.sub_mat_end_col

            self.mat_size = self.mat_end_col - self.mat_start_col + 1

            self.basic_col_length = pc.sub_basic_col_length

        elif sheet == 'Meta Data':

            self.first_col_name = pc.meta_1st_col_name

            self.find_series_col()

        else:

            self.first_col_name = pc.error_bin_e_1st_col_name

            self.find_series_col()

            self.first_col_split = pc.error_total_e_1st_col_name

            split_row_num = self.df.loc[self.df[self.first_col_name] == self.first_col_split].index[0]

            if bin_or_total == 'bin_e':
                self.split_df = self.df.iloc[0:split_row_num - 1, self.first_col_num:]

            if bin_or_total == 'total_e':
                self.split_df = self.df.iloc[split_row_num:, self.first_col_num:]

            self.error_name_col_num = self.target_col_num

            self.error_bin_col_num = self.error_name_col_num + 1

    def find_series_col(self):
        """Find column number."""

        self.first_col_num = self.df.columns.get_loc(self.first_col_name)

        self.target_col_num = self.first_col_num + 1

    def locate_1to1_entry(self, col_name):

        target_row_num = self.df.loc[self.df[self.first_col_name] == col_name].index[0]
        out_entry = self.df.iloc[target_row_num, self.target_col_num]

        return out_entry

    def assert_col_num(self, corner_col, mat_start_col, left_col, mat_end_col, right_col):

        check_mat = self.df.loc[self.df[self.first_col_name] == corner_col]
        assert check_mat.iloc[0, mat_start_col] == left_col, 'Starting column number incorrect!'
        assert check_mat.iloc[0, mat_end_col] == right_col, 'Total column number incorrect!'

    def locate_mat_entry(self, col_name):

        target_row_num = self.df.loc[self.df[self.first_col_name] == col_name].index[0]
        out_entry = self.df.iloc[target_row_num, self.mat_col_num:self.mat_col_num + self.mat_size]

        return out_entry.values

    def find_extra_rows(self, col_name):

        if not self.df.loc[self.df[self.first_col_name] == col_name].empty:
            data_ref = read_submission(self.file).find_data_ref()

            print(data_ref + ' has ' + col_name)

            return True

        return False

    def read_xls_submission(self):

        var_names_top = pc.sub_var_names_top

        xls_names_top = pc.sub_xls_names_top

        sub_list_top = {key: self.locate_1to1_entry(value) for key, value in zip(var_names_top, xls_names_top)}

        self.assert_col_num(pc.sub_mat_assert_corner_col, self.mat_start_col, pc.sub_mat_assert_left_col,
                            self.mat_end_col, pc.sub_mat_assert_right_col)

        xls_names_bot = pc.sub_xls_names_bot

        var_head_bot = pc.sub_var_head_bot

        var_end_bot = pc.sub_var_end_bot

        sub_list_bot = {}

        for row in range(len(xls_names_bot)):

            one_row_data = self.locate_mat_entry(xls_names_bot[row])

            for col in range(len(var_end_bot)):
                sub_list_bot[var_head_bot[row] + var_end_bot[col]] = one_row_data[col]

        self.sub_dict = {**sub_list_top, **sub_list_bot}

        submission_series = pd.Series(self.sub_dict, name=self.sub_dict['data_ref'])

        if self.df.shape[0] > self.basic_col_length:
            print('suggestion: check ' + sub_list_top['data_ref'] + ' for extra sheets in file!')

        return submission_series

    def read_xls_metadata(self):

        data_ref = read_submission(self.file).find_data_ref()

        var_names = pc.meta_var_names

        xls_names = pc.meta_xls_names

        result_list = {key: self.locate_1to1_entry(value) for key, value in zip(var_names, xls_names)}

        meta_series = pd.Series(result_list, name=data_ref)

        return meta_series

    def read_xls_matrix(self):
        """Turn sheets of error values into matrix."""

        error_cat = pc.error_cat

        error_cat_short = pc.error_cat_short

        error_bin_len = pc.error_bin_len

        error_bin_start_name = pc.error_bin_start_name

        error_bin_end_name = pc.error_bin_end_name

        for i in range(len(error_cat)):
            self.assert_col_num(error_cat[i], self.error_bin_col_num, error_bin_start_name[i],
                                self.error_bin_col_num + error_bin_len[i], error_bin_end_name[i])

        data_ref = read_submission(self.file).find_data_ref()

        error_name = pc.error_name

        out_df = []

        for ec_count in range(len(error_cat)):

            out_df_ec = []

            error_row_num = self.split_df.loc[self.split_df[self.first_col_name] == error_cat[ec_count]].index[0]

            for en_count in range(len(error_name)):
                bin_name = self.df.iloc[error_row_num, self.error_bin_col_num:self.error_bin_col_num
                                                                              + error_bin_len[ec_count] + 1]
                error_value = self.df.iloc[error_row_num + en_count + 1,
                              self.error_bin_col_num:self.error_bin_col_num + error_bin_len[ec_count] + 1]

                bin_name = pd.Series(bin_name.values, name='bin_name')
                error_value = pd.Series(error_value.values, name='error_value')

                out_df_en = pd.concat([bin_name, error_value], axis=1)

                out_df_en['error_name'] = pd.Series(np.repeat(error_name[en_count], out_df_en.shape[0], axis=0),
                                                    index=out_df_en.index)

                out_df_ec.append(out_df_en)

            out_df_ec = pd.concat(out_df_ec, axis=0)

            out_df_ec['error_cat'] = pd.Series(np.repeat(error_cat_short[ec_count], out_df_ec.shape[0], axis=0),
                                               index=out_df_ec.index)

            out_df.append(out_df_ec)

        out_df = pd.concat(out_df, axis=0)

        out_df['file_name'] = pd.Series(np.repeat(data_ref, out_df.shape[0], axis=0),
                                        index=out_df.index)

        out_df.reset_index(inplace=True, drop=True)

        return out_df

    def read_xls_extra_matrix(self, correction_name):

        if self.find_extra_rows(correction_name) == True:
            data_ref = read_submission(self.file).find_data_ref()

            return data_ref
