import os
import numpy as np
import pandas as pd
import itertools

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

import geopandas as gpd
from descartes import PolygonPatch

import pcwg03_initialize as p_init

import pcwg03_config as pc
import pcwg03_convert_df as pcd
import pcwg03_energy_fraction as pef
import pcwg03_slice_df as psd

meta_df = p_init.meta_df

save_fig = pc.save_fig

dpi_choice = 600  # output plot resolution

fs, f14, f15, f16 = 12, 14, 15, 16

plt.rcParams.update({'font.size': fs})

xp_f1, yp_f1 = 0.04, 0.93  # x, y positions

fmt_code = '.1f'  # for bootstrap t-test mean heatmap


def save_plot(sub_dir, var, plot_type, pdf=True):
    """Export figure to either pdf or png file."""

    if not os.path.exists(p_init.out_plot_path+'/'+sub_dir):

        os.makedirs(p_init.out_plot_path+'/'+sub_dir)

    if pdf is True:

        plt.savefig(p_init.out_plot_path+sub_dir+'/'+var+'_'+plot_type+'.pdf',
                    bbox_inches='tight', dpi=dpi_choice)

    else:

        plt.savefig(p_init.out_plot_path+'/'+sub_dir+'/'+var+'_'+plot_type+'.png',
                    bbox_inches='tight', dpi=dpi_choice)


def finish_plot(sub_dir, var, plot_type, tight_layout=True, save_fig=False, pdf=True):
    """Terminating procedures for plotting."""

    if tight_layout is True:

        plt.tight_layout()

    if save_fig is True:

        save_plot(sub_dir, var, plot_type, pdf)

    plt.show()


def plot_pdm_example():
    """Plot example power deviation matrix.
    Using input from Excel file.
    """

    file = p_init.py_file_path+'/pdm_example.xls'

    df = pd.read_excel(file)
    df.set_index('TI', inplace=True)

    ax = sns.heatmap(df, cmap='RdYlBu', robust=True, center=0, cbar_kws={'label': 'Power deviation(%)'})

    # ax.set_xlabel(r'Wind speed (m s$^{-1}$)')
    ax.set_xlabel('Normalized wind speed')
    ax.set_ylabel('TI (%)')

    finish_plot('meta', 'pdm_example', 'heatmap')


def plot_wsti_energy_fraction_box():
    """Plot 4 box plots for WS-TI, ITI-OS, and Inner-Outer Ranges.
    A pair of box plots on energy and data fractions, and a pair box plots on NMEs.
    Similar to `plot_outws_energy_fraction_box`.
    """

    ef_filter_df1, ef_filter_df2, wsti_nme_df1, wsti_nme_df2 = pef.get_wsti_ef_nme()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [3, 1]})

    ax1 = sns.boxplot(x='bin_name', y='value', data=ef_filter_df1, hue='error_name',
                      palette='colorblind', ax=ax1)
    ax1 = sns.swarmplot(x='bin_name', y='value', data=ef_filter_df1, hue='error_name',
                        alpha=0.7, dodge=True, ax=ax1)
    ax1.set_ylabel('Fraction (%)')
    ax1.set_xlabel('')
    ax1.set_ylim([-5, 95])

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:2], ['Data fraction', 'Energy fraction'], loc='upper left')

    ax2 = sns.boxplot(x='bin_name', y='value', data=ef_filter_df2, hue='error_name',
                      palette='colorblind', ax=ax2)
    ax2 = sns.swarmplot(x='bin_name', y='value', data=ef_filter_df2, hue='error_name',
                        alpha=0.7, dodge=True, ax=ax2)
    ax2.legend_.remove()
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.set_ylim([-5, 95])
    ax2.set_yticklabels([])

    ax3 = sns.boxplot(x='bin_name', y='error_value', data=wsti_nme_df1, hue='error_name',
                      palette='BuGn_r', ax=ax3)
    ax3.axhline(0, ls='--', color='grey')
    ax3.set_ylabel('NME (%)')
    ax3.set_xlabel('WS-TI bins')
    ax3.legend_.remove()
    ax3.set_ylim([-2.5, 2.5])

    ax4 = sns.boxplot(x='bin_name', y='error_value', data=wsti_nme_df2, hue='error_name',
                      palette='BuGn_r', ax=ax4)
    ax4.axhline(0, ls='--', color='grey')
    ax4.set_ylabel('')
    ax4.set_xlabel('Inner-Outer Range')
    ax4.legend_.remove()
    ax4.set_ylim([-2.5, 2.5])
    ax4.set_yticklabels([])

    ax1.text(0.94, 0.9, '(a)', color='k', fontsize=12, transform=ax1.transAxes)
    ax2.text(0.82, 0.9, '(b)', color='k', fontsize=12, transform=ax2.transAxes)
    ax3.text(0.94, 0.9, '(c)', color='k', fontsize=12, transform=ax3.transAxes)
    ax4.text(0.82, 0.9, '(d)', color='k', fontsize=12, transform=ax4.transAxes)

    finish_plot('meta', 'wsti_energyfraction', 'box')


def plot_outws_energy_fraction_box():
    """Plot 2 box plots for Outer Range WS.
    A box plot on energy and data fractions, and a box plot on NMEs.
    Similar to `plot_wsti_energy_fraction_box`.
    """

    dc_ef_all_df_s, outws_nme_df_s = pef.get_outws_ef_nme()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1 = sns.boxplot(x='bin_name', y='value', data=dc_ef_all_df_s, hue='error_name',
                      palette='colorblind', ax=ax1)

    ax1.set_ylabel('Fraction (%)', fontsize=f15)
    ax1.set_xlabel('')

    ax1.tick_params(labelsize=f15)
    ax1.xaxis.set_tick_params(rotation=45)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:2], ['Data fraction', 'Energy fraction'], fontsize=f15)

    ax2 = sns.boxplot(x='bin_name', y='error_value', data=outws_nme_df_s, hue='error_name',
                      palette='BuGn_r', ax=ax2)

    ax2.axhline(0, ls='--', color='grey')
    ax2.set_ylabel('NME (%)', fontsize=f15)
    ax2.set_xlabel('Normalized wind speed category', fontsize=f15)
    ax2.set_ylim([-1.2, 1.2])
    ax2.legend_.remove()

    ax2.tick_params(labelsize=f15)
    ax2.xaxis.set_tick_params(rotation=45)

    ax1.text(0.94, 0.91, '(a)', color='k', fontsize=f15, transform=ax1.transAxes)
    ax2.text(0.94, 0.91, '(b)', color='k', fontsize=f15, transform=ax2.transAxes)

    finish_plot('meta', 'outws_energyfraction', 'box')


def plot_hist_series(df, var, name):
    """Plot histogram for series of meta data."""

    p_series = pd.Series(df[var])

    p_series.value_counts(dropna=False).plot(kind='bar', rot=45)
    plt.ylabel('Count')
    plt.title(name)

    finish_plot('meta', 'meta_' + var, 'hist')


def loop_meta_hist():
    """Generate histograms from available meta data."""

    for var, name in zip(pc.meta_var_names_turb, pc.meta_xls_names_turb):
        plot_hist_series(p_init.meta_df, var, name)


def plot_group_meta_hist():
    """Plot 4 histograms using grouped bins on x-axis."""

    hist_df1 = pd.DataFrame({'turbi_dia_grouped': meta_df['turbi_dia_grouped'].value_counts(dropna=False)})
    hist_df1.reset_index(inplace=True)
    hist_df1.replace('143 - 154', '120+', inplace=True)
    x_sorted1, x_nozero1 = psd.remove_0_in_label(hist_df1)

    hist_df2 = pd.DataFrame({'turbi_hh_grouped': meta_df['turbi_hh_grouped'].value_counts(dropna=False)})
    hist_df2.reset_index(inplace=True)
    hist_df2.replace('132 - 143', '110+', inplace=True)
    hist_df2.replace('110 - 121', '110+', inplace=True)
    hist_df22 = hist_df2.groupby(['index'], as_index=False).agg('sum')
    x_sorted2, x_nozero2 = psd.remove_0_in_label(hist_df22)

    hist_df3 = pd.DataFrame({'turbi_spower_grouped': meta_df['turbi_spower_grouped'].value_counts(dropna=False)})
    hist_df3.reset_index(inplace=True)
    hist_df3.replace('489 - 536', '441+', inplace=True)
    hist_df3.replace('536 - 583', '441+', inplace=True)
    hist_df33 = hist_df3.groupby(['index'], as_index=False).agg('sum')
    x_sorted3, x_nozero3 = psd.remove_0_in_label(hist_df33)

    meta_var_names_array = np.array(pc.meta_var_names)
    year_measure_idx = np.where(meta_var_names_array == 'year_measuremt')
    year_str = pc.meta_var_names[int(year_measure_idx[0])]

    hist_df4 = pd.DataFrame({year_str: meta_df[year_str].value_counts(dropna=False)})
    hist_df4.reset_index(inplace=True)
    x_sorted4, x_nozero4 = psd.remove_0_in_label(hist_df4)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    sns.barplot(x='index', y='turbi_dia_grouped', data=hist_df1, ax=ax1, order=x_sorted1, color='c')
    ax1.set_xticklabels(labels=x_nozero1['index'], rotation=45)
    ax1.set_xlabel('Turbine rotor diameter (m)', fontsize=fs+1)
    ax1.set_ylabel('Count', fontsize=fs+1)

    sns.barplot(x='index', y='turbi_hh_grouped', data=hist_df22, ax=ax2, order=x_sorted2, color='c')
    ax2.set_xticklabels(labels=x_nozero2['index'], rotation=45)
    ax2.set_xlabel('Turbine hub height (m)', labelpad=10, fontsize=fs+1)
    ax2.set_ylabel('')

    sns.barplot(x='index', y='turbi_spower_grouped', data=hist_df33, ax=ax3, order=x_sorted3, color='c')
    ax3.set_xticklabels(labels=x_nozero3['index'], rotation=45)
    ax3.set_xlabel(r'Turbine specific power (W m$^{-2}$)', fontsize=fs+1)
    ax3.set_ylabel('Count', fontsize=fs+1)

    sns.barplot(x='index', y=year_str, data=hist_df4, ax=ax4, order=x_sorted4, color='c')
    ax4.set_xticklabels(labels=x_nozero4['index'], rotation=45)
    ax4.set_xlabel('Year of measurement', labelpad=25, fontsize=fs+1)
    ax4.set_ylabel('')

    ax1.text(xp_f1, yp_f1, '(a)', color='k', fontsize=fs, transform=ax1.transAxes)
    ax2.text(xp_f1, yp_f1, '(b)', color='k', fontsize=fs, transform=ax2.transAxes)
    ax3.text(xp_f1, yp_f1, '(c)', color='k', fontsize=fs, transform=ax3.transAxes)
    ax4.text(xp_f1, yp_f1, '(d)', color='k', fontsize=fs, transform=ax4.transAxes)

    finish_plot('meta', 'meta_', 'hist_ranked')


def plot_map():
    """Map submission origins, if available."""

    country_series = meta_df['geog_country'].value_counts(dropna=False)
    country_series = country_series.rename_axis('country').reset_index()
    country_na = country_series.loc[country_series['country'].isnull()].index
    country_series_plot = country_series.drop(country_na[0])  # drop NaN

    # count of NaN
    nan_country = str(country_series.loc[country_series['country'].isnull()]['geog_country'][0])
    total_country = str(len(meta_df['geog_country']))

    colmap = plt.cm.get_cmap('viridis')
    colmap(1)

    country_num_max = max(country_series_plot['geog_country'])

    country_num = np.linspace(0, 1, country_num_max + 1)

    country_num_on_map = country_num[country_series_plot['geog_country']]

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    def plot_country_patch(axes, country_name, fcolor):
        # plot a country on the provided axes
        nami = world[world.name == country_name]
        namigm = nami.__geo_interface__['features']  # geopandas's geo_interface
        namig0 = {'type': namigm[0]['geometry']['type'], 'coordinates': namigm[0]['geometry']['coordinates']}
        axes.add_patch(PolygonPatch(namig0, fc=fcolor, ec='black', alpha=0.85, zorder=2))

    colmap = plt.cm.get_cmap('viridis')

    ax = world.plot(figsize=(8, 4), edgecolor=u'gray', color='w')

    for x, y in zip(country_series_plot['country'].values, country_num_on_map):
        plot_country_patch(ax, x, colmap(y))

    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    plt.title(nan_country + ' of ' + total_country + ' submissions have unknown countries')

    fig = ax.get_figure()
    cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    sm = plt.cm.ScalarMappable(cmap=colmap)
    sm._A = []
    cbr = fig.colorbar(sm, cax=cax)
    cbr.set_ticks(np.linspace(0, 1, country_num_max))
    cbr.ax.set_yticklabels(np.linspace(1, country_num_max, country_num_max))

    finish_plot('meta', 'meta', 'map', tight_layout=False)


def plot_nme_hist():
    """Plot NME histograms: pre-NME-filtering and post-NME-filtering."""

    plt.figure(1, figsize=(8, 6))

    plt.rcParams.update({'font.size': 14})

    gridspec.GridSpec(2, 2)

    # need to get pre-nme-filter error data frame
    df1 = pcd.get_error_df_dict(p_init.data_file)['base_total_e']

    df1p = df1.loc[(df1['error_cat'] == 'by_range') & (df1['error_name'] == 'nme')]

    inner_nme = df1p['error_value'].loc[df1p['bin_name'] == 'Inner'] * 100
    outer_nme = df1p['error_value'].loc[df1p['bin_name'] == 'Outer'] * 100

    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=1)

    a_value = 0.7

    sns.distplot(list(outer_nme.values), color='#73c0c4', label='Outer Range', bins=10, kde=False, ax=ax1,
                 hist_kws={'alpha': a_value})
    sns.distplot(list(inner_nme.values), color='#3c758b', label='Inner Range', bins=5, kde=False, ax=ax1,
                 hist_kws={'alpha': a_value})

    plt.ylabel('Count')
    plt.xlabel('NME (%)')

    plt.legend()

    sheet_bt_choice = 'base_total_e'
    df2 = p_init.error_df

    def choose_in_out_def(in_or_out, in_out_def):

        selection = ((df2[sheet_bt_choice]['error_cat'] == 'by_range')
                     & (df2[sheet_bt_choice]['error_name'] == 'nme')
                     & (df2[sheet_bt_choice]['bin_name'] == in_or_out)
                     & (df2[sheet_bt_choice]['file_name']
                        .isin(meta_df.loc[meta_df['inner_def'] == in_out_def]['file_name'])))

        return selection

    inner_a = df2[sheet_bt_choice].loc[choose_in_out_def('Inner', 'A')]['error_value'] * 100
    inner_b = df2[sheet_bt_choice].loc[choose_in_out_def('Inner', 'B')]['error_value'] * 100
    inner_c = df2[sheet_bt_choice].loc[choose_in_out_def('Inner', 'C')]['error_value'] * 100

    outer_a = df2[sheet_bt_choice].loc[choose_in_out_def('Outer', 'A')]['error_value'] * 100
    outer_b = df2[sheet_bt_choice].loc[choose_in_out_def('Outer', 'B')]['error_value'] * 100
    outer_c = df2[sheet_bt_choice].loc[choose_in_out_def('Outer', 'C')]['error_value'] * 100

    p23_c = ['seagreen', 'limegreen', 'lawngreen']
    # p23_c = ['red', 'darkorange', 'gold']

    ax2 = plt.subplot2grid((2, 2), (1, 0))

    ax2.hist([inner_a, inner_b, inner_c], stacked=True, color=p23_c)
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Filtered Inner Range NME (%)')

    ax3 = plt.subplot2grid((2, 2), (1, 1))

    ax3.hist([outer_a, outer_b, outer_c], stacked=True, color=p23_c)
    ax3.set_ylabel('Count')
    ax3.set_xlabel('Filtered Outer Range NME (%)')

    labels = ['A', 'B', 'C']
    plt.legend(labels, title='Definition', loc='upper left')

    ax1.text(0.03, 0.89, '(a)', color='k', fontsize=12, transform=ax1.transAxes)
    ax2.text(0.05, 0.88, '(b)', color='k', fontsize=12, transform=ax2.transAxes)
    ax3.text(0.89, 0.88, '(c)', color='k', fontsize=12, transform=ax3.transAxes)

    finish_plot('error_hist', 'nme', '3def_hist')

    plt.rcParams.update({'font.size': fs})


def plot_wsti_nme_box():
    """Plot 4 panel box plots for WS-TI NME."""

    box_plot_y_scale = 0.5  # zoom in
    # box_plot_y_scale = 1

    def ws_ti_df_by_sheet(sheet_name_short, df, bt_choice, error_name, file_num):

        sheet_bt_choice = sheet_name_short + bt_choice

        ws_ti_df = df[sheet_bt_choice].loc[(df[sheet_bt_choice]['error_cat'] == 'by_ws_ti')
                                           & (df[sheet_bt_choice]['error_name'] == error_name)]

        if file_num is True:

            sheet_name_end = ': ' + str(round(len(ws_ti_df) / 4))

        else:

            sheet_name_end = ''

        ws_ti_df.insert(0, 'sheet_name', str(sheet_name_short)[:-1] + sheet_name_end)

        return ws_ti_df

    def loop_box_plot(bt_choice, error_name, error_df, file_num_choice=False, extra_error_df=None):

        for i, i_short in zip(pc.matrix_sheet_name, pc.matrix_sheet_name_short):

            dum_df = ws_ti_df_by_sheet(i_short, error_df, bt_choice, error_name, file_num=file_num_choice)

            if pc.matrix_sheet_name.index(i) == 0:
                ws_ti_df = dum_df

            else:
                ws_ti_df = ws_ti_df.append(dum_df)

        if extra_error_df is not None:

            for i, i_short in zip(pc.correction_list, pc.extra_matrix_sheet_name_short):

                dum_df = ws_ti_df_by_sheet(i_short, extra_error_df, bt_choice, error_name,
                                           file_num=file_num_choice)

                if pc.correction_list.index(i) == 0:
                    ws_ti_df_extra = dum_df

                else:
                    ws_ti_df_extra = ws_ti_df_extra.append(dum_df)

            ws_ti_df = ws_ti_df.append(ws_ti_df_extra)

        ws_ti_df['error_value'] = ws_ti_df['error_value'].astype(float) * 100

        ws_ti_error_min, ws_ti_error_max = ws_ti_df['error_value'].min(), ws_ti_df['error_value'].max()
        ws_ti_error_abs_max = np.max([abs(ws_ti_error_min), abs(ws_ti_error_max)])

        lws_lti_df = ws_ti_df.loc[ws_ti_df['bin_name'] == 'LWS-LTI']
        lws_hti_df = ws_ti_df.loc[ws_ti_df['bin_name'] == 'LWS-HTI']
        hws_lti_df = ws_ti_df.loc[ws_ti_df['bin_name'] == 'HWS-LTI']
        hws_hti_df = ws_ti_df.loc[ws_ti_df['bin_name'] == 'HWS-HTI']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))

        def plot_box_by_sheet(df, ax, sub_t):

            # add grey to colorblind... manually
            # list(sns.color_palette(['grey']))
            # sns.color_palette('colorblind')

            new_p = [(0.5019607843137255, 0.5019607843137255, 0.5019607843137255),
                     (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
                     (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
                     (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
                     (0.8352941176470589, 0.3686274509803922, 0.0),
                     (0.8, 0.47058823529411764, 0.7372549019607844),
                     (0.792156862745098, 0.5686274509803921, 0.3803921568627451),
                     (0.984313725490196, 0.6862745098039216, 0.8941176470588236),
                     (0.5803921568627451, 0.5803921568627451, 0.5803921568627451),
                     (0.9254901960784314, 0.8823529411764706, 0.2),
                     (0.33725490196078434, 0.7058823529411765, 0.9137254901960784)]

            # ax = sns.boxplot(x='sheet_name', y='error_value', data=df, ax=ax, palette=new_p)
            ax = sns.boxplot(x='sheet_name', y='error_value', data=df, ax=ax, palette='colorblind')
            # ax = sns.swarmplot(x='sheet_name', y='error_value', data=df, ax=ax, palette='colorblind')

            if error_name == 'nme':

                ax.axhline(0, ls='--', color='grey')

            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            ax.set_title(df['bin_name'].iloc[0])
            ax.set_ylabel(bt_choice + 'nergy ' + error_name + ' (%)')

            # ax.set_ylim([ws_ti_error_min*box_plot_y_scale, ws_ti_error_max*box_plot_y_scale])
            ax.set_ylim([-ws_ti_error_abs_max * box_plot_y_scale, ws_ti_error_abs_max * box_plot_y_scale])

            ax.text(0.95, 0.92, sub_t, color='k', fontsize=12, transform=ax.transAxes)

            return ax

        plot_box_by_sheet(lws_lti_df, ax1, '(a)')
        plot_box_by_sheet(lws_hti_df, ax2, '(b)')
        plot_box_by_sheet(hws_lti_df, ax3, '(c)')
        plot_box_by_sheet(hws_hti_df, ax4, '(d)')

        if extra_error_df is not None:
            var = 'wsti_nme_boxplot_extra'

        else:
            var = 'wsti_nme_boxplot'

        finish_plot('results', var, bt_choice + '_' + error_name)

    loop_box_plot('total_e', 'nme', p_init.error_df, extra_error_df=p_init.extra_error_df)


def plot_nme_avg_spread_heatmap(ee_df=None, rr_choice=None):
    """Mass generate heatmaps of NME average and NME spread."""

    def loop_nme_avg_spread_heatmap(by_bin, bt_choice, error_name, e_df=p_init.error_df, ee_df=None,
                                    rr_choice=pc.robust_resistant_choice):

        for idx, i_short in enumerate(pc.matrix_sheet_name_short):

            df = psd.get_error_in_bin(e_df, i_short+bt_choice, by_bin, error_name)

            u_bin, average, spread = psd.find_unique_bin_create_dum(df['bin_name'])

            if idx == 0:

                average_df = pd.DataFrame(index=u_bin, columns=[pc.matrix_sheet_name_short])
                spread_df = pd.DataFrame(index=u_bin, columns=[pc.matrix_sheet_name_short])

            psd.cal_average_spread(df, u_bin, average_df, spread_df, i_short, rr_choice)

        if ee_df is not None:

            for idx, i_short in enumerate(pc.extra_matrix_sheet_name_short):

                df = psd.get_error_in_bin(p_init.extra_error_df, i_short+bt_choice, by_bin, error_name)

                u_bin, average, spread = psd.find_unique_bin_create_dum(df['bin_name'])

                psd.cal_average_spread(df, u_bin, average_df, spread_df, i_short, rr_choice)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        if error_name == 'nme':

            sns.heatmap(average_df, linewidths=.5, cmap='RdBu_r', center=0, ax=ax1)

        elif error_name == 'nmae':

            sns.heatmap(average_df, linewidths=.5, ax=ax1)

        if rr_choice is None:

            avg_title, spd_title, out_name = 'mean', 'standard deviation', 'mean_sd'

        else:

            avg_title, spd_title, out_name = 'median', 'interquartile range', 'median_iqr'

        ax1.set_xlabel('correction methods')
        ax1.set_ylabel(by_bin)
        ax1.yaxis.set_tick_params(rotation=0)
        ax1.set_title(bt_choice + 'nergy ' + avg_title + ' ' + error_name + ' (%)')

        sns.heatmap(spread_df, linewidths=.5, cmap='viridis', ax=ax2)
        ax2.set_xlabel('correction methods')
        ax2.set_ylabel(by_bin)
        ax2.yaxis.set_tick_params(rotation=0)
        ax2.set_title(bt_choice + 'nergy ' + spd_title + ' of ' + error_name + ' (%)')

        if ee_df is not None:
            var = 'nme_avg_spread_boxplot_extra'
        else:
            var = 'nme_avg_spread_boxplot'

        finish_plot('results', var, bt_choice + '_heatmap')

    for idx, (cat_i, bt_j, error_k) in enumerate(itertools.product(pc.error_cat_short[1:],
                                                                   pc.bt_choice, pc.error_name[1:])):
        loop_nme_avg_spread_heatmap(cat_i, bt_j, error_k,
                                    e_df=p_init.error_df, ee_df=ee_df,
                                    rr_choice=rr_choice)


def plot_inner_outer_data_count_box_hist():
    """Plot Inner Range and Outer Range data count"""

    box_io_df = p_init.error_df['base_total_e'].loc[(p_init.error_df['base_total_e']['error_cat'] == 'by_range')
                                                    & (p_init.error_df['base_total_e']['error_name'] == 'data_count')]

    u_bin = box_io_df['bin_name'].unique()

    for idx, val in enumerate(u_bin):

        data_count = box_io_df.loc[box_io_df['bin_name'] == val]['error_value'].values

        if idx == 0:

            box_dict = {val: data_count}

        else:

            box_dict[val] = data_count

    box_df = pd.DataFrame(box_dict)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    sns.boxplot(data=box_df, ax=ax1, palette='GnBu_d')
    ax1.set_xticklabels(labels=['Inner Range', 'Outer Range'])

    ax1.set_ylabel('10-minute data count')

    base_inner_count = p_init.error_df['base_bin_e'].loc[(p_init.error_df['base_bin_e']['error_cat'] == 'by_range')
                                                         & (p_init.error_df['base_bin_e']['bin_name'] == 'Inner')
                                                         & (p_init.error_df['base_bin_e']['error_name']
                                                            == 'data_count')]

    base_outer_count = p_init.error_df['base_bin_e'].loc[(p_init.error_df['base_bin_e']['error_cat'] == 'by_range')
                                                         & (p_init.error_df['base_bin_e']['bin_name'] == 'Outer')
                                                         & (p_init.error_df['base_bin_e']['error_name']
                                                            == 'data_count')]

    ratio_inout_count = pd.Series([base_outer_count['error_value'].values / base_inner_count['error_value'].values])

    ax2.hist(ratio_inout_count.values, color='grey')
    ax2.set_ylabel('File count')
    ax2.set_xlabel(r'$\mathrm{\mathsf{\frac{Outer\/\/Range\/\/data\/\/count}{Inner\/\/Range\/\/data\/\/count}}}$',
                   fontsize=fs + 5)

    ax1.text(xp_f1, yp_f1, '(a)', color='k', fontsize=fs, transform=ax1.transAxes)
    ax2.text(xp_f1, yp_f1, '(b)', color='k', fontsize=fs, transform=ax2.transAxes)

    finish_plot('meta', 'data_count', 'box_hist')


def plot_file_data_count_hist_box():
    """Mass generate histogram of file count and box plot of data count for each category.
    Check if every sheet/method has the same file count for each inflow category.
    """

    def loop_count_histbox(by_bin, e_df, sheet_name, no_hist=None):

        for idx, i_short in enumerate(sheet_name):

            do_plot = False

            bt_choice = 'bin_e'  # same data count for total_e

            df = e_df[i_short + bt_choice].loc[(e_df[i_short + bt_choice]['error_cat'] == by_bin)
                                               & (e_df[i_short + bt_choice]['error_name'] == 'data_count')]

            u_bin = df['bin_name'].unique()

            nan_bin_count = np.zeros(len(u_bin))

            for i in range(len(df)):

                for index, val in enumerate(u_bin):

                    if df.iloc[i]['bin_name'] == val:

                        if pd.isnull(df.iloc[i]['error_value']):

                            nan_bin_count[index] += 1

            file_num = len(df) / len(u_bin)
            bin_count = file_num - nan_bin_count

            try:

                lump_bin_count  # see if data duplicates already

            except NameError:

                lump_bin_count = bin_count
                lump_bin_count = np.expand_dims(lump_bin_count, axis=0)
                do_plot = True

            if lump_bin_count.shape[0] > 1:
                # for i in range(lump_bin_count.shape[0]):
                for i in lump_bin_count:

                    if np.array_equal(bin_count, i):
                        do_plot = False
                    else:
                        do_plot = True

            # if data count not duplicate
            if do_plot:

                print(i_short + ' contains unique file count distribution for ' + by_bin)

                file_data = {'d_bin': u_bin, 'd_count': bin_count}

                hist_df = pd.DataFrame(file_data)

                for index, val in enumerate(u_bin):

                    data_count = df.loc[df['bin_name'] == val]['error_value'].values

                    if index == 0:
                        box_dict = {val: data_count}
                    else:
                        box_dict[val] = data_count

                box_df = pd.DataFrame(box_dict)

                if no_hist is None:

                    no_hist = ''

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                    sns.barplot(x='d_bin', y='d_count', data=hist_df, ax=ax1, color='orchid')
                    ax1.set_xticklabels(labels=hist_df['d_bin'], rotation=45)
                    ax1.set_title(str(i_short)[:-1] + ', max: ' + str(int(np.max(bin_count))) + ' files')
                    ax1.set_xlabel(by_bin)
                    ax1.set_ylabel('file count')

                    sns.boxplot(data=box_df, ax=ax2, palette='colorblind')
                    ax2.set_xticklabels(labels=hist_df['d_bin'], rotation=45)
                    ax2.set_title(str(i_short)[:-1])
                    ax2.set_xlabel(by_bin)
                    ax2.set_ylabel('data count')

                    if len(hist_df['d_bin']) > 15:  # every other xtick label
                        for label_ax1, label_ax2 in zip(ax1.xaxis.get_ticklabels()[::2],
                                                        ax2.xaxis.get_ticklabels()[::2]):

                            label_ax1.set_visible(False)
                            label_ax2.set_visible(False)

                elif no_hist == 'box':

                    no_hist = no_hist + '_'

                    fig, ax2 = plt.subplots(figsize=(6, 5))

                    sns.boxplot(data=box_df, ax=ax2, palette='colorblind')
                    ax2.set_xticklabels(labels=hist_df['d_bin'], rotation=45)
                    ax2.set_title(str(i_short)[:-1])
                    ax2.set_xlabel(by_bin)
                    ax2.set_ylabel('data count')

                    if len(hist_df['d_bin']) > 15:  # every other xtick label
                        for label_ax2 in ax2.xaxis.get_ticklabels()[::2]:
                            label_ax2.set_visible(False)

                finish_plot('meta', no_hist + by_bin + '_' + str(i_short)[:-1], 'count_hist_box')

                # if i > 1:
                lump_bin_count = np.concatenate((lump_bin_count, np.expand_dims(bin_count, axis=0)), axis=0)

            else:

                print(i_short + ' contains duplicating file count distribution for ' + by_bin)

        del lump_bin_count

    for i in pc.error_cat_short[1:]:

        loop_count_histbox(i, p_init.error_df, pc.matrix_sheet_name_short)
        loop_count_histbox(i, p_init.extra_error_df, pc.extra_matrix_sheet_name_short)


def loop_outer_diff_scatter(meta_var, one_plot=None, diff_choice=None):
    """Plot scatter plot between meta data and error, or between meta data and error difference.
    Pair of plots: NME and NMAE
    """

    if diff_choice is None:

        y_var = 'error_value'
        y_title_end = ''

    else:

        y_var = 'diff'
        y_title_end = ' - Baseline'

    def add_corr_text(ax, corr):

        add_x = 0.13

        if corr.size != 0:

            ax.text(0.7, 0.94, 'correlation', color='k',
                    weight='semibold', transform=ax.transAxes)  # ratio of axes

            for idx, val in enumerate(corr):
                ax.text(0.75, 1 - add_x, val, color=sheet_color[idx],
                        weight='semibold', transform=ax.transAxes)

                add_x += 0.07

    for bt_c in pc.bt_choice:

        ########################
        ##### SKIPPING ONE #####
        ########################
        if meta_var != 'year_operatn':  # HARD CODED -- no useful data there

            nme_df, nme_corr = psd.get_outer_meta('nme', meta_var, bt_c, y_var)
            nmae_df, nmae_corr = psd.get_outer_meta('nmae', meta_var, bt_c, y_var)

            if diff_choice is not None:

                nme_df = nme_df.loc[~(nme_df['sheet'] == 'base')]
                nmae_df = nmae_df.loc[~(nmae_df['sheet'] == 'base')]

            nme_file_num = str(round(nme_df[meta_var].count()/len(nme_df['sheet'].unique())))[:-2]
            nmae_file_num = str(round(nmae_df[meta_var].count()/len(nmae_df['sheet'].unique())))[:-2]

            if meta_var == 'turbi_spower':  # change units for specific power
                nme_df[meta_var] = nme_df[meta_var] * 1e3
                nmae_df[meta_var] = nmae_df[meta_var] * 1e3

            if one_plot is None:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            else:
                fig, ax1 = plt.subplots(figsize=(6, 4))

            axx = sns.color_palette(palette='colorblind')
            sheet_color = axx.as_hex()

            if one_plot is None:

                sns.scatterplot(x=meta_var, y=y_var, hue='sheet', data=nme_df, alpha=0.5,
                                palette='colorblind', ax=ax1, legend=False)

                sns.scatterplot(x=meta_var, y=y_var, hue='sheet', data=nmae_df, alpha=0.5,
                                palette='colorblind', ax=ax2)
                ax2.set_xlabel(meta_var+': '+nmae_file_num+' files')
                ax2.set_ylabel(bt_c+' nmae'+y_title_end)
                ax2.axhline(y=0, linestyle='--', color='grey')
                ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                add_corr_text(ax2, nmae_corr)

                op_text = ''

            # plot NMEs only
            else:

                sns.scatterplot(x=meta_var, y=y_var, hue='sheet', data=nme_df, alpha=0.5,
                                palette='colorblind', ax=ax1)
                ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

                op_text = '_one'

            ax1.set_xlabel(meta_var+': '+nme_file_num+' files')
            ax1.set_ylabel(bt_c+' nme'+y_title_end)
            ax1.axhline(y=0, linestyle='--', color='grey')
            add_corr_text(ax1, nme_corr)

            finish_plot('results', 'scatter_diff_outer', meta_var+op_text+'_'+bt_c)


def plot_outer_diff_scatter(one_plot=None, diff_choice=None):
    """Mass generate scatter plot of error or error difference."""

    for meta_var in pc.meta_var_names_turb:

        loop_outer_diff_scatter(meta_var, one_plot=one_plot, diff_choice=diff_choice)


def plot_outer_nme_inner_dc_scatter():
    """Generate scatter plot between Outer Range NME and Inner Range data count."""

    for sheet in pc.matrix_sheet_name_short:

        target_df = p_init.error_df[sheet+'total_e']

        outer_nme = target_df.loc[(target_df['error_cat'] == 'by_range') & (target_df['error_name'] == 'nme')
                                  & (target_df['bin_name'] == 'Outer')]
        inner_dc = target_df.loc[(target_df['error_cat'] == 'by_range') & (target_df['error_name'] == 'data_count')
                                 & (target_df['bin_name'] == 'Inner')]

        two_df_out_nme_in_dc = pd.merge(outer_nme, inner_dc, on='file_name')

        ax = sns.scatterplot(x='error_value_y', y='error_value_x', data=two_df_out_nme_in_dc)
        ax.set_title(sheet)
        ax.set_ylabel('outer range nme')
        ax.set_xlabel('inner range data count')

        finish_plot('results', 'scatter_outer_nme', 'inner_dc')


def plot_nme_diff_box_range_scatter_hist():
    """Produce a panel of box plot, scatter, and histogram.
    Plot absolute NME difference box plots for each submission.
    Plot statistical range of absolute NME differences for each submission.
    Plot histogram for the statistical ranges.
    """

    nme_diff_df, nme_range_p_df = psd.get_nme_diff_range()

    gs = gridspec.GridSpec(105, 100)

    plt.subplots(figsize=[16, 6])

    top_start, top_end = 0, 50
    center_left = 75

    ax1 = plt.subplot(gs[top_start:top_end, 0:center_left])
    ax21 = plt.subplot(gs[top_end + 5::, 0:center_left])
    ax22 = plt.subplot(gs[top_end + 5:, center_left + 4:])

    sns.boxplot(data=nme_diff_df, color='orange', ax=ax1)
    # fy_cp = ['orange'] * nme_diff_df.shape[1]
    # sns.swarmplot(data=nme_diff_df, palette=fy_cp, ax=ax1)
    ax1.set(xticklabels=[])
    ax1.tick_params(bottom=False)
    ax1.set_ylabel('|NME| difference (%)')
    ax1.axhline(0, ls='--', color='grey')

    pfy_c = ['dodgerblue', 'grey', 'red']
    pf_m = ['^', 'o', 'v']
    pf_ms = []
    for item in nme_range_p_df['all']:
        if item == 'Improved' or item == 'Worse':
            pf_ms.append(110)
        else:
            pf_ms.append(80)

    sns.scatterplot(x='index', y='nme', data=nme_range_p_df, hue='all', palette=pfy_c, style='all',
                    markers=pf_m, s=pf_ms, ax=ax21)
    handles, labels = ax21.get_legend_handles_labels()
    ax21.legend(handles[1:], labels[1:], ncol=3)
    ax21.set_xlim([-0.6, 51.6])
    ax21.set(xticklabels=[])
    # ax2.set_markersizes = pf_ms
    ax21.tick_params(bottom=False)
    ax21.set_ylabel('Range of \n|NME| differences (%)')
    ax21.set_xlabel('Data set submission')
    ax21.xaxis.labelpad = 10

    nme_range_p_df['nme'].hist(orientation='horizontal', color='k', ax=ax22)
    # ax22 = plt.hist(nme_range_p_df['nme'].values, orientation='horizontal', color='k')
    ax22.set_xlabel('Count')
    ax22.grid(False)

    xp_fx, yp_fx = 0.97, 0.88

    ax1.text(xp_fx, yp_fx, '(a)', color='k', fontsize=fs, transform=ax1.transAxes)
    ax21.text(xp_fx, yp_fx, '(b)', color='k', fontsize=fs, transform=ax21.transAxes)
    plt.text(xp_fx - 0.07, yp_fx, '(c)', color='k', fontsize=fs, transform=ax22.transAxes)

    finish_plot('results', 'nme_diff_box', 'range_scatter_hist', tight_layout=False)


def plot_wsti_nme_avg_spread_heatmap():
    """Plot 1 average and spread NME heatmap for WS-TI, ITI-OS, and Inner-Outer Ranges."""

    average_df, spread_df, dum = psd.get_wsti_nme_stat()

    average_df = psd.sort_plot_wsti_df_index(average_df)
    spread_df = psd.sort_plot_wsti_df_index(spread_df)

    average_df.rename(columns=pc.method_dict, inplace=True)
    average_df1 = average_df.iloc[0:5]
    average_df3 = average_df.iloc[5:]

    ax13_max = average_df.values.max()
    ax13_min = average_df.values.min()

    ax13_mm = np.max([abs(ax13_max), abs(ax13_min)])

    spread_df.rename(columns=pc.method_dict, inplace=True)
    spread_df2 = spread_df.iloc[0:5]
    spread_df4 = spread_df.iloc[5:]

    ax24_max = spread_df.values.max()
    ax24_min = spread_df.values.min()

    plt.rcParams.update({'font.size': f16})

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 9), gridspec_kw={'height_ratios': [1, 1.2]})

    sns.heatmap(average_df1, linewidths=.5, cmap='RdBu_r', center=0, ax=ax1, annot=True, vmin=-ax13_mm, vmax=ax13_mm,
                cbar=False)

    ax1.set_xlabel('')
    ax1.set(xticklabels=[])
    ax1.set_ylabel('WS-TI bin')
    ax1.yaxis.set_tick_params(rotation=0)
    ax1.tick_params(bottom=False)

    sns.heatmap(spread_df2, linewidths=.5, cmap='viridis', ax=ax2, annot=True, vmin=ax24_min, vmax=ax24_max,
                cbar=False)
    ax2.set_xlabel('')
    ax2.set(xticklabels=[])
    ax2.set_ylabel('')
    ax2.yaxis.set_tick_params(rotation=0)
    ax2.tick_params(bottom=False)

    sns.heatmap(average_df3, linewidths=.5, cmap='RdBu_r', center=0, ax=ax3, annot=True, vmin=-ax13_mm, vmax=ax13_mm,
                cbar=False)
    ax3.set_xlabel('Correction method')
    ax3.yaxis.set_tick_params(rotation=0)
    ax3.xaxis.set_tick_params(rotation=30)
    ax3.set_ylabel('Inner-Outer Range')

    sns.heatmap(spread_df4, linewidths=.5, cmap='viridis', ax=ax4, annot=True, vmin=ax24_min, vmax=ax24_max,
                cbar=False)
    ax4.set_xlabel('Correction method')
    ax4.yaxis.set_tick_params(rotation=0)
    ax4.xaxis.set_tick_params(rotation=30)

    sm = plt.cm.ScalarMappable(cmap='RdBu_r')
    sm.set_array([-ax13_mm, ax13_mm])
    cb3 = fig.colorbar(sm, ax=ax3, orientation='horizontal', pad=0.35)
    cb3.set_label('Mean NME (%)')

    sm = plt.cm.ScalarMappable(cmap='viridis')
    sm.set_array([ax24_min, ax24_max])
    cb4 = fig.colorbar(sm, ax=ax4, orientation='horizontal', pad=0.35)
    cb4.set_label('NME standard deviation (%)')

    ax3.text(0.0, -1.15, '(a)', color='k', fontsize=f16, transform=ax3.transAxes)
    ax4.text(0.0, -1.15, '(b)', color='k', fontsize=f16, transform=ax4.transAxes)

    finish_plot('results', 'wsti_nme', 'avg_spread_heatmap')

    plt.rcParams.update({'font.size': fs})


def plot_wsti_nme_ef_dot():
    """Plot for NME scatter vs WS-TI bins each correction method.
    Dot size proportionate to energy fraction.
    """

    for idx, i_short in enumerate(pc.matrix_sheet_name_short):
        error_cat = 'by_ws_ti'

        ws_ti_o_nme_df = psd.get_wsti_outer_nme(i_short, error_cat)

        ef_df = pef.cal_wsti_ef(error_cat)

        problem_file = pef.check_problematic_file(ef_df, error_cat)

        out_df = pef.remove_problematic_files(ws_ti_o_nme_df, problem_file, error_cat, pc.wsti_new_bin)

        energy_pct = pef.get_energy_pct(ef_df, error_cat)

        merge_df = pd.merge(out_df, energy_pct, how='left', on=['bin_name', 'file_name'])

        merge_df.drop(['error_name_y'], axis=1, inplace=True)

        merge_df['value'].fillna(100, inplace=True)

        merge_df.rename(columns={'value': 'Energy fraction (%)'}, inplace=True)

        merge_df_rev = pd.concat([merge_df.loc[merge_df['bin_name'] == 'Outer'].reset_index(),
                                  merge_df.loc[merge_df['bin_name'] == 'ITI-OS'].reset_index(),
                                  merge_df.loc[merge_df['bin_name'] == 'HWS-HTI'].reset_index(),
                                  merge_df.loc[merge_df['bin_name'] == 'HWS-LTI'].reset_index(),
                                  merge_df.loc[merge_df['bin_name'] == 'LWS-HTI'].reset_index(),
                                  merge_df.loc[merge_df['bin_name'] == 'LWS-LTI'].reset_index()], axis=0)

        plt.subplots(figsize=(10, 4))

        ax = sns.scatterplot(x='error_value', y='bin_name', size='Energy fraction (%)', data=merge_df_rev,
                             sizes=(10, 200), alpha=0.5, color='black')

        ax.set_title(i_short)
        plt.axvline(0, linestyle='--', c='k')

        plt.legend(loc='center left', bbox_to_anchor=(1., 0.5))

        finish_plot('results', 'wsti_nme', 'ef_dot')


def get_removal_num(remove_outlier_choice, remove_quantile, diff_removal_num, percent_thres):
    """Determine the number of removed submissions from outlier filtering."""

    if remove_outlier_choice is True:
        if remove_quantile is True:
            ax1_title_head = ('Removing top ' + str(pc.quantile_cut * 100)[:-2] + '% of improvement ('
                              + str(np.max(diff_removal_num))[:-2] + ' files max):\n')
            plot_name_ro = 'q' + str(pc.quantile_cut * 100)[:-2]

        else:
            ax1_title_head = ('Removing data with improvement > ' + str(percent_thres)
                              + '%' + str(np.max(diff_removal_num))[:-2] + ' files max):\n')
            plot_name_ro = str(percent_thres) + '%'

    else:
        ax1_title_head = ''
        plot_name_ro = ''

    return ax1_title_head, plot_name_ro


def plot_wsti_pct_ttest_ltest_heatmap(remove_outlier_choice=False, remove_quantile=False, bonferroni=None,
                                      percent_thres=None):
    """Plot 3 heatmaps of statistical results for WS-TI and Inner-Outer Range.
    Plot: percentage of data sets that improve from Baseline;
    whether method's mean NME less than Baseline's or not, and its statistical significance;
    whether method's NME variance less than Baseline's or not, and its statistical significance.
    Similar to `plot_ecat_pct_ttest_ltest_heatmap`
    """

    plot_choice, pc_df, diff_ttest_df, ltest_df, \
    diff_removal_num = psd.perform_stat_test(wsti=True, remove_outlier_choice=remove_outlier_choice,
                                             remove_quantile=remove_quantile, bonferroni=bonferroni,
                                             percent_thres=percent_thres)

    if plot_choice is True:

        pc_df11 = pc_df.iloc[0:5]
        pc_df12 = pc_df.iloc[5:]

        diff_ttest_df1 = diff_ttest_df.iloc[0:5]
        diff_ttest_df2 = diff_ttest_df.iloc[5:]

        ltest_df1 = ltest_df.iloc[0:5]
        ltest_df2 = ltest_df.iloc[5:]

        plt.rcParams.update({'font.size': f14})

        gs = gridspec.GridSpec(190, 32)

        fig, ax = plt.subplots(figsize=[14, 10])

        top_start, top_end = 0, 84
        center_left, center_right = 10, 22

        ax11 = plt.subplot(gs[top_start:top_end, 0:center_left])
        ax12 = plt.subplot(gs[top_end + 4:, 0:center_left])
        ax21 = plt.subplot(gs[top_start:top_end, center_left + 1:center_right - 1])
        ax22 = plt.subplot(gs[top_end + 4:-51, center_left + 1:center_right - 1])
        ax31 = plt.subplot(gs[top_start:top_end, center_right:])
        ax32 = plt.subplot(gs[top_end + 4:-51, center_right:])

        ax1_title_head, plot_name_ro = get_removal_num(remove_outlier_choice, remove_quantile,
                                                       diff_removal_num, percent_thres)

        sns.heatmap(pc_df11, linewidths=.5, cmap='YlOrRd', vmin=0, vmax=100, ax=ax11, annot=True, cbar=False)
        ax11.yaxis.set_tick_params(rotation=0)
        ax11.set_ylabel('WS-TI bin')
        ax11.set_xlabel('')
        ax11.set(xticklabels=[])
        ax11.tick_params(bottom=False)

        sns.heatmap(pc_df12, linewidths=.5, cmap='YlOrRd', vmin=0, vmax=100, ax=ax12, annot=True, cbar=False)
        ax12.yaxis.set_tick_params(rotation=0)
        ax12.set_ylabel('Inner-Outer Range')
        ax12.set_xlabel('Correction method')
        ax12.xaxis.set_tick_params(rotation=30)

        sm = plt.cm.ScalarMappable(cmap='YlOrRd')
        sm.set_array([0, 100])
        cb3 = fig.colorbar(sm, ax=ax12, orientation='horizontal', pad=0.35)
        cb3.set_label('Data sets with improvement (%)')

        sns.heatmap(diff_ttest_df1, linewidths=.5, cmap='Greys', cbar=False, vmin=0, vmax=2, ax=ax21)
        ax21.yaxis.set_tick_params(rotation=0)
        ax21.set_xlabel('')
        ax21.set(xticklabels=[])
        ax21.tick_params(bottom=False, left=False)
        ax21.set(yticklabels=[])
        # ax21.tick_params(left=False)

        sns.heatmap(diff_ttest_df2, linewidths=.5, cmap='Greys', cbar=False, vmin=0, vmax=2, ax=ax22)
        ax22.yaxis.set_tick_params(rotation=0)
        ax22.xaxis.set_tick_params(rotation=30)
        ax22.tick_params(left=False)
        ax22.set(yticklabels=[])
        ax22.set_xlabel('Correction method')

        ax2_c = sns.color_palette(palette='Greys').as_hex()
        ax22.text(0.5, -0.9, 'Method improves \nfrom Baseline', color=ax2_c[3], fontsize=f14, ha='center',
                  weight='semibold', transform=ax22.transAxes)
        ax22.text(0.5, -1.0, 'Method improves significantly', color=ax2_c[-1], fontsize=f14, ha='center',
                  weight='semibold', transform=ax22.transAxes)

        sns.heatmap(ltest_df1, linewidths=.5, cmap='Purples', cbar=False, vmin=0, vmax=2, ax=ax31)
        ax31.yaxis.set_tick_params(rotation=0)
        ax31.yaxis.tick_right()
        ax31.set_xlabel('')
        ax31.set(xticklabels=[])
        ax31.tick_params(bottom=False)

        sns.heatmap(ltest_df2, linewidths=.5, cmap='Purples', cbar=False, vmin=0, vmax=2, ax=ax32)
        ax32.yaxis.set_tick_params(rotation=0)
        ax32.xaxis.set_tick_params(rotation=30)
        ax32.yaxis.tick_right()
        ax32.set_xlabel('Correction method')

        ax3_c = sns.color_palette(palette='Purples').as_hex()
        ax32.text(0.5, -0.85, 'Method variance < Baseline variance', color=ax3_c[3], fontsize=f14, ha='center',
                  weight='semibold', transform=ax32.transAxes)
        ax32.text(0.5, -0.95, 'Method improves significantly', color=ax3_c[-1], fontsize=f14, ha='center',
                  weight='semibold', transform=ax32.transAxes)

        f8_xp, f8_yp = 0., -0.62

        ax12.text(f8_xp, f8_yp, '(a)', color='k', fontsize=f14, transform=ax12.transAxes)
        ax22.text(f8_xp, f8_yp, '(b)', color='k', fontsize=f14, transform=ax22.transAxes)
        ax32.text(f8_xp, f8_yp, '(c)', color='k', fontsize=f14, transform=ax32.transAxes)

        finish_plot('results', 'wsti_pct_ttest_ltest_'+plot_name_ro, 'heatmap', tight_layout=False)


def plot_ecat_pct_ttest_ltest_heatmap(error_cat=None, remove_outlier_choice=False, remove_quantile=False,
                                      bonferroni=None, percent_thres=None):
    """Plot 3 heatmaps of statistical results for each error category.
    Similar to `plot_wsti_pct_ttest_ltest_heatmap`
    """

    plot_choice, pc_df, diff_ttest_df, ltest_df, \
    diff_removal_num = psd.perform_stat_test(error_cat=error_cat,
                                             remove_outlier_choice=remove_outlier_choice,
                                             remove_quantile=remove_quantile, bonferroni=bonferroni,
                                             percent_thres=percent_thres)

    if plot_choice is True:

        pc_df.rename(columns=pc.method_dict, inplace=True)
        diff_ttest_df.rename(columns=pc.method_dict, inplace=True)
        ltest_df.rename(columns=pc.method_dict, inplace=True)

        plt.rcParams.update({'font.size': f14})

        gs = gridspec.GridSpec(100, 32)

        fig, ax = plt.subplots(figsize=[14, 11])

        top_start, top_end = 0, 100
        center_left, center_right = 10, 22

        ax1 = plt.subplot(gs[:top_end, 0:center_left])
        ax2 = plt.subplot(gs[:top_end - 31, center_left + 1:center_right - 1])
        ax3 = plt.subplot(gs[:top_end - 31, center_right:])

        ax1_title_head, plot_name_ro = get_removal_num(remove_outlier_choice, remove_quantile,
                                                       diff_removal_num, percent_thres)

        sns.heatmap(pc_df, linewidths=.5, cmap='YlOrRd', vmin=0, vmax=100, ax=ax1, annot=True, cbar=False)
        ax1.yaxis.set_tick_params(rotation=0)

        if error_cat == 'by_ws_bin_outer':
            ax1.set_ylabel('Normalized Outer Range wind-speed bin')
        else:
            ax1.set_ylabel(error_cat)

        ax1.set_xlabel('Correction method')
        ax1.xaxis.set_tick_params(rotation=30)

        sm = plt.cm.ScalarMappable(cmap='YlOrRd')
        sm.set_array([0, 100])
        cb3 = fig.colorbar(sm, ax=ax1, orientation='horizontal', pad=0.16)
        cb3.set_label('Data sets with improvement (%)')

        sns.heatmap(diff_ttest_df, linewidths=.5, cmap='Greys', cbar=False, vmin=0, vmax=2, ax=ax2)
        ax2.yaxis.set_tick_params(rotation=0)
        ax2.xaxis.set_tick_params(rotation=30)
        ax2.tick_params(left=False)
        ax2.set(yticklabels=[])
        ax2.set_xlabel('Correction method')

        ax2_c = sns.color_palette(palette='Greys').as_hex()
        ax2.text(0.5, -0.3, 'Method improves \nfrom Baseline', color=ax2_c[3], fontsize=f14, ha='center',
                 weight='semibold', transform=ax2.transAxes)
        ax2.text(0.5, -0.34, 'Method improves significantly', color=ax2_c[-1], fontsize=f14, ha='center',
                 weight='semibold', transform=ax2.transAxes)

        sns.heatmap(ltest_df, linewidths=.5, cmap='Purples', cbar=False, vmin=0, vmax=2, ax=ax3)
        ax3.yaxis.set_tick_params(rotation=0)
        ax3.xaxis.set_tick_params(rotation=30)
        ax3.yaxis.tick_right()
        ax3.set_xlabel('Correction method')

        ax3_c = sns.color_palette(palette='Purples').as_hex()
        ax3.text(0.5, -0.28, 'Method variance < Baseline variance', color=ax3_c[3], fontsize=f14, ha='center',
                 weight='semibold', transform=ax3.transAxes)
        ax3.text(0.5, -0.32, 'Method improves significantly', color=ax3_c[-1], fontsize=f14, ha='center',
                 weight='semibold', transform=ax3.transAxes)

        f8_xp, f8_yp = 0., -0.2

        ax1.text(f8_xp, f8_yp, '(a)', color='k', fontsize=f14, transform=ax1.transAxes)
        ax2.text(f8_xp, f8_yp, '(b)', color='k', fontsize=f14, transform=ax2.transAxes)
        ax3.text(f8_xp, f8_yp, '(c)', color='k', fontsize=f14, transform=ax3.transAxes)

        finish_plot('results', error_cat+'_pct_ttest_ltest_'+plot_name_ro, 'heatmap', tight_layout=False)

        plt.rcParams.update({'font.size': fs})


def loop_ecat_pct_ttest_ltest_heatmap():
    """Mass generate 3-panel heatmaps."""

    for i in pc.error_cat_short[1:]:
        plot_ecat_pct_ttest_ltest_heatmap(error_cat=i)


def plot_ediff_hist_kstest():
    """Mass generate histograms of |NME| difference from Baseline.
    Perform K-S test for each distribution.
    Print error bins of methods that do not statistically differ from Gaussian distribution.
    """

    for idx, (b_or_t, error, error_cat) in enumerate(itertools.product(pc.bt_choice, pc.error_name[1:],
                                                                       pc.error_cat_short)):

        base_df = p_init.error_df['base_' + b_or_t]
        base_df_s = base_df.loc[(base_df['error_cat'] == error_cat) & (base_df['error_name'] == error)]
        u_bin = base_df_s['bin_name'].unique()

        for b_name in u_bin:

            fig, ax = plt.subplots(figsize=(6, 5))
            ks_str = None

            for method_num, method_sheet in enumerate(pc.matrix_sheet_name_short[1:]):

                method_df = p_init.error_df[method_sheet+b_or_t]
                method_df_s = method_df.loc[(method_df['error_cat'] == error_cat)
                                            & (method_df['error_name'] == error)]

                base_array = psd.get_bin_array(base_df_s, b_name)
                method_array = psd.get_bin_array(method_df_s, b_name)

                base_na = psd.drop_array_na(base_array)
                method_na = psd.drop_array_na(method_array)

                if all(base_na) == all(method_na):  # ensure the nan's are at the same indices

                    # need 2 samples to do stat tests
                    if (len(base_array.dropna()) > 1) and (len(method_array.dropna()) > 1):

                        # individual improvement, negative means improved
                        # compare absolute value of NME
                        diff_array = (abs(method_array.dropna()) - abs(base_array.dropna())) * 100

                        sns.distplot(list(diff_array.values), ax=ax, kde=False, label=str(method_sheet)[:-1])
                        ax.set_xlabel('Error difference from Baseline (%)')
                        ax.set_ylabel('Count')

                        if psd.perform_kstest(diff_array) is True:
                            pass
                        else:
                            print(method_sheet+' '+error_cat+' '+b_name+' '+b_or_t+' '+error
                                  +' does NOT differ from Gaussian with statistical significance')

                            if ks_str is None:
                                ks_str = '\n NOT differ from Gasussian: \n'
                            ks_str += method_sheet+' '

                    else:
                        print(method_sheet+' '+error_cat+' '+b_name+' '+b_or_t+' '+error
                              +' has insufficient samples to perform KS test')

            if ks_str is None:
                ks_str = '\n all methods differ from Gaussian'

            ax.set_title(error_cat+' '+b_name+' '+b_or_t+' '+error+': '+ks_str)

            plt.legend()

            finish_plot('results', error_cat+' '+b_name+' '+b_or_t+' '+error, 'hist_kstest')


def plot_filter_outlier_kde():
    """Produce kde plot for NME distributions.
    2 plots, before and after outlier filtering."""

    plt.rcParams.update({'font.size': f14})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    line_fill_c = ['black', 'blue', 'dodgerblue', 'grey']
    str_f6 = ['Den-Turb', 'Den-2DPDM', 'Den-Augturb', 'Den-3DPDM']

    f6_xlim = [-2.5, 1.5]
    f6_ylim = [0, 1.4]

    def print_kstest_result(arr, method_sheet):

        if psd.perform_kstest(arr) is True:
            print(method_sheet + ' differs from Gaussian with statistical significance')
        else:
            print(method_sheet + ' does NOT differ from Gaussian with statistical significance')

    for method_num, method_sheet in enumerate(pc.matrix_sheet_name_short[1:]):

        base_df = p_init.error_df['base_' + 'total_e']
        base_df_s = base_df.loc[(base_df['error_cat'] == 'overall') & (base_df['error_name'] == 'nme')]

        method_df = p_init.error_df[method_sheet + 'total_e']
        method_df_s = method_df.loc[(method_df['error_cat'] == 'overall') & (method_df['error_name'] == 'nme')]

        base_array = psd.get_bin_array(base_df_s, 'ALL')
        method_array = psd.get_bin_array(method_df_s, 'ALL')

        base_na = psd.drop_array_na(base_array)
        method_na = psd.drop_array_na(method_array)

        if (all(base_na) == all(method_na)) and (len(base_array.dropna()) > 1) and (len(method_array.dropna()) > 1):
            # individual improvement, negative means improved
            # compare absolute value of NME
            diff_array = (abs(method_array.dropna()) - abs(base_array.dropna())) * 100

            # sns.distplot(list(diff_array.values), ax=ax1, kde=False, hist=None, fit=stats.norm,
            #              label=str(method_sheet)[:-1])
            sns.kdeplot(list(diff_array.values), ax=ax1, color=line_fill_c[method_num], shade=True,
                        label=str_f6[method_num])

            ax1.set_xlabel('|NME| difference (%)')
            ax1.set_ylabel('Probability density')
            # ax1.set_title(error_cat+' '+bin_name+' '+error)
            ax1.set_xlim(f6_xlim)
            ax1.set_ylim(f6_ylim)
            ax1.axvline(0, ls='--', color='grey')
            ax1.text(0.9, 0.93, '(a)', color='k', fontsize=f14, transform=ax1.transAxes)

            print_kstest_result(diff_array, method_sheet)

            diff_data_no_outlier = psd.remove_quantile_in_array(diff_array)

            # sns.distplot(list(diff_data_no_outlier.values), ax=ax2, kde=False, hist=None, fit=stats.norm,
            #              label=str(method_sheet)[:-1])
            sns.kdeplot(list(diff_data_no_outlier.values), ax=ax2, color=line_fill_c[method_num], shade=True,
                        label=str_f6[method_num], legend=False)

            ax2.set_xlabel('|NME| difference (%)')
            ax2.set_ylabel('')
            # ax2.set_title(error_cat+' '+bin_name+' '+b_or_t+' '+error)
            ax2.set_xlim(f6_xlim)
            ax2.set_ylim(f6_ylim)
            ax2.axvline(0, ls='--', color='grey')
            ax2.text(0.9, 0.93, '(b)', color='k', fontsize=f14, transform=ax2.transAxes)

            print_kstest_result(diff_data_no_outlier, method_sheet)

    ax1.legend()

    finish_plot('results', 'filter_overall_nme', '2kde')

    plt.rcParams.update({'font.size': fs})


def plot_filter_outlier_panel_kde_hist():
    """2 pairs of plots on NME and |NME| difference distributions.
    Demonstrate the effects of outlier filtering on |NME| improvement over Baseline.
    Choose the Den-Augturb method and 2 bins in Outer Range WS as example.
    """

    def get_outer_2bins_nme(method):

        te_df = p_init.error_df[method+'_total_e']
        outws01 = te_df.loc[(te_df['error_cat'] == 'by_ws_bin_outer') & (te_df['bin_name'] == '0.0-0.1')
                            & (te_df['error_name'] == 'nme')]
        outws15 = te_df.loc[(te_df['error_cat'] == 'by_ws_bin_outer') & (te_df['bin_name'] == '1.4-1.5')
                            & (te_df['error_name'] == 'nme')]
        outws01_val = outws01['error_value'].dropna() * 100
        outws15_val = outws15['error_value'].dropna() * 100

        return outws01_val, outws15_val

    dat_outws01_val, dat_outws15_val = get_outer_2bins_nme('den_augturb')
    base_outws01_val, base_outws15_val = get_outer_2bins_nme('base')

    nme_diff_01 = pd.to_numeric((abs(dat_outws01_val) - abs(base_outws01_val)))
    nme_diff_15 = pd.to_numeric((abs(dat_outws15_val) - abs(base_outws15_val)))

    bottom_01 = nme_diff_01.quantile(pc.quantile_cut)  # bottom x%
    # nme_diff_01_cut = nme_diff_01.drop((nme_diff_01[nme_diff_01.values < bottom_01].index))

    bottom_15 = nme_diff_15.quantile(pc.quantile_cut)

    plt.rcParams.update({'font.size': f14})

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 7))

    dat_c, base_c, diff_c = 'dodgerblue', 'darkviolet', 'orange'

    sns.kdeplot(pd.to_numeric(base_outws01_val), ax=ax1, label='Baseline', color=base_c, shade=True)
    sns.kdeplot(pd.to_numeric(dat_outws01_val), ax=ax1, label='Den-Augturb', color=dat_c, shade=True)
    ax1.set_ylabel('Probability density')
    ax1.set_xlabel('NME (%)')

    sns.kdeplot(pd.to_numeric(base_outws15_val), ax=ax2, label='Baseline', legend=False, color=base_c, shade=True)
    sns.kdeplot(pd.to_numeric(dat_outws15_val), ax=ax2, label='Den-Augturb', legend=False, color=dat_c, shade=True)
    ax2.set_ylabel('Probability density')
    ax2.set_xlabel('NME (%)')

    sns.distplot(nme_diff_01, ax=ax3, kde=False, hist_kws={'color': diff_c, 'alpha': 0.8}, bins=10)
    # sns.distplot(nme_diff_01_cut, ax=ax3, kde=False, hist_kws={'color':'k', 'alpha':0.8}, bins=9)
    # ax3.set_xlim(nme_diff_01.min(), nme_diff_01.max())
    # ax3.fill_between(0, 1, where=(<0), interpolate=True, color='#EF9A9A')
    # ax3.set_facecolor('k')
    ax3.axvline(0, ls='--', color='grey')
    ax3.axvline(bottom_01, ls='--', color='red')
    ax3.set_ylabel('Count')
    ax3.set_xlabel('|NME| difference (%)')

    sns.distplot(nme_diff_15, ax=ax4, kde=False, hist_kws={'color': diff_c, 'alpha': 0.8})
    ax4.axvline(0, ls='--', color='grey')
    ax4.axvline(bottom_15, ls='--', color='red')
    ax4.set_ylabel('Count')
    ax4.set_xlabel('|NME| difference (%)')

    fo_xp, fo_yp = 0.04, 0.9

    ax1.text(fo_xp, fo_yp, '(a)', color='k', fontsize=f14, transform=ax1.transAxes)
    ax2.text(fo_xp, fo_yp, '(b)', color='k', fontsize=f14, transform=ax2.transAxes)
    ax3.text(fo_xp, fo_yp, '(c)', color='k', fontsize=f14, transform=ax3.transAxes)
    ax4.text(fo_xp, fo_yp, '(d)', color='k', fontsize=f14, transform=ax4.transAxes)

    ax1.set_title('Normalized wind speed: 0-0.1', fontsize=f14)
    ax2.set_title('Normalized wind speed: 1.4-1.5', fontsize=f14)

    finish_plot('results', 'filter_overall_nme', '2kde')

    plt.rcParams.update({'font.size': fs})


def plot_wsti_outws_boot_ttest_mean(wsti_df, outws_df):
    """Plot 2 panels of heatmaps of t-test on means from bootstrapped samples.
    Error categories: WS-TI, Inner-Outer Range, and Outer-Range WS.
    """

    plt.rcParams.update({'font.size': f14})

    gs = gridspec.GridSpec(190, 28)

    plt.subplots(figsize=[12, 10])

    top_start, top_end = 0, 120
    center_left = 12

    ax11 = plt.subplot(gs[top_start:top_end, 0:center_left])
    ax12 = plt.subplot(gs[top_end + 4:, 0:center_left])
    ax2 = plt.subplot(gs[top_start:, center_left + 4:])

    wsti_df1 = wsti_df.iloc[0:5]
    wsti_df2 = wsti_df.iloc[5:]

    sns.heatmap(wsti_df1, linewidths=.5, cmap='Greys', vmin=0, vmax=2, ax=ax11, annot=False, cbar=False,
                fmt=fmt_code)
    ax11.yaxis.set_tick_params(rotation=0)
    ax11.set_ylabel('WS-TI bin')
    ax11.set_xlabel('')
    ax11.set(xticklabels=[])
    ax11.tick_params(bottom=False)

    sns.heatmap(wsti_df2, linewidths=.5, cmap='Greys', vmin=0, vmax=2, ax=ax12, annot=False, cbar=False,
                fmt=fmt_code)
    ax12.yaxis.set_tick_params(rotation=0)
    # ax12.set_title('Percentage of individual improvement')
    ax12.set_ylabel('Inner-Outer Range')
    ax12.set_xlabel('Correction method')
    ax12.xaxis.set_tick_params(rotation=30)

    sns.heatmap(outws_df, linewidths=.5, cmap='Greys', vmin=0, vmax=2, ax=ax2, annot=False, cbar=False,
                fmt=fmt_code, cbar_kws={'label': 'Fraction of bootstrapped means passing t-test (%)'})
    ax2.yaxis.set_tick_params(rotation=0)
    ax2.set_ylabel('Normalized Outer-Range wind-speed bin')
    ax2.set_xlabel('Correction method')
    # ax2.set(yticklabels=[])
    ax2.xaxis.set_tick_params(rotation=30)

    f9_xp, f9_yp = 0., -0.14

    ax12.text(f9_xp, -0.415, '(a)', color='k', fontsize=f14, transform=ax12.transAxes)
    ax2.text(f9_xp, -0.143, '(b)', color='k', fontsize=f14, transform=ax2.transAxes)

    ax2_c = sns.color_palette(palette='Greys').as_hex()
    ax12.text(0.5, -0.57, 'Method improves from Baseline on average', color=ax2_c[3], fontsize=f14, ha='center',
              weight='semibold', transform=ax12.transAxes)
    ax2.text(0.5, -0.2, 'Method improves significantly on average', color=ax2_c[-1], fontsize=f14, ha='center',
             weight='semibold', transform=ax2.transAxes)

    finish_plot('results', 'bootstrap_ttest_wsti_outws', 'heatmap', tight_layout=False)

    plt.rcParams.update({'font.size': fs})


def plot_wsti_outws_bootstrap_ttest_heatmap(remove_outlier=None, wilcoxon=None, hypo_test=None):
    """Prepare for plotting bootstrapped t-test results, and implement plotting."""

    dum1, dum2, wsti_nme_df = psd.get_wsti_nme_stat()
    outws_nme_df = psd.get_methods_nme('by_ws_bin_outer')

    wsti_df = psd.do_ttest_boot(wsti_nme_df,
                                psd.cal_bootstrap_means(wsti_nme_df, remove_outlier=remove_outlier,
                                                        hypo_test=hypo_test),
                                wsti=True, wilcoxon=wilcoxon, hypo_test=hypo_test)

    outws_df = psd.do_ttest_boot(outws_nme_df,
                                 psd.cal_bootstrap_means(outws_nme_df, remove_outlier=remove_outlier,
                                                         hypo_test=hypo_test),
                                 wilcoxon=wilcoxon, hypo_test=hypo_test)

    plot_wsti_outws_boot_ttest_mean(wsti_df, outws_df)


def plot_removed_outlier_wsti_outws_bootstrap_ttest_heatmap():
    """Prepare for plotting bootstrapped t-test results, and implement plotting."""

    dum1, dum2, wsti_nme_df = psd.get_wsti_nme_stat()
    outws_nme_df = psd.get_methods_nme('by_ws_bin_outer')

    wsti_mean_df = psd.cal_bootstrap_means(wsti_nme_df, remove_outlier=True)
    outws_mean_df = psd.cal_bootstrap_means(outws_nme_df, remove_outlier=True)

    wsti_do_boot = psd.do_ttest_boot(wsti_nme_df, wsti_mean_df, wsti=True)
    outws_do_boot = psd.do_ttest_boot(outws_nme_df, outws_mean_df)

    plot_wsti_outws_boot_ttest_mean(wsti_do_boot, outws_do_boot)


def plot_ecat_boot_ttest_mean(df, error_cat):
    """Plot heatmap of t-test on means from bootstrapped samples for any error category."""

    fig, ax = plt.subplots(figsize=(6, 7))

    sns.heatmap(df, linewidths=.5, cmap='Greys', vmin=0, vmax=2, ax=ax, annot=False, cbar=False,
                fmt=fmt_code, cbar_kws={'label': 'Fraction of bootstrapped means passing t-test (%)'})
    ax.yaxis.set_tick_params(rotation=0)
    ax.set_ylabel(error_cat)
    ax.set_xlabel('Correction method')
    ax.xaxis.set_tick_params(rotation=30)

    ax_c = sns.color_palette(palette='Greys').as_hex()
    ax.text(0.5, -0.25, 'Method improves from Baseline on average', color=ax_c[3], fontsize=fs, ha='center',
            weight='semibold', transform=ax.transAxes)
    ax.text(0.5, -0.3, 'Method improves significantly on average', color=ax_c[-1], fontsize=fs, ha='center',
            weight='semibold', transform=ax.transAxes)

    finish_plot('results', 'bootstrap_ttest_'+error_cat, 'heatmap', tight_layout=False)


def plot_ecat_bootstrap_ttest_heatmap(error_cat, remove_outlier=None, wilcoxon=None, hypo_test=None):
    """Prepare for plotting bootstrapped results."""

    nme_df = psd.get_methods_nme(error_cat)

    plot_df = psd.do_ttest_boot(nme_df,
                                psd.cal_bootstrap_means(nme_df, remove_outlier=remove_outlier, hypo_test=hypo_test),
                                wilcoxon=wilcoxon, hypo_test=hypo_test)

    plot_ecat_boot_ttest_mean(plot_df, error_cat)


def loop_ecat_bootstrap_ttest_heatmap(remove_outlier=None):
    """Mass generate 1-panel t-test heatmap."""

    for i in pc.error_cat_short[1:]:
        plot_ecat_bootstrap_ttest_heatmap(error_cat=i, remove_outlier=remove_outlier)


def plot_1in2_ltest_heatmap(df, error_cat, ax, cmap_in, sub_t, p_title, cut_color=False):
    """Plot 1 heatmap of Levene's test results."""

    if cut_color is True:

        cmap_all = plt.get_cmap(cmap_in)
        colors_half = cmap_all(np.linspace(0, 0.5, cmap_all.N // 2))
        cmap = LinearSegmentedColormap.from_list('Upper Half', colors_half)

    else:

        cmap = cmap_in

    sns.heatmap(df, linewidths=.5, cmap=cmap, vmin=0, vmax=100, ax=ax)
    ax.set_title(p_title, pad=10)
    ax.yaxis.set_tick_params(rotation=0)
    ax.set_xlabel('correction methods')
    ax.set_ylabel(error_cat)
    ax.text(-0.25, -0.3, sub_t, color='k', fontsize=12, transform=ax.transAxes)


def plot_ecat_bootstrap_ltest_heatmap(error_cat):
    """Prepare for plotting bootstrapped Levene's test results, and implement plotting."""

    count_var_df, count_ltest_df = psd.cal_bootstrap_ltest_pct(psd.get_methods_nme(error_cat))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    plot_1in2_ltest_heatmap(count_var_df, error_cat, ax1, 'Purples', '(a)',
                            'Percentage of reduced variance in \nbootstrapped samples from Baseline', cut_color=True)
    plot_1in2_ltest_heatmap(count_ltest_df, error_cat, ax2, 'Purples', '(b)',
                            'Percentage of significantly reduced variance in \nbootstrapped samples from Baseline')

    finish_plot('results', 'bootstrap_ltest_'+error_cat, 'heatmap')


def loop_ecat_bootstrap_ltest_heatmap():
    """Mass generate 2-panel Levene's test heatmaps."""

    for i in pc.error_cat_short[1:]:
        plot_ecat_bootstrap_ltest_heatmap(error_cat=i)
