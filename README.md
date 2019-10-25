# PCWG-Share-3
Jupyter notebook and Python code for post processing of PCWG-Share-03

# Requirements
Python 3; specific libraries are listed in `requirements.txt`

# Input files
* Excel files from the PCWG analysis tool (to be converted to `pkl` 
files in the code)
* pdm_example.xls: input file for plotting a power deviation matrix as
an example

# Components
* `demo` notebook: generates plots in the manuscript, and  
additional plots for reference
* `pcwg03_config`: lists strings/constants used in other code that 
are not supposed to change frequently
    * `data_file_path` - requires user input, the file path 
    containing the Excel files generated from the PCWG analysis tool
    * `save_fig` - requires user input, determines whether the code 
    outputs figures as png/pdf files or not
* `pcwg03_convert_df`: converts data in Excel file into Pandas 
data frames
* `pcwg03_energy_fraction`: calculates energy fractions from data 
counts and error values
* `pcwg03_initialize`: sets up the analysis environment
* `pcwg03_plot`: contains all the plotting functions
* `pcwg03_read_data`: reads in Excel data
* `pcwg03_slice_df`: contains all the data analysis and statistics 
functions

# How To Use This Code
Follow the `demo` Jupyter notebook

# Project Contributors
* Joseph C. Y. Lee, NREL
* Peter Stuart, RES
* Andrew Clifton, Stuttgart Wind Energy
* M. Jason Fields, NREL
* Jordan Perr-Sauer, NREL
* Lindy Williams, NREL
* Lee Cameron, RES
* Taylor Geer, DNV GL
* Paul Housley, SSE plc