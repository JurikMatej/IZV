#!/usr/bin/env python3.11
# coding=utf-8

"""
IZV project part 2
Author: xjurik12
Python version: 3.10
"""

from io import BytesIO
import zipfile

from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd
import seaborn as sns
import numpy as np


def load_data(filename: str) -> pd.DataFrame:
    """
    Load data from a zip file into a pandas dataframe
    The zip file must have a predetermined structure as per the ASSIGNMENT
    of this project - nested zip archives with csv files with specific names encoded in cp1250

    :param filename: Filename of the data to be loaded
    :return: Pandas Dataframe containing the loaded csv data and an added column named 'region'
    with a specific region's name abbreviation
    """
    headers = ["p1", "p36", "p37", "p2a", "weekday(p2a)", "p2b", "p6",
               "p7", "p8", "p9", "p10", "p11", "p12", "p13a", "p13b",
               "p13c", "p14", "p15", "p16", "p17", "p18", "p19", "p20",
               "p21", "p22", "p23", "p24", "p27", "p28", "p34", "p35",
               "p39", "p44", "p45a", "p47", "p48a", "p49", "p50a", "p50b",
               "p51", "p52", "p53", "p55a", "p57", "p58", "a", "b", "d",
               "e", "f", "g", "h", "i", "j", "k", "l", "n", "o", "p", "q",
               "r", "s", "t", "p5a"]

    regions = {
        "00": "PHA",
        "01": "STC",
        "02": "JHC",
        "03": "PLK",
        "04": "ULK",
        "05": "HKK",
        "06": "JHM",
        "07": "MSK",
        "14": "OLK",
        "15": "ZLK",
        "16": "VYS",
        "17": "PAK",
        "18": "LBK",
        "19": "KVK",
    }

    df = pd.DataFrame()

    # As per the assignment, datasource is a zipped archive of nested zipped archives
    # containing csv files by which the data source will be populated
    with zipfile.ZipFile(filename, "r") as data_archive:
        for nested_archive_name in data_archive.namelist():
            nested_archive_rawcontent = BytesIO(data_archive.read(nested_archive_name))

            with zipfile.ZipFile(nested_archive_rawcontent) as csv_archive:
                for csvfile_zipped_name in csv_archive.namelist():

                    # Check whether the filename equals one of the region codes

                    # File name without the extension (remove '.csv')
                    current_region_code = csvfile_zipped_name.split('.')[0]
                    if current_region_code in regions:
                        csvfile_rawcontent = BytesIO(csv_archive.read(csvfile_zipped_name))

                        # Create a dataframe with the csv file contents
                        df_to_append = pd.read_csv(csvfile_rawcontent,
                                                   dtype=np.string_,
                                                   encoding="cp1250",
                                                   sep=";",
                                                   names=headers,
                                                   low_memory=False)
                        # Add the region abbreviated name to the dataframe
                        df_to_append['region'] = regions[current_region_code]

                        # Add resulting dataframe to the result
                        df = pd.concat([df, df_to_append], ignore_index=True)

    return df


def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Take the loaded dataframe and type cast its columns into desired types.
    Parse necessary dates to pandas datetime format.
    Also remove duplicate entries marked by the p1 'Identification Numbers'.

    :param df: Loaded dataframe
    :param verbose: Flag to make the function also report the original dataframe size,
                    and it's size after parsing

    :return: Parsed data frame
    """
    df_to_format = df.copy()

    if verbose:
        size = np.sum(df_to_format.memory_usage(index=False, deep=True))
        print(f"orig_size={(size / 1_000_000):.1f} MB")

    # Store 'p2a' as a new datetime-typed column 'date'
    df_to_format["p2a"] = pd.to_datetime(df_to_format["p2a"])
    df_to_format.rename(columns={"p2a": "date"}, inplace=True)

    # Define column groups based on their desired datatype
    all_cols = set(df_to_format)
    processed_cols = {"region", "date"}
    float_cols = {"a", "b", "d", "e", "f", "g"}
    not_category_cols = processed_cols.union(float_cols)

    category_cols = all_cols.difference(not_category_cols)

    # Set types of the columns as needed
    for col in category_cols.difference(processed_cols):
        if col in category_cols:
            df_to_format[col] = df_to_format[col].astype('category')  # Cast to category

        elif col in float_cols:
            df_to_format[col] = [str(row).replace(",", ".") for row in df_to_format[col]]
            df_to_format[col] = pd.to_numeric(df_to_format[col], errors='coerce')  # Cast to float

        else:
            pass  # Already processed

    # Drop entries with duplicate identification numbers
    df_to_format = df_to_format.drop_duplicates(subset=['p1'])

    if verbose:
        size = np.sum(df_to_format.memory_usage(index=False, deep=True))
        print(f"new_size={(size / 1_000_000):.1f} MB")

    return df_to_format


def plot_state(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    """
    Make a plot visualizing the state of the driver who caused an accident.
    Each subplot represents a count of drivers for a specific driver state counted across
    different regions

    :param df: Parsed dataframe
    :param fig_location: Path where to save the resulting figure.
                         If set, save figure, else do not save
    :param show_figure: Show the figure if true, else do not show
    """

    # Define an enumeration of the driver state indexes mapped to their textual description
    # For a full overview, all the possible driver states are defined here,
    # but some are commented out because only states with index 4 and above are being analysed
    # as per the ASSIGNMENT
    driver_states = {
        # 1: 'dobrý',
        # 2: 'unaven, usnul, náhlá fyzická indispozice',
        # 3: 'pod vlivem léků, narkotik',
        4: 'pod vlivem alkoholu do 0,99 ‰',
        5: 'pod vlivem alkoholu 1 ‰ a více',
        6: 'nemoc, úraz apod.',
        7: 'invalida',
        8: 'řidič při jízdě zemřel',
        9: 'sebevražda'
    }

    # Make a copy of the passed dataset to work on
    data = df.copy()

    # p57 data col = driver state
    # Filter p57 to only contain relevant states
    data = data.dropna(subset=['p57'])
    data['p57'] = data['p57'].astype(int)
    data = data[(data['p57'] > 3)]

    # Store a count of the driver states before grouping the data
    data['region_p57_count'] = data['p57']

    data = data.replace({'p57': driver_states})
    data = data.groupby(['region', 'p57']).agg({'region_p57_count': 'count'}).reset_index()

    # Make a barplot for the resulting dataset
    sns.set_style("whitegrid")
    graphs = sns.catplot(data=data,
                         x='region',
                         y='region_p57_count',
                         col='p57',
                         col_wrap=2,
                         hue='region',
                         kind='bar',
                         height=3,
                         aspect=1.8,
                         sharey=False,
                         legend=False)

    # Modify specific parameters of the graph
    graphs.set_xlabels("Kraj")
    graphs.set_ylabels("Počet nehod")
    graphs.set_titles("Stav řidiče: {col_name}")
    plt.suptitle("Počet nehod dle stavu řidiče při nedobrém stavu", y=1.02)
    for ax in graphs.axes.ravel():
        ax.margins(y=0.1)  # Add y margin to each ax

    if show_figure:
        plt.show()

    if fig_location:
        graphs.savefig(fig_location)

    plt.close()


# Ukol4: alkohol v jednotlivých hodinách
def plot_alcohol(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    """
    Make a plot visualizing how many accidents are caused by drivers who have drunk alcohol,
    and how many are caused by drivers who have not drunk any alcohol

    Data is limited to 4 arbitrarily selected regions and shows sums of drivers
    (under the influence or sober) who caused an accident during a day (0-23 hours)

    :param df: Parsed dataframe
    :param fig_location: Path where to save the resulting figure.
                         If set, save figure, else do not save
    :param show_figure: Show the figure if true, else do not show
    """
    # Make a copy of the passed dataset to work on
    data = df.copy()

    # Select four regions to create the graph for
    selected_regions = ['JHM', 'MSK', 'OLK', 'ZLK']
    data = data[data['region'].isin(selected_regions)]

    # p11 data col = info concerning alcohol use of the person who caused an accident
    # Prepare the p11 data col for further use - cast as 'int' column
    data['p11'] = data['p11'].astype(int)
    # ...and filter out the entries for which alcohol use was not tested (p11 = 0)
    data = data[data['p11'] != 0]

    # Distinguish alcohol use in a new category column 'alcohol'
    # NOTE to the reviewer:
    #   p11=1 means ALCOHOL=YES,
    #   but ACCORDING to the ASSIGNMENT, it is treated as ALCOHOL=NO here
    data.loc[data['p11'] < 3, 'alcohol'] = "Ne"
    data.loc[data['p11'] >= 3, 'alcohol'] = "Ano"

    # Prepare x-axis - time of the day
    data['p2b'] = [x[:2] if x[:2] not in ['24', '25'] else None for x in data['p2b']]
    data['p2b'] = data['p2b'].dropna()

    # Group data by region, time of day and alcohol use and add
    # the number of accidents to each group
    data = data.groupby(['region', 'p2b', 'alcohol']).agg({'p11': 'count'}).reset_index()

    # Make a barplot for the resulting dataset
    sns.set_style("whitegrid")
    graphs = sns.catplot(data=data,
                         x='p2b',
                         y='p11',
                         col='region',
                         col_wrap=2,
                         hue='alcohol',
                         kind="bar",
                         legend=True,
                         legend_out=True,
                         sharex=False,
                         sharey=False,
                         height=3.2,
                         aspect=1.7)

    # Modify specific parameters of the graph
    graphs.set_titles("Kraj: {col_name}")
    graphs.set(xlabel="Hodina", ylabel="Počet nehod")
    graphs.legend.set(title="Alkohol")
    graphs.fig.subplots_adjust(hspace=0.4, wspace=0.2)

    if fig_location:
        graphs.savefig(fig_location)

    if show_figure:
        plt.show()

    plt.close()


# Ukol 5: Zavinění nehody v čase
def plot_fault(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    """
    Makes a plot visualising four of the various causes for an accident.
    Data is limited to 4 arbitrarily picked regions and shows counts of each
    accident cause type for the last 7 years across those regions.

    :param df: Parsed dataframe
    :param fig_location: Path where to save the resulting figure.
                         If set, save figure, else do not save
    :param show_figure: Show the figure if true, else do not show
    """
    # Make a copy of the passed dataset to work on
    data = df.copy()

    # Select four regions to create the graph for
    selected_regions = ['JHM', 'MSK', 'OLK', 'ZLK']
    data = data[data['region'].isin(selected_regions)]

    # Limit data to a date range
    date_from = np.datetime64("2016-01-01")
    date_to = np.datetime64("2023-01-01")
    data = data[(date_from <= data['date']) & (data['date'] <= date_to)]

    # p10 data col = the cause of an accident
    # Select a specific range of accidents (with indexes ranging from 1 to 4)
    data['p10'] = data['p10'].astype(int)
    data = data[(data['p10'] > 0) & (data['p10'] < 5)]

    # Define respective accident causes as separate dataframe columns
    data['Řidičem motorového vozidla'] = data[data['p10'] == 1]['p10']
    data['Řidičem nemotorového vozidla'] = data[data['p10'] == 2]['p10']
    data['Chodcem'] = data[data['p10'] == 3]['p10']
    data['Zvířetem'] = data[data['p10'] == 4]['p10']

    # Pivot relevant data (count occurrences of each caused accident by its causes)
    data = pd.pivot_table(data,
                          index=["region", "date"],
                          values=['Zvířetem',
                                  'Řidičem motorového vozidla',
                                  'Řidičem nemotorového vozidla',
                                  'Chodcem'],
                          aggfunc="count")

    # Resample dates by months and stack the data
    data = data.groupby([pd.Grouper(level='region'),
                         pd.Grouper(level='date', freq="M")]).sum()
    data = data.stack().to_frame()

    # Give the aggregated accident count column a name
    data.columns = ['Počet nehod']

    # Make a lineplot for the resulting dataset
    sns.set_style("whitegrid")
    graphs = sns.relplot(data=data,
                         x="date",
                         y="Počet nehod",
                         col=data.index.get_level_values('region'),
                         col_wrap=2,
                         hue=data.index.get_level_values(2),
                         kind="line",
                         height=3,
                         aspect=1.6,
                         facet_kws={"sharex": False})

    # Modify specific parameters of the graph
    graphs.set_ylabels("Počet nehod")
    graphs.set_xlabels("")
    graphs.set_titles("Kraj: {col_name}")
    graphs.legend.set(title="Zavinění")

    # Per-ax modifications
    for ax in graphs.axes.ravel():
        # Set x-axis major tick formatter
        fmt = DateFormatter('%m/%y')
        ax.xaxis.set_major_formatter(fmt)

    if fig_location:
        graphs.savefig(fig_location)

    if show_figure:
        plt.show()

    plt.close()


if __name__ == "__main__":
    _df = load_data("data/data.zip")
    _df2 = parse_data(_df, True)

    plot_state(_df2, "01_state.png")
    plot_alcohol(_df2, "02_alcohol.png")
    plot_fault(_df2, "03_fault.png")
