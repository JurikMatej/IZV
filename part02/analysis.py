#!/usr/bin/env python3.11
# coding=utf-8
from io import BytesIO

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile

# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz

# Ukol 1: nacteni dat ze ZIP souboru
def load_data(filename: str) -> pd.DataFrame:
    # tyto konstanty nemente, pomuzou vam pri nacitani
    headers = ["p1", "p36", "p37", "p2a", "weekday(p2a)", "p2b", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13a",
               "p13b", "p13c", "p14", "p15", "p16", "p17", "p18", "p19", "p20", "p21", "p22", "p23", "p24", "p27", "p28",
               "p34", "p35", "p39", "p44", "p45a", "p47", "p48a", "p49", "p50a", "p50b", "p51", "p52", "p53", "p55a",
               "p57", "p58", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", "o", "p", "q", "r", "s", "t", "p5a"]

    # def get_dataframe(filename: str, verbose: bool = False) -> pd.DataFrame:
    regions = {
        "PHA": "00",
        "STC": "01",
        "JHC": "02",
        "PLK": "03",
        "ULK": "04",
        "HKK": "05",
        "JHM": "06",
        "MSK": "07",
        "OLK": "14",
        "ZLK": "15",
        "VYS": "16",
        "PAK": "17",
        "LBK": "18",
        "KVK": "19",
    }

    # TODO skip regions not used
    # TODO skip chodci.csv

    df = pd.DataFrame()
    with zipfile.ZipFile("data.zip", "r") as data_archive:

        for nested_archive_name in data_archive.namelist():
            nested_archive_raw = BytesIO(data_archive.read(nested_archive_name))
            with zipfile.ZipFile(nested_archive_raw) as csv_archive:

                for csvfile_zipped_name in csv_archive.namelist():
                    # read one csv from one inner zip
                    csvfile_raw = BytesIO(csv_archive.read(csvfile_zipped_name))
                    current_region_code = csvfile_zipped_name.split('.')[
                        0]  # File name without the extension (remove '.csv')

                    for region_abbrev, region_code in regions.items():
                        if current_region_code == region_code:
                            df_to_append = pd.read_csv(csvfile_raw, encoding="cp1250", sep=";",
                                                       names=headers, low_memory=False)

                            # Add a region column to the DataFrame
                            df_to_append['region'] = region_abbrev

                            df = pd.concat([df, df_to_append], ignore_index=True)

    return df


# Ukol 2: zpracovani dat
def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    df2 = df.copy()

    if verbose:
        size = np.sum(df2.memory_usage(index=False, deep=True))
        print("orig_size={:.1f} MB".format(size / 1_000_000))

    # Store 'p2a' as new 'date' column
    df2['date'] = pd.to_datetime(df2['p2a'])

    # Define column groups based on their desired datatype
    all_cols = set(df2)

    processed_cols = set(("region", "p2a"))
    float_cols = set(("a", "b", "d", "e", "f", "g"))
    not_category_cols = processed_cols.union(float_cols)

    category_cols = all_cols.difference(not_category_cols)

    # Set types of the columns as needed
    for col in category_cols.difference(processed_cols):
        if col in category_cols:
            df[col] = df2[col].astype('category')  # Cast to category

        elif col in float_cols:
            df2[col] = [str(row).replace(",", ".") for row in df2[col]]
            df2[col] = pd.to_numeric(df2[col], errors='coerce')  # Cast to float

    # Drop any duplicates in 'p1' column
    df2 = df2.drop_duplicates(subset=['p1'])

    if verbose:
        size = np.sum(df2.memory_usage(index=False, deep=True))
        print("new_size={:.1f} MB".format(size / 1_000_000))

    return df2


# Ukol 3: počty nehod oidke stavu řidiče
def plot_state(df: pd.DataFrame, fig_location: str = None,
               show_figure: bool = False):
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

    df2 = df.copy()

    df2 = df2.dropna(subset=['p57'])
    df2['p57'] = df2['p57'].astype("int")
    df2 = df2[(df2['p57'] > 3)]
    df2['region_p57_count'] = df2['p57']
    df2['region_p57_max'] = df2['p57']

    data_to_plot = df2.copy()
    data_to_plot = data_to_plot.replace({'p57': driver_states})
    data_to_plot = data_to_plot.groupby(['region', 'p57']).agg({'region_p57_count': 'count'}).reset_index()

    ## TODO HOW TO GET Y MAX... Rewrite so each AX has a unique max_y
    # for region in data_to_plot['region'].unique():
    # data_to_plot[data_to_plot['region'] == 'HKK']['region_p57_count'].max()

    # Generate graphs
    # sns.set_style("darkgrid")

    graphs = sns.catplot(data=data_to_plot,
                         x='region',
                         y='region_p57_count',
                         col='p57',
                         col_wrap=2,
                         kind="bar",
                         height=3,
                         aspect=1.6)

    graphs.set_xlabels("Kraj")
    graphs.set_ylabels("Počet nehod")
    graphs.set_titles("{col_name}")
    plt.suptitle("Počet nehod dle stavu řidiče při nedobrém stavu", y=1.03)

    # iterate through axes
    for ax in graphs.axes.ravel():
        # add annotations
        for c in ax.containers:
            labels = [int((v.get_height())) for v in c]
            ax.bar_label(c, labels=labels, label_type='edge')

        ax.margins(y=0.2)

    # TODO fig_location & show_figure

    plt.show()
    plt.close()



# Ukol4: alkohol v jednotlivých hodinách
def plot_alcohol(df: pd.DataFrame, fig_location: str = None,
                 show_figure: bool = False):
    regions_to_plot = ['HKK', 'JHC', 'JHM', 'KVK']

    data_to_plot = df.copy()
    data_to_plot = data_to_plot[data_to_plot['region'].isin(regions_to_plot)]

    # Add info about alcohol presence
    data_to_plot['p11'] = data_to_plot['p11'].astype(int)
    data_to_plot.loc[data_to_plot['p11'] < 3, 'alcohol'] = "Ne"
    data_to_plot.loc[data_to_plot['p11'] >= 3, 'alcohol'] = "Ano"

    data_to_plot['p2b'] = [x[:2] if x[:2] not in ['24', '25'] else None for x in data_to_plot['p2b']]
    data_to_plot['p2b'] = data_to_plot['p2b'].dropna()  # .astype(int)

    data_to_plot = data_to_plot.groupby(['region', 'p2b', 'alcohol']).agg({'p11': 'count'}).reset_index()
    # data_to_plot

    # data_to_plot = data_to_plot.groupby(['region', 'p7', data_to_plot.date.dt.month]).agg({'p1': 'count'}).reset_index()
    # data_to_plot = data_to_plot.replace({'p7': crash_type_titles})
    # p7cat = data_to_plot['p7'].cat.remove_categories([0, 3])
    # data_to_plot['p7'] = p7cat
    #
    # TODO Set better ylim
    graphs = sns.catplot(data=data_to_plot,
                         x='p2b',
                         y='p11',
                         col='region',
                         col_wrap=2,
                         kind="bar",
                         legend=True,
                         legend_out=True,
                         sharex=False,
                         sharey=False,
                         hue='alcohol',
                         height=3,
                         aspect=1.6,
                         )

    graphs.fig.subplots_adjust(hspace=0.3, wspace=0.2)
    graphs.set_xlabels("Hodina")
    graphs.set_titles("Kraj: {col_name}")
    graphs.legend.set(title="Alkohol")
    graphs.set(ylabel="Počet nehod")

    # TODO fig_location & show_figure

    plt.show()
    plt.close()


# Ukol 5: Zavinění nehody v čase
def plot_fault(df: pd.DataFrame, fig_location: str = None,
               show_figure: bool = False):
    # regions_to_plot = ['HKK', 'JHC', 'JHM', 'KVK']
    regions_to_plot = ['JHM', 'MSK', 'OLK', 'ZLK']

    data_to_plot = df.copy()
    data_to_plot = data_to_plot[data_to_plot['region'].isin(regions_to_plot)]

    # Limit data to a date range
    date_from = np.datetime64("2016-01-01")
    date_to = np.datetime64("2023-01-01")  # But the max date is 2022-12-31 either way
    data_to_plot = data_to_plot[(date_from <= data_to_plot["date"]) & (data_to_plot["date"] <= date_to)]

    data_to_plot['p10'] = data_to_plot['p10'].astype(int)
    data_to_plot = data_to_plot[(data_to_plot['p10'] > 0) & (data_to_plot['p10'] < 5)]
    # data_to_plot

    data_to_plot['caused_by_motorized'] = data_to_plot[data_to_plot['p10'] == 1]['p10']
    data_to_plot['caused_by_non_motorized'] = data_to_plot[data_to_plot['p10'] == 2]['p10']
    data_to_plot['caused_by_walker'] = data_to_plot[data_to_plot['p10'] == 3]['p10']
    data_to_plot['caused_by_animal'] = data_to_plot[data_to_plot['p10'] == 4]['p10']

    # data_to_plot.loc[data_to_plot['p10'] == 1, "caused_by"] = "Řidičem motorového vozidla"
    # data_to_plot.loc[data_to_plot['p10'] == 2, "caused_by"] = "Řidičem nemotorového vozidla"
    # data_to_plot.loc[data_to_plot['p10'] == 3, "caused_by"] = "Chodcem"
    # data_to_plot.loc[data_to_plot['p10'] == 4, "caused_by"] = "Zvířetem"

    # data_to_plot = data_to_plot.groupby(['region', 'date']).agg({'caused_by': 'count'}).reset_index()
    data_to_plot = (data_to_plot.groupby(['region', 'date']).agg(
        {'caused_by_motorized': 'sum', 'caused_by_non_motorized': 'sum', 'caused_by_walker': 'sum',
         'caused_by_animal': 'sum', 'p10': 'sum'}).reset_index())
    # data_to_plot

    data_to_plot = pd.pivot_table(data_to_plot,
                                  values=["caused_by_motorized", "caused_by_non_motorized", "caused_by_walker",
                                          "caused_by_animal"],
                                  index=["region", "date"],
                                  aggfunc="sum")
    # data_to_plot

    data_to_plot = data_to_plot.groupby([data_to_plot.index.get_level_values('region'),
                                         data_to_plot.index.get_level_values('date').year,
                                         data_to_plot.index.get_level_values('date').month]).sum()
    # data_to_plot
    data_to_plot = data_to_plot.stack().to_frame()
    data_to_plot.columns = ["Počet nehod"]
    # new_array = np.char.add([x for x in data_to_plot.index.get_level_values(2).astype(str)],
    #                         ['/'])
    # new_array = np.char.add(new_array, [x[2:] for x in data_to_plot.index.get_level_values(1).astype(str)])
    # new_array

    # data_to_plot['datecol'] = [x[2:] for x in data_to_plot.index.get_level_values(1).astype(str)]
    # data_to_plot['datecol'] = new_array
    data_to_plot['datecol'] = pd.to_datetime(data_to_plot.index.get_level_values(1).astype(str) + "/"
                                             + data_to_plot.index.get_level_values(2).astype(str) + "/01")

    data_to_plot.index.names = ['region', 'year', 'month', None]

    # data_to_plot['datecol']

    # sns.set_style("darkgrid")
    graphs = sns.relplot(data=data_to_plot,
                         x="datecol",
                         y="Počet nehod",
                         kind="line",
                         col=data_to_plot.index.get_level_values('region'),
                         col_wrap=2,
                         height=4,
                         aspect=1.6,
                         facet_kws={"sharex": False},
                         hue=data_to_plot.index.get_level_values(3)
                         )

    graphs.set_ylabels("Počet nehod")
    graphs.set_xlabels("")
    graphs.set_titles("Kraj: {col_name}")
    graphs.legend.set(title="Zavinění")

    # TODO savefig and show_fig

    plt.show()
    plt.close()


if __name__ == "__main__":
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni
    # funkce.
    df = load_data("data/data.zip")
    df2 = parse_data(df, True)

    plot_state(df2, "01_state.png")
    plot_alcohol(df2, "02_alcohol.png", True)
    plot_fault(df2, "03_fault.png")


# Poznamka:
# pro to, abyste se vyhnuli castemu nacitani muzete vyuzit napr
# VS Code a oznaceni jako bunky (radek #%%% )
# Pak muzete data jednou nacist a dale ladit jednotlive funkce
# Pripadne si muzete vysledny dataframe ulozit nekam na disk (pro ladici
# ucely) a nacitat jej naparsovany z disku
