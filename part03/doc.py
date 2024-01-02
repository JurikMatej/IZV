#!/usr/bin/python3.10
# coding=utf-8

"""
IZV project part 3
Author: xjurik12
Python version tested on: 3.12
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.core.indexes.frozen import FrozenList


def prepare_data(path_to_input: str) -> pd.DataFrame:
    """
    Read prepared data from the provided path and prepare it for processing

    :param path_to_input: path to prepared data (pickle-dumped dataframe)
    """
    df = pd.read_pickle(path_to_input)

    # Keep only relevant data
    # df = df[['p1', 'p10', 'p11', 'date', 'region', 'd', 'e']]

    return df


def interesting_data_table(df: pd.DataFrame):
    """
    Nehody policajtov a ministerstva vnutra v porovnani s celkovymi nehodami

    TODO docstring
    """

    VEHICLE_OWNERS = {
        11: "Ministerstvo vnútra",
        12: "Polícia ČR",
        13: "Mestská, obecná polícia",
        15: "Ministerstvo obrany",
    }

    data = df.copy()

    # Filter out data not needed or empty
    data = data[['p1', 'p48a', 'date']]
    data = data.dropna(subset=['p48a'])
    data = data[data['p48a'].isin([11, 12, 13, 15])]  # Filter out the police and government vehicles

    data['year'] = data['date'].dt.year

    data['total'] = data['p1']  # Prepare the accident fraction column
    totals_table = data.groupby(['year']).agg({'total': 'count'})

    data['p48a'] = data['p48a'].map(VEHICLE_OWNERS)
    data = data.groupby(['year', 'p48a']).agg({'p1': 'count'}).reset_index()

    data = totals_table.merge(data, on='year', how='outer')

    # Calculate the fraction of total accidents for different vehicle owner caused accidents
    data['%ofTotal'] = data['p1'] / data['total'] * 100
    data['%ofTotal'] = [f"{x:.2f}\\%" for x in data['%ofTotal']]  # NOTE: the \\% escape is necessary for .tex compiler

    # TABLE 1: Overview - Total numbers of per-year accidents caused by the police and government
    table_overview = data.groupby(['year']).agg({'total': 'first'})
    # Rename the columns before exporting them as a table
    table_overview.index.names = FrozenList(['Rok'])
    table_overview.columns = ['Nehody celkom']

    # TABLE 2: Specifics
    #          - Numbers and the fraction of total accidents per specific vehicle owner (police or gov) by year
    table_specific = data.groupby(['year', 'p48a']).agg({'p1': 'first', '%ofTotal': 'first'})
    # Rename the columns before exporting them as a table
    table_specific.index.names = FrozenList(['Rok', 'Majiteľ havarovaného vozidla'])
    table_specific.columns = ['Nehody podľa vozidla', 'Pomer k celku za rok']

    print("Tables used in the final report: ")
    print(table_overview)
    print(table_specific)

    # Generate source latex for the tables to be used in the doc.pdf
    # print(table_overview.to_latex())
    # print(table_specific.to_latex())


def interesting_data_graph(df: pd.DataFrame):
    """
    Do coho policajti radi buraju

    TODO docstring
    """
    ACCIDENT_TYPE = {
        1: "srážka s jedoucím nekolejovým vozidlem",
        2: "srážka s vozidlem zaparkovaným, odstaveným",
        3: "srážka s pevnou překážkou",
        4: "srážka s chodcem",
        5: "srážka s lesní zvěří",
        6: "srážka s domácím zvířetem",
        7: "srážka s vlakem",
        8: "srážka s tramvají",
        9: "havárie",
        0: "jiný druh nehody",
    }

    data = df.copy()
    data = data[['p1', 'p6', 'p48a']]

    # Drop empty data for accident types
    data = data.dropna(subset=['p6'])

    # Map accident type categories to their string descriptions
    data['p6'] = data['p6'].map(ACCIDENT_TYPE)
    data.rename(columns={'p6': 'Typ nehody'}, inplace=True)

    # Filter out only the accidents caused by the police
    data = data[data['p48a'].isin([12, 13])]

    g = sns.displot(data, x="Typ nehody", hue="Typ nehody",
                    height=5, aspect=2)

    # Fine tune the axis
    g.ax.set(ylabel="Počet nehod", xticks=[])

    # Set bar labels
    for container in g.ax.containers:
        g.ax.bar_label(container, fmt=lambda x: int(x) if x > 0 else '')

    plt.suptitle("I policisté mají nehody")

    g.savefig("fig.png")
    plt.show()


def interesting_data_stats(df: pd.DataFrame):
    """
    1. Najburanejsie policajtske vozidlo

        p48a
        ----
        12 - policie ČR
        13 - městská, obecní policie

        p45a
        ----
        1 - Alfa Romeo
        ...

    2. Pocet policajtov pod vplyvom

        p48a
        ----
        12 - policie ČR
        13 - městská, obecní policie

        p11
        ---
        3, 5-9 - ANO


    3. Priemerna skoda na policajtskom aute

        p48a
        ----
        12 - policie ČR
        13 - městská, obecní policie

        p53
        ---
        skoda na vozidle v stokorunach

        p14
        ---
        celkova hmotna skoda
    """

    pass


if __name__ == '__main__':
    # pd.options.display.float_format = '{:.2f}'.format

    df = prepare_data('accidents.pkl.gz')
    interesting_data_table(df)
    # interesting_data_graph(df)
    interesting_data_stats(df)
