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

    p48a
    ----
    11 - ministerstvo vnitra
    12 - policie ČR
    13 - městská, obecní policie
    15 - ministerstvo obrany

    """
    data = df.copy()


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
    df = prepare_data('accidents.pkl.gz')
    interesting_data_table(df)
    interesting_data_graph(df)
    interesting_data_stats(df)
