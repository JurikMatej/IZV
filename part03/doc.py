#!/usr/bin/python3.10
# coding=utf-8

"""
IZV project part 3
Author: xjurik12
Python version tested on: 3.12
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.core.indexes.frozen import FrozenList

GRAPH_OUTPUT_FILE = "fig.png"


def prepare_data(path_to_input: str) -> pd.DataFrame:
    """
    Read prepared data from the provided path and parse it for processing

    :param path_to_input: path to prepared data (pickle-dumped dataframe)
    """
    return pd.read_pickle(path_to_input)


def interesting_data_table(df: pd.DataFrame, generate_latex: bool = False):
    """
    Create a table displaying the number of accidents that were caused
    by either police or other government member

    This report contains 2 tables:
        1. Total accident count across all years
        2. Specific accident type counts across all years

    :param df: Source dataframe prepared by prepare_data()
    :param generate_latex: If true, export the table in .tex format
                           (must be selected and copied from the standard output,
                           because STDOUT is already in use as per the assignment)
    """
    vehicle_owners = {
        11: "Ministerstvo vnútra",
        12: "Polícia ČR",
        13: "Mestská, obecná polícia",
        15: "Ministerstvo obrany",
    }

    data = df.copy()

    # Filter out data not needed or empty
    data = data[['p1', 'p10', 'p48a', 'date']]
    data = data.dropna(subset=['p48a'])

    # Filter to only the accidents which involved the police or government vehicles
    data = data[data['p48a'].isin([11, 12, 13, 15])]
    # Filter to only the accidents that were caused by the police or the gov drivers
    data = data[data['p10'] == 1]

    data['p48a'] = data['p48a'].map(vehicle_owners)

    # Create a column storing only the year of the accident
    data['year'] = data['date'].dt.year

    # Prepare the 'total accidents' column
    data['total'] = data['p1']
    totals_table = data.groupby(['year']).agg({'total': 'count'})
    data = data.groupby(['year', 'p48a']).agg({'p1': 'count'}).reset_index()
    data = totals_table.merge(data, on='year', how='outer')

    # Calculate the fraction of total accidents for different vehicle owner caused accidents
    data['%ofTotal'] = data['p1'] / data['total'] * 100
    # NOTE: the \\% escape is necessary for .tex compiler
    data['%ofTotal'] = [f"{x:.2f}\\%" for x in data['%ofTotal']]

    # TABLE 1: Overview - Total numbers of per-year accidents caused by the police and government
    table_overview = data.groupby(['year']).agg({'total': 'first'})
    # Rename the columns before exporting them as a table
    table_overview.index.names = FrozenList(['Rok'])
    table_overview.columns = ['Nehody celkom']

    # TABLE 2: Specifics
    #          - Numbers and the fraction of total accidents per specific vehicle owner
    #            (police or gov) by year
    table_specific = data.groupby(['year', 'p48a']).agg({'p1': 'first', '%ofTotal': 'first'})
    # Rename the columns before exporting them as a table
    table_specific.index.names = FrozenList(['Rok', 'Nehody spôsobil'])
    table_specific.columns = ['Počet nehôd', 'Pomer k celku za rok']

    print("Tabuľky použité vo finálnej správe:")
    print(table_overview)
    print(table_specific)

    if generate_latex:
        # Generate source latex for the tables to be used in the doc.pdf
        print(table_overview.to_latex())
        print(table_specific.to_latex())


def interesting_data_graph(df: pd.DataFrame):
    """
    Generate a histogram displaying what the police crashes into the most
    and save it in the pre-configured format (see GRAPH_OUTPUT_FILE)

    :param df: Source dataframe prepared by prepare_data()
    """
    accident_type = {
        1: "Zrážka s idúcim nekoľajovým vozidlom",
        2: "Zrážka s vozidlom zaparkovaným/odstaveným",
        3: "Zrážka s pevnou prekážkou",
        4: "Zrážka s chodcom",
        5: "Zrážka s lesnou zverou",
        6: "Zrážka s domácim zvieraťom",
        7: "Zrážka s vlakom",
        8: "Zrážka s električkou (tramvaj)",
        9: "Havária",
    }

    data = df.copy()
    data = data[['p1', 'p6', 'p10', 'p48a']]

    # Drop empty data for accident types
    data = data.dropna(subset=['p6'])

    # Filter out the unspecified types of an accident
    data = data[data['p6'] != 0]
    # Map accident type categories to their string descriptions
    data['p6'] = data['p6'].map(accident_type)
    data.rename(columns={'p6': 'Typ nehody'}, inplace=True)

    # Filter to only the accidents which involved the police
    data = data[data['p48a'].isin([12, 13])]

    # Filter to only the accidents that were caused by the police
    data = data[data['p10'] == 1]

    g = sns.displot(data,
                    x="Typ nehody",
                    hue="Typ nehody",
                    height=5, aspect=2)

    # Fine tune the axis
    g.ax.set(ylabel="Počet nehôd", xticks=[])

    # Set bar labels
    for container in g.ax.containers:
        g.ax.bar_label(container, fmt=lambda x: int(x) if x > 0 else '')

    plt.suptitle("Aj policajti spôsobujú nehody")

    g.savefig(GRAPH_OUTPUT_FILE)
    print(f"Graf použitý vo finálnej správe sa úspešne uložil do súboru '{GRAPH_OUTPUT_FILE}'")

    plt.show()


def _most_crashed_police_vehicle(df: pd.DataFrame):
    """
    Display the most crashed police vehicle based on the data

    :param df: Source data filtered to only the police owned vehicles in Czech Republic
    """
    vehicles = {  # Vehicles without Buses and special types of vehicle (train, tractor etc.)
        1: "ALFA-ROMEO",
        2: "AUDI",
        3: "AVIA",
        4: "BMW",
        5: "CHEVROLET",
        6: "CHRYSLER",
        7: "CITROEN",
        8: "DACIA",
        9: "DAEWOO",
        10: "DAF",
        11: "DODGE",
        12: "FIAT ",
        13: "FORD",
        14: "GAZ, VOLHA",
        15: "FERRARI",
        16: "HONDA",
        17: "HYUNDAI",
        18: "IFA",
        19: "IVECO",
        20: "JAGUAR",
        21: "JEEP",
        22: "LANCIA",
        23: "LAND ROVER",
        24: "LIAZ",
        25: "MAZDA",
        26: "MERCEDES",
        27: "MITSUBISHI",
        28: "MOSKVIČ",
        29: "NISSAN",
        30: "OLTCIT",
        31: "OPEL",
        32: "PEUGEOT",
        33: "PORSCHE",
        34: "PRAGA",
        35: "RENAULT",
        36: "ROVER",
        37: "SAAB",
        38: "SEAT",
        39: "ŠKODA",
        40: "SCANIA",
        41: "SUBARU",
        42: "SUZUKI",
        43: "TATRA",
        44: "TOYOTA",
        45: "TRABANT",
        46: "VAZ",
        47: "VOLKSWAGEN",
        48: "VOLVO",
        49: "WARTBURG",
        50: "ZASTAVA",
        51: "AGM",
        52: "ARO",
        53: "AUSTIN",
        54: "BARKAS",
        55: "DAIHATSU",
        56: "DATSUN",
        57: "DESTACAR",
        58: "ISUZU",
        59: "KAROSA",
        60: "KIA",
        61: "LUBLIN",
        62: "MAN",
        63: "MASERATI",
        64: "MULTICAR",
        65: "PONTIAC",
        66: "ROSS",
        67: "SIMCA",
        68: "SSANGYONG",
        69: "TALBOT",
        70: "TAZ",
        71: "ZAZ",
        79: "APRILIA",
        80: "CAGIVA",
        81: "ČZ",
        82: "DERBI",
        83: "DUCATI",
        84: "GILERA",
        85: "HARLEY",
        86: "HERO",
        87: "HUSQVARNA",
        88: "JAWA",
        89: "KAWASAKI",
        90: "KTM",
        91: "MALAGUTI",
        92: "MANET",
        93: "MZ",
        94: "PIAGGIO",
        95: "SIMSON",
        96: "VELOREX",
        97: "YAMAHA",
        98: "jiné vyrobené v ČR",
        99: "jiné vyrobené mimo ČR"
    }

    fact_data = df.copy()
    fact_data = fact_data[['p1', 'p45a']]
    # Filter out the buses and the 'none of these' field
    fact_data = fact_data[~fact_data['p45a'].isin([0, *list(range(72, 79))])]
    fact_data['p45a'] = fact_data['p45a'].map(vehicles)

    fact_data = fact_data.groupby(['p45a']).agg({'p1': 'count'}).reset_index()
    # print(fact1_data)

    the_car = fact_data[fact_data['p1'] == fact_data['p1'].max()].iloc[0]['p45a']
    the_amount = fact_data['p1'].max()

    print(f"Najbúranejšie policajné auto za posledné roky: {the_car} ({the_amount} havárií)")


def _count_of_police_officers_under_the_influence(df: pd.DataFrame):
    """
    Display the number of police members that were a part of an accident
    while under the influence of alcohol

    :param df: Source data filtered to only the police owned vehicles in Czech Republic
    """
    fact_data = df.copy()
    fact_data = fact_data[['p1', 'p11']]

    # Filter data to alcohol influence only -> p11=[3, 5..9] because p11=4 means use of drugs
    fact_data = fact_data[fact_data['p11'].isin([3, *list(range(5, 10))])]

    the_count = fact_data['p1'].count()

    print(f"Počet policajtov zúčastnených pri nehodách, ktorí boli pod vplyvom alkoholu: "
          f"{the_count}")


def _average_damages_caused_with_police_involved(df: pd.DataFrame):
    """
    Display the average damages caused to:
        1. The police cars involved in an accident (damage only to car itself)
        2. The total infrastructure involved (the average of total damages caused)

    :param df: Source data filtered to only the police owned vehicles in Czech Republic
    """
    fact_data = df.copy()
    fact_data = fact_data[['p1', 'p53', 'p14']]

    # The damages themselves are represented as number of 100 czech koruna (czk) bills
    the_vehicle_damage = int(fact_data['p53'].mean().round(2) * 100)
    the_total_damage = int(fact_data['p14'].mean().round(2) * 100)

    print(f"Priemerná škoda na policajnom aute: {the_vehicle_damage} czk")
    print(f"Priemerná celková škoda spôsobená nehodou, na ktorej sa účastnil aj policajt: "
          f"{the_total_damage} czk")


def interesting_data_stats(df: pd.DataFrame):
    """
    Display any additional interesting stats found in the data
    that are relevant to the generated table and graph

    :param df: Source dataframe prepared by prepare_data()
    """
    data = df.copy()
    data = data[['p1', 'p11', 'p14', 'p45a', 'p53', 'p48a']]

    # Filter to the police vehicles only
    data = data[data['p48a'].isin([12, 13])]

    _most_crashed_police_vehicle(data)
    _count_of_police_officers_under_the_influence(data)
    _average_damages_caused_with_police_involved(data)


if __name__ == '__main__':
    _df = prepare_data('accidents.pkl.gz')
    interesting_data_table(_df, generate_latex=False)
    print()
    interesting_data_graph(_df)
    print()
    interesting_data_stats(_df)
