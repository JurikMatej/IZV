#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: xjurik12
Python verze: 3.10, tested also on 3.12

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""
import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup
from numpy.typing import NDArray
from typing import List, Callable, Dict, Any


def integrate(f: Callable[[NDArray], NDArray], a: float, b: float, steps=1000) -> float:
    """
    Uloha c.1: Numericky vypocet integralu
    Funkcia vypocita integral pomocou tzv. obdlznikovej metody

    :param f: funkcia pouzita v integracii
    :param a: zaciatok intervalu urciteho integralu
    :param b: koniec intervalu urciteho integralu
    :param steps: pocet krokov
    :return: hodnota predstavujuca vypocitany integral
    """
    x = np.linspace(a, b, steps)
    return np.sum((x[1:] - x[:-1]) * f((x[:-1] + x[1:]) / 2))


def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None = None):
    """
    Uloha c.2: Generovanie grafu funkcie nasobenej zoznamom konstant na preddefinovanom intervale

    :param a: zoznam konstant, ktorymi sa funkcia ma nasobit
    :param show_figure: prepinac zobrazenia grafu - graf sa zobrazi, ak je nastaveny na True
    :param save_path: cesta, do ktorej sa ulozi vysledny graf, pokial je definovana
    """
    x = np.linspace(-3, 3, 200, dtype=np.float32)
    y = np.array(a).reshape(3, 1).astype(np.float32)**2 * x**3 * np.sin(x)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    ax.spines["right"].set_position(("data", 5))

    ax.set_xlabel("x")
    ax.xaxis.set_label_coords(0.5, -0.09)
    ax.set_ylabel(r"$f_a(x)$")

    ax.set_xlim([-3, 5])
    ax.set_ylim([0, 40])

    ax.set_xticks(np.arange(-3, 4, 1))

    ax.plot(x, y[0], color="blue", label=r"$y_{1.0}(x)$")
    ax.fill_between(x, y[0], color="blue", alpha=0.11)

    ax.plot(x, y[1], color="orange", label=r"$y_{1.5}(x)$")
    ax.fill_between(x, y[1], color="orange", alpha=0.11)

    ax.plot(x, y[2], color="green", label=r"$y_{2.0}(x)$")
    ax.fill_between(x, y[2], color="green", alpha=0.11)

    f_20_x = np.trapz(y[2], x)
    f_15_x = np.trapz(y[1], x)
    f_10_x = np.trapz(y[0], x)
    ax.annotate(r"$\int f_{2.0}(x)dx$ = " + f"{f_20_x:.2f}", xy=(3, 10), xytext=(3, 14.5))
    ax.annotate(r"$\int f_{1.5}(x)$dx = " + f"{f_15_x:.2f}", xy=(3, 10), xytext=(3, 8))
    ax.annotate(r"$\int f_{1.0}(x)dx$ = " + f"{f_10_x:.2f}", xy=(3, 10), xytext=(3, 3))

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncols=3)

    if show_figure:
        fig.show()

    if save_path:
        fig.savefig(save_path)

    plt.close(fig)


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """
    Uloha c.3: Generovanie grafov dvoch samostatnych funkcii a ich suctu
    Funkcia vygeneruje grafy dvoch preddefinovanych funkcii zo zadania a potom graf treti, ktory je suctom predoslych
    dvoch
    V tomto tretom grafe budu zvyraznene na zeleno jeho casti s hodnotami vyssimi ako tymi,
    ktore v rovnakom bode X nadobuda prva funkcia a na cerveno jeho casti s hodnotami nizsimi ako tymi,
    ktore v rovnakom bode X nadobuda prva funkcia

    :param show_figure: prepinac zobrazenia grafu - graf sa zobrazi, ak je nastaveny na True
    :param save_path: cesta, do ktorej sa ulozi vysledny graf, pokial je v tomto parametri definovana
    """
    t = np.linspace(0, 100, 25000)
    f1 = 0.5 * np.cos(0.02 * np.pi * t)
    f2 = 0.25 * (np.sin(np.pi * t) + np.sin(3 / 2 * np.pi * t))
    f3 = f1 + f2

    fig = plt.figure(figsize=(10, 12))
    ax1, ax2, ax3 = fig.subplots(nrows=3)

    ax1.plot(t, f1)
    ax1.margins(x=0, y=0)
    ax1.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax1.set_xlabel(r"t")
    ax1.set_ylabel(r"$f_1(t)$")

    ax2.plot(t, f2)
    ax2.margins(x=0, y=0)
    ax2.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax2.set_xlabel(r"t")
    ax2.set_ylabel(r"$f_2(t)$")

    mask_f3_over_f1 = np.ma.masked_greater(f3, f1)
    mask_f3_under_f1 = np.ma.masked_less(f3, f1)
    ax3.plot(t, mask_f3_over_f1, color='red')
    ax3.plot(t, mask_f3_under_f1, color='green')
    ax3.set_ylim([-0.8, 0.8])
    ax3.margins(x=0, y=0)
    ax3.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax3.set_xlabel(r"t")
    ax3.set_ylabel(r"$f_1(t) + f_2(t)$")

    # show or save the plot
    if show_figure:
        plt.show()

    if save_path:
        plt.savefig(save_path)

    plt.close()

def download_data() -> List[Dict[str, Any]]:
    """
    Uloha c.3: Stahovanie meteorologickych dat dostupnych na web stranke predlozenej v zadani a ich nasledne spracovanie
    do pozadovaneho formatu

    :return: Data spracovane do predom zadaneho formatu List[Dict[str, Any]]
    """
    meteo_content_raw = requests.get("https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html").content
    soup = BeautifulSoup(meteo_content_raw, "html.parser")
    meteo_table_raw = soup.find_all("table")[1]

    meteo_table_trs = iter(meteo_table_raw.find_all("tr"))
    next(meteo_table_trs)  # Skip the table headers

    result = list()
    for tr in meteo_table_trs:
        tds = tr.find_all("td")

        pos = tds[0].strong.string
        lat = f"{tds[2].string[:-1]}".replace(",", ".")
        long = f"{tds[4].string[:-1]}".replace(",", ".")
        height = str(tds[6].string).replace(",", ".")

        new_data_entry = {
            'position': pos,
            'lat': np.float32(lat),
            'long': np.float32(long),
            'height': np.float32(height.strip())
        }
        result.append(new_data_entry)

    return result
