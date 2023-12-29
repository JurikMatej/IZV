#!/usr/bin/python3.10
# coding=utf-8
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as ctx
import sklearn.cluster
import numpy as np


def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """ Konvertovani dataframe do geopandas.GeoDataFrame se spravnym kodovani"""

    df = df.dropna(subset=['d', 'e'])
    df.reset_index()
    return geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df['d'], df['e']), crs='epsg:5514')

def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """ Vykresleni grafu s nehodami  """

    region = "JHM"
    years_to_plot = [2021, 2022]
    # basemap_provider = ctx.providers.Stamen.Terrain
    basemap_provider = ctx.providers.OpenStreetMap.Mapnik

    data = gdf.copy()
    data = data[data['region'] == region]

    data = data[data['p10'] == 4]
    data = data[data['date'].dt.year.isin(years_to_plot)]

    data.to_crs("epsg:3857")

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # source=ctx.providers.Stamen.TonerLite

    for ax, year in zip(axs, [2021, 2022]):
        gdf[gdf['date'].dt.year == year].plot(ax=ax)
        print("Downloading the basemap")
        ctx.add_basemap(ax, crs=data.crs.to_string(), source=basemap_provider, zoom=15)  # TODO hangs
        print("Hotovinko sak")

    if show_figure:
        plt.show()

    if fig_location:
        plt.savefig(fig_location)

    plt.close()

def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """ Vykresleni grafu s lokalitou vsech nehod v kraji shlukovanych do clusteru """

    # TODO clustering

    if show_figure:
        plt.show()

    if fig_location:
        plt.savefig(fig_location)

    plt.close()

if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    gdf = make_geo(pd.read_pickle("accidents.pkl.gz"))
    plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)
