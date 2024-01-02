#!/usr/bin/python3.10
# coding=utf-8

"""
IZV project part 3
Author: xjurik12
Python version tested on: 3.12
"""

import contextily as ctx
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import cluster

# The region selected for all plots
REGION = "JHM"
# Basemap provider for geo-plots
BASEMAP_PROVIDER = ctx.providers.OpenStreetMap.Mapnik

# Format - xmin, ymin, xmax, ymax
EPSG_5514_CZECHIA_BOUNDARIES = (-911041.21, -1244158.89, -418597.43, -935719.38)


def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """
    Convert the source dataframe to geopandas.GeoDataFrame with the right coordinate system
    applied

    :param df: source
    :return: geopandas.GeoDataFrame with the contents of the source df filtered to relevant
             columns to the assignment
    """

    # Keep only relevant data
    df = df.dropna(subset=['d', 'e'])
    df.reset_index()

    df = df[['p1', 'p10', 'p11', 'date', 'region', 'd', 'e']]

    # Create a GeoDataFrame with the Krovak projection
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas
                                 .points_from_xy(x=df['d'], y=df['e']), crs='EPSG:5514')

    # Some of the X,Y coords were invalid - outside the boundaries of Czechia
    # This clipping takes the EPSG:5514 projected boundaries of Czechia
    # and limits the dataset to entries within those boundaries
    # (ref: https://epsg.io/5514-5239)
    gdf.geometry = gdf.clip_by_rect(*EPSG_5514_CZECHIA_BOUNDARIES)

    # Convert to Web Mercator projection to fix potential graph distortion
    return gdf.to_crs('EPSG:3857')


def _get_graph_pair_limits(axs):
    """
    Compute common limits for a pair of axes (two axes)

    :param axs: Two axes to get common limits of
    """
    (xmin1, xmax1), (ymin1, ymax1) = axs[0].get_xlim(), axs[0].get_ylim()
    (xmin2, xmax2), (ymin2, ymax2) = axs[1].get_xlim(), axs[1].get_ylim()
    final_xlim = (min(xmin1, xmin2), max(xmax1, xmax2))
    final_ylim = (min(ymin1, ymin2), max(ymax1, ymax2))
    return final_xlim, final_ylim


def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """
    Plot a graph with geographical marks of accidents that were caused by an animal
    in pre-selected region over years 2021 and 2022

    :param gdf: geopandas.GeoDataFrame provided by make_geo() function
    :param fig_location: path to save the result graph to
    :param show_figure: display the result graph if True
    """
    years_to_plot = [2021, 2022]

    data = gdf.copy()
    data = data[data['region'] == REGION]

    data = data[data['p10'] == 4]
    data = data[data['date'].dt.year.isin(years_to_plot)]

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    # Plot maps with the accidents marked as colored dots
    for ax, year in zip(axs, years_to_plot):
        data[data['date'].dt.year == year].plot(ax=ax, markersize=4, color="tab:red")
        ctx.add_basemap(ax, crs=data.crs.to_string(), source=BASEMAP_PROVIDER, alpha=0.9)

        ax.set_title(f"{REGION} kraj ({year})")

    # Finally, configure axis for both plots (so they share the same size)
    final_xlim, final_ylim = _get_graph_pair_limits(axs)

    for ax in axs:
        ax.set_xlim(final_xlim)
        ax.set_ylim(final_ylim)
        ax.axis("off")

    fig.tight_layout()

    if show_figure:
        plt.show()

    if fig_location:
        fig.savefig(fig_location)

    plt.close()


def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """
    Plot a graph showing the accidents caused while the driver was under the
    influence of alcohol.

    Data is used from all the years the data is available from, but limited
    to a pre-selected region.

    Furthermore, the plots in the graph are clustered by their respective location
    to form groups of accidents in close proximity.

    :param gdf: geopandas.GeoDataFrame provided by make_geo() function
    :param fig_location: path to save the result graph to
    :param show_figure: display the result graph if True
    """
    data = gdf.copy()
    # NOTE: Not filtering out irrelevant regions in make_geo is a part of the assignment
    data = data[
        data['region'] == REGION
        ]

    # Proved significant alcohol intake
    data = data[data['p11'] >= 4]
    # Remove entries where the geometry is empty
    data = data[~(data.geometry.is_empty | data.geometry.isna())]

    data_coords = np.dstack([data.geometry.x, data.geometry.y]).reshape((-1, 2))

    # Clustering was tested on 4 different algorithms
    # - BisectingKMeans
    #     -> Produced results where multiple clusters shared a city
    #        (Brno divided into 2 clusters)
    #
    # - AgglomerativeClustering
    #     -> Produced results where the clusters slightly overlapped
    #
    # - MiniBatchKMeans
    #     -> Produced results where many of the accidents looked like outliers
    #        for the given cluster (clusters over cities in close proximity
    #        picked up even accidents that seem too remote to be picked up)
    #
    # - KMeans
    #     -> Personal preference of all the results.
    #        Clusters are generally well-placed over close cities in close proximity
    #
    # The parameters (n_clusters=15 & n_init=14) make for
    # the most visually pleasing result I found.
    #
    # More clusters result in clusters that are too small and lessening
    # the number of clusters leads to areas too large
    #
    # In my opinion, n_init=14 hits the sweet spot for the clusters to contain
    # just the right accidents based on proximity to their cities
    model = cluster.KMeans(n_clusters=15, n_init=14)
    model = model.fit(data_coords)

    # Add clusters column as the ordinal cluster number
    data['cluster'] = model.labels_
    # Dissolve data into clusters by accident ID
    data = data.dissolve(by="cluster", aggfunc={"p1": "count"})

    # Plot the clusters with accidents and colormap legend
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Clusters
    data.convex_hull.plot(ax=ax, lw=1, color="tab:grey", alpha=0.4)
    # Accidents
    data.plot(ax=ax, markersize=4, column="p1", legend=True,
              # Legend config
              legend_kwds={
                  "orientation": "horizontal",
                  "shrink": 0.744,
                  "pad": 0.014,
                  "label": "Počet nehod v úseku"})

    # Add basemap
    ctx.add_basemap(ax, crs=data.crs.to_string(), alpha=0.9, source=BASEMAP_PROVIDER)

    ax.set_title(f"Nehody v {REGION} kraji s výynamnou měrou alkoholu")
    ax.set_axis_off()

    fig.tight_layout()

    if show_figure:
        plt.show()

    if fig_location:
        fig.savefig(fig_location)

    plt.close()


if __name__ == "__main__":
    _gdf = make_geo(pd.read_pickle("accidents.pkl.gz"))
    plot_geo(_gdf, "geo1.png", True)
    plot_cluster(_gdf, "geo2.png", True)
