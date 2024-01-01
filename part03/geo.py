#!/usr/bin/python3.10
# coding=utf-8
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as ctx
import sklearn.cluster as cluster
import numpy as np


REGION = "JHM"  # The region selected for all plots
BASEMAP_PROVIDER = ctx.providers.OpenStreetMap.Mapnik  # Basemap provider for geo-plots

EPSG_5514_CZECHIA_BOUNDARIES = (-911041.21, -1244158.89, -418597.43, -935719.38)  # Format - xmin, ymin, xmax, ymax


def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """ Konvertovani dataframe do geopandas.GeoDataFrame se spravnym kodovani """

    df = df.dropna(subset=['d', 'e'])
    df.reset_index()

    # Keep only relevant data
    df = df[['p1', 'p10', 'p11', 'date', 'region', 'd', 'e']]

    # Create a GeoDataFrame with the Krovak projection
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(x=df['d'], y=df['e']), crs='EPSG:5514')

    # Some of the X,Y coords were invalid - outside the boundaries of Czechia
    # This clipping takes the EPSG:5514 projected boundaries of Czechia and limits the dataset to entries within those
    # boundaries (ref: https://epsg.io/5514-5239)
    gdf.geometry = gdf.clip_by_rect(*EPSG_5514_CZECHIA_BOUNDARIES)

    return gdf.to_crs('EPSG:3857')  # Convert to Web Mercator projection to fix potential graph distortion


def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """ Vykresleni grafu s nehodami  """

    years_to_plot = [2021, 2022]
    plot_color_by_year = ["tab:red", "tab:red"]

    data = gdf.copy()
    data = data[data['region'] == REGION]

    data = data[data['p10'] == 4]
    data = data[data['date'].dt.year.isin(years_to_plot)]

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    # Plot maps with the accidents marked as colored dots
    for ax, year, color in zip(axs, years_to_plot, plot_color_by_year):
        data[data['date'].dt.year == year].envelope.plot(ax=ax, markersize=4, color=color)
        ctx.add_basemap(ax, crs=data.crs.to_string(), source=BASEMAP_PROVIDER, alpha=0.9)

        ax.set_title(f"{REGION} kraj ({year})")

    # Finally, configure axis for both plots
    (xmin1, xmax1), (ymin1, ymax1) = axs[0].get_xlim(), axs[0].get_ylim()
    (xmin2, xmax2), (ymin2, ymax2) = axs[1].get_xlim(), axs[1].get_ylim()

    final_xlim = (min(xmin1, xmin2), max(xmax1, xmax2))
    final_ylim = (min(ymin1, ymin2), max(ymax1, ymax2))

    for ax in axs:
        ax.set_xlim(final_xlim)
        ax.set_ylim(final_ylim)
        ax.axis("off")

    fig.tight_layout()

    if show_figure:
        plt.show()

    if fig_location:
        plt.savefig(fig_location)

    plt.close()


def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """ Vykresleni grafu s lokalitou vsech nehod v kraji shlukovanych do clusteru """

    data = gdf.copy()
    data = data[
        data['region'] == REGION  # NOTE: Not filtering out irrelevant regions in make_geo is a part of the assignment
    ]  

    data = data[data['p11'] >= 4]  # Proved significant alcohol intake
    data = data[~(data.geometry.is_empty | data.geometry.isna())]  # Remove entries where the geometry is empty
    
    data_coords = np.dstack([data.geometry.x, data.geometry.y]).reshape((-1, 2))

    # Clustering was tested on 4 different algorithms
    # - BisectingKMeans -> Produced results where multiple clusters shared a city (Brno divided into 2 clusters)
    # - AgglomerativeClustering
    #                   -> Produced results where the clusters slightly overlapped
    # - MiniBatchKMeans -> Produced results where many of the accidents looked like outliers for the given cluster
    #                      (clusters over cities in close proximity picked up even accidents
    #                       that seem too remote to be picked up)
    # - KMeans          -> Personal preference of all the results.
    #                      Clusters are generally well-placed over close cities in close proximity
    model = cluster.KMeans(n_clusters=15, n_init=14)
    model = model.fit(data_coords)

    data['cluster'] = model.labels_  # Add clusters column as the ordinal cluster number
    data = data.dissolve(by="cluster", aggfunc={"p1": "count"})  # Dissolve data into clusters by accident ID

    # Plot the clusters with accidents and colormap legend
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    data.convex_hull.plot(ax=ax, lw=1, color="tab:grey", alpha=0.4)  # Clusters
    data.plot(ax=ax, markersize=4, column="p1", legend=True,   # Accidents
              legend_kwds={  # Legend
                  "orientation": "horizontal",
                  "shrink": 0.744,
                  "pad": 0.014,
                  "label": "Počet nehod v úseku"})

    ctx.add_basemap(ax, crs=data.crs.to_string(), alpha=0.9, source=BASEMAP_PROVIDER)  # Basemap

    ax.set_title(f"Nehody v {REGION} kraji s výynamnou měrou alkoholu")
    ax.set_axis_off()

    fig.tight_layout()

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
