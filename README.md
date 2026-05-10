# Airbnb-Analysis-Amsterdam

# The purpose of this analysis was to investigate whether Airbnb review scores are primarily influenced by the actual quality of the property or whether other factors such as price, property size (bedrooms), and spatial location also play an important role.  Using Airbnb listings from Amsterdam (airbnb-listings.geojson) please download the file from here:(https://www.kaggle.com/datasets/darkmatternet/airbnb-listings-nyc-london-paris-tokyo-and-more) before running the code and change the name of the name of the geojson

import numpy as np
import geopandas as gpd
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import folium as f
from folium import Marker

import contextily

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score

from kneed import KneeLocator
from collections import Counter
gdf = gpd.read_file('airbnb-listings.geojson')

print(gdf.head())
#FILTER CITY

Abnb = gdf.loc[

    gdf["jurisdiction_names"] == 'Amsterdam',

    (
        "id",
        "jurisdiction_names",
        "bedrooms",
        "price",
        "review_scores_value",
        "geometry"
    )

]

print(Abnb.head())
#CLEAN DATA

print(Abnb.dtypes)

print("\nMissing values:")
print(Abnb.isnull().sum())

# Remove missing values

Abnb.dropna(

    subset=[
        'bedrooms',
        'price',
        'review_scores_value'
    ],

    inplace=True
)

# Remove unrealistic values

Abnb = Abnb[Abnb['price'] > 0]

Abnb = Abnb[Abnb['bedrooms'] > 0]

print("\nDataset shape:")
print(Abnb.shape)
print(Abnb.isnull().sum())
#INTERACTIVE MAP

m = f.Map(

    location=[52.3558, 4.8781],

    zoom_start=12
)

def plotDot(point):

    f.CircleMarker(

        location=[
            point.geometry.y,
            point.geometry.x
        ],

        radius=2,

        tooltip=(
            f"Price: {point.price} | "
            f"Review: {point.review_scores_value}"
        ),

        color='red',

        fill=True

    ).add_to(m)

Abnb.apply(plotDot, axis=1)

m
#EXPLORATORY ANALYSIS

sns.pairplot(

    Abnb[
        [
            'bedrooms',
            'price',
            'review_scores_value'
        ]
    ],

    height=4
)

plt.show()
# CORRELATION MATRIX

corr = Abnb[
    [
        'bedrooms',
        'price',
        'review_scores_value'
    ]
].corr()

plt.figure(figsize=(8,6))

sns.heatmap(

    corr,

    annot=True,

    cmap='coolwarm'
)

plt.title('Correlation Matrix')

plt.show()
#SPATIAL DISTRIBUTION

ax = Abnb.plot(

    column='price',

    figsize=(12,8),

    legend=True,

    markersize=5
)

contextily.add_basemap(

    ax,

    crs=Abnb.crs
)

plt.title('Airbnb Price Distribution')

plt.show()
#BEDROOM MAP

ax = Abnb.plot(

    column='bedrooms',

    figsize=(12,8),

    legend=True,

    markersize=5
)

contextily.add_basemap(

    ax,

    crs=Abnb.crs
)

plt.title('Bedroom Distribution')

plt.show()
#REVIEW SCORES MAP

ax = Abnb.plot(

    column='review_scores_value',

    figsize=(12,8),

    legend=True,

    markersize=5
)

contextily.add_basemap(

    ax,

    crs=Abnb.crs
)

plt.title('Review Score Distribution')

plt.show()
#LINEAR REGRESSION

#REVIEWS VS PRICES

LR = LinearRegression()

x = Abnb['review_scores_value'].values.reshape(-1,1)

y = Abnb['price'].values

LR.fit(x, y)

y_pred = LR.predict(x)

plt.figure(figsize=(12,8))

plt.scatter(x, y, alpha=0.4)

plt.plot(x, y_pred)

plt.xlabel('Review Scores')

plt.ylabel('Price')

plt.title('Review Scores vs Price')

plt.show()

print("Coefficient:", LR.coef_)

print("Intercept:", LR.intercept_)

print("R²:", LR.score(x,y))
#REVIEWS VS BEDROOMS

x1 = Abnb['review_scores_value'].values.reshape(-1,1)

y1 = Abnb['bedrooms'].values

LR.fit(x1, y1)

y1_pred = LR.predict(x1)

plt.figure(figsize=(12,8))

plt.scatter(x1, y1, alpha=0.4)

plt.plot(x1, y1_pred)

plt.xlabel('Review Scores')

plt.ylabel('Bedrooms')

plt.title('Review Scores vs Bedrooms')

plt.show()

print("R²:", LR.score(x1,y1))
#DBSCAN PREPARATION AND ANALYSIS

Abnb1 = Abnb.copy()

Abnb1 = Abnb1.drop(

    [
        'jurisdiction_names',
        'geometry'
    ],

    axis=1
)

# Features

X_train = Abnb1[
    [
        'bedrooms',
        'price',
        'review_scores_value'
    ]
]

# STANDARDISE VARIABLES

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_train)
#KNN ESTIMATION

n_nb = NearestNeighbors(

    n_neighbors=5
).fit(X_scaled)

neigh_dist, neigh_ind = n_nb.kneighbors(X_scaled)

sort_neigh_dist = np.sort(

    neigh_dist,

    axis=0
)

k_dist = sort_neigh_dist[:,4]

plt.figure(figsize=(10,6))

plt.plot(k_dist)

plt.ylabel("5-NN Distance")

plt.xlabel("Sorted Observations")

plt.title("KNN Distance Plot")

plt.show()
#KNEE DETECTION

kneedle = KneeLocator(

    x=range(len(k_dist)),

    y=k_dist,

    S=1.0,

    curve='convex',

    direction='increasing'

)

eps_value = kneedle.knee_y

if eps_value is None or eps_value <= 0:

    eps_value = np.percentile(
        k_dist,
        95
    )

print(
    f"Estimated epsilon: {eps_value}"
)

kneedle.plot_knee()

plt.show()
#DBSCAN CLUSTERING

# SAFE EPS VALUE

eps_value = kneedle.knee_y

# If KneeLocator fails and returns 0 or None
# assign a default epsilon

if eps_value is None or eps_value <= 0:

    eps_value = 0.5

print(f"Using eps value: {eps_value}")

# RUN DBSCAN

clusters = DBSCAN(

    eps=float(eps_value),

    min_samples=5

).fit(X_scaled)

# GET LABELS

c = clusters.labels_

print(Counter(c))
#CLUSTER PLOTS

DBSCAN_dataset = X_train.copy()

DBSCAN_dataset['Cluster'] = c

Abnb['cluster'] = c


#PRICE VS REVIEW

plt.figure(figsize=(12,8))

sns.scatterplot(

    data=DBSCAN_dataset,

    x='price',

    y='review_scores_value',

    hue='Cluster',

    palette='Set2'
)

plt.title('DBSCAN Clusters: Price vs Review Scores')

plt.show()
#BEDROOM VS REVIEW

plt.figure(figsize=(12,8))

sns.scatterplot(

    data=DBSCAN_dataset,

    x='bedrooms',

    y='review_scores_value',

    hue='Cluster',

    palette='Set2'
)

plt.title('DBSCAN Clusters: Bedrooms vs Review Scores')

plt.show()
#SPATIAL CLUSTER MAP

ax = Abnb.plot(

    column='cluster',

    categorical=True,

    figsize=(14,10),

    legend=True,

    markersize=6
)

contextily.add_basemap(

    ax,

    crs=Abnb.crs
)

plt.title('Spatial Distribution of DBSCAN Clusters')

plt.show()
#OUTLIER ANALYSIS

outliers = Abnb[Abnb['cluster'] == -1]

print("\nNumber of outliers:")

print(len(outliers))
#CLUSTER SUMMARY

cluster_summary = Abnb.groupby('cluster')[

    [
        'price',
        'bedrooms',
        'review_scores_value'
    ]

].mean().round(2)

cluster_summary['count'] = (

    Abnb.groupby('cluster').size()
)

print(cluster_summary)
#SILHOUETE SCORE

valid_clusters = DBSCAN_dataset[
    DBSCAN_dataset['Cluster'] != -1
]

if len(valid_clusters['Cluster'].unique()) > 1:

    score = silhouette_score(

        X_scaled[c != -1],

        c[c != -1]
    )

    print("\nSilhouette Score:")

    print(round(score, 3))
#LOG PLOTS

Abnb['log_price'] = np.log1p(Abnb['price'])

plt.figure(figsize=(12,8))

sns.scatterplot(

    data=Abnb,

    x='log_price',

    y='review_scores_value',

    hue='cluster',

    palette='Set2'
)

plt.title('Log Price vs Review Scores')

plt.show()
