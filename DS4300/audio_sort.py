from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from audio_features import mfcc_features
import pandas as pd
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def KMeans_init(sample_data, clusters):

    mfcc_data = [np.array(mfcc_features(signal, 48000)).flatten() for signal in sample_data]

    scaler = StandardScaler()
    standardized = scaler.fit_transform(mfcc_data)

    kmeans = KMeans(n_clusters=clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(standardized)

    return kmeans.labels_


def edge_distance(data):
    # train
    df = pd.DataFrame(data)
    all_features = df.iloc[:, [1,2,3,4,5,6,7]]

    scaler = StandardScaler()
    edges = []

    scaled_df = pd.DataFrame(scaler.fit_transform(all_features), columns=all_features.columns)

    for idx, node1 in scaled_df.iterrows():
        for idx_, node in scaled_df.iterrows():
            if idx == idx_:
                pass
            else:
                # calculate cos distance
                edges.append((idx, idx_, (cosine_similarity([node], [node1]))))


    sorted_edges = sorted(edges, key=lambda x: x[2])

    recommended_edges_close = sorted_edges[0:math.floor((len(sorted_edges)+1)/4)]
    recommended_edges_far = sorted_edges[-1][0:math.floor((len(sorted_edges)+1)/8)]

    return recommended_edges_close, recommended_edges_far


