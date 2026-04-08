# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 18:17:05 2026

@author: gerar
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class SegmentationPipeline:
        def __init__(self, n_clusters=6, random_state=42, pca_components=2):
                self.n_clusters = n_clusters
                self.random_state = random_state
                self.pca_components = pca_components

                self.preprocessor = None
                self.pca = None
                self.kmeans = None

        def fit_predict(self, df_raw):

                drop_cols = [c for c in ["income_gt_50k", "label", "weight"] if c in df_raw.columns]
                X = df_raw.drop(columns=drop_cols).copy()

                num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
                cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

                self.preprocessor = ColumnTransformer(
                        transformers=[
                                ("num", StandardScaler(), num_cols),
                                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                        ]
                )

                X_mat = self.preprocessor.fit_transform(X)

                self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
                X_pca = self.pca.fit_transform(X_mat)

                self.kmeans = KMeans(
                        n_clusters=self.n_clusters,
                        random_state=self.random_state,
                        n_init="auto"
                )

                clusters = self.kmeans.fit_predict(X_pca)

                return clusters, X_pca


        def save_cluster_plot(self, X_pca, clusters, out_png):
                plt.figure(figsize=(9, 6))
                plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, alpha=0.6)
                plt.title("Segmentation using KMeans + PCA")
                plt.xlabel("Principal Component 1")
                plt.ylabel("Principal Component 2")
                plt.savefig(out_png, bbox_inches="tight")
                plt.close()


        def save_cluster_sizes(self, clusters, out_png):
                vals, counts = np.unique(clusters, return_counts=True)

                plt.figure(figsize=(9, 5))
                plt.bar(vals, counts)
                plt.title("Cluster Size Distribution")
                plt.xlabel("cluster")
                plt.ylabel("count")
                plt.savefig(out_png, bbox_inches="tight")
                plt.close()


        def save_cluster_profile(self, df_raw, clusters, out_csv):
                df = df_raw.copy()
                df["cluster"] = clusters

                agg = {}
                for col in ["age", "wage_per_hour", "weeks_worked_in_year", "capital_gains"]:
                        if col in df.columns:
                                agg[col] = "mean"
                if "income_gt_50k" in df.columns:
                        agg["income_gt_50k"] = "mean"

                profile = df.groupby("cluster").agg(agg)
                profile.to_csv(out_csv)
                return profile