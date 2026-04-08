# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 18:31:47 2026

@author: gerar
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

class EDAPlotter:
        def __init__(self, output_dir="outputs/eda"):
                self.output_dir = output_dir
                os.makedirs(self.output_dir, exist_ok=True)

        def plot_target_distribution(self, df, target_col="income_gt_50k"):
                counts = df[target_col].value_counts(dropna=False)
                plt.figure(figsize=(6,4))
                counts.plot(kind="bar")
                plt.title("Target Distribution")
                plt.xlabel(target_col)
                plt.ylabel("count")
                plt.savefig(os.path.join(self.output_dir, "target_distribution.png"), bbox_inches="tight")
                plt.close()
        def plot_numeric_distribution(self, df, col):
                plt.figure(figsize=(7,4))
                df[col].dropna().hist(bins=40)
                plt.title(f"Distribution: {col}")
                plt.xlabel(col)
                plt.ylabel("count")
                plt.savefig(os.path.join(self.output_dir, f"dist_{col}.png"), bbox_inches="tight")
                plt.close()

        def plot_categorical_topk(self, df, col, top_k=15):
                vc = df[col].value_counts().head(top_k)

                plt.figure(figsize=(9,5))
                vc.plot(kind="bar")
                plt.title(f"Top {top_k} Categories: {col}")
                plt.xlabel(col)
                plt.ylabel("count")
                plt.xticks(rotation=45, ha="right")
                plt.savefig(os.path.join(self.output_dir, f"cat_{col}.png"), bbox_inches="tight")
                plt.close()

        def plot_feature_vs_target_rate(self, df, col, target_col="income_gt_50k", top_k=15):
                tmp = df[[col, target_col]].dropna()
                rates = tmp.groupby(col)[target_col].mean().sort_values(ascending=False).head(top_k)

                plt.figure(figsize=(9,5))
                rates.plot(kind="bar")
                plt.title(f"P({target_col}=1) by {col} (Top {top_k})")
                plt.xlabel(col)
                plt.ylabel("rate")
                plt.xticks(rotation=45, ha="right")
                plt.savefig(os.path.join(self.output_dir, f"rate_{col}.png"), bbox_inches="tight")
                plt.close()

        def run_quick_eda(self, df_raw, feature_list):
                if "income_gt_50k" in df_raw.columns:
                        self.plot_target_distribution(df_raw)

                for col in feature_list:
                        if col not in df_raw.columns:
                                continue
                        if pd.api.types.is_numeric_dtype(df_raw[col]):
                                self.plot_numeric_distribution(df_raw, col)
                        else:
                                self.plot_categorical_topk(df_raw, col, top_k=15)
                                if "income_gt_50k" in df_raw.columns:
                                        self.plot_feature_vs_target_rate(df_raw, col, top_k=15)