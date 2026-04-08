# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 18:33:11 2026

@author: gerar
"""

import os

from LoadAndProcessDataset import LoadDataset
from segmentation import SegmentationPipeline
from xgb_model import XGBPipeline
from evaluator import Evaluator
from eda_plots import EDAPlotter

def main():

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        loader = LoadDataset(random_state=42)

        # Raw dataset (for EDA + fairness grouping + segmentation)
        raw_df = loader.load_data(encode=False)["census-bureau"]

        # Encoded dataset (for XGBoost)
        model_df = loader.load_data(encode=True)["census-bureau"]

        # ---------------------------
        # EDA plots (choose features)
        # ---------------------------
        top_features = [
                "education",
                "detailed_occupation_recode",
                "wage_per_hour",
                "weeks_worked_in_year",
                "age",
                "capital_gains",
                "detailed_industry_recode",
                "marital_stat",
                "dividends_from_stocks",
                "class_of_worker"
        ]

        eda = EDAPlotter(output_dir=os.path.join(OUTPUT_DIR, "eda"))
        eda.run_quick_eda(raw_df, top_features)

        # ---------------------------
        # Segmentation (KMeans + PCA)
        # ---------------------------
        seg = SegmentationPipeline(n_clusters=6, pca_components=2)
        clusters, X_pca = seg.fit_predict(raw_df)

        seg_df = raw_df.copy()
        seg_df.to_csv(os.path.join(OUTPUT_DIR, "segmentation_with_clusters.csv"), index=False)

        seg.save_cluster_plot(
    X_pca, clusters,
    os.path.join(OUTPUT_DIR, "cluster_plot.png"))

        seg.save_cluster_sizes(
            clusters,
            os.path.join(OUTPUT_DIR, "cluster_sizes.png"))

        seg.save_cluster_profile(
            raw_df, clusters,
            os.path.join(OUTPUT_DIR, "cluster_profile.csv"))
        

        # ---------------------------
        # XGBoost training + outputs
        # ---------------------------
        xgb_pipe = XGBPipeline(random_state=42, threshold=0.26)
        raw_test, model_test, y_score_test, best_threshold = xgb_pipe.run_train_val_test(
                df_raw=raw_df,
                df_model=model_df,
                output_importance_csv=os.path.join(OUTPUT_DIR, "xgb_feature_importance.csv")
        )

        # ---------------------------
        # Evaluate on TEST only
        # ---------------------------
        evaluator = Evaluator(output_dir=OUTPUT_DIR, threshold=best_threshold, decimals=4)

        results = evaluator.evaluate_and_save(
                df_raw=raw_test,
                df_model=model_test,
                y_score=y_score_test,
                sensitive_cols=("sex", "race")
        )


        print("Done. Saved results to outputs/")
        print(results)

if __name__ == "__main__":
        main()