# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 18:20:12 2026

@author: gerar
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


class XGBPipeline:
        def __init__(self, random_state=42, threshold=0.26):
                self.random_state = random_state
                self.threshold_ = float(threshold)

                self.label_col = "income_gt_50k"
                self.weight_col = "weight"

                # Keep defaults mostly, only set the minimum for correctness + reproducibility and only changed the learning rate
                self.model = XGBClassifier(
                        tree_method="hist",
                        enable_categorical=True,
                        random_state=self.random_state, learning_rate=0.09,
                        eval_metric="auc"
                )

        def _prepare_Xyw(self, df_model):
                """
                Returns X, y, w from df_model.
                Drops label / weight / optional string label column if present.
                """
                df = df_model.copy()

                y = df[self.label_col].astype(int).to_numpy()

                w = None
                if self.weight_col in df.columns:
                        w = df[self.weight_col].to_numpy()

                drop_cols = [self.label_col]
                if self.weight_col in df.columns:
                        drop_cols.append(self.weight_col)

                # some versions keep a string label column called "label"
                if "label" in df.columns:
                        drop_cols.append("label")

                X = df.drop(columns=drop_cols, errors="ignore")
                return X, y, w

        def split_data(self, df_raw, df_model, test_size=0.15, val_size=0.15):
                """
                Stratified split on model labels; applies same indices to raw_df.
                """
                y = df_model[self.label_col].astype(int).to_numpy()
                idx = np.arange(len(df_model))

                idx_trainval, idx_test = train_test_split(
                        idx,
                        test_size=test_size,
                        random_state=self.random_state,
                        stratify=y
                )

                y_trainval = y[idx_trainval]
                # val_size is fraction of total; convert to fraction of trainval
                val_frac_of_trainval = val_size / (1.0 - test_size)

                idx_train, idx_val = train_test_split(
                        idx_trainval,
                        test_size=val_frac_of_trainval,
                        random_state=self.random_state,
                        stratify=y_trainval
                )

                raw_train = df_raw.iloc[idx_train].reset_index(drop=True)
                raw_val = df_raw.iloc[idx_val].reset_index(drop=True)
                raw_test = df_raw.iloc[idx_test].reset_index(drop=True)

                model_train = df_model.iloc[idx_train].reset_index(drop=True)
                model_val = df_model.iloc[idx_val].reset_index(drop=True)
                model_test = df_model.iloc[idx_test].reset_index(drop=True)

                return raw_train, raw_val, raw_test, model_train, model_val, model_test

        def fit(self, df_model_train):
                X_train, y_train, w_train = self._prepare_Xyw(df_model_train)
                self.model.fit(X_train, y_train, sample_weight=w_train)
                return self

        def predict_proba(self, df_model):
                X, _, _ = self._prepare_Xyw(df_model)
                return self.model.predict_proba(X)[:, 1]

        def save_feature_importance(self, df_model_train, output_csv_path):
                """
                Saves gain-based feature importance.
                """
                X_train, _, _ = self._prepare_Xyw(df_model_train)
                booster = self.model.get_booster()
                score = booster.get_score(importance_type="gain")

                # Convert to dataframe aligned to columns
                imp = pd.DataFrame({
                        "feature": list(score.keys()),
                        "importance_gain": list(score.values())
                }).sort_values("importance_gain", ascending=False)

                imp.to_csv(output_csv_path, index=False)

        def run_train_val_test(self, df_raw, df_model, output_importance_csv=None):
                """
                Train on TRAIN only; evaluate on TEST only.
                Threshold is fixed to self.threshold_ (no tuning).
                """
                raw_train, raw_val, raw_test, model_train, model_val, model_test = self.split_data(df_raw, df_model)

                self.fit(model_train)

                y_score_test = self.predict_proba(model_test)

                if output_importance_csv is not None:
                        self.save_feature_importance(model_train, output_importance_csv)

                return raw_test, model_test, y_score_test, self.threshold_