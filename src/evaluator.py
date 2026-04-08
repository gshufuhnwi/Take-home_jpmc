# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 18:31:02 2026

@author: gerar
"""

import os
import json
import numpy as np
import pandas as pd
from metrics import Metrics


class Evaluator:
        def __init__(self, output_dir="outputs", threshold=0.26, decimals=4):
                self.output_dir = output_dir
                self.threshold = float(threshold)
                self.decimals = int(decimals)

                os.makedirs(self.output_dir, exist_ok=True)
                self.metrics = Metrics()

        def round_dict(self, d):
                """
                Recursively round floats inside nested dicts to self.decimals.
                """
                if isinstance(d, dict):
                        return {k: self.round_dict(v) for k, v in d.items()}
                elif isinstance(d, float):
                        return round(d, self.decimals)
                elif isinstance(d, np.floating):
                        return round(float(d), self.decimals)
                else:
                        return d

        def _normalize_columns(self, df):
                """
                Normalize column names to match your preprocessing convention:
                - trim
                - spaces -> underscores
                - remove apostrophes
                """
                df = df.copy()
                df.columns = [c.strip().replace(" ", "_").replace("'", "") for c in df.columns]
                return df

        def evaluate_and_save(self, df_raw, df_model, y_score, sensitive_cols=("sex", "race")):
                """
                df_raw: raw dataframe aligned to df_model index (used for fairness attributes)
                df_model: processed dataframe used for ML (contains income_gt_50k + weight)
                y_score: predicted probabilities for the positive class
                sensitive_cols: tuple of columns in df_raw to use for fairness evaluation
                """

                # defensive column normalization for fairness columns
                df_raw = self._normalize_columns(df_raw)

                # y_true and weights
                if "income_gt_50k" not in df_model.columns:
                        raise ValueError("df_model must contain 'income_gt_50k'.")

                y_true = df_model["income_gt_50k"].astype(int).to_numpy()
                y_score = np.asarray(y_score).astype(float)
                y_pred = (y_score >= self.threshold).astype(int)

                # classification metrics (from your Metrics class)
                core = self.metrics.classification_metrics(y_true, y_pred, y_score=y_score)

                # fairness metrics
                fairness = {}
                for col in sensitive_cols:
                        if col not in df_raw.columns:
                                fairness[col] = {"error": f"Column '{col}' not found in df_raw"}
                                continue

                        # ensure alignment by index
                        sens = df_raw.loc[df_model.index, col].to_numpy()

                        fairness[col] = {
                                "SPD": self.metrics.spd(y_pred, sens),
                                "EOD": self.metrics.eod(y_true, y_pred, sens)
                        }

                # --------------------------
                # Save predictions
                # --------------------------
                pred_df = pd.DataFrame({
                        "y_true": y_true,
                        "y_score": np.round(y_score, self.decimals),
                        "y_pred": y_pred
                }, index=df_model.index)

                pred_df.to_csv(os.path.join(self.output_dir, "xgb_predictions.csv"), index=False)

                # --------------------------
                # Save metrics CSV (rounded)
                # --------------------------
                metrics_rows = [{"metric": k, "value": round(float(v), self.decimals)} for k, v in core.items()]
                pd.DataFrame(metrics_rows).to_csv(os.path.join(self.output_dir, "xgb_metrics.csv"), index=False)

                # --------------------------
                # Save fairness summary CSV
                # --------------------------
                fairness_rows = []
                for s_col, vals in fairness.items():
                        if isinstance(vals, dict) and "error" in vals:
                                fairness_rows.append({"attribute": s_col, "metric": "error", "value": vals["error"]})
                                continue

                        # SPD
                        spd_val = vals["SPD"]["spd"]
                        fairness_rows.append({
                                "attribute": s_col,
                                "metric": "SPD",
                                "value": round(float(spd_val), self.decimals)
                        })

                        # EOD
                        eod_val = vals["EOD"]["eod"]
                        fairness_rows.append({
                                "attribute": s_col,
                                "metric": "EOD",
                                "value": round(float(eod_val), self.decimals)
                        })

                pd.DataFrame(fairness_rows).to_csv(os.path.join(self.output_dir, "fairness_summary.csv"), index=False)

                # --------------------------
                # Save detailed JSON (rounded)
                # --------------------------
                results = {
                        "threshold": round(self.threshold, self.decimals),
                        "classification_metrics": self.round_dict(core),
                        "fairness_metrics": self.round_dict(fairness)
                }

                with open(os.path.join(self.output_dir, "xgb_metrics_and_fairness.json"), "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2)

                return results