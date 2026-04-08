# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 18:23:12 2026

@author: gerar
"""

import numpy as np
from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        balanced_accuracy_score,
        precision_score,
        recall_score
)

class Metrics:
        def __init__(self):
                pass

        def classification_metrics(self, y_true, y_pred, y_score=None):
                y_true = np.asarray(y_true).astype(int)
                y_pred = np.asarray(y_pred).astype(int)

                out = {}
                out["f1"] = f1_score(y_true, y_pred, zero_division=0)
                out["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
                out["precision"] = precision_score(y_true, y_pred, zero_division=0)
                out["recall"] = recall_score(y_true, y_pred, zero_division=0)

                if y_score is not None:
                        y_score = np.asarray(y_score).astype(float)
                        out["roc_auc"] = roc_auc_score(y_true, y_score)
                        out["pr_auc"] = average_precision_score(y_true, y_score)

                return out


        def spd(self, y_pred, sensitive):
                """
                Statistical Parity Difference:
                max(P(pred=1|group)) - min(P(pred=1|group))
                """
                y_pred = np.asarray(y_pred).astype(int)
                s = np.asarray(sensitive)

                groups = [g for g in np.unique(s) if str(g) != "nan"]
                rates = {}

                for g in groups:
                        mask = (s == g)
                        if mask.sum() == 0:
                                continue
                        rates[str(g)] = float(y_pred[mask].mean())

                if len(rates) < 2:
                        return {"spd": np.nan, "group_positive_rates": rates}

                spd_val = max(rates.values()) - min(rates.values())
                return {"spd": float(spd_val), "group_positive_rates": rates}


        def eod(self, y_true, y_pred, sensitive):
                """
                Equal Opportunity Difference:
                max(TPR_group) - min(TPR_group)
                """
                y_true = np.asarray(y_true).astype(int)
                y_pred = np.asarray(y_pred).astype(int)
                s = np.asarray(sensitive)

                groups = [g for g in np.unique(s) if str(g) != "nan"]
                tprs = {}

                for g in groups:
                        mask = (s == g)
                        pos = mask & (y_true == 1)
                        denom = pos.sum()
                        if denom == 0:
                                tprs[str(g)] = np.nan
                        else:
                                tprs[str(g)] = float(y_pred[pos].mean())

                tprs_clean = {k: v for k, v in tprs.items() if not np.isnan(v)}
                if len(tprs_clean) < 2:
                        return {"eod": np.nan, "group_tpr": tprs}

                eod_val = max(tprs_clean.values()) - min(tprs_clean.values())
                return {"eod": float(eod_val), "group_tpr": tprs}