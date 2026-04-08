# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 00:22:37 2026

@author: gerar
"""

import pandas as pd
import numpy as np
import os
import re


class LoadDataset:
        def __init__(self, random_state=42):
                # Get project root = parent of src/
                self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.directory = os.path.join(self.base_dir, "dataset")

                # Census data and columns
                self.datafiles = ["census-bureau.data"]
                self.column_file = "census-bureau.columns"

                self.alldataset = {}

                # Make random generator deterministic
                self.random_state = random_state
                self.rng = np.random.default_rng(self.random_state)

        def load_data(self, encode=False):
                # Reset RNG each call so repeated load_data(encode=False) then load_data(encode=True)
                # produces consistent imputations
                self.rng = np.random.default_rng(self.random_state)

                # Reset container each call to avoid stale data
                self.alldataset = {}

                for files in self.datafiles:
                        key = files.split(".")[0]  # "census-bureau"

                        # load column names
                        columns = self.load_columns(self.column_file)

                        # read data
                        data = pd.read_csv(
                                os.path.join(self.directory, files),
                                header=None,
                                names=columns,
                                sep=",",
                                skipinitialspace=True,
                                na_values=["?", "??", "?,?,?", ""]
                        )

                        # preprocess and store
                        self.alldataset[key] = self.PreprocessingData(key, data, encode=encode)

                return self.alldataset

        def load_columns(self, columnfile):
                with open(os.path.join(self.directory, columnfile), "r", encoding="utf-8", errors="ignore") as f:
                        columns = [line.strip() for line in f if line.strip()]
                return columns

        def probability_distribution_by_column(self, data, column_name):
                """
                Impute missing values for a categorical column using probability distribution sampling.
                Uses self.rng (seeded) for determinism.
                """
                # Normalize frequency of unique values in the feature column (excluding NaN)
                p_values = data[column_name].dropna().value_counts(normalize=True)

                # Find missing locations in the column
                missing_values = data[column_name].isnull()

                # If nothing to sample from, or nothing missing, skip safely
                if p_values.empty or missing_values.sum() == 0:
                        return data

                # Sample values equal to # missing values with probability obtained in p_values
                data.loc[missing_values, column_name] = self.rng.choice(
                        p_values.index.to_numpy(),
                        size=int(missing_values.sum()),
                        p=p_values.values
                )
                return data

        def PreprocessingData(self, key, data, encode=False):
                # 1) Normalize column names: spaces -> underscore, remove apostrophes
                data.columns = [col.strip().replace(" ", "_").replace("'", "") for col in data.columns]

                # 2) Clean string columns: strip and remove trailing periods, preserve NaN
                obj_cols = data.select_dtypes(include=["object"]).columns
                for col in obj_cols:
                        data[col] = (
                                data[col]
                                .astype(str)
                                .str.strip()
                                .str.replace(r"\.$", "", regex=True)
                                .replace("nan", np.nan)
                        )

                # 3) Label and weight columns
                label_col = "label"
                weight_col = "weight"

                # 4) Convert weight to numeric and fill missing with median (NOT 1 unless median is NaN)
                if weight_col in data.columns:
                        data[weight_col] = pd.to_numeric(data[weight_col], errors="coerce")
                        median_weight = data[weight_col].median()
                        if pd.isna(median_weight):
                                median_weight = 1.0
                        data[weight_col] = data[weight_col].fillna(median_weight)

                # 4b) Numeric median imputation (skip weight + target)
                num_cols = data.select_dtypes(include=["int64", "float64"]).columns
                for col in num_cols:
                        if col in [weight_col, "income_gt_50k"]:
                                continue
                        if data[col].isnull().sum() > 0:
                                median_value = data[col].median()
                                data[col] = data[col].fillna(median_value)

                # 5) Probability-distribution imputation ONLY for categorical features (objects)
                cat_cols = data.select_dtypes(include=["object"]).columns
                for col in cat_cols:
                        if col == label_col:
                                continue
                        if data[col].isnull().sum() > 0:
                                data = self.probability_distribution_by_column(data, col)

                # 6) Convert label column to binary target
                def convert_label(value):
                        if pd.isna(value):
                                return 0
                        value = str(value).strip()
                        if "50000+" in value or "+" in value or re.search(r">\s*50", value):
                                return 1
                        return 0

                if label_col in data.columns:
                        data["income_gt_50k"] = data[label_col].apply(convert_label).astype(int)
                else:
                        # fallback if label column missing
                        data["income_gt_50k"] = 0

                # 7) Encode categorical features: encode=False for EDA, encode=True for modeling
                if encode is True:
                        cat_cols = data.select_dtypes(include=["object"]).columns
                        for col in cat_cols:
                                if col == label_col:
                                        continue
                                # IMPORTANT: assign back (otherwise no-op)
                                data[col] = data[col].astype("category")

                return data