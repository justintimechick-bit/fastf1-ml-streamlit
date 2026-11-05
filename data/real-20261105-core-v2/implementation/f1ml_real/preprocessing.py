"""Fold-local core preprocessing for the reviewed real-data training path.

The blocked Phase 9 audit does not fit this transformer or any estimator. Its
selection is deliberately part of fit, so later/held-out observations cannot
promote an entirely missing training feature. Legacy synthetic models stay intact.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from f1ml.features.contract import (
    categorical_columns,
    model_matrix,
    numeric_columns,
    schema_version,
)


class ObservedCorePreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, task):
        self.task = task

    def fit(self, X, y=None):
        matrix = model_matrix(X, self.task)
        numeric = numeric_columns(self.task, schema_version(X))
        self.excluded_features_ = [c for c in numeric if not matrix[c].notna().any()]
        self.exclusion_reasons_ = dict.fromkeys(
            self.excluded_features_, "entirely_missing_in_training"
        )
        for column in numeric:
            if column not in self.excluded_features_ and matrix[column].nunique(dropna=False) <= 1:
                self.excluded_features_.append(column)
                self.exclusion_reasons_[column] = "constant_in_training"
        # A missingness flag for a never-observed input is not an observation.
        flags = [
            c
            for c in numeric
            if c.endswith("_missing") and c.removesuffix("_missing") in self.excluded_features_
        ]
        self.excluded_features_ = list(dict.fromkeys(self.excluded_features_ + flags))
        self.exclusion_reasons_.update(
            dict.fromkeys(flags, "parent_feature_unobserved_or_constant")
        )
        self.numeric_features_ = [c for c in numeric if c not in self.excluded_features_]
        self.categorical_features_ = list(categorical_columns(self.task, schema_version(X)))
        self.transformer_ = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    self.numeric_features_,
                ),
                (
                    "category",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.categorical_features_,
                ),
            ]
        )
        self.transformer_.fit(matrix, y)
        self.n_training_rows_ = len(matrix)
        return self

    def transform(self, X):
        check_is_fitted(self, "transformer_")
        values = self.transformer_.transform(model_matrix(X, self.task))
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite core matrix after train-only preprocessing")
        return values

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "transformer_")
        return self.transformer_.get_feature_names_out()
