"""Monthly economic covariates with a fixed, fold-local information cutoff."""

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from src.cv import DATE_COL
from src.data_cleaning import load
from .base import FeatureBlock
from .registry import register_feature

COE_COLUMNS = [f"MACRO_COE_{category}_LAG1" for category in ("A", "B", "C", "E")]
STOCK_COLUMNS = ["MACRO_SGX_MONTHLY_PRICE_INDEX_LAG1"]


class EconomyFeatures(FeatureBlock):
    """Use t-1 observations for training and forecasts beyond the fitted cutoff.

    At origin month M, only complete months through M-1 are read. There is no
    access to later real COE/stock observations, including during transform.
    """

    def __init__(self, include_coe=True, include_stocks=True, forecast="last",
                 trend_window=12, damping=0.8, max_log_slope=0.03):
        self.include_coe = include_coe
        self.include_stocks = include_stocks
        self.forecast = forecast
        self.trend_window = trend_window
        self.damping = damping
        self.max_log_slope = max_log_slope

    def fit(self, train_df):
        if self.forecast not in ("last", "damped"):
            raise ValueError("forecast must be last or damped")
        if self.trend_window < 2 or not 0 < self.damping < 1 or self.max_log_slope < 0:
            raise ValueError("Invalid economic forecast settings")
        self.origin_ = pd.Period(train_df[DATE_COL].max(), freq="M")
        self.available_through_ = self.origin_ - 1
        self.audit_ = {"origin": str(self.origin_), "available_through": str(self.available_through_),
                       "publication_assumption": "one complete calendar month of delay",
                       "future_real_observations_used": False, "forecast": self.forecast,
                       "trend_window": self.trend_window, "damping": self.damping,
                       "max_log_slope": self.max_log_slope}
        tables = []
        if self.include_coe:
            coe = load("auxiliary/coe_prices.csv")
            # Filter BEFORE any aggregation, selection, imputation, or forecasting.
            coe = coe.loc[coe.YEAR_MONTH.le(str(self.available_through_))].copy()
            monthly = coe.groupby(["YEAR_MONTH", "CATEGORY"]).QUOTA_PREMIUM.mean().unstack("CATEGORY")
            monthly = monthly.reindex(columns=["a", "b", "c", "e"])
            monthly.columns = COE_COLUMNS
            monthly.index = pd.PeriodIndex(monthly.index, freq="M")
            tables.append(monthly)
            self.audit_["coe_last_observation"] = coe.YEAR_MONTH.max()
        if self.include_stocks:
            stocks = load("auxiliary/stock_prices.csv")
            stocks = stocks.loc[stocks.MARKET.eq("sgx") & stocks.YEAR_MONTH.le(str(self.available_through_))].copy()
            stocks = stocks.loc[stocks.CLOSE.gt(0) & stocks.CLOSE.notna()]
            monthly = stocks.groupby(["YEAR_MONTH", "SYMBOL"]).CLOSE.mean().unstack("SYMBOL")
            monthly.index = pd.PeriodIndex(monthly.index, freq="M")
            monthly = monthly.sort_index()
            # Fix membership and normalization in the first observed month, not
            # by future performance/coverage. All prices are SGD and unadjusted.
            anchor = monthly.index.min()
            panel = monthly.columns[monthly.loc[anchor].notna()].tolist()
            months = pd.period_range(anchor, self.available_through_, freq="M")
            prices = monthly.reindex(index=months, columns=panel).ffill()
            normalized = prices.div(prices.loc[anchor], axis=1) * 100.0
            index = normalized.mean(axis=1).to_frame(STOCK_COLUMNS[0])
            tables.append(index)
            self.audit_.update(sgx_panel=panel, stock_anchor_month=str(anchor),
                               stock_aggregation="daily CLOSE mean per symbol/month, then equal-weight normalized index",
                               stock_last_observation=stocks.YEAR_MONTH.max(),
                               adjusted_close_used=False, mixed_currencies=False)
        if not tables:
            raise ValueError("Enable at least one economy source")
        history = pd.concat(tables, axis=1).sort_index()
        months = pd.period_range(history.index.min(), self.available_through_, freq="M")
        self.history_ = history.reindex(months).ffill()
        if self.history_.iloc[-1].isna().any():
            raise ValueError("Economic series missing at fitted information cutoff")
        self.slopes_ = {}
        for column in self.history_:
            values = self.history_[column].dropna().iloc[-self.trend_window:]
            if (values <= 0).any():
                raise ValueError("Economic levels must be positive")
            slope = np.polyfit(np.arange(len(values)), np.log(values.to_numpy()), 1)[0] if len(values) > 1 else 0.0
            self.slopes_[column] = float(np.clip(slope, -self.max_log_slope, self.max_log_slope))
        self.audit_["log_slopes"] = self.slopes_.copy()
        self.audit_["columns"] = list(self.history_.columns)
        return self

    def transform(self, df):
        check_is_fitted(self, "history_")
        months = pd.PeriodIndex(df[DATE_COL], freq="M") - 1
        result = self.history_.reindex(months).copy()
        future = months > self.available_through_
        if future.any():
            horizons = months[future].asi8 - self.available_through_.ordinal
            for column in self.history_:
                last = self.history_[column].iloc[-1]
                if self.forecast == "last":
                    values = np.full(future.sum(), last)
                else:
                    accumulated = self.damping * (1 - self.damping ** horizons) / (1 - self.damping)
                    values = last * np.exp(self.slopes_[column] * accumulated)
                result.loc[future, column] = values
        result.index = df.index
        return result


register_feature("macro", EconomyFeatures)
register_feature("macro_coe", lambda: EconomyFeatures(include_stocks=False))
register_feature("macro_stocks", lambda: EconomyFeatures(include_coe=False))
register_feature("macro_damped", lambda: EconomyFeatures(forecast="damped"))
