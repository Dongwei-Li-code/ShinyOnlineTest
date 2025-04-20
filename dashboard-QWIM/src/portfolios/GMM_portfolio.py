import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from pathlib import Path
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
from pypfopt.cla import CLA
from pypfopt.objective_functions import L2_reg
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


# Features preparation

def compute_ewm_DD(ret_ser, hl):
    ret_ser_neg = np.minimum(ret_ser, 0)
    sq_mean = ret_ser_neg.pow(2).ewm(halflife=hl).mean()
    return np.sqrt(sq_mean)


def feature_engineer(ret_ser, ver="v0"):
    if ver == "v0":
        feat_dict = {}
        hls = [5, 20, 30]
        for hl in hls:
            feat_dict[f"ret_{hl}"] = ret_ser.ewm(halflife=hl).mean()
            DD = compute_ewm_DD(ret_ser, hl)
            feat_dict[f"DD-log_{hl}"] = np.log(DD)
            feat_dict[f"sortino_{hl}"] = feat_dict[f"ret_{hl}"].div(DD)
        return pd.DataFrame(feat_dict)

    elif ver == "v1":
        feat_dict = {}
        windows = [6, 14]

        # Feature 1: Observation
        feat_dict["obs"] = ret_ser

        # Feature 2 & 3: Absolute changes
        feat_dict["abs_change"] = ret_ser.diff().abs()
        feat_dict["prev_abs_change"] = ret_ser.diff().shift(1).abs()

        for w in windows:
            half_w = w // 2

            # Feature 4 & 5: Centered Mean & Std (using t-w+1 to t)
            feat_dict[f"centered_mean_{w}"] = ret_ser.rolling(window=w).mean()
            feat_dict[f"centered_std_{w}"] = ret_ser.rolling(window=w).std()

            # Feature 6 & 7: Left Mean & Std (first half of t-w+1 to t)
            feat_dict[f"left_mean_{w}"] = ret_ser.rolling(window=w).apply(lambda x: x[:half_w].mean(), raw=False)
            feat_dict[f"left_std_{w}"] = ret_ser.rolling(window=w).apply(lambda x: x[:half_w].std(), raw=False)

            # Feature 8 & 9: Right Mean & Std (second half of t-w+1 to t)
            feat_dict[f"right_mean_{w}"] = ret_ser.rolling(window=w).apply(lambda x: x[half_w:].mean(), raw=False)
            feat_dict[f"right_std_{w}"] = ret_ser.rolling(window=w).apply(lambda x: x[half_w:].std(), raw=False)

        return pd.DataFrame(feat_dict)


def backtest_portfolio(
        price_data,
        test_period,
        lookback_period=252,
        rebalancing_freq='Y',
        strategy='max_sharpe'
):

    returns = price_data.pct_change().dropna()

    whole_dates = price_data.index

    start_date, end_date = test_period
    test_data = price_data.loc[start_date:end_date]

    # rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalancing_freq)
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq="YE").tolist()
    rebalance_dates.insert(0, whole_dates[whole_dates.searchsorted(start_date, side="right") - 1])
    rebalance_dates = pd.DatetimeIndex(rebalance_dates)

    weights_dict = {}

    for rebalance_date in rebalance_dates:
        train_end = rebalance_date
        # train_start = train_end - pd.Timedelta(days=lookback_period)
        train_start = whole_dates[whole_dates.searchsorted(start_date, side="right") - 1 - lookback_period]

        train_data = price_data.loc[train_start:train_end]

        print(
            f"Rebalancing on {rebalance_date.strftime('%Y-%m-%d')} | Train period: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")

        mu = mean_historical_return(train_data)
        # S = train_data.pct_change().dropna().cov()
        S = train_data.dropna().pct_change().ewm(span=252).cov(pairwise=True).loc[train_data.index[-1]] * 252
        # S = CovarianceShrinkage(train_data).ledoit_wolf()

        # Select the optimal strategy
        ef = EfficientFrontier(mu, S)
        # ef.add_constraint(lambda w: w <= 0.4)
        if strategy == 'max_sharpe':
            ef.max_sharpe()
        elif strategy == 'min_volatility':
            ef.min_volatility()
        elif strategy == 'max_quadratic_utility':
            ef.max_quadratic_utility(risk_aversion=10.0)
        else:
            raise ValueError("Unsupported strategy. Use 'max_sharpe', 'min_volatility', or 'max_quadratic_utility'.")

        # Weights of each asset
        cleaned_weights = ef.clean_weights()
        weights_dict[rebalance_date] = cleaned_weights

    # backtest
    portfolio_returns = []
    dates = []

    for i in range(len(rebalance_dates) - 1):
        start = rebalance_dates[i]
        end = rebalance_dates[i + 1]


        period_returns = returns.loc[start:end]

        # weights of last rebalance
        last_weights = weights_dict[rebalance_dates[i]]
        weight_vector = np.array([last_weights[asset] for asset in price_data.columns])


        period_portfolio_returns = period_returns @ weight_vector
        portfolio_returns.extend(period_portfolio_returns.tolist())
        dates.extend(period_returns.index.tolist())


    portfolio_returns = pd.DataFrame(portfolio_returns, index=dates, columns=['Portfolio Return'])
    portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep='first')]  # 删掉重复的行


    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    portfolio_cumulative = portfolio_cumulative.reindex(price_data.loc[start_date:end_date].index, method='ffill')

    return portfolio_cumulative, portfolio_returns, weights_dict


def rolling_backtest(raw_data, test_start, num_years):
    # rolling return
    return_list = []

    for i in range(num_years):
        train_start = test_start - pd.DateOffset(years=10) + pd.DateOffset(days=1)  # 训练集起始时间
        train_end = test_start  # 训练集结束时间
        test_end = test_start + pd.DateOffset(years=1)  # 测试集结束时间

        # print(f"Iteration {i+1}:")
        # print(f"train_start = {train_start.strftime('%Y-%m-%d')}")
        # print(f"train_end = {train_end.strftime('%Y-%m-%d')}")
        # print(f"test_start = {test_start.strftime('%Y-%m-%d')}")
        # print(f"test_end = {test_end.strftime('%Y-%m-%d')}")
        # print("-" * 40)

        ## strategy
        portfolio_cumulative_without_regime, portfolio_returns, weights_dict = backtest_portfolio(
            raw_data.fillna(method="ffill"),
            test_period=(test_start, test_end),
            lookback_period=2520,
            rebalancing_freq='YE',
            strategy='min_volatility'
        )
        return_list.append(portfolio_returns)

        # test_start move to next year
        test_start = test_start + pd.DateOffset(years=1)

    rolling_returns = pd.concat(return_list)
    rolling_returns = rolling_returns[~rolling_returns.index.duplicated(keep='first')]  # 删掉重复的行

    rolling_cumulative = (1 + rolling_returns).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(rolling_cumulative, label="Backtest Portfolio", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Backtest Portfolio Performance")
    plt.legend()
    plt.grid()
    plt.show()

    return rolling_cumulative, rolling_returns, return_list




def backtest_regime_portfolio(
        labeled_data,
        test_period,
        train_period,
        strategy_name="strategy_1",
        rebalancing_freq='Y',
        strategy='max_sharpe',
        whole=False
):
    price_data = labeled_data.drop(columns=['regime'])

    regime_data = labeled_data['regime']

    returns = price_data.pct_change().dropna()

    start_date, end_date = pd.to_datetime(test_period)
    train_start, train_end = pd.to_datetime(train_period)

    train_returns = returns.loc[train_start:train_end].copy()
    train_regimes = regime_data.loc[train_start:train_end]

    regime_mu_cov = {}
    for regime in train_regimes.unique():
        regime_mask = train_regimes == regime

        regime_returns = train_returns[regime_mask]

        # if strategy_name == "strategy_1":
        #   continue
        if strategy_name == "strategy_2":
            mean_returns = regime_returns.mean()
            selected_assets = mean_returns[mean_returns > 0].index.tolist()
            regime_returns = regime_returns[selected_assets]

        if len(regime_returns) < 30:
            print(f"Skipping regime {regime}")
            continue

        mu = mean_historical_return(regime_returns, returns_data=True)
        S = CovarianceShrinkage(regime_returns, returns_data=True).ledoit_wolf()

        regime_mu_cov[regime] = (mu, S)

    if whole:
        whole_dates = price_data.index

        rebalance_dates = whole_dates[
            (whole_dates.searchsorted(pd.date_range(start=start_date, end=end_date, freq="YE")) - 1).clip(0)]
        rebalance_dates = rebalance_dates.tolist()
        rebalance_dates.insert(0, whole_dates[whole_dates.searchsorted(start_date, side="right") - 1])
        rebalance_dates = pd.DatetimeIndex(rebalance_dates)

    else:

        rebalance_dates = [d for d in pd.date_range(start=start_date, end=end_date, freq=rebalancing_freq) if
                           d in returns.index]
        regime_changes = regime_data.loc[start_date:end_date].shift(1) != regime_data.loc[start_date:end_date]
        regime_change_dates = regime_data.loc[start_date:end_date].index[regime_changes]
        rebalance_dates.extend(regime_change_dates)
        rebalance_dates = sorted(set(rebalance_dates))

    weights_dict = {}
    portfolio_returns = []
    dates = []

    for i in range(len(rebalance_dates)):
        rebalance_date = rebalance_dates[i]

        current_regime = regime_data.loc[rebalance_date]

        if current_regime in regime_mu_cov:
            mu, S = regime_mu_cov[current_regime]

        else:
            print(f"Skipping rebalancing on {rebalance_date} (no training data for regime {current_regime})")
            weights_dict[rebalance_date] = {asset: 0 for asset in price_data.columns}
            continue

        selected_assets = mu.index.tolist()

        ef = EfficientFrontier(mu, S)
        # ef.add_constraint(lambda w: w <= 0.4)  # maximum weights: 0.4
        if strategy == 'max_sharpe':
            ef.max_sharpe()
        elif strategy == 'min_volatility':
            ef.min_volatility()
        elif strategy == 'max_quadratic_utility':
            ef.max_quadratic_utility(risk_aversion=10.0)
        else:
            raise ValueError("Unsupported strategy. Use 'max_sharpe', 'min_volatility', or 'max_quadratic_utility'.")
        cleaned_weights = ef.clean_weights()

        # **unselected asset weight will be 0**
        full_weights = {asset: cleaned_weights.get(asset, 0) for asset in price_data.columns}
        weights_dict[rebalance_date] = full_weights

        # print(f"Rebalancing on {rebalance_date.strftime('%Y-%m-%d')} | Regime: {current_regime}")

        end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else end_date
        period_returns = returns.loc[rebalance_date:end]
        weight_vector = np.array([full_weights.get(asset, 0) for asset in price_data.columns])

        period_portfolio_returns = period_returns @ weight_vector
        portfolio_returns.extend(period_portfolio_returns.tolist())
        dates.extend(period_returns.index.tolist())

    portfolio_returns = pd.Series(portfolio_returns, index=dates).dropna()
    portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep='first')]  # 删掉重复的行

    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    portfolio_cumulative = portfolio_cumulative.reindex(price_data.loc[start_date:end_date].index, method='ffill')

    return portfolio_cumulative, portfolio_returns, weights_dict


def rolling_backtest_regime(features, raw_data, test_start, num_years, strategy_name="strategy_1",
                            whole=False, train_start = "2002-01-03", train_end = "2011-12-31", test_end = "2012-12-31"):
    return_list = []
    weights_list = []

    for i in range(num_years):
        train_start = test_start - pd.DateOffset(years=10) + pd.DateOffset(days=1)
        train_end = test_start
        test_end = test_start + pd.DateOffset(years=1)

        X_train = features.loc[train_start:test_start]
        X_test = features.loc[test_start:test_end]

        train_start, train_end = X_train.index[0], X_train.index[-1]
        test_start, test_end = X_test.index[0], X_test.index[-1]

        """
        market regimes cluster for each year
        """
        gmm = GaussianMixture(n_components=2)
        gmm.fit(X_train)
        gmm_labels = gmm.predict(X_train)
        gmm_labels = pd.Series(gmm_labels, index=X_train.index, name="regime")

        labels = gmm.predict(X_test)
        labels_test_online = pd.Series(labels, index=X_test.index, name="regime")

        # labeled_data = pd.DataFrame(labels_test_online).join(raw_data, how='left')
        labeled_data = pd.DataFrame(pd.concat([gmm_labels, labels_test_online], axis=0), columns=["regime"]).join(
            raw_data, how='left')

        labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]

        labeled_start, labeled_end = labeled_data.index[[0, -1]]

        portfolio_cumulative_with_regime, portfolio_returns, weights_dict = backtest_regime_portfolio(
            labeled_data.fillna(method="ffill"),
            test_period=(test_start, test_end),
            train_period=(train_start, train_end),
            strategy_name=strategy_name,
            rebalancing_freq='YE',
            strategy='min_volatility',
            whole=whole
        )

        return_list.append(portfolio_returns)
        weights_list.append(pd.DataFrame(weights_dict).T)

        test_start = test_start + pd.DateOffset(years=1)

    rolling_returns = pd.concat(return_list)
    rolling_weights = pd.concat(weights_list)
    rolling_returns = rolling_returns[~rolling_returns.index.duplicated(keep='first')]  # 删掉重复的行
    rolling_weights = rolling_weights[~rolling_weights.index.duplicated(keep='first')]  # 删掉重复的行

    rolling_cumulative = (1 + rolling_returns).cumprod()
    return rolling_cumulative, rolling_returns, rolling_weights


def portfolio_metrics(cumulative_returns: pd.Series, risk_free_rate: float = 0.02):

    daily_returns = cumulative_returns.pct_change().dropna()

    total_return = cumulative_returns.iloc[-1] / cumulative_returns.iloc[0] - 1

    annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1

    annualized_volatility = daily_returns.std() * np.sqrt(252)

    # maximum drawdown
    rolling_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / rolling_max - 1
    max_drawdown = drawdown.min()

    end_of_max_dd = drawdown.idxmin()  # 取最小值对应的日期
    drawdown_recovery = (cumulative_returns == rolling_max) & (cumulative_returns.index <= end_of_max_dd)
    start_of_max_dd = drawdown_recovery[drawdown_recovery].index.min()  # 找到第一个最大值回撤点

    if pd.isna(start_of_max_dd):
        max_drawdown_duration = 0
    else:
        max_drawdown_duration = (end_of_max_dd - start_of_max_dd).days

    excess_daily_returns = daily_returns - risk_free_rate / 252
    sharpe_ratio = excess_daily_returns.mean() / excess_daily_returns.std() * np.sqrt(252)

    downside_returns = daily_returns[daily_returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252)
    sortino_ratio = excess_daily_returns.mean() / downside_volatility if downside_volatility != 0 else np.nan

    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Max Drawdown": max_drawdown,
        "Max Drawdown Duration (days)": max_drawdown_duration,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
    }


def main():

    current_file = Path(__file__).resolve()
    # project_root = current_file.parents[2]
    project_root = current_file.parents[2]  # Mac: /Users/waywardxiao/PycharmProjects/qwim
    excel_path = project_root / 'data' / 'raw' / 'indices.xlsx'
    GMM_benchmark_output_path = project_root / "data" / "processed" / "JM_benchmark_portfolio_values.csv"
    GMM_portfolio_output_path = project_root / "data" / "processed" / "GMM_portfolio_values.csv"
    GMM_weights_output_path = project_root / "data" / "raw" / "GMM_portfolio_weights_ETFs.csv"

    raw_data = pd.read_excel(excel_path, index_col=0, parse_dates=True)
    raw_data = raw_data.drop("SP500", axis=1)
    SP500 = raw_data["SPTR"]
    SP500_ret = np.log(SP500 / SP500.shift(1))
    features = feature_engineer(ret_ser=SP500_ret, ver="v1")



    rolling_cumulative, rolling_returns, return_list = rolling_backtest(raw_data, test_start=pd.to_datetime("2008-01-01"),
                                                                        num_years=14)

    rolling_cumulative_regime_1, rolling_returns1, rolling_weights1 = rolling_backtest_regime(features, raw_data, test_start=pd.to_datetime("2008-01-01"), num_years=14,strategy_name="strategy_1",whole=False)

    rolling_cumulative_regime_2, rolling_returns2,rolling_weights2 = rolling_backtest_regime(features, raw_data, test_start=pd.to_datetime("2008-01-01"), num_years=14,strategy_name="strategy_2",whole=False)

    rolling_cumulative_regime_3, rolling_returns3, rolling_weights3 = rolling_backtest_regime(features, raw_data, test_start=pd.to_datetime("2008-01-01"), num_years=14,strategy_name="strategy_1",whole=True)

    rolling_cumulative_regime_4, rolling_returns4, rolling_weights4 = rolling_backtest_regime(features, raw_data, test_start=pd.to_datetime("2008-01-01"), num_years=14, strategy_name="strategy_2", whole=True)

    strategy_returns = pd.DataFrame(
        {
            "annually without regime": portfolio_metrics(rolling_cumulative["Portfolio Return"]),
            "regime-based—strategy1": portfolio_metrics(rolling_cumulative_regime_1),
            "regime-based—strategy2": portfolio_metrics(rolling_cumulative_regime_2),
            "regime-based—strategy3": portfolio_metrics(rolling_cumulative_regime_3),
            "regime-based—strategy4": portfolio_metrics(rolling_cumulative_regime_4)

        }
    )

    rolling_cumulative.index.name = "Date"
    rolling_cumulative_regime_2.index.name = "Date"
    rolling_cumulative_regime_2.name = 'Portfolio Return'
    rolling_weights2.index.name = 'Date'


    GMM_benchmark = pd.DataFrame(index=rolling_cumulative.index, data={"Value":rolling_cumulative['Portfolio Return']})

    GMM_portfolio = pd.DataFrame(index=rolling_cumulative_regime_2.index,
                                data={"Value": rolling_cumulative_regime_2})

    rolling_weights2 = pd.DataFrame(rolling_weights2)

    #GMM_benchmark.to_csv(GMM_benchmark_output_path)
    GMM_portfolio.to_csv(GMM_portfolio_output_path)
    rolling_weights2.to_csv(GMM_weights_output_path)

if __name__ == "__main__":
    print("Hello from dashboard-qwim!")
    main()

