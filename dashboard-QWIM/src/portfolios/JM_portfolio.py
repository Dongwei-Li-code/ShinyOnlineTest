import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
from pypfopt.cla import CLA
from pypfopt.objective_functions import L2_reg
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

from jumpmodels.utils import filter_date_range        # useful helpers
from jumpmodels.jump import JumpModel                 # class of JM & CJM
from jumpmodels.sparse_jump import SparseJumpModel    # class of Sparse JM


current_file = Path(__file__).resolve()
project_root = current_file.parents[2]  # 向上两层：src → dashboard-QWIM
excel_path = project_root / 'data' / 'raw' / 'indices.xlsx'
JM_benchmark_output_path = project_root / "data" / "processed" / "JM_benchmark_portfolio_values.csv"
JM_portfolio_output_path = project_root / "data" / "processed" / "JM_portfolio_values.csv"
JM_weights_output_path = project_root / "data" / "raw" / "JM_portfolio_weights_ETFs.csv"

raw_data = pd.read_excel(excel_path, index_col=0, parse_dates=True)
raw_data = raw_data.drop("SP500",axis=1)
SP500 = raw_data["SPTR"]
SP500_ret = np.log(SP500 / SP500.shift(1))

def backtest_portfolio(
    price_data, 
    test_period, 
    lookback_period=252, 
    rebalancing_freq='Y', 
    strategy='max_sharpe'
):
    """
    Perform dynamic portfolio optimization based on pypfopt and backtest the returns.

    Parameters:
    - price_data: DataFrame, daily price data of multiple assets, with index as dates and columns as asset names
    - test_period: tuple, (start date of test set, end date of test set)
    - lookback_period: int, length of the training lookback window (default is 252 trading days)
    - rebalancing_freq: str, rebalancing frequency ('M': monthly, 'Q': quarterly, 'Y': yearly)
    - strategy: str, portfolio optimization strategy ('min_volatility': minimum volatility, 'max_sharpe': maximum Sharpe ratio)

    Returns:
    - Plots the return curve and returns the final performance
    """

    # calculate daily returns
    returns = price_data.pct_change().dropna()

    # extract dates
    whole_dates = price_data.index
    
    # Specify the time range for the test set
    start_date, end_date = test_period
    test_data = price_data.loc[start_date:end_date]
    
    # get the rebalance dates
    # rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalancing_freq)
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq="YE").tolist()
    rebalance_dates.insert(0, whole_dates[whole_dates.searchsorted(start_date, side="right") - 1])
    rebalance_dates = pd.DatetimeIndex(rebalance_dates)
    rebalance_dates
    
    # initialize the weights dict
    weights_dict = {}
    
    for rebalance_date in rebalance_dates:
        # get the training data
        train_end = rebalance_date
        # train_start = train_end - pd.Timedelta(days=lookback_period)
        train_start = whole_dates[whole_dates.searchsorted(start_date, side="right") - 1 -lookback_period]
        
        train_data = price_data.loc[train_start:train_end]


        # rebalance and optimization
        print(f"Rebalancing on {rebalance_date.strftime('%Y-%m-%d')} | Train period: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        
        # calcualte return vector and covariance matrix
        mu = mean_historical_return(train_data)
        # S = train_data.pct_change().dropna().cov()
        S = train_data.dropna().pct_change().ewm(span=252).cov(pairwise=True).loc[train_data.index[-1]]*252
        # S = CovarianceShrinkage(train_data).ledoit_wolf()
        
        # choose one optimization method
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
        
        # Store the portfolio weights for each asset
        cleaned_weights = ef.clean_weights()
        weights_dict[rebalance_date] = cleaned_weights
    
    # initialize the strategy returns
    portfolio_returns = []
    dates = []
    
    for i in range(len(rebalance_dates) - 1):
        start = rebalance_dates[i]
        end = rebalance_dates[i + 1]
        
        # if end not in returns.index:
        #     continue
        
        # Get asset returns over the current time window
        period_returns = returns.loc[start:end]
        
        # Get asset weights from the last rebalancing
        last_weights = weights_dict[rebalance_dates[i]]
        weight_vector = np.array([last_weights[asset] for asset in price_data.columns])
        
        # calculate the returns of the current time window
        period_portfolio_returns = period_returns @ weight_vector
        portfolio_returns.extend(period_portfolio_returns.tolist())
        dates.extend(period_returns.index.tolist())
    
    # convert to DataFrame
    portfolio_returns = pd.DataFrame(portfolio_returns, index=dates, columns=['Portfolio Return'])
    portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep='first')]  # delete the repeating rows
    
    # calculate the cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    portfolio_cumulative = portfolio_cumulative.reindex(price_data.loc[start_date:end_date].index, method='ffill')  
    
    # # plotting
    # plt.figure(figsize=(12, 6))
    # plt.plot(portfolio_cumulative, label="Backtest Portfolio", linewidth=2)
    # plt.xlabel("Date")
    # plt.ylabel("Cumulative Return")
    # plt.title("Backtest Portfolio Performance")
    # plt.legend()
    # plt.grid()
    # plt.show()
    
    # # calculate quant metrics
    # total_return = portfolio_cumulative.iloc[-1, 0] - 1
    # annualized_return = portfolio_returns.mean()[0] * 252
    # annualized_volatility = portfolio_returns.std()[0] * np.sqrt(252)
    # sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else np.nan
    
    # print(f"Total Return: {total_return:.2%}")
    # print(f"Annualized Return: {annualized_return:.2%}")
    # print(f"Annualized Volatility: {annualized_volatility:.2%}")
    # print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    return portfolio_cumulative,portfolio_returns,weights_dict


def rolling_backtest(test_start, num_years):
    # rolling return
    return_list = []

    for i in range(num_years):
        train_start = test_start - pd.DateOffset(years=10) + pd.DateOffset(days=1)  # start date of the training set
        train_end = test_start  # end date of the training set
        test_end = test_start + pd.DateOffset(years=1)  # end date of the test set
        
        # print(f"Iteration {i+1}:")
        # print(f"train_start = {train_start.strftime('%Y-%m-%d')}")
        # print(f"train_end = {train_end.strftime('%Y-%m-%d')}")
        # print(f"test_start = {test_start.strftime('%Y-%m-%d')}")
        # print(f"test_end = {test_end.strftime('%Y-%m-%d')}")
        # print("-" * 40)

        ## strategy
        portfolio_cumulative_without_regime,portfolio_returns,weights_dict = backtest_portfolio(
            raw_data.fillna(method="ffill"), 
            test_period=(test_start, test_end), 
            lookback_period=2520, 
            rebalancing_freq='YE', 
            strategy='min_volatility'
        )
        return_list.append(portfolio_returns)


        # test_start Shift the time window forward by one year
        test_start = test_start + pd.DateOffset(years=1)

    rolling_returns = pd.concat(return_list)
    rolling_returns = rolling_returns[~rolling_returns.index.duplicated(keep='first')]  # deltet the duplicate rows

    # calculate the cumulative returns
    rolling_cumulative = (1 + rolling_returns).cumprod()

    # plt.figure(figsize=(12, 6))
    # plt.plot(rolling_cumulative, label="Backtest Portfolio", linewidth=2)
    # plt.xlabel("Date")
    # plt.ylabel("Cumulative Return")
    # plt.title("Backtest Portfolio Performance")
    # plt.legend()
    # plt.grid()
    # plt.show()

    return rolling_cumulative, rolling_returns, return_list

def compute_ewm_DD(ret_ser: pd.Series, hl: float) -> pd.Series:
    """
    Compute the exponentially weighted moving downside deviation (DD) for a return series.

    The downside deviation is calculated as the square root of the exponentially 
    weighted second moment of negative returns.

    Parameters
    ----------
    ret_ser : pd.Series
        The input return series.

    hl : float
        The halflife parameter for the exponentially weighted moving average.

    Returns
    -------
    pd.Series
        The exponentially weighted moving downside deviation for the return series.
    """
    ret_ser_neg: pd.Series = np.minimum(ret_ser, 0.)
    sq_mean = ret_ser_neg.pow(2).ewm(halflife=hl).mean()
    return np.sqrt(sq_mean)


# reviewed
def feature_engineer(ret_ser: pd.Series, ver: str = "v0") -> pd.DataFrame:
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
    

features = feature_engineer(ret_ser=SP500_ret, ver="v1")



def sharpe_tario(cumulative_returns: pd.Series, risk_free_rate: float = 0.02):
    """
    calculate sharpe ratio
    """
    # calculate daily_returns
    daily_returns = cumulative_returns.pct_change().dropna()
    
    # calculate the sharpe ratio
    excess_daily_returns = daily_returns - risk_free_rate / 252
    sharpe_ratio = excess_daily_returns.mean() / excess_daily_returns.std() * np.sqrt(252)
    
    # # calculate the sortino_ratio
    # downside_returns = daily_returns[daily_returns < 0]
    # downside_volatility = downside_returns.std() * np.sqrt(252)
    # sortino_ratio = excess_daily_returns.mean() / downside_volatility if downside_volatility != 0 else np.nan
    
    return sharpe_ratio

def backtest_regime_portfolio_1(
    labeled_data, 
    test_period, 
    train_period,  
    rebalancing_freq='Y', 
    strategy='max_sharpe'
):
    """
    Perform dynamic portfolio optimization based on pypfopt and backtest the returns 
    (supports rebalancing based on changing market regimes).

    Parameters:
    - labeled_data: DataFrame, contains asset prices and market regimes; index is time, columns include asset prices and a "regime" column
    - test_period: tuple, (start date of test set, end date of test set)
    - train_period: tuple, (start date of training set, end date of training set) 
      used to calculate `mu` and `covariance` for each `regime`
    - rebalancing_freq: str, regular rebalancing frequency ('M': monthly, 'Q': quarterly, 'Y': yearly)
    - strategy: str, portfolio optimization strategy ('min_volatility', 'max_sharpe', 'max_quadratic_utility')

    Returns:
    - portfolio_cumulative: DataFrame, cumulative returns of the portfolio
    - portfolio_returns: DataFrame, daily returns of the portfolio
    - weights_dict: dict, portfolio weights for each rebalancing
    """
    # Split the price data and regime labels
    price_data = labeled_data.drop(columns=['regime'])
    regime_data = labeled_data['regime']
    
    # calculate the daily returns
    returns = price_data.pct_change().dropna()

    # extract the time range
    start_date, end_date = test_period
    train_start, train_end = train_period  

    # Calculate mu and covariance for each regime within the training period
    train_returns = returns.loc[train_start:train_end].copy()
    train_regimes = regime_data.loc[train_start:train_end]

    regime_mu_cov = {}  # store the mu and covariance for each regime
    for regime in train_regimes.unique():
        regime_mask = train_regimes == regime
        # print(len(train_returns),len(regime_mask))
        # print("the index of price_data:", price_data.index[0], price_data.index[-1])
        # print("the index of regime_data:", regime_data.index[0], regime_data.index[-1])
        # print("the index of returns:", returns.index[0], returns.index[-1])
        # print("the index of regime_data:", regime_data.index[0], regime_data.index[-1])
        # print("the index of train_returns:", train_returns.index[0], train_returns.index[-1])
        # print("the index of train_regimes:", train_regimes.index[0], train_regimes.index[-1])
        # print("the index of regime_mask:", regime_mask.index[0], regime_mask.index[-1])
        regime_returns = train_returns[regime_mask[train_returns.index[0]:train_returns.index[-1]]]  # get the returns for this specific regime

        if len(regime_returns) < 30:
            print(f"Skipping regime {regime} in train_period (not enough data)")
            continue

        mu = mean_historical_return(regime_returns, returns_data=True)
        S = CovarianceShrinkage(regime_returns, returns_data=True).ledoit_wolf()
        regime_mu_cov[regime] = (mu, S)

    # specify the rebalance dates
    rebalance_dates = [d for d in pd.date_range(start=start_date, end=end_date, freq=rebalancing_freq) if d in returns.index]

    # Rebalance only on the day after regime changes occurring within the test period
    regime_changes = regime_data.loc[start_date:end_date].shift(1) != regime_data.loc[start_date:end_date]
    regime_change_dates = regime_data.loc[start_date:end_date].index[regime_changes]
    rebalance_dates.extend(regime_change_dates)
    rebalance_dates = sorted(set(rebalance_dates))  # Remove duplicates & sort
    
    # initialize portfolio weights and returns
    weights_dict = {}
    portfolio_returns = []
    dates = []

    for i in range(len(rebalance_dates)):
        rebalance_date = rebalance_dates[i]
        # get the current regime
        current_regime = regime_data.loc[rebalance_date]
        # Use `mu` and `covariance` from the training period for the current regime
        if current_regime in regime_mu_cov:
            mu, S = regime_mu_cov[current_regime]
        else:
            print(f"Skipping rebalancing on {rebalance_date} (no training data for regime {current_regime})")
            weights_dict[rebalance_date] = {asset: 0 for asset in price_data.columns}  # Go to cash
            continue

        selected_assets = mu.index.tolist()

        # strategy optimization
        ef = EfficientFrontier(mu, S)
        # ef.add_constraint(lambda w: w <= 0.4)  # Cap individual asset weights at 40%
        if strategy == 'max_sharpe':
            ef.max_sharpe()
        elif strategy == 'min_volatility':
            ef.min_volatility()
        elif strategy == 'max_quadratic_utility':
            ef.max_quadratic_utility(risk_aversion=10.0)
        else:
            raise ValueError("Unsupported strategy. Use 'max_sharpe', 'min_volatility', or 'max_quadratic_utility'.")
        cleaned_weights = ef.clean_weights()

        # ensure weights of unselected assets are zero
        full_weights = {asset: cleaned_weights.get(asset, 0) for asset in price_data.columns}
        weights_dict[rebalance_date] = full_weights

        # print(f"Rebalancing on {rebalance_date.strftime('%Y-%m-%d')} | Regime: {current_regime}")

        # calculate the returns for all assets
        end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else end_date
        period_returns = returns.loc[rebalance_date:end]
        weight_vector = np.array([full_weights.get(asset, 0) for asset in price_data.columns])

        # calculate the portfolio returns
        period_portfolio_returns = period_returns @ weight_vector
        portfolio_returns.extend(period_portfolio_returns.tolist())
        dates.extend(period_returns.index.tolist())

    # organize the portfolio return series
    portfolio_returns = pd.Series(portfolio_returns, index=dates).dropna()
    portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep='first')]  # delete the duplicate rows

    # calculate the cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    portfolio_cumulative = portfolio_cumulative.reindex(price_data.loc[start_date:end_date].index, method='ffill')

    # # plotting
    # plt.figure(figsize=(12, 6))
    # plt.plot(portfolio_cumulative, label="Backtest Portfolio", linewidth=2)
    # plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7) 
    # plt.xlabel("Date"), plt.ylabel("Cumulative Return"), plt.title("Backtest Portfolio Performance")
    # plt.legend(), plt.grid(), plt.show()

    return portfolio_cumulative, portfolio_returns, weights_dict


def rolling_dynamic_backtest_regime_1(test_start, num_years):
    # rolling return
    return_list = []
    weights_list = []

    for i in range(num_years):
        train_start = test_start - pd.DateOffset(years=10) + pd.DateOffset(days=1)  # start date of the training set
        train_end = test_start  # end date of the training set
        test_end = test_start + pd.DateOffset(years=1)  # end date of the test set
        
        # print(f"Iteration {i+1}:")
        # print(f"train_start = {train_start.strftime('%Y-%m-%d')}")
        # print(f"train_end = {train_end.strftime('%Y-%m-%d')}")
        # print(f"test_start = {test_start.strftime('%Y-%m-%d')}")
        # print(f"test_end = {test_end.strftime('%Y-%m-%d')}")
        # print("-" * 40)
        
        ## train-test split
        X_train = filter_date_range(features, start_date=train_start, end_date=test_start)
        X_test = filter_date_range(features, start_date=test_start, end_date=test_end)
        # print time split
        train_start, train_end = X_train.index[[0, -1]]
        test_start, test_end = X_test.index[[0, -1]]
        # print("Training starts at:", train_start, "and ends at:", train_end)
        # print("Testing starts at:", test_start, "and ends at:", test_end)
        # print("-" * 40)

        ## preproscessing
        from jumpmodels.preprocess import StandardScalerPD, DataClipperStd
        clipper = DataClipperStd(mul=3.)
        scalar = StandardScalerPD()
        # fit on training data
        X_train_processed = scalar.fit_transform(clipper.fit_transform(X_train))
        # transform the test data
        X_test_processed = scalar.transform(clipper.transform(X_test))

        # ## regime detection
        # jump_penalty=60.
        # jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )
        # jm.fit(X_train_processed, SP500_ret, sort_by="cumret")
        # labels_test_online = jm.predict_online(X_test_processed.dropna())  # make online inference
        # labeled_data = pd.DataFrame(pd.concat([jm.labels_,labels_test_online]),columns=["regime"]).join(raw_data, how='left')
        # labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]
        # labeled_start, labeled_end = labeled_data.index[[0, -1]]
        # # print("Labeled starts at:", labeled_start, "and ends at:", labeled_end)
        # # print("-" * 40)

        ## optimize the jump penalty
        jump_penalty_list = [50.0, 55.0, 60.0]
        train_score = []
        for jump_penalty in jump_penalty_list:
            ## regime detection
            jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )
            jm.fit(X_train_processed, SP500_ret, sort_by="cumret")
            labels_test_online = jm.predict_online(X_test_processed.dropna())  # make online inference
            labeled_data = pd.DataFrame(pd.concat([jm.labels_,labels_test_online]),columns=["regime"]).join(raw_data, how='left')
            labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]
            labeled_start, labeled_end = labeled_data.index[[0, -1]]
            portfolio_cumulative_with_regime, portfolio_returns, weights_dict = backtest_regime_portfolio_1(
                labeled_data.fillna(method="ffill"), 
                test_period=(train_start, train_end), 
                train_period=(train_start, train_end), 
                rebalancing_freq='YE', 
                strategy='min_volatility'
            )
            train_score.append(sharpe_tario(portfolio_cumulative_with_regime))
        jump_penalty = jump_penalty_list[train_score.index(max(train_score))]

        ## strategy
        jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )
        jm.fit(X_train_processed, SP500_ret, sort_by="cumret")
        labels_test_online = jm.predict_online(X_test_processed.dropna())  # make online inference
        labeled_data = pd.DataFrame(pd.concat([jm.labels_,labels_test_online]),columns=["regime"]).join(raw_data, how='left')
        labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]
        portfolio_cumulative_with_regime, portfolio_returns, weights_dict = backtest_regime_portfolio_1(
            labeled_data.fillna(method="ffill"), 
            test_period=(test_start, test_end), 
            train_period=(train_start, train_end), 
            rebalancing_freq='YE', 
            strategy='min_volatility'
        )
        return_list.append(portfolio_returns)
        weights_list.append(weights_dict)

        # test_start Slide the window forward by one year
        test_start = test_start + pd.DateOffset(years=1)


    rolling_returns = pd.concat(return_list)
    rolling_returns = rolling_returns[~rolling_returns.index.duplicated(keep='first')]  # 删掉重复的行

    # calculate the cumulative returns
    rolling_cumulative = (1 + rolling_returns).cumprod()

    # organize the weights df
    merged = {}
    for d in weights_list:
        merged.update(d)
    rolling_dynamic_weights = pd.DataFrame.from_dict(merged, orient='index')
    rolling_dynamic_weights.index.name = 'date'

    return rolling_cumulative, rolling_returns, rolling_dynamic_weights

def backtest_regime_portfolio_2(
    labeled_data, 
    test_period, 
    train_period,  
    rebalancing_freq='Y', 
    strategy='max_sharpe'
):
    """
    Perform dynamic portfolio optimization using pypfopt and backtest its performance 
    (supports rebalancing based on market regime changes).

    Parameters:
    - labeled_data: DataFrame, contains asset prices and market regime labels; index is datetime, columns include asset prices and a "regime" column
    - test_period: tuple, (start date of the test set, end date of the test set)
    - train_period: tuple, (start date of the training set, end date of the training set) 
      **used to compute `mu` and `covariance` under different `regimes`**
    - rebalancing_freq: str, regular rebalancing frequency ('M': monthly, 'Q': quarterly, 'Y': yearly)
    - strategy: str, portfolio optimization strategy ('min_volatility', 'max_sharpe', 'max_quadratic_utility')

    Returns:
    - portfolio_cumulative: DataFrame, cumulative returns of the portfolio
    - portfolio_returns: DataFrame, daily returns of the portfolio
    - weights_dict: dict, portfolio weights at each rebalancing point
    """
    # split asset price data and regime labels
    price_data = labeled_data.drop(columns=['regime'])
    regime_data = labeled_data['regime']
    
    # calculate daily returns
    returns = price_data.pct_change().dropna()

    # extract date range
    start_date, end_date = test_period
    train_start, train_end = train_period  

    # Compute mu and covariance for each regime within the train_period
    train_returns = returns.loc[train_start:train_end].copy()
    train_regimes = regime_data.loc[train_start:train_end]

    regime_mu_cov = {}  # store the mu and covariance for each regime
    for regime in train_regimes.unique():
        regime_mask = train_regimes == regime
        regime_returns = train_returns[regime_mask[train_returns.index[0]:train_returns.index[-1]]]  # get the returns for this specific regime 

        # keep only assets with positive mean return under the current regime in the train_period
        mean_returns = regime_returns.mean()
        selected_assets = mean_returns[mean_returns > 0].index.tolist()
        regime_returns = regime_returns[selected_assets]  # keep only assets with positive mean return under the current regime in the train_period

        if len(regime_returns) < 30:
            print(f"Skipping regime {regime} in train_period (not enough data)")
            continue

        mu = mean_historical_return(regime_returns, returns_data=True)
        S = CovarianceShrinkage(regime_returns, returns_data=True).ledoit_wolf()
        regime_mu_cov[regime] = (mu, S)

    # determine rebalancing time points
    rebalance_dates = [d for d in pd.date_range(start=start_date, end=end_date, freq=rebalancing_freq) if d in returns.index]

    # rebalance on the day after regime changes occurring within the test period
    regime_changes = regime_data.loc[start_date:end_date].shift(1) != regime_data.loc[start_date:end_date]
    regime_change_dates = regime_data.loc[start_date:end_date].index[regime_changes]
    rebalance_dates.extend(regime_change_dates)
    rebalance_dates = sorted(set(rebalance_dates))  # remove duplicates and sort
    
    # Initialize portfolio weights and returns
    weights_dict = {}
    portfolio_returns = []
    dates = []

    for i in range(len(rebalance_dates)):
        rebalance_date = rebalance_dates[i]
        # get the current regime
        current_regime = regime_data.loc[rebalance_date]

        # select only the assets filtered for the current regime within the train_period
        if current_regime in regime_mu_cov:
            mu, S = regime_mu_cov[current_regime]
        else:
            print(f"Skipping rebalancing on {rebalance_date} (no training data for regime {current_regime})")
            weights_dict[rebalance_date] = {asset: 0 for asset in price_data.columns}  # go to cash
            continue

        # strategy optimization
        ef = EfficientFrontier(mu, S)
        # ef.add_constraint(lambda w: w <= 0.4)  # limit maximum asset weight to 40%
        if strategy == 'max_sharpe':
            ef.max_sharpe()
        elif strategy == 'min_volatility':
            ef.min_volatility()
        elif strategy == 'max_quadratic_utility':
            ef.max_quadratic_utility(risk_aversion=10.0)
        else:
            raise ValueError("Unsupported strategy. Use 'max_sharpe', 'min_volatility', or 'max_quadratic_utility'.")
        cleaned_weights = ef.clean_weights()

        # ensure weights of unselected assets are set to 0
        full_weights = {asset: cleaned_weights.get(asset, 0) for asset in price_data.columns}
        weights_dict[rebalance_date] = full_weights

        # print(f"Rebalancing on {rebalance_date.strftime('%Y-%m-%d')} | Regime: {current_regime}")

        # compute the weights
        end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else end_date
        period_returns = returns.loc[rebalance_date:end]
        weight_vector = np.array([full_weights.get(asset, 0) for asset in price_data.columns])

        # calculate the portfolio returns for this time windoe
        period_portfolio_returns = period_returns @ weight_vector
        portfolio_returns.extend(period_portfolio_returns.tolist())
        dates.extend(period_returns.index.tolist())

    # organize the portfolio returns
    portfolio_returns = pd.Series(portfolio_returns, index=dates).dropna()
    portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep='first')]  # delete the duplicated rows

    # calculate the cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    portfolio_cumulative = portfolio_cumulative.reindex(price_data.loc[start_date:end_date].index, method='ffill')

    # # plotting
    # plt.figure(figsize=(12, 6))
    # plt.plot(portfolio_cumulative, label="Backtest Portfolio", linewidth=2)
    # plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    # plt.xlabel("Date"), plt.ylabel("Cumulative Return"), plt.title("Backtest Portfolio Performance")
    # plt.legend(), plt.grid(), plt.show()

    return portfolio_cumulative, portfolio_returns, weights_dict


def rolling_dynamic_backtest_regime_2(test_start, num_years):
    # rolling return
    return_list = []
    weights_list = []

    for i in range(num_years):
        train_start = test_start - pd.DateOffset(years=10) + pd.DateOffset(days=1)  # start date of the training set
        train_end = test_start  # end date of the training set
        test_end = test_start + pd.DateOffset(years=1)  # end date of the test set
        
        # print(f"Iteration {i+1}:")
        # print(f"train_start = {train_start.strftime('%Y-%m-%d')}")
        # print(f"train_end = {train_end.strftime('%Y-%m-%d')}")
        # print(f"test_start = {test_start.strftime('%Y-%m-%d')}")
        # print(f"test_end = {test_end.strftime('%Y-%m-%d')}")
        # print("-" * 40)
        
        ## train-test split
        X_train = filter_date_range(features, start_date=train_start, end_date=test_start)
        X_test = filter_date_range(features, start_date=test_start, end_date=test_end)
        # print time split
        train_start, train_end = X_train.index[[0, -1]]
        test_start, test_end = X_test.index[[0, -1]]
        # print("Training starts at:", train_start, "and ends at:", train_end)
        # print("Testing starts at:", test_start, "and ends at:", test_end)
        # print("-" * 40)

        ## preproscessing
        from jumpmodels.preprocess import StandardScalerPD, DataClipperStd
        clipper = DataClipperStd(mul=3.)
        scalar = StandardScalerPD()
        # fit on training data
        X_train_processed = scalar.fit_transform(clipper.fit_transform(X_train))
        # transform the test data
        X_test_processed = scalar.transform(clipper.transform(X_test))

        # ## regime detection
        # jump_penalty=60.
        # jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )
        # jm.fit(X_train_processed, SP500_ret, sort_by="cumret")
        # labels_test_online = jm.predict_online(X_test_processed.dropna())  # make online inference
        # labeled_data = pd.DataFrame(pd.concat([jm.labels_,labels_test_online]),columns=["regime"]).join(raw_data, how='left')
        # labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]
        # labeled_start, labeled_end = labeled_data.index[[0, -1]]
        # # print("Labeled starts at:", labeled_start, "and ends at:", labeled_end)
        # # print("-" * 40)

        ## optimize the jump penalty
        jump_penalty_list = [50.0, 55.0, 60.0]
        train_score = []
        for jump_penalty in jump_penalty_list:
            ## regime detection
            jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )
            jm.fit(X_train_processed, SP500_ret, sort_by="cumret")
            labels_test_online = jm.predict_online(X_test_processed.dropna())  # make online inference
            labeled_data = pd.DataFrame(pd.concat([jm.labels_,labels_test_online]),columns=["regime"]).join(raw_data, how='left')
            labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]
            labeled_start, labeled_end = labeled_data.index[[0, -1]]
            portfolio_cumulative_with_regime, portfolio_returns, weights_dict = backtest_regime_portfolio_2(
                labeled_data.fillna(method="ffill"), 
                test_period=(train_start, train_end), 
                train_period=(train_start, train_end), 
                rebalancing_freq='YE', 
                strategy='min_volatility'
            )
            train_score.append(sharpe_tario(portfolio_cumulative_with_regime))
        jump_penalty = jump_penalty_list[train_score.index(max(train_score))]

        ## strategy
        jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )
        jm.fit(X_train_processed, SP500_ret, sort_by="cumret")
        labels_test_online = jm.predict_online(X_test_processed.dropna())  # make online inference
        labeled_data = pd.DataFrame(pd.concat([jm.labels_,labels_test_online]),columns=["regime"]).join(raw_data, how='left')
        labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]
        portfolio_cumulative_with_regime, portfolio_returns, weights_dict = backtest_regime_portfolio_2(
            labeled_data.fillna(method="ffill"), 
            test_period=(test_start, test_end), 
            train_period=(train_start, train_end), 
            rebalancing_freq='YE', 
            strategy='min_volatility'
        )
        return_list.append(portfolio_returns)
        weights_list.append(weights_dict)

        # test_start Shift the time window forward by one year
        test_start = test_start + pd.DateOffset(years=1)

    
    rolling_returns = pd.concat(return_list)
    rolling_returns = rolling_returns[~rolling_returns.index.duplicated(keep='first')]  # delete the duplicated rows

    # compute the cumulative returns
    rolling_cumulative = (1 + rolling_returns).cumprod()

    # organize the weights df
    merged = {}
    for d in weights_list:
        merged.update(d)
    rolling_dynamic_weights = pd.DataFrame.from_dict(merged, orient='index')
    rolling_dynamic_weights.index.name = 'date'

    return rolling_cumulative, rolling_returns, rolling_dynamic_weights

def backtest_regime_portfolio_3(
    labeled_data, 
    test_period, 
    train_period,  
    rebalancing_freq='Y', 
    strategy='max_sharpe'
):
    """
    Perform dynamic portfolio optimization using pypfopt and backtest its performance 
    (supports rebalancing based on changes in market regimes).

    Parameters:
    - labeled_data: DataFrame, contains asset prices and market regime labels; index is datetime, columns include asset prices and a "regime" column
    - test_period: tuple, (start date of the test set, end date of the test set)
    - train_period: tuple, (start date of the training set, end date of the training set) 
      **used to compute `mu` and `covariance` under different `regimes`**
    - rebalancing_freq: str, rebalancing frequency ('M': monthly, 'Q': quarterly, 'Y': yearly)
    - strategy: str, portfolio optimization strategy ('min_volatility', 'max_sharpe', 'max_quadratic_utility')

    Returns:
    - portfolio_cumulative: DataFrame, cumulative returns of the portfolio
    - portfolio_returns: DataFrame, daily returns of the portfolio
    - weights_dict: dict, portfolio weights at each rebalancing point
    """
    # split asset price data and regime labels
    price_data = labeled_data.drop(columns=['regime'])
    regime_data = labeled_data['regime']
    
    # compute the daily returns
    returns = price_data.pct_change().dropna()

    # extract the date range
    start_date, end_date = test_period
    train_start, train_end = train_period  

    # compute mu and covariance for each regime within the `train_period`
    train_returns = returns.loc[train_start:train_end].copy()
    train_regimes = regime_data.loc[train_start:train_end]

    regime_mu_cov = {}  # store the mu and covariance for each regime
    for regime in train_regimes.unique():
        regime_mask = train_regimes == regime
        # print(len(train_returns),len(regime_mask))
        # print("the index of price_data:", price_data.index[0], price_data.index[-1])
        # print("the index of regime_data:", regime_data.index[0], regime_data.index[-1])
        # print("the index of returns:", returns.index[0], returns.index[-1])
        # print("the index of regime_data:", regime_data.index[0], regime_data.index[-1])
        # print("the index of train_returns:", train_returns.index[0], train_returns.index[-1])
        # print("the index of train_regimes:", train_regimes.index[0], train_regimes.index[-1])
        # print("the index of regime_mask:", regime_mask.index[0], regime_mask.index[-1])
        regime_returns = train_returns[regime_mask[train_returns.index[0]:train_returns.index[-1]]]  # select the returns for this specific regime

        if len(regime_returns) < 30:
            print(f"Skipping regime {regime} in train_period (not enough data)")
            continue

        mu = mean_historical_return(regime_returns, returns_data=True)
        S = CovarianceShrinkage(regime_returns, returns_data=True).ledoit_wolf()
        regime_mu_cov[regime] = (mu, S)

    # # determine rebalancing time points
    # rebalance_dates = [d for d in pd.date_range(start=start_date, end=end_date, freq=rebalancing_freq) if d in returns.index]

    # # rebalance only on the day after regime changes occurring within the test period
    # regime_changes = regime_data.loc[start_date:end_date].shift(1) != regime_data.loc[start_date:end_date]
    # regime_change_dates = regime_data.loc[start_date:end_date].index[regime_changes]
    # rebalance_dates.extend(regime_change_dates)
    # rebalance_dates = sorted(set(rebalance_dates))  # remove duplicates and sort

    # determine rebalancing time points
    whole_dates = price_data.index
    # rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalancing_freq)
    rebalance_dates = whole_dates[(whole_dates.searchsorted(pd.date_range(start=start_date, end=end_date, freq="YE")) - 1).clip(0)]
    rebalance_dates = rebalance_dates.tolist()
    rebalance_dates.insert(0, whole_dates[whole_dates.searchsorted(start_date, side="right") - 1])
    rebalance_dates = pd.DatetimeIndex(rebalance_dates)
    rebalance_dates
    
    # initialize portfolio weights and returns
    weights_dict = {}
    portfolio_returns = []
    dates = []

    for i in range(len(rebalance_dates)):
        rebalance_date = rebalance_dates[i]
        # get the current regime
        current_regime = regime_data.loc[rebalance_date]
        # current_regime = regime_data.loc[whole_dates[whole_dates.searchsorted(start_date, side="right") - 1 -1]]
        # use mu and covariance from the training period for the current regime
        if current_regime in regime_mu_cov:
            mu, S = regime_mu_cov[current_regime]
        else:
            print(f"Skipping rebalancing on {rebalance_date} (no training data for regime {current_regime})")
            weights_dict[rebalance_date] = {asset: 0 for asset in price_data.columns}  # 直接空仓
            continue

        selected_assets = mu.index.tolist()

        # strategy optimization
        ef = EfficientFrontier(mu, S)
        # ef.add_constraint(lambda w: w <= 0.4)  # limit maximum asset weight to 40%
        if strategy == 'max_sharpe':
            ef.max_sharpe()
        elif strategy == 'min_volatility':
            ef.min_volatility()
        elif strategy == 'max_quadratic_utility':
            ef.max_quadratic_utility(risk_aversion=10.0)
        else:
            raise ValueError("Unsupported strategy. Use 'max_sharpe', 'min_volatility', or 'max_quadratic_utility'.")
        cleaned_weights = ef.clean_weights()

        # ensure weights of unselected assets are set to 0
        full_weights = {asset: cleaned_weights.get(asset, 0) for asset in price_data.columns}
        weights_dict[rebalance_date] = full_weights

        # print(f"Rebalancing on {rebalance_date.strftime('%Y-%m-%d')} | Regime: {current_regime}")

        # compute the weights and returns of all assets for this time window
        end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else end_date
        period_returns = returns.loc[rebalance_date:end]
        weight_vector = np.array([full_weights.get(asset, 0) for asset in price_data.columns])

        # compute the startegy returns for this time window
        period_portfolio_returns = period_returns @ weight_vector
        portfolio_returns.extend(period_portfolio_returns.tolist())
        dates.extend(period_returns.index.tolist())

    # organize the portfolio return series
    portfolio_returns = pd.Series(portfolio_returns, index=dates).dropna()
    portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep='first')]  # delete the duplicated returns

    # compute the cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    portfolio_cumulative = portfolio_cumulative.reindex(price_data.loc[start_date:end_date].index, method='ffill')

    # # plotting
    # plt.figure(figsize=(12, 6))
    # plt.plot(portfolio_cumulative, label="Backtest Portfolio", linewidth=2)
    # plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    # plt.xlabel("Date"), plt.ylabel("Cumulative Return"), plt.title("Backtest Portfolio Performance")
    # plt.legend(), plt.grid(), plt.show()

    return portfolio_cumulative, portfolio_returns, weights_dict


def rolling_dynamic_backtest_regime_3(test_start, num_years):
    # rolling return
    return_list = []
    weights_list = []

    for i in range(num_years):
        train_start = test_start - pd.DateOffset(years=10) + pd.DateOffset(days=1)  # start date of the training set
        train_end = test_start  # end date of the training set
        test_end = test_start + pd.DateOffset(years=1)  # end date of the test set
        
        # print(f"Iteration {i+1}:")
        # print(f"train_start = {train_start.strftime('%Y-%m-%d')}")
        # print(f"train_end = {train_end.strftime('%Y-%m-%d')}")
        # print(f"test_start = {test_start.strftime('%Y-%m-%d')}")
        # print(f"test_end = {test_end.strftime('%Y-%m-%d')}")
        # print("-" * 40)
        
        ## train-test split
        X_train = filter_date_range(features, start_date=train_start, end_date=test_start)
        X_test = filter_date_range(features, start_date=test_start, end_date=test_end)
        # print time split
        train_start, train_end = X_train.index[[0, -1]]
        test_start, test_end = X_test.index[[0, -1]]
        # print("Training starts at:", train_start, "and ends at:", train_end)
        # print("Testing starts at:", test_start, "and ends at:", test_end)
        # print("-" * 40)

        ## preproscessing
        from jumpmodels.preprocess import StandardScalerPD, DataClipperStd
        clipper = DataClipperStd(mul=3.)
        scalar = StandardScalerPD()
        # fit on training data
        X_train_processed = scalar.fit_transform(clipper.fit_transform(X_train))
        # transform the test data
        X_test_processed = scalar.transform(clipper.transform(X_test))

        # ## regime detection
        # jump_penalty=60.
        # jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )
        # jm.fit(X_train_processed, SP500_ret, sort_by="cumret")
        # labels_test_online = jm.predict_online(X_test_processed.dropna())  # make online inference
        # labeled_data = pd.DataFrame(pd.concat([jm.labels_,labels_test_online]),columns=["regime"]).join(raw_data, how='left')
        # labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]
        # labeled_start, labeled_end = labeled_data.index[[0, -1]]
        # # print("Labeled starts at:", labeled_start, "and ends at:", labeled_end)
        # # print("-" * 40)

        ## optimize the jump penalty
        jump_penalty_list = [50.0, 55.0, 60.0]
        train_score = []
        for jump_penalty in jump_penalty_list:
            ## regime detection
            jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )
            jm.fit(X_train_processed, SP500_ret, sort_by="cumret")
            labels_test_online = jm.predict_online(X_test_processed.dropna())  # make online inference
            labeled_data = pd.DataFrame(pd.concat([jm.labels_,labels_test_online]),columns=["regime"]).join(raw_data, how='left')
            labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]
            labeled_start, labeled_end = labeled_data.index[[0, -1]]
            portfolio_cumulative_with_regime, portfolio_returns, weights_dict = backtest_regime_portfolio_3(
                labeled_data.fillna(method="ffill"), 
                test_period=(train_start, train_end), 
                train_period=(train_start, train_end), 
                rebalancing_freq='YE', 
                strategy='min_volatility'
            )
            train_score.append(sharpe_tario(portfolio_cumulative_with_regime))
        jump_penalty = jump_penalty_list[train_score.index(max(train_score))]

        ## strategy
        jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )
        jm.fit(X_train_processed, SP500_ret, sort_by="cumret")
        labels_test_online = jm.predict_online(X_test_processed.dropna())  # make online inference
        labeled_data = pd.DataFrame(pd.concat([jm.labels_,labels_test_online]),columns=["regime"]).join(raw_data, how='left')
        labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]
        portfolio_cumulative_with_regime, portfolio_returns, weights_dict = backtest_regime_portfolio_3(
            labeled_data.fillna(method="ffill"), 
            test_period=(test_start, test_end), 
            train_period=(train_start, train_end), 
            rebalancing_freq='YE', 
            strategy='min_volatility'
        )
        return_list.append(portfolio_returns)
        weights_list.append(weights_dict)

        # test_start Shift the time window forward by one year
        test_start = test_start + pd.DateOffset(years=1)


    rolling_returns = pd.concat(return_list)
    rolling_returns = rolling_returns[~rolling_returns.index.duplicated(keep='first')]  # 删掉重复的行

    # compute the cumulative returns
    rolling_cumulative = (1 + rolling_returns).cumprod()

    # organize the weights df
    merged = {}
    for d in weights_list:
        merged.update(d)
    rolling_dynamic_weights = pd.DataFrame.from_dict(merged, orient='index')
    rolling_dynamic_weights.index.name = 'date'

    return rolling_cumulative, rolling_returns, rolling_dynamic_weights



def main():
    print("Hello from dashboard-qwim!")

    rolling_cumulative, rolling_returns, return_list = rolling_backtest(test_start=pd.to_datetime("2008-01-01"), num_years=14)
    rolling_dynamic_cumulative_regime_1, rolling_dynamic_returns, rolling_dynamic_weights_1 = rolling_dynamic_backtest_regime_1(test_start = pd.to_datetime("2008-01-01"), num_years = 14)
    rolling_dynamic_cumulative_regime_2, rolling_dynamic_returns, rolling_dynamic_weights_2 = rolling_dynamic_backtest_regime_2(test_start = pd.to_datetime("2008-01-01"), num_years = 14)
    rolling_dynamic_cumulative_regime_3, rolling_dynamic_returns, rolling_dynamic_weights_3 = rolling_dynamic_backtest_regime_3(test_start = pd.to_datetime("2008-01-01"), num_years = 14)

    strategy_returns = pd.DataFrame(
        {
            "annually without regime": rolling_cumulative["Portfolio Return"],
            "annually regime-based": rolling_dynamic_cumulative_regime_3,
            "dynamic regime-based": rolling_dynamic_cumulative_regime_1,
            "regime-based with positive assets": rolling_dynamic_cumulative_regime_2
        }
    )
    strategy_returns = strategy_returns.dropna()
    strategy_returns.index.name = "Date"
    JM_benchmark = pd.DataFrame(index=strategy_returns.index, data={"Value": strategy_returns["annually without regime"]})
    JM_portfolio = pd.DataFrame(index=strategy_returns.index, data={"Value": strategy_returns["regime-based with positive assets"]})
    JM_benchmark.to_csv(JM_benchmark_output_path)
    JM_portfolio.to_csv(JM_portfolio_output_path)
    rolling_dynamic_weights_2.to_csv(JM_weights_output_path)



if __name__ == "__main__":
    # print(JM_benchmark_output_path)
    main()
