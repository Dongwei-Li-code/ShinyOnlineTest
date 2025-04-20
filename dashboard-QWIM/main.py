def main():
    print("Hello from dashboard-qwim!")

if __name__ == "__main__":
    main()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pypfopt.efficient_frontier import EfficientFrontier
# from pypfopt.risk_models import CovarianceShrinkage
# from pypfopt.expected_returns import mean_historical_return
# from pypfopt.cla import CLA
# from pypfopt.objective_functions import L2_reg
# from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# from jumpmodels.utils import filter_date_range        # useful helpers
# from jumpmodels.jump import JumpModel                 # class of JM & CJM
# from jumpmodels.sparse_jump import SparseJumpModel    # class of Sparse JM

# raw_data = pd.read_excel("data/raw/indices.xlsx", index_col=0, parse_dates=True)
# raw_data = raw_data.drop("SP500",axis=1)
# SP500 = raw_data["SPTR"]
# SP500_ret = np.log(SP500 / SP500.shift(1))

# def backtest_portfolio(
#     price_data, 
#     test_period, 
#     lookback_period=252, 
#     rebalancing_freq='Y', 
#     strategy='max_sharpe'
# ):
#     """
#     基于 pypfopt 进行动态投资组合优化，并回测其收益情况。
    
#     参数：
#     - price_data: DataFrame, 多资产价格日度数据，index为时间，columns为资产
#     - test_period: tuple, (测试集开始时间, 测试集结束时间)
#     - lookback_period: int, 训练集回溯的长度（默认为252个交易日）
#     - rebalancing_freq: str, 调仓频率 ('M'：月度, 'Q'：季度, 'Y'：年度)
#     - strategy: str, 投资组合优化策略 ('min_volatility'：最小方差, 'max_sharpe'：最大夏普比率)

#     返回：
#     - 绘制收益曲线并返回最终收益表现
#     """

#     # 计算每日收益率
#     returns = price_data.pct_change().dropna()

#     # 提取全部时间
#     whole_dates = price_data.index
    
#     # 确定测试集时间范围
#     start_date, end_date = test_period
#     test_data = price_data.loc[start_date:end_date]
    
#     # 确定调仓时间点
#     # rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalancing_freq)
#     rebalance_dates = pd.date_range(start=start_date, end=end_date, freq="YE").tolist()
#     rebalance_dates.insert(0, whole_dates[whole_dates.searchsorted(start_date, side="right") - 1])
#     rebalance_dates = pd.DatetimeIndex(rebalance_dates)
#     rebalance_dates
    
#     # 初始化投资组合权重
#     weights_dict = {}
    
#     for rebalance_date in rebalance_dates:
#         # 选择回溯训练集数据
#         train_end = rebalance_date
#         # train_start = train_end - pd.Timedelta(days=lookback_period)
#         train_start = whole_dates[whole_dates.searchsorted(start_date, side="right") - 1 -lookback_period]
        
#         train_data = price_data.loc[train_start:train_end]


#         # 进行优化
#         print(f"Rebalancing on {rebalance_date.strftime('%Y-%m-%d')} | Train period: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        
#         # 计算期望收益率和协方差矩阵
#         mu = mean_historical_return(train_data)
#         # S = train_data.pct_change().dropna().cov()
#         S = train_data.dropna().pct_change().ewm(span=252).cov(pairwise=True).loc[train_data.index[-1]]*252
#         # S = CovarianceShrinkage(train_data).ledoit_wolf()
        
#         # 选择优化策略
#         ef = EfficientFrontier(mu, S)
#         # ef.add_constraint(lambda w: w <= 0.4)
#         if strategy == 'max_sharpe':
#             ef.max_sharpe()
#         elif strategy == 'min_volatility':
#             ef.min_volatility()
#         elif strategy == 'max_quadratic_utility': 
#             ef.max_quadratic_utility(risk_aversion=10.0)
#         else:
#             raise ValueError("Unsupported strategy. Use 'max_sharpe', 'min_volatility', or 'max_quadratic_utility'.")
        
#         # 记录资产权重
#         cleaned_weights = ef.clean_weights()
#         weights_dict[rebalance_date] = cleaned_weights
    
#     # 回测收益
#     portfolio_returns = []
#     dates = []
    
#     for i in range(len(rebalance_dates) - 1):
#         start = rebalance_dates[i]
#         end = rebalance_dates[i + 1]
        
#         # if end not in returns.index:
#         #     continue
        
#         # 获取当前区间的资产收益率
#         period_returns = returns.loc[start:end]
        
#         # 获取上次调仓时的权重
#         last_weights = weights_dict[rebalance_dates[i]]
#         weight_vector = np.array([last_weights[asset] for asset in price_data.columns])
        
#         # 计算该期间的组合收益
#         period_portfolio_returns = period_returns @ weight_vector
#         portfolio_returns.extend(period_portfolio_returns.tolist())
#         dates.extend(period_returns.index.tolist())
    
#     # 转换为 DataFrame
#     portfolio_returns = pd.DataFrame(portfolio_returns, index=dates, columns=['Portfolio Return'])
#     portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep='first')]  # 删掉重复的行
    
#     # 计算累积收益
#     portfolio_cumulative = (1 + portfolio_returns).cumprod()
#     portfolio_cumulative = portfolio_cumulative.reindex(price_data.loc[start_date:end_date].index, method='ffill')  
    
#     # # 绘制收益曲线
#     # plt.figure(figsize=(12, 6))
#     # plt.plot(portfolio_cumulative, label="Backtest Portfolio", linewidth=2)
#     # plt.xlabel("Date")
#     # plt.ylabel("Cumulative Return")
#     # plt.title("Backtest Portfolio Performance")
#     # plt.legend()
#     # plt.grid()
#     # plt.show()
    
#     # # 计算最终收益表现
#     # total_return = portfolio_cumulative.iloc[-1, 0] - 1
#     # annualized_return = portfolio_returns.mean()[0] * 252
#     # annualized_volatility = portfolio_returns.std()[0] * np.sqrt(252)
#     # sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else np.nan
    
#     # print(f"Total Return: {total_return:.2%}")
#     # print(f"Annualized Return: {annualized_return:.2%}")
#     # print(f"Annualized Volatility: {annualized_volatility:.2%}")
#     # print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
#     return portfolio_cumulative,portfolio_returns,weights_dict


# def rolling_backtest(test_start, num_years):
#     # rolling return
#     return_list = []

#     for i in range(num_years):
#         train_start = test_start - pd.DateOffset(years=10) + pd.DateOffset(days=1)  # 训练集起始时间
#         train_end = test_start  # 训练集结束时间
#         test_end = test_start + pd.DateOffset(years=1)  # 测试集结束时间
        
#         # print(f"Iteration {i+1}:")
#         # print(f"train_start = {train_start.strftime('%Y-%m-%d')}")
#         # print(f"train_end = {train_end.strftime('%Y-%m-%d')}")
#         # print(f"test_start = {test_start.strftime('%Y-%m-%d')}")
#         # print(f"test_end = {test_end.strftime('%Y-%m-%d')}")
#         # print("-" * 40)

#         ## strategy
#         portfolio_cumulative_without_regime,portfolio_returns,weights_dict = backtest_portfolio(
#             raw_data.fillna(method="ffill"), 
#             test_period=(test_start, test_end), 
#             lookback_period=2520, 
#             rebalancing_freq='YE', 
#             strategy='min_volatility'
#         )
#         return_list.append(portfolio_returns)


#         # test_start 每次向后移动一年
#         test_start = test_start + pd.DateOffset(years=1)

#     rolling_returns = pd.concat(return_list)
#     rolling_returns = rolling_returns[~rolling_returns.index.duplicated(keep='first')]  # 删掉重复的行

#     # 计算累积收益
#     rolling_cumulative = (1 + rolling_returns).cumprod()

#     # plt.figure(figsize=(12, 6))
#     # plt.plot(rolling_cumulative, label="Backtest Portfolio", linewidth=2)
#     # plt.xlabel("Date")
#     # plt.ylabel("Cumulative Return")
#     # plt.title("Backtest Portfolio Performance")
#     # plt.legend()
#     # plt.grid()
#     # plt.show()

#     return rolling_cumulative, rolling_returns, return_list

# def compute_ewm_DD(ret_ser: pd.Series, hl: float) -> pd.Series:
#     """
#     Compute the exponentially weighted moving downside deviation (DD) for a return series.

#     The downside deviation is calculated as the square root of the exponentially 
#     weighted second moment of negative returns.

#     Parameters
#     ----------
#     ret_ser : pd.Series
#         The input return series.

#     hl : float
#         The halflife parameter for the exponentially weighted moving average.

#     Returns
#     -------
#     pd.Series
#         The exponentially weighted moving downside deviation for the return series.
#     """
#     ret_ser_neg: pd.Series = np.minimum(ret_ser, 0.)
#     sq_mean = ret_ser_neg.pow(2).ewm(halflife=hl).mean()
#     return np.sqrt(sq_mean)


# # reviewed
# def feature_engineer(ret_ser: pd.Series, ver: str = "v0") -> pd.DataFrame:
#     if ver == "v0":
#         feat_dict = {}
#         hls = [5, 20, 30]
#         for hl in hls:
#             feat_dict[f"ret_{hl}"] = ret_ser.ewm(halflife=hl).mean()
#             DD = compute_ewm_DD(ret_ser, hl)
#             feat_dict[f"DD-log_{hl}"] = np.log(DD)
#             feat_dict[f"sortino_{hl}"] = feat_dict[f"ret_{hl}"].div(DD)
#         return pd.DataFrame(feat_dict)
    
#     elif ver == "v1":
#         feat_dict = {}
#         windows = [6, 14]
        
#         # Feature 1: Observation
#         feat_dict["obs"] = ret_ser
        
#         # Feature 2 & 3: Absolute changes
#         feat_dict["abs_change"] = ret_ser.diff().abs()
#         feat_dict["prev_abs_change"] = ret_ser.diff().shift(1).abs()
        
#         for w in windows:
#             half_w = w // 2
            
#             # Feature 4 & 5: Centered Mean & Std (using t-w+1 to t)
#             feat_dict[f"centered_mean_{w}"] = ret_ser.rolling(window=w).mean()
#             feat_dict[f"centered_std_{w}"] = ret_ser.rolling(window=w).std()
            
#             # Feature 6 & 7: Left Mean & Std (first half of t-w+1 to t)
#             feat_dict[f"left_mean_{w}"] = ret_ser.rolling(window=w).apply(lambda x: x[:half_w].mean(), raw=False)
#             feat_dict[f"left_std_{w}"] = ret_ser.rolling(window=w).apply(lambda x: x[:half_w].std(), raw=False)
            
#             # Feature 8 & 9: Right Mean & Std (second half of t-w+1 to t)
#             feat_dict[f"right_mean_{w}"] = ret_ser.rolling(window=w).apply(lambda x: x[half_w:].mean(), raw=False)
#             feat_dict[f"right_std_{w}"] = ret_ser.rolling(window=w).apply(lambda x: x[half_w:].std(), raw=False)
        
#         return pd.DataFrame(feat_dict)
    

# features = feature_engineer(ret_ser=SP500_ret, ver="v1")



# def sharpe_tario(cumulative_returns: pd.Series, risk_free_rate: float = 0.02):
#     """
#     calculate sharpe ratio
#     """
#     # 计算每日收益率
#     daily_returns = cumulative_returns.pct_change().dropna()
    
#     # 夏普比率（假设无风险利率是年化的）
#     excess_daily_returns = daily_returns - risk_free_rate / 252
#     sharpe_ratio = excess_daily_returns.mean() / excess_daily_returns.std() * np.sqrt(252)
    
#     # # 索提诺比率（使用负收益的标准差）
#     # downside_returns = daily_returns[daily_returns < 0]
#     # downside_volatility = downside_returns.std() * np.sqrt(252)
#     # sortino_ratio = excess_daily_returns.mean() / downside_volatility if downside_volatility != 0 else np.nan
    
#     return sharpe_ratio

# def backtest_regime_portfolio_1(
#     labeled_data, 
#     test_period, 
#     train_period,  
#     rebalancing_freq='Y', 
#     strategy='max_sharpe'
# ):
#     """
#     基于 pypfopt 进行动态投资组合优化，并回测其收益情况（支持市场状态 Regime 变化调仓）。
    
#     参数：
#     - labeled_data: DataFrame, 包含资产价格和市场状态 (regime) 的数据，index 为时间，columns 为资产价格 + "regime"
#     - test_period: tuple, (测试集开始时间, 测试集结束时间)
#     - train_period: tuple, (训练集开始时间, 训练集结束时间) **计算不同 `regime` 下的 `mu` 和 `covariance`**
#     - rebalancing_freq: str, 正常调仓频率 ('M'：月度, 'Q'：季度, 'Y'：年度)
#     - strategy: str, 投资组合优化策略 ('min_volatility', 'max_sharpe', 'max_quadratic_utility')

#     返回：
#     - portfolio_cumulative: DataFrame, 投资组合的累积收益
#     - portfolio_returns: DataFrame, 投资组合的每日收益
#     - weights_dict: dict, 每次调仓的权重信息
#     """
#     # 分离价格数据和 Regime 数据
#     price_data = labeled_data.drop(columns=['regime'])
#     regime_data = labeled_data['regime']
    
#     # 计算每日收益率
#     returns = price_data.pct_change().dropna()

#     # 提取时间范围
#     start_date, end_date = test_period
#     train_start, train_end = train_period  

#     # **计算 `train_period` 内，每个 `regime` 的 `mu` 和 `covariance`**
#     train_returns = returns.loc[train_start:train_end].copy()
#     train_regimes = regime_data.loc[train_start:train_end]

#     regime_mu_cov = {}  # 存储每个 regime 的 mu 和 covariance
#     for regime in train_regimes.unique():
#         regime_mask = train_regimes == regime
#         # print(len(train_returns),len(regime_mask))
#         # print("the index of price_data:", price_data.index[0], price_data.index[-1])
#         # print("the index of regime_data:", regime_data.index[0], regime_data.index[-1])
#         # print("the index of returns:", returns.index[0], returns.index[-1])
#         # print("the index of regime_data:", regime_data.index[0], regime_data.index[-1])
#         # print("the index of train_returns:", train_returns.index[0], train_returns.index[-1])
#         # print("the index of train_regimes:", train_regimes.index[0], train_regimes.index[-1])
#         # print("the index of regime_mask:", regime_mask.index[0], regime_mask.index[-1])
#         regime_returns = train_returns[regime_mask[train_returns.index[0]:train_returns.index[-1]]]  # 只取该 `regime` 下的收益

#         if len(regime_returns) < 30:
#             print(f"Skipping regime {regime} in train_period (not enough data)")
#             continue

#         mu = mean_historical_return(regime_returns, returns_data=True)
#         S = CovarianceShrinkage(regime_returns, returns_data=True).ledoit_wolf()
#         regime_mu_cov[regime] = (mu, S)

#     # **确定调仓时间点**
#     rebalance_dates = [d for d in pd.date_range(start=start_date, end=end_date, freq=rebalancing_freq) if d in returns.index]

#     # 仅考虑 test_period 内的 Regime 变化调仓点（变化的第二天）
#     regime_changes = regime_data.loc[start_date:end_date].shift(1) != regime_data.loc[start_date:end_date]
#     regime_change_dates = regime_data.loc[start_date:end_date].index[regime_changes]
#     rebalance_dates.extend(regime_change_dates)
#     rebalance_dates = sorted(set(rebalance_dates))  # 去重 & 排序
    
#     # 初始化投资组合权重和收益
#     weights_dict = {}
#     portfolio_returns = []
#     dates = []

#     for i in range(len(rebalance_dates)):
#         rebalance_date = rebalance_dates[i]
#         # 获取当前 Regime
#         current_regime = regime_data.loc[rebalance_date]
#         # **✅ 直接使用 `train_period` 内该 `regime` 计算的 `mu` 和 `covariance`**
#         if current_regime in regime_mu_cov:
#             mu, S = regime_mu_cov[current_regime]
#         else:
#             print(f"Skipping rebalancing on {rebalance_date} (no training data for regime {current_regime})")
#             weights_dict[rebalance_date] = {asset: 0 for asset in price_data.columns}  # 直接空仓
#             continue

#         selected_assets = mu.index.tolist()

#         # **进行投资组合优化**
#         ef = EfficientFrontier(mu, S)
#         # ef.add_constraint(lambda w: w <= 0.4)  # 限制最大权重 40%
#         if strategy == 'max_sharpe':
#             ef.max_sharpe()
#         elif strategy == 'min_volatility':
#             ef.min_volatility()
#         elif strategy == 'max_quadratic_utility':
#             ef.max_quadratic_utility(risk_aversion=10.0)
#         else:
#             raise ValueError("Unsupported strategy. Use 'max_sharpe', 'min_volatility', or 'max_quadratic_utility'.")
#         cleaned_weights = ef.clean_weights()

#         # **确保未选资产的权重为 0**
#         full_weights = {asset: cleaned_weights.get(asset, 0) for asset in price_data.columns}
#         weights_dict[rebalance_date] = full_weights

#         # print(f"Rebalancing on {rebalance_date.strftime('%Y-%m-%d')} | Regime: {current_regime}")

#         # **计算投资组合收益**
#         end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else end_date
#         period_returns = returns.loc[rebalance_date:end]
#         weight_vector = np.array([full_weights.get(asset, 0) for asset in price_data.columns])

#         # 计算该期间的组合收益
#         period_portfolio_returns = period_returns @ weight_vector
#         portfolio_returns.extend(period_portfolio_returns.tolist())
#         dates.extend(period_returns.index.tolist())

#     # **整理投资组合收益**
#     portfolio_returns = pd.Series(portfolio_returns, index=dates).dropna()
#     portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep='first')]  # 删掉重复的行

#     # **计算累积收益**
#     portfolio_cumulative = (1 + portfolio_returns).cumprod()
#     portfolio_cumulative = portfolio_cumulative.reindex(price_data.loc[start_date:end_date].index, method='ffill')

#     # # **绘制收益曲线**
#     # plt.figure(figsize=(12, 6))
#     # plt.plot(portfolio_cumulative, label="Backtest Portfolio", linewidth=2)
#     # plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)  # 参考线
#     # plt.xlabel("Date"), plt.ylabel("Cumulative Return"), plt.title("Backtest Portfolio Performance")
#     # plt.legend(), plt.grid(), plt.show()

#     return portfolio_cumulative, portfolio_returns, weights_dict


# def rolling_dynamic_backtest_regime_1(test_start, num_years):
#     # rolling return
#     return_list = []
#     weights_list = []

#     for i in range(num_years):
#         train_start = test_start - pd.DateOffset(years=10) + pd.DateOffset(days=1)  # 训练集起始时间
#         train_end = test_start  # 训练集结束时间
#         test_end = test_start + pd.DateOffset(years=1)  # 测试集结束时间
        
#         # print(f"Iteration {i+1}:")
#         # print(f"train_start = {train_start.strftime('%Y-%m-%d')}")
#         # print(f"train_end = {train_end.strftime('%Y-%m-%d')}")
#         # print(f"test_start = {test_start.strftime('%Y-%m-%d')}")
#         # print(f"test_end = {test_end.strftime('%Y-%m-%d')}")
#         # print("-" * 40)
        
#         ## train-test split
#         X_train = filter_date_range(features, start_date=train_start, end_date=test_start)
#         X_test = filter_date_range(features, start_date=test_start, end_date=test_end)
#         # print time split
#         train_start, train_end = X_train.index[[0, -1]]
#         test_start, test_end = X_test.index[[0, -1]]
#         # print("Training starts at:", train_start, "and ends at:", train_end)
#         # print("Testing starts at:", test_start, "and ends at:", test_end)
#         # print("-" * 40)

#         ## preproscessing
#         from jumpmodels.preprocess import StandardScalerPD, DataClipperStd
#         clipper = DataClipperStd(mul=3.)
#         scalar = StandardScalerPD()
#         # fit on training data
#         X_train_processed = scalar.fit_transform(clipper.fit_transform(X_train))
#         # transform the test data
#         X_test_processed = scalar.transform(clipper.transform(X_test))

#         # ## regime detection
#         # jump_penalty=60.
#         # jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )
#         # jm.fit(X_train_processed, SP500_ret, sort_by="cumret")
#         # labels_test_online = jm.predict_online(X_test_processed.dropna())  # make online inference
#         # labeled_data = pd.DataFrame(pd.concat([jm.labels_,labels_test_online]),columns=["regime"]).join(raw_data, how='left')
#         # labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]
#         # labeled_start, labeled_end = labeled_data.index[[0, -1]]
#         # # print("Labeled starts at:", labeled_start, "and ends at:", labeled_end)
#         # # print("-" * 40)

#         ## optimize the jump penalty
#         jump_penalty_list = [50.0, 55.0, 60.0]
#         train_score = []
#         for jump_penalty in jump_penalty_list:
#             ## regime detection
#             jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )
#             jm.fit(X_train_processed, SP500_ret, sort_by="cumret")
#             labels_test_online = jm.predict_online(X_test_processed.dropna())  # make online inference
#             labeled_data = pd.DataFrame(pd.concat([jm.labels_,labels_test_online]),columns=["regime"]).join(raw_data, how='left')
#             labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]
#             labeled_start, labeled_end = labeled_data.index[[0, -1]]
#             portfolio_cumulative_with_regime, portfolio_returns, weights_dict = backtest_regime_portfolio_1(
#                 labeled_data.fillna(method="ffill"), 
#                 test_period=(train_start, train_end), 
#                 train_period=(train_start, train_end), 
#                 rebalancing_freq='YE', 
#                 strategy='min_volatility'
#             )
#             train_score.append(sharpe_tario(portfolio_cumulative_with_regime))
#         jump_penalty = jump_penalty_list[train_score.index(max(train_score))]

#         ## strategy
#         jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )
#         jm.fit(X_train_processed, SP500_ret, sort_by="cumret")
#         labels_test_online = jm.predict_online(X_test_processed.dropna())  # make online inference
#         labeled_data = pd.DataFrame(pd.concat([jm.labels_,labels_test_online]),columns=["regime"]).join(raw_data, how='left')
#         labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]
#         portfolio_cumulative_with_regime, portfolio_returns, weights_dict = backtest_regime_portfolio_1(
#             labeled_data.fillna(method="ffill"), 
#             test_period=(test_start, test_end), 
#             train_period=(train_start, train_end), 
#             rebalancing_freq='YE', 
#             strategy='min_volatility'
#         )
#         return_list.append(portfolio_returns)
#         weights_list.append(weights_dict)

#         # test_start 每次向后移动一年
#         test_start = test_start + pd.DateOffset(years=1)


#     rolling_returns = pd.concat(return_list)
#     rolling_returns = rolling_returns[~rolling_returns.index.duplicated(keep='first')]  # 删掉重复的行

#     # 计算累积收益
#     rolling_cumulative = (1 + rolling_returns).cumprod()

#     # 处理weights df
#     merged = {}
#     for d in weights_list:
#         merged.update(d)
#     rolling_dynamic_weights = pd.DataFrame.from_dict(merged, orient='index')
#     rolling_dynamic_weights.index.name = 'date'

#     return rolling_cumulative, rolling_returns, rolling_dynamic_weights

# def backtest_regime_portfolio_2(
#     labeled_data, 
#     test_period, 
#     train_period,  
#     rebalancing_freq='Y', 
#     strategy='max_sharpe'
# ):
#     """
#     基于 pypfopt 进行动态投资组合优化，并回测其收益情况（支持市场状态 Regime 变化调仓）。
    
#     参数：
#     - labeled_data: DataFrame, 包含资产价格和市场状态 (regime) 的数据，index 为时间，columns 为资产价格 + "regime"
#     - test_period: tuple, (测试集开始时间, 测试集结束时间)
#     - train_period: tuple, (训练集开始时间, 训练集结束时间) **计算不同 `regime` 下的 `mu` 和 `covariance`**
#     - rebalancing_freq: str, 正常调仓频率 ('M'：月度, 'Q'：季度, 'Y'：年度)
#     - strategy: str, 投资组合优化策略 ('min_volatility', 'max_sharpe', 'max_quadratic_utility')

#     返回：
#     - portfolio_cumulative: DataFrame, 投资组合的累积收益
#     - portfolio_returns: DataFrame, 投资组合的每日收益
#     - weights_dict: dict, 每次调仓的权重信息
#     """
#     # 分离价格数据和 Regime 数据
#     price_data = labeled_data.drop(columns=['regime'])
#     regime_data = labeled_data['regime']
    
#     # 计算每日收益率
#     returns = price_data.pct_change().dropna()

#     # 提取时间范围
#     start_date, end_date = test_period
#     train_start, train_end = train_period  

#     # **计算 `train_period` 内，每个 `regime` 的 `mu` 和 `covariance`**
#     train_returns = returns.loc[train_start:train_end].copy()
#     train_regimes = regime_data.loc[train_start:train_end]

#     regime_mu_cov = {}  # 存储每个 regime 的 mu 和 covariance
#     for regime in train_regimes.unique():
#         regime_mask = train_regimes == regime
#         regime_returns = train_returns[regime_mask[train_returns.index[0]:train_returns.index[-1]]]  # 只取该 `regime` 下的收益

#         # **✅ 仅保留 `train_period` 该 `regime` 均值收益 > 0 的资产**
#         mean_returns = regime_returns.mean()
#         selected_assets = mean_returns[mean_returns > 0].index.tolist()
#         regime_returns = regime_returns[selected_assets]  # 只保留均值收益 > 0 的资产

#         if len(regime_returns) < 30:
#             print(f"Skipping regime {regime} in train_period (not enough data)")
#             continue

#         mu = mean_historical_return(regime_returns, returns_data=True)
#         S = CovarianceShrinkage(regime_returns, returns_data=True).ledoit_wolf()
#         regime_mu_cov[regime] = (mu, S)

#     # **确定调仓时间点**
#     rebalance_dates = [d for d in pd.date_range(start=start_date, end=end_date, freq=rebalancing_freq) if d in returns.index]

#     # 仅考虑 test_period 内的 Regime 变化调仓点（变化的第二天）
#     regime_changes = regime_data.loc[start_date:end_date].shift(1) != regime_data.loc[start_date:end_date]
#     regime_change_dates = regime_data.loc[start_date:end_date].index[regime_changes]
#     rebalance_dates.extend(regime_change_dates)
#     rebalance_dates = sorted(set(rebalance_dates))  # 去重 & 排序
    
#     # 初始化投资组合权重和收益
#     weights_dict = {}
#     portfolio_returns = []
#     dates = []

#     for i in range(len(rebalance_dates)):
#         rebalance_date = rebalance_dates[i]
#         # 获取当前 Regime
#         current_regime = regime_data.loc[rebalance_date]

#         # **✅ 仅使用 `train_period` 内该 `regime` 筛选后的资产**
#         if current_regime in regime_mu_cov:
#             mu, S = regime_mu_cov[current_regime]
#         else:
#             print(f"Skipping rebalancing on {rebalance_date} (no training data for regime {current_regime})")
#             weights_dict[rebalance_date] = {asset: 0 for asset in price_data.columns}  # 直接空仓
#             continue

#         # **进行投资组合优化**
#         ef = EfficientFrontier(mu, S)
#         # ef.add_constraint(lambda w: w <= 0.4)  # 限制最大权重 40%
#         if strategy == 'max_sharpe':
#             ef.max_sharpe()
#         elif strategy == 'min_volatility':
#             ef.min_volatility()
#         elif strategy == 'max_quadratic_utility':
#             ef.max_quadratic_utility(risk_aversion=10.0)
#         else:
#             raise ValueError("Unsupported strategy. Use 'max_sharpe', 'min_volatility', or 'max_quadratic_utility'.")
#         cleaned_weights = ef.clean_weights()

#         # **确保未选资产的权重为 0**
#         full_weights = {asset: cleaned_weights.get(asset, 0) for asset in price_data.columns}
#         weights_dict[rebalance_date] = full_weights

#         # print(f"Rebalancing on {rebalance_date.strftime('%Y-%m-%d')} | Regime: {current_regime}")

#         # **计算投资组合收益**
#         end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else end_date
#         period_returns = returns.loc[rebalance_date:end]
#         weight_vector = np.array([full_weights.get(asset, 0) for asset in price_data.columns])

#         # 计算该期间的组合收益
#         period_portfolio_returns = period_returns @ weight_vector
#         portfolio_returns.extend(period_portfolio_returns.tolist())
#         dates.extend(period_returns.index.tolist())

#     # **整理投资组合收益**
#     portfolio_returns = pd.Series(portfolio_returns, index=dates).dropna()
#     portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep='first')]  # 删掉重复的行

#     # **计算累积收益**
#     portfolio_cumulative = (1 + portfolio_returns).cumprod()
#     portfolio_cumulative = portfolio_cumulative.reindex(price_data.loc[start_date:end_date].index, method='ffill')

#     # # **绘制收益曲线**
#     # plt.figure(figsize=(12, 6))
#     # plt.plot(portfolio_cumulative, label="Backtest Portfolio", linewidth=2)
#     # plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)  # 参考线
#     # plt.xlabel("Date"), plt.ylabel("Cumulative Return"), plt.title("Backtest Portfolio Performance")
#     # plt.legend(), plt.grid(), plt.show()

#     return portfolio_cumulative, portfolio_returns, weights_dict


# def rolling_dynamic_backtest_regime_2(test_start, num_years):
#     # rolling return
#     return_list = []
#     weights_list = []

#     for i in range(num_years):
#         train_start = test_start - pd.DateOffset(years=10) + pd.DateOffset(days=1)  # 训练集起始时间
#         train_end = test_start  # 训练集结束时间
#         test_end = test_start + pd.DateOffset(years=1)  # 测试集结束时间
        
#         # print(f"Iteration {i+1}:")
#         # print(f"train_start = {train_start.strftime('%Y-%m-%d')}")
#         # print(f"train_end = {train_end.strftime('%Y-%m-%d')}")
#         # print(f"test_start = {test_start.strftime('%Y-%m-%d')}")
#         # print(f"test_end = {test_end.strftime('%Y-%m-%d')}")
#         # print("-" * 40)
        
#         ## train-test split
#         X_train = filter_date_range(features, start_date=train_start, end_date=test_start)
#         X_test = filter_date_range(features, start_date=test_start, end_date=test_end)
#         # print time split
#         train_start, train_end = X_train.index[[0, -1]]
#         test_start, test_end = X_test.index[[0, -1]]
#         # print("Training starts at:", train_start, "and ends at:", train_end)
#         # print("Testing starts at:", test_start, "and ends at:", test_end)
#         # print("-" * 40)

#         ## preproscessing
#         from jumpmodels.preprocess import StandardScalerPD, DataClipperStd
#         clipper = DataClipperStd(mul=3.)
#         scalar = StandardScalerPD()
#         # fit on training data
#         X_train_processed = scalar.fit_transform(clipper.fit_transform(X_train))
#         # transform the test data
#         X_test_processed = scalar.transform(clipper.transform(X_test))

#         # ## regime detection
#         # jump_penalty=60.
#         # jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )
#         # jm.fit(X_train_processed, SP500_ret, sort_by="cumret")
#         # labels_test_online = jm.predict_online(X_test_processed.dropna())  # make online inference
#         # labeled_data = pd.DataFrame(pd.concat([jm.labels_,labels_test_online]),columns=["regime"]).join(raw_data, how='left')
#         # labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]
#         # labeled_start, labeled_end = labeled_data.index[[0, -1]]
#         # # print("Labeled starts at:", labeled_start, "and ends at:", labeled_end)
#         # # print("-" * 40)

#         ## optimize the jump penalty
#         jump_penalty_list = [50.0, 55.0, 60.0]
#         train_score = []
#         for jump_penalty in jump_penalty_list:
#             ## regime detection
#             jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )
#             jm.fit(X_train_processed, SP500_ret, sort_by="cumret")
#             labels_test_online = jm.predict_online(X_test_processed.dropna())  # make online inference
#             labeled_data = pd.DataFrame(pd.concat([jm.labels_,labels_test_online]),columns=["regime"]).join(raw_data, how='left')
#             labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]
#             labeled_start, labeled_end = labeled_data.index[[0, -1]]
#             portfolio_cumulative_with_regime, portfolio_returns, weights_dict = backtest_regime_portfolio_2(
#                 labeled_data.fillna(method="ffill"), 
#                 test_period=(train_start, train_end), 
#                 train_period=(train_start, train_end), 
#                 rebalancing_freq='YE', 
#                 strategy='min_volatility'
#             )
#             train_score.append(sharpe_tario(portfolio_cumulative_with_regime))
#         jump_penalty = jump_penalty_list[train_score.index(max(train_score))]

#         ## strategy
#         jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )
#         jm.fit(X_train_processed, SP500_ret, sort_by="cumret")
#         labels_test_online = jm.predict_online(X_test_processed.dropna())  # make online inference
#         labeled_data = pd.DataFrame(pd.concat([jm.labels_,labels_test_online]),columns=["regime"]).join(raw_data, how='left')
#         labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]
#         portfolio_cumulative_with_regime, portfolio_returns, weights_dict = backtest_regime_portfolio_2(
#             labeled_data.fillna(method="ffill"), 
#             test_period=(test_start, test_end), 
#             train_period=(train_start, train_end), 
#             rebalancing_freq='YE', 
#             strategy='min_volatility'
#         )
#         return_list.append(portfolio_returns)
#         weights_list.append(weights_dict)

#         # test_start 每次向后移动一年
#         test_start = test_start + pd.DateOffset(years=1)

    
#     rolling_returns = pd.concat(return_list)
#     rolling_returns = rolling_returns[~rolling_returns.index.duplicated(keep='first')]  # 删掉重复的行

#     # 计算累积收益
#     rolling_cumulative = (1 + rolling_returns).cumprod()

#     # 处理weights df
#     merged = {}
#     for d in weights_list:
#         merged.update(d)
#     rolling_dynamic_weights = pd.DataFrame.from_dict(merged, orient='index')
#     rolling_dynamic_weights.index.name = 'date'

#     return rolling_cumulative, rolling_returns, rolling_dynamic_weights

# def backtest_regime_portfolio_3(
#     labeled_data, 
#     test_period, 
#     train_period,  
#     rebalancing_freq='Y', 
#     strategy='max_sharpe'
# ):
#     """
#     基于 pypfopt 进行动态投资组合优化，并回测其收益情况（支持市场状态 Regime 变化调仓）。
    
#     参数：
#     - labeled_data: DataFrame, 包含资产价格和市场状态 (regime) 的数据，index 为时间，columns 为资产价格 + "regime"
#     - test_period: tuple, (测试集开始时间, 测试集结束时间)
#     - train_period: tuple, (训练集开始时间, 训练集结束时间) **计算不同 `regime` 下的 `mu` 和 `covariance`**
#     - rebalancing_freq: str, 正常调仓频率 ('M'：月度, 'Q'：季度, 'Y'：年度)
#     - strategy: str, 投资组合优化策略 ('min_volatility', 'max_sharpe', 'max_quadratic_utility')

#     返回：
#     - portfolio_cumulative: DataFrame, 投资组合的累积收益
#     - portfolio_returns: DataFrame, 投资组合的每日收益
#     - weights_dict: dict, 每次调仓的权重信息
#     """
#     # 分离价格数据和 Regime 数据
#     price_data = labeled_data.drop(columns=['regime'])
#     regime_data = labeled_data['regime']
    
#     # 计算每日收益率
#     returns = price_data.pct_change().dropna()

#     # 提取时间范围
#     start_date, end_date = test_period
#     train_start, train_end = train_period  

#     # **计算 `train_period` 内，每个 `regime` 的 `mu` 和 `covariance`**
#     train_returns = returns.loc[train_start:train_end].copy()
#     train_regimes = regime_data.loc[train_start:train_end]

#     regime_mu_cov = {}  # 存储每个 regime 的 mu 和 covariance
#     for regime in train_regimes.unique():
#         regime_mask = train_regimes == regime
#         # print(len(train_returns),len(regime_mask))
#         # print("the index of price_data:", price_data.index[0], price_data.index[-1])
#         # print("the index of regime_data:", regime_data.index[0], regime_data.index[-1])
#         # print("the index of returns:", returns.index[0], returns.index[-1])
#         # print("the index of regime_data:", regime_data.index[0], regime_data.index[-1])
#         # print("the index of train_returns:", train_returns.index[0], train_returns.index[-1])
#         # print("the index of train_regimes:", train_regimes.index[0], train_regimes.index[-1])
#         # print("the index of regime_mask:", regime_mask.index[0], regime_mask.index[-1])
#         regime_returns = train_returns[regime_mask[train_returns.index[0]:train_returns.index[-1]]]  # 只取该 `regime` 下的收益

#         if len(regime_returns) < 30:
#             print(f"Skipping regime {regime} in train_period (not enough data)")
#             continue

#         mu = mean_historical_return(regime_returns, returns_data=True)
#         S = CovarianceShrinkage(regime_returns, returns_data=True).ledoit_wolf()
#         regime_mu_cov[regime] = (mu, S)

#     # # **确定调仓时间点**
#     # rebalance_dates = [d for d in pd.date_range(start=start_date, end=end_date, freq=rebalancing_freq) if d in returns.index]

#     # # 仅考虑 test_period 内的 Regime 变化调仓点（变化的第二天）
#     # regime_changes = regime_data.loc[start_date:end_date].shift(1) != regime_data.loc[start_date:end_date]
#     # regime_change_dates = regime_data.loc[start_date:end_date].index[regime_changes]
#     # rebalance_dates.extend(regime_change_dates)
#     # rebalance_dates = sorted(set(rebalance_dates))  # 去重 & 排序

#     # 确定调仓时间点
#     whole_dates = price_data.index
#     # rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalancing_freq)
#     rebalance_dates = whole_dates[(whole_dates.searchsorted(pd.date_range(start=start_date, end=end_date, freq="YE")) - 1).clip(0)]
#     rebalance_dates = rebalance_dates.tolist()
#     rebalance_dates.insert(0, whole_dates[whole_dates.searchsorted(start_date, side="right") - 1])
#     rebalance_dates = pd.DatetimeIndex(rebalance_dates)
#     rebalance_dates
    
#     # 初始化投资组合权重和收益
#     weights_dict = {}
#     portfolio_returns = []
#     dates = []

#     for i in range(len(rebalance_dates)):
#         rebalance_date = rebalance_dates[i]
#         # 获取当前 Regime
#         current_regime = regime_data.loc[rebalance_date]
#         # current_regime = regime_data.loc[whole_dates[whole_dates.searchsorted(start_date, side="right") - 1 -1]]
#         # **✅ 直接使用 `train_period` 内该 `regime` 计算的 `mu` 和 `covariance`**
#         if current_regime in regime_mu_cov:
#             mu, S = regime_mu_cov[current_regime]
#         else:
#             print(f"Skipping rebalancing on {rebalance_date} (no training data for regime {current_regime})")
#             weights_dict[rebalance_date] = {asset: 0 for asset in price_data.columns}  # 直接空仓
#             continue

#         selected_assets = mu.index.tolist()

#         # **进行投资组合优化**
#         ef = EfficientFrontier(mu, S)
#         # ef.add_constraint(lambda w: w <= 0.4)  # 限制最大权重 40%
#         if strategy == 'max_sharpe':
#             ef.max_sharpe()
#         elif strategy == 'min_volatility':
#             ef.min_volatility()
#         elif strategy == 'max_quadratic_utility':
#             ef.max_quadratic_utility(risk_aversion=10.0)
#         else:
#             raise ValueError("Unsupported strategy. Use 'max_sharpe', 'min_volatility', or 'max_quadratic_utility'.")
#         cleaned_weights = ef.clean_weights()

#         # **确保未选资产的权重为 0**
#         full_weights = {asset: cleaned_weights.get(asset, 0) for asset in price_data.columns}
#         weights_dict[rebalance_date] = full_weights

#         # print(f"Rebalancing on {rebalance_date.strftime('%Y-%m-%d')} | Regime: {current_regime}")

#         # **计算投资组合收益**
#         end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else end_date
#         period_returns = returns.loc[rebalance_date:end]
#         weight_vector = np.array([full_weights.get(asset, 0) for asset in price_data.columns])

#         # 计算该期间的组合收益
#         period_portfolio_returns = period_returns @ weight_vector
#         portfolio_returns.extend(period_portfolio_returns.tolist())
#         dates.extend(period_returns.index.tolist())

#     # **整理投资组合收益**
#     portfolio_returns = pd.Series(portfolio_returns, index=dates).dropna()
#     portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep='first')]  # 删掉重复的行

#     # **计算累积收益**
#     portfolio_cumulative = (1 + portfolio_returns).cumprod()
#     portfolio_cumulative = portfolio_cumulative.reindex(price_data.loc[start_date:end_date].index, method='ffill')

#     # # **绘制收益曲线**
#     # plt.figure(figsize=(12, 6))
#     # plt.plot(portfolio_cumulative, label="Backtest Portfolio", linewidth=2)
#     # plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)  # 参考线
#     # plt.xlabel("Date"), plt.ylabel("Cumulative Return"), plt.title("Backtest Portfolio Performance")
#     # plt.legend(), plt.grid(), plt.show()

#     return portfolio_cumulative, portfolio_returns, weights_dict


# def rolling_dynamic_backtest_regime_3(test_start, num_years):
#     # rolling return
#     return_list = []
#     weights_list = []

#     for i in range(num_years):
#         train_start = test_start - pd.DateOffset(years=10) + pd.DateOffset(days=1)  # 训练集起始时间
#         train_end = test_start  # 训练集结束时间
#         test_end = test_start + pd.DateOffset(years=1)  # 测试集结束时间
        
#         # print(f"Iteration {i+1}:")
#         # print(f"train_start = {train_start.strftime('%Y-%m-%d')}")
#         # print(f"train_end = {train_end.strftime('%Y-%m-%d')}")
#         # print(f"test_start = {test_start.strftime('%Y-%m-%d')}")
#         # print(f"test_end = {test_end.strftime('%Y-%m-%d')}")
#         # print("-" * 40)
        
#         ## train-test split
#         X_train = filter_date_range(features, start_date=train_start, end_date=test_start)
#         X_test = filter_date_range(features, start_date=test_start, end_date=test_end)
#         # print time split
#         train_start, train_end = X_train.index[[0, -1]]
#         test_start, test_end = X_test.index[[0, -1]]
#         # print("Training starts at:", train_start, "and ends at:", train_end)
#         # print("Testing starts at:", test_start, "and ends at:", test_end)
#         # print("-" * 40)

#         ## preproscessing
#         from jumpmodels.preprocess import StandardScalerPD, DataClipperStd
#         clipper = DataClipperStd(mul=3.)
#         scalar = StandardScalerPD()
#         # fit on training data
#         X_train_processed = scalar.fit_transform(clipper.fit_transform(X_train))
#         # transform the test data
#         X_test_processed = scalar.transform(clipper.transform(X_test))

#         # ## regime detection
#         # jump_penalty=60.
#         # jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )
#         # jm.fit(X_train_processed, SP500_ret, sort_by="cumret")
#         # labels_test_online = jm.predict_online(X_test_processed.dropna())  # make online inference
#         # labeled_data = pd.DataFrame(pd.concat([jm.labels_,labels_test_online]),columns=["regime"]).join(raw_data, how='left')
#         # labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]
#         # labeled_start, labeled_end = labeled_data.index[[0, -1]]
#         # # print("Labeled starts at:", labeled_start, "and ends at:", labeled_end)
#         # # print("-" * 40)

#         ## optimize the jump penalty
#         jump_penalty_list = [50.0, 55.0, 60.0]
#         train_score = []
#         for jump_penalty in jump_penalty_list:
#             ## regime detection
#             jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )
#             jm.fit(X_train_processed, SP500_ret, sort_by="cumret")
#             labels_test_online = jm.predict_online(X_test_processed.dropna())  # make online inference
#             labeled_data = pd.DataFrame(pd.concat([jm.labels_,labels_test_online]),columns=["regime"]).join(raw_data, how='left')
#             labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]
#             labeled_start, labeled_end = labeled_data.index[[0, -1]]
#             portfolio_cumulative_with_regime, portfolio_returns, weights_dict = backtest_regime_portfolio_3(
#                 labeled_data.fillna(method="ffill"), 
#                 test_period=(train_start, train_end), 
#                 train_period=(train_start, train_end), 
#                 rebalancing_freq='YE', 
#                 strategy='min_volatility'
#             )
#             train_score.append(sharpe_tario(portfolio_cumulative_with_regime))
#         jump_penalty = jump_penalty_list[train_score.index(max(train_score))]

#         ## strategy
#         jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )
#         jm.fit(X_train_processed, SP500_ret, sort_by="cumret")
#         labels_test_online = jm.predict_online(X_test_processed.dropna())  # make online inference
#         labeled_data = pd.DataFrame(pd.concat([jm.labels_,labels_test_online]),columns=["regime"]).join(raw_data, how='left')
#         labeled_data = labeled_data[~labeled_data.index.duplicated(keep='first')]
#         portfolio_cumulative_with_regime, portfolio_returns, weights_dict = backtest_regime_portfolio_3(
#             labeled_data.fillna(method="ffill"), 
#             test_period=(test_start, test_end), 
#             train_period=(train_start, train_end), 
#             rebalancing_freq='YE', 
#             strategy='min_volatility'
#         )
#         return_list.append(portfolio_returns)
#         weights_list.append(weights_dict)

#         # test_start 每次向后移动一年
#         test_start = test_start + pd.DateOffset(years=1)


#     rolling_returns = pd.concat(return_list)
#     rolling_returns = rolling_returns[~rolling_returns.index.duplicated(keep='first')]  # 删掉重复的行

#     # 计算累积收益
#     rolling_cumulative = (1 + rolling_returns).cumprod()

#     # 处理weights df
#     merged = {}
#     for d in weights_list:
#         merged.update(d)
#     rolling_dynamic_weights = pd.DataFrame.from_dict(merged, orient='index')
#     rolling_dynamic_weights.index.name = 'date'

#     return rolling_cumulative, rolling_returns, rolling_dynamic_weights



# def main():
#     print("Hello from dashboard-qwim!")

#     rolling_cumulative, rolling_returns, return_list = rolling_backtest(test_start=pd.to_datetime("2008-01-01"), num_years=14)
#     rolling_dynamic_cumulative_regime_1, rolling_dynamic_returns, rolling_dynamic_weights_1 = rolling_dynamic_backtest_regime_1(test_start = pd.to_datetime("2008-01-01"), num_years = 14)
#     rolling_dynamic_cumulative_regime_2, rolling_dynamic_returns, rolling_dynamic_weights_2 = rolling_dynamic_backtest_regime_2(test_start = pd.to_datetime("2008-01-01"), num_years = 14)
#     rolling_dynamic_cumulative_regime_3, rolling_dynamic_returns, rolling_dynamic_weights_3 = rolling_dynamic_backtest_regime_3(test_start = pd.to_datetime("2008-01-01"), num_years = 14)

#     strategy_returns = pd.DataFrame(
#         {
#             "annually without regime": rolling_cumulative["Portfolio Return"],
#             "annually regime-based": rolling_dynamic_cumulative_regime_3,
#             "dynamic regime-based": rolling_dynamic_cumulative_regime_1,
#             "regime-based with positive assets": rolling_dynamic_cumulative_regime_2
#         }
#     )
#     strategy_returns = strategy_returns.dropna()
#     strategy_returns.index.name = "Date"
#     JM_benchmark = pd.DataFrame(index=strategy_returns.index, data={"Value": strategy_returns["annually without regime"]})
#     JM_portfolio = pd.DataFrame(index=strategy_returns.index, data={"Value": strategy_returns["regime-based with positive assets"]})
#     JM_benchmark.to_csv("JM_benchmark_portfolio_values.csv")
#     JM_portfolio.to_csv("JM_portfolio_values.csv")
#     rolling_dynamic_weights_2.to_csv("JM_portfolio_weights_ETFs.csv")



# if __name__ == "__main__":
#     main()
