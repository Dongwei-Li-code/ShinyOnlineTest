import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from scipy.stats import poisson, gamma
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
from pypfopt.cla import CLA
from pypfopt.objective_functions import L2_reg
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]  # 向上两层：src → dashboard-QWIM
excel_path =project_root / 'data' / 'raw' / 'indices.xlsx'
feature_path=project_root / 'data' / 'raw' / 'feature.csv'
hmm_benchmark_output_path =  project_root / "data" / "processed" /"hmm_benchmark_portfolio_values.csv"
hmm_portfolio_output_path =  project_root / "data" / "processed" / "hmm_portfolio_values.csv"
hmm_weights_output_path = project_root / 'data' / 'raw' / "hmm_portfolio_weights.csv"
raw_data = pd.read_excel(excel_path, index_col=0, parse_dates=True)
raw_data = raw_data.drop("SP500",axis=1)


def backtest_portfolio(
    price_data, 
    test_period, 
    lookback_period=252, 
    rebalancing_freq='Y', 
    strategy='max_sharpe'
):
    

    # 计算每日收益率
    returns = price_data.pct_change().dropna()

    # 提取全部时间
    whole_dates = price_data.index
    
    # 确定测试集时间范围
    start_date, end_date = test_period
    test_data = price_data.loc[start_date:end_date]
    
    # 确定调仓时间点
    # rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalancing_freq)
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq="YE").tolist()
    rebalance_dates.insert(0, whole_dates[whole_dates.searchsorted(start_date, side="right") - 1])
    rebalance_dates = pd.DatetimeIndex(rebalance_dates)
    rebalance_dates
    
    # 初始化投资组合权重
    weights_dict = {}
    
    for rebalance_date in rebalance_dates:
        # 选择回溯训练集数据
        train_end = rebalance_date
        # train_start = train_end - pd.Timedelta(days=lookback_period)
        train_start = whole_dates[whole_dates.searchsorted(start_date, side="right") - 1 -lookback_period]
        
        train_data = price_data.loc[train_start:train_end]


        # 进行优化
        print(f"Rebalancing on {rebalance_date.strftime('%Y-%m-%d')} | Train period: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        
        # 计算期望收益率和协方差矩阵
        mu = mean_historical_return(train_data)
        # S = train_data.pct_change().dropna().cov()
        S = train_data.dropna().pct_change().ewm(span=252).cov(pairwise=True).loc[train_data.index[-1]]*252
        # S = CovarianceShrinkage(train_data).ledoit_wolf()
        
        # 选择优化策略
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
        
        # 记录资产权重
        cleaned_weights = ef.clean_weights()
        weights_dict[rebalance_date] = cleaned_weights
    
    # 回测收益
    portfolio_returns = []
    dates = []
    
    for i in range(len(rebalance_dates) - 1):
        start = rebalance_dates[i]
        end = rebalance_dates[i + 1]
        
        # if end not in returns.index:
        #     continue
        
        # 获取当前区间的资产收益率
        period_returns = returns.loc[start:end]
        
        # 获取上次调仓时的权重
        last_weights = weights_dict[rebalance_dates[i]]
        weight_vector = np.array([last_weights[asset] for asset in price_data.columns])
        
        # 计算该期间的组合收益
        period_portfolio_returns = period_returns @ weight_vector
        portfolio_returns.extend(period_portfolio_returns.tolist())
        dates.extend(period_returns.index.tolist())
    
    # 转换为 DataFrame
    portfolio_returns = pd.DataFrame(portfolio_returns, index=dates, columns=['Portfolio Return'])
    portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep='first')]  # 删掉重复的行
    
    # 计算累积收益
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    portfolio_cumulative = portfolio_cumulative.reindex(price_data.loc[start_date:end_date].index, method='ffill')  
    
    
    return portfolio_cumulative,portfolio_returns,weights_dict


class CustomHSMM:
    def __init__(self, n_states, duration_model="poisson"):
        self.n_states = n_states
        self.duration_model = duration_model
        self.lambda_ = None
        self.alpha = None
        self.beta = None
        self.hmm = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000)

    def fit(self, X):
        self.hmm.fit(X)
        self.hidden_states = self.hmm.predict(X)
        self.log_likelihood = self.hmm.score(X)

        # 计算持续时间
        change_points = np.where(np.diff(self.hidden_states) != 0)[0]
        durations = np.diff(np.concatenate(([0], change_points, [len(self.hidden_states)])))

        if self.duration_model == "poisson":
            self.lambda_ = np.mean(durations)
        elif self.duration_model == "gamma":
            self.alpha, _, self.beta = gamma.fit(durations, floc=0)

    def predict(self, X):
        return self.hmm.predict(X)

    def score(self, X):
        return self.hmm.score(X)

# ============ 模型训练函数 ============ #
def train_hmm(n_states, features):
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000)
    model.fit(features)
    hidden_states = model.predict(features)
    log_likelihood = model.score(features)
    n_params = n_states * (n_states - 1 + features.shape[1])
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + np.log(features.shape[0]) * n_params
    return model, hidden_states, log_likelihood, aic, bic, model.transmat_

def train_hsmm(n_states, features, duration_model):
    hsmm = CustomHSMM(n_states=n_states, duration_model=duration_model)
    hsmm.fit(features)
    log_likelihood = hsmm.score(features)
    aic = -2 * log_likelihood + 2 * n_states
    bic = -2 * log_likelihood + np.log(features.shape[0]) * n_states
    return hsmm, log_likelihood, aic, bic, hsmm.lambda_, hsmm.alpha, hsmm.beta


def compute_state_statistics(name, hidden_states, df, n_states):
    stats = []
    for s in range(n_states):
        state_returns = df['Return'].values[hidden_states == s]
        mean_ret = np.mean(state_returns)
        vol = np.std(state_returns)
        sharpe = mean_ret / vol if vol > 0 else np.nan
        stats.append({'State': s, 'Mean Return': mean_ret, 'Volatility': vol, 'Sharpe Ratio': sharpe})
    return pd.DataFrame(stats)

df = pd.read_csv(feature_path, parse_dates=["Date"])
df = df.set_index("Date")


df['MA_5'] = df['Return'].rolling(5).mean()
df['Vol_10'] = df['Return'].rolling(10).std()
df['Return_5D'] = df['Return'].rolling(5).sum()
df['VIX_Change'] = df['VIX'].diff()


feature_cols = ['Return', 'VIX', '10Y Treasury Rate', 'MA_5', 'Vol_10', 'Return_5D', 'VIX_Change']


df = df.dropna()


features_df = df[feature_cols].copy()






assets = [
    "DBLCDBCE",
    "DJUSRET",
    "GOLDLNPM",
    "IBOXHY",
    "LBUSTRUU",
    "LUACTRUU",
    "LUTLTRUU",
    "NDDUEAFE",
    "NDUEEGF",
    "RU20INTR Index",
    "SP500",
    "SPTR",
    "SPTRMDCP"
]

price_cols = assets

price_df = df[price_cols].copy()

labeled_data = price_df

def rolling_backtest(test_start, num_years):
    # rolling return
    return_list = []

    for i in range(num_years):
        train_start = test_start - pd.DateOffset(years=10) + pd.DateOffset(days=1)  # 训练集起始时间
        train_end = test_start  # 训练集结束时间
        test_end = test_start + pd.DateOffset(years=1)  # 测试集结束时间
        
        
        portfolio_cumulative_without_regime,portfolio_returns,weights_dict = backtest_portfolio(
            raw_data.fillna(method="ffill"), 
            test_period=(test_start, test_end), 
            lookback_period=2520, 
            rebalancing_freq='YE', 
            strategy='min_volatility'
        )
        return_list.append(portfolio_returns)


        # test_start 每次向后移动一年
        test_start = test_start + pd.DateOffset(years=1)

    rolling_returns = pd.concat(return_list)
    rolling_returns = rolling_returns[~rolling_returns.index.duplicated(keep='first')]  # 删掉重复的行
    rolling_cumulative = (1 + rolling_returns).cumprod()
    
    return rolling_cumulative, rolling_returns, return_list

def backtest_regime_portfolio(labeled_data, test_period, train_period, rebalancing_freq='QE', strategy='max_quadratic_utility'):
    """
    基于 HMM Regime 的回测函数。
    
    参数：
      - labeled_data: DataFrame，包含资产价格数据及 'regime' 列（index 为日期）
      - test_period: tuple, (测试开始日期, 测试结束日期)（字符串格式，如 ("2015-01-01", "2020-12-31")）
      - train_period: tuple, (训练开始日期, 训练结束日期)
      - rebalancing_freq: str, 调仓频率（例如 'M', 'Q' 或 'Y'）
      - strategy: str, 优化目标 ('max_sharpe', 'min_volatility' 或 'max_quadratic_utility')
      
    返回：
      - portfolio_cumulative: Series，组合累计收益
      - portfolio_returns: Series，每日组合收益
      - weights_dict: dict，每个调仓日对应的资产权重
    """
    # 分离价格数据与 regime 信息
    price_data = labeled_data.drop(columns=['regime'])
    print("price_data columns:", price_data.columns)
    regime_data = labeled_data['regime']
    
    # 计算每日收益率
    returns = price_data.pct_change().dropna()
    
    start_date, end_date = test_period
    train_start, train_end = train_period
    
    # 在训练期内，根据 regime 筛选正收益资产，并计算 mu 与协方差矩阵
    train_returns = returns.loc[train_start:train_end].copy()
    train_regimes = regime_data.loc[train_start:train_end]
    
    print("Training period regime distribution:")
    print(train_regimes.value_counts(dropna=False))
    
    regime_mu_cov = {}
    for reg in train_regimes.unique():
        # 跳过 NaN 值
        if pd.isna(reg):
            print("Skipping regime NaN in train_period")
            continue
            
        # 统一转换为 int 类型
        reg_int = int(reg)
        regime_mask = (train_regimes == reg)
        regime_returns = train_returns[regime_mask]
        
        if len(regime_returns) < 30:
            print(f"Skipping regime {reg_int} in train_period (not enough data: {len(regime_returns)} samples)")
            continue
        
        asset_mean = regime_returns.mean()
        positive_assets = asset_mean[asset_mean > 0].index.tolist()
        if 'SP500' not in positive_assets:
            positive_assets.append('SP500')
        if len(positive_assets) == 0:
            print(f"Regime {reg_int} in train_period: 无正收益资产，跳过")
            continue
        
        regime_returns_filtered = regime_returns[positive_assets]
        mu = mean_historical_return(regime_returns_filtered, returns_data=True)
        S = CovarianceShrinkage(regime_returns_filtered, returns_data=True).ledoit_wolf()
        regime_mu_cov[reg_int] = (mu, S)
        print(f"Regime {reg_int}: {len(regime_returns_filtered)} samples, assets used: {positive_assets}")
        
    # 确定调仓日期：合并固定调仓日期和 regime 变化触发日期
    candidate_dates = [d for d in pd.date_range(start=start_date, end=end_date, freq=rebalancing_freq) if d in returns.index]
    regime_change = regime_data.loc[start_date:end_date].shift(1) != regime_data.loc[start_date:end_date]
    regime_change_dates = regime_data.loc[start_date:end_date].index[regime_change].tolist()
    rebalance_dates = sorted(list(set(candidate_dates + regime_change_dates)))
    print("Rebalance dates:", rebalance_dates)
    
    weights_dict = {}
    port_returns = []
    ret_dates = []
    
    for i in range(len(rebalance_dates)):
        reb_date = rebalance_dates[i]
        current_regime = regime_data.loc[reb_date]
        if pd.isna(current_regime):
            print(f"Rebalance date {reb_date} has NaN regime, skipping")
            weights_dict[reb_date] = {asset: 0 for asset in price_data.columns}
            continue
        current_regime = int(current_regime)
        
        if current_regime in regime_mu_cov:
            mu, S = regime_mu_cov[current_regime]
        else:
            print(f"Skipping rebalancing on {reb_date} (no training data for regime {current_regime})")
            weights_dict[reb_date] = {asset: 0 for asset in price_data.columns}
            continue
        
        # 均值–方差优化
        ef = EfficientFrontier(mu, S)
        if strategy == 'max_sharpe':
            ef.max_sharpe()
        elif strategy == 'min_volatility':
            ef.min_volatility()
        elif strategy == 'max_quadratic_utility':
            ef.max_quadratic_utility(risk_aversion=10.0)
        else:
            raise ValueError("Unsupported strategy")
        
        cleaned_weights = ef.clean_weights()
        full_weights = {asset: cleaned_weights.get(asset, 0) for asset in price_data.columns}
        weights_dict[reb_date] = full_weights
        
        # 持仓期：从当前调仓日至下一个调仓日（最后一次到测试期末）
        period_end = rebalance_dates[i+1] if i+1 < len(rebalance_dates) else end_date
        period_returns = returns.loc[reb_date:period_end]
        w_vec = np.array([full_weights.get(asset, 0) for asset in price_data.columns])
        port_ret = period_returns.dot(w_vec)
        ret_dates.extend(port_ret.index.tolist())
        port_returns.extend(port_ret.tolist())
    
    portfolio_returns = pd.Series(port_returns, index=ret_dates).dropna()
    portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep='first')]
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    portfolio_cumulative = portfolio_cumulative.reindex(price_data.loc[start_date:end_date].index, method='ffill')
    
    return portfolio_cumulative, portfolio_returns, weights_dict


def rolling_backtest_regime_1(test_start, num_years):
    """
    滚动回测：每次以过去10年数据作为训练集回测未来1年，并拼接所有结果。
    """
    return_list = []
    weights_list = []

    def filter_date_range(df, start_date, end_date):
        return df.loc[start_date:end_date]
    
    for i in range(num_years):
        train_start = test_start - pd.DateOffset(years=10) + pd.DateOffset(days=1)
        train_end = test_start
        test_end = test_start + pd.DateOffset(years=1)
        
        # 切分出训练集和测试集（features_df 为原始特征，不含未来数据）
        X_train = filter_date_range(features_df, train_start, train_end)
        X_test  = filter_date_range(features_df, test_start, test_end)
        train_start_eff, train_end_eff = X_train.index[0], X_train.index[-1]
        test_start_eff, test_end_eff = X_test.index[0], X_test.index[-1]
        
        # --- 用训练集归一化与模型拟合 ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        hsmm = CustomHSMM(n_states=4, duration_model="gamma")
        hsmm.fit(X_train_scaled)
        market_states_train = hsmm.predict(X_train_scaled)
        
        # 使用训练期得到的归一化参数转换测试集，并预测测试期 regime
        X_test_scaled = scaler.transform(X_test)
        market_states_test = hsmm.predict(X_test_scaled)
        
        # 更新窗口内 labeled_data 的 regime 列：将训练期与测试期均用当前窗口预测结果
        labeled_data_period = labeled_data.loc[train_start:test_end].copy()
        # 拼接训练期与测试期预测得到的 regime（确保索引顺序一致）
        all_regimes = np.concatenate([market_states_train, market_states_test])
        # 注意：这里假设 X_train.index 与 X_test.index 的顺序自然连续，不存在重叠
        labeled_data_period.loc[train_start:test_end, 'regime'] = all_regimes

        # 输出训练期 regime 分布以便调试
        train_regime_dist = labeled_data_period.loc[train_start:train_end, 'regime'].value_counts(dropna=False)
        print(f"Rolling window {train_start.date()} to {train_end.date()} regime distribution:")
        print(train_regime_dist)
        
        # 调用回测函数：训练期和测试期均有最新的 regime 列
        port_cum, port_ret, w_dict = backtest_regime_portfolio(
            labeled_data=labeled_data_period,
            test_period=(test_start_eff.strftime("%Y-%m-%d"), test_end_eff.strftime("%Y-%m-%d")),
            train_period=(train_start_eff.strftime("%Y-%m-%d"), train_end_eff.strftime("%Y-%m-%d")),
            rebalancing_freq='QE',
            strategy='max_quadratic_utility'
        )
        return_list.append(port_ret)
        weights_list.append(w_dict)
        test_start = test_start + pd.DateOffset(years=1)
    
    rolling_returns = pd.concat(return_list)
    rolling_returns = rolling_returns[~rolling_returns.index.duplicated(keep='first')]
    rolling_cumulative = (1 + rolling_returns).cumprod()
    
    merged = {}
    for d in weights_list:
        merged.update(d)
    rolling_dynamic_weights = pd.DataFrame.from_dict(merged, orient='index')
    rolling_dynamic_weights.index.name = 'date'
    rolling_dynamic_weights.sort_index(inplace=True)
    
    return rolling_cumulative, rolling_returns, rolling_dynamic_weights



def main():
    train_start = "2002-01-03"
    train_end = "2011-12-31"
    test_start = "2011-12-31"
    test_end = "2012-12-31"

    rolling_cumulative, rolling_returns, return_list = rolling_backtest(test_start=pd.to_datetime("2008-01-01"), num_years=14)
    rolling_cum_regime_1, rolling_ret_regime_1,rolling_dynamic_weights_1 = rolling_backtest_regime_1(
    test_start=pd.to_datetime("2008-01-01"),
    num_years=14)
    
    
    strategy_returns=pd.DataFrame(
    {
        "annually without regime": rolling_cumulative["Portfolio Return"],
        " regime-based": rolling_cum_regime_1
        
    })
    strategy_returns = strategy_returns.dropna()
    strategy_returns.index.name = "Date"
    hmm_benchmark = pd.DataFrame(index=strategy_returns.index, data={"Value": strategy_returns["annually without regime"]})
    hmm_portfolio = pd.DataFrame(index=strategy_returns.index, data={"Value": strategy_returns["regime-based "]})
    hmm_benchmark.to_csv(hmm_benchmark_output_path)
    hmm_portfolio.to_csv(hmm_portfolio_output_path)
    rolling_dynamic_weights_1.to_csv(hmm_weights_output_path)
if __name__ == "__main__":
    main()
