Based on the analysis of your exogenous features, here's how to determine which safe transformations are suitable for each feature category:

## **Technical Indicators** (All Differencing + Rolling Statistics)

**Suitable for all transformations:**
- `btc_sma_*`, `btc_ema_*` series, `btc_macd*`, `btc_bb_*`, `btc_rsi_14`, `btc_atr_14`, `btc_volatility_index`, `btc_trading_volume`

**Reasoning**: Technical indicators are typically time series with trends and seasonality patterns, making them ideal candidates for differencing operations and rolling statistics.

**Recommended transformations:**
- **First difference**: Captures short-term momentum changes
- **Log returns**: For price-derived indicators (moving averages, Bollinger bands)
- **Rolling statistics**: Smooths volatility and identifies regime changes

## **Blockchain Indicators** (Differencing + Rolling Statistics, No Log Returns)

**Features:** `active_addresses_blockchain`, `hash_rate_blockchain`, `miner_revenue_blockchain`, `difficulty_blockchain`, `estimated_transaction_volume_usd_blockchain`, `*_cbbi` indicators

**Avoid log returns** because blockchain metrics are often count data or cumulative measures where log transformations may not be meaningful.

**Recommended transformations:**
- **First difference**: Captures network growth changes
- **Seasonal difference**: Useful for mining difficulty adjustments
- **Rolling statistics**: Smooths network activity fluctuations

## **Sentiment Indicators** (Rolling Statistics Only)

**Features:** All `*_sentiment` variables, `Fear Greed`

**Only rolling statistics recommended** because sentiment scores are often bounded or normalized, making differencing potentially problematic.

**Recommended transformations:**
- **Rolling mean**: Captures sentiment trends
- **Rolling std**: Measures sentiment volatility
- **Rolling min/max**: Identifies sentiment extremes

## **Cross-Market Indicators** (All Transformations)

**Features:** `Gold_Price`, `Gold_*`, `Oil_*`, `DJI`, `GSPC`, `IXIC`, `NYFANG`, `CBOE_Volatility`, `EM_ETF`, `DXY`, `EURUSD`

**All transformations suitable** since these are traditional financial time series.

**Recommended transformations:**
- **Log returns**: Ideal for price indices and currency pairs
- **First difference**: For volatility indices
- **Rolling statistics**: For regime detection

## **Special Considerations**

**Exclude from transformations:**
- **`Date`**: Not suitable for any mathematical transformations
- **Already derived features**: Features like `btc_sma_14_50_diff`, `btc_macd_diff` are already differenced

**Priority transformations by feature type:**

| Feature Type | Primary | Secondary | Avoid |
|--------------|---------|-----------|-------|
| Price-based (SMA, EMA, Gold, Oil) | Log returns, First difference | Rolling mean | None |
| Volume-based (trading_volume, blockchain volumes) | First difference | Rolling statistics | Log returns for zero values |
| Ratio/Index (RSI, CBBI indicators) | Rolling statistics | First difference | Log returns |
| Sentiment scores | Rolling statistics only | None | All differencing |

**Implementation tip**: Start with the primary transformations for each category, then use feature selection methods to determine which secondary transformations add predictive value without introducing multicollinearity.