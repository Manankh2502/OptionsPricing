from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

# we focus on a few liquid symbols for now. 
DEFAULT_SYMBOLS = ["SPY", "QQQ", "AAPL"]


@dataclass
class YahooOptionsClient:
    """
    Client for getting and cleaning options data
    """
    raw_data_dir: Path = Path("data/raw")
    target_dte: int = 30
    min_moneyness: float = 0.8
    max_moneyness: float = 1.2
    min_open_interest: int = 10
    max_spread_pct: float = 0.5

    def get_ticker(self, symbol: str) -> yf.Ticker:
        return yf.Ticker(symbol.upper())
    
    def get_expirations(self, symbol: str) -> list[str]:
        ticker = self.get_ticker(symbol)
        return list(ticker.options)
    
    def select_expiration(
        self,
        symbol: str,
        target_dte: Optional[int] = None,
    ) -> str:
        expirations = self.get_expirations(symbol)

        # fail proof cause yahoo is sometimes bad
        if not expirations:
            raise ValueError(f"No option expirations found for {symbol}")
        
        if target_dte is None:
            target_dte = self.target_dte

        today = date.today()

        def days_to_expiry(expiration: str) -> int:
            expiry_date = datetime.strptime(expiration, "%Y-%m-%d").date()
            return (expiry_date - today).days
        
        future_expirations = [
            expiration
            for expiration in expirations
            if days_to_expiry(expiration) >= 0
        ]

        if not future_expirations:
            raise ValueError(f"No future option expirations found for {symbol}")
        
        return min(
            future_expirations,
            key=lambda expiration: abs(days_to_expiry(expiration) - target_dte),
        )
    
    def get_underlying_price(self, symbol: str) -> float:
        ticker = self.get_ticker(symbol)
        # idk if fast_info is going ot succeed
        try:
            price = ticker.fast_info.get("last_price")
            if price is not None:
                return float(price)
        except Exception:
            pass

        history = ticker.history(period="1d")

        if history.empty:
            raise ValueError(f"Could not fetch underlying price for {symbol}")

        return float(history["Close"].iloc[-1])

    def fetch_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None,
        apply_filters: bool = True,
    ) -> pd.DataFrame:
        symbol = symbol.upper()

        ticker = self.get_ticker(symbol)
        expirations = list(ticker.options)

        if not expirations:
            raise ValueError(f"No option expirations found for {symbol}")

        if expiration is None:
            expiration = self.select_expiration(symbol)

        if expiration not in expirations:
            raise ValueError(
                f"Expiration {expiration} not available for {symbol}. "
                f"Available expirations: {expirations}"
            )
        
        underlying_price = self.get_underlying_price(symbol)
        snapshot_time = datetime.now(timezone.utc)

        raw_chain = ticker.option_chain(expiration)

        calls = self._prepare_side(
            df=raw_chain.calls,
            symbol=symbol,
            expiration=expiration,
            option_type="call",
            underlying_price=underlying_price,
            snapshot_time=snapshot_time,
        )

        puts = self._prepare_side(
            df=raw_chain.puts,
            symbol=symbol,
            expiration=expiration,
            option_type="put",
            underlying_price=underlying_price,
            snapshot_time=snapshot_time,
        )

        chain = pd.concat([calls, puts], ignore_index=True)

        chain = chain.sort_values(
            by=["expiration", "option_type", "strike"],
            ignore_index=True,
        )

        if apply_filters:
            chain = self.filter_chain(chain)

        return chain
    
    def filter_chain(self, chain: pd.DataFrame) -> pd.DataFrame:
        filtered = chain.copy()

        filtered = filtered[
            (filtered["moneyness"] >= self.min_moneyness)
            & (filtered["moneyness"] <= self.max_moneyness)
        ]

        filtered = filtered[
            (filtered["bid"] > 0)
            & (filtered["ask"] > 0)
        ]

        if "open_interest" in filtered.columns:
            filtered = filtered[
                filtered["open_interest"].fillna(0) >= self.min_open_interest
            ]

        filtered = filtered[
            filtered["spread_pct"] <= self.max_spread_pct
        ]

        return filtered.reset_index(drop=True)
    
    def save_snapshot(self, chain: pd.DataFrame, symbol: str, expiration: str) -> Path:
        symbol = symbol.upper()

        snapshot_time = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        output_dir = self.raw_data_dir / symbol / expiration
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"options_chain_{snapshot_time}.csv"

        chain.to_csv(output_path, index=False)

        return output_path
    
    def fetch_and_save(
        self,
        symbol: str,
        expiration: Optional[str] = None,
        apply_filters: bool = True,
    ) -> tuple[pd.DataFrame, Path]:
        chain = self.fetch_chain(
            symbol=symbol,
            expiration=expiration,
            apply_filters=apply_filters,
        )

        if chain.empty:
            raise ValueError(
                f"No contracts left after filtering for {symbol}. "
                "Try apply_filters=False or loosen the filter settings."
            )

        selected_expiration = str(chain["expiration"].iloc[0])

        output_path = self.save_snapshot(
            chain=chain,
            symbol=symbol,
            expiration=selected_expiration,
        )

        return chain, output_path
    
    def _prepare_side(
        self,
        df: pd.DataFrame,
        symbol: str,
        expiration: str,
        option_type: str,
        underlying_price: float,
        snapshot_time: datetime,
    ) -> pd.DataFrame:
        normalized = df.copy()

        normalized = normalized.rename(
            columns={
                "contractSymbol": "contract_symbol",
                "lastTradeDate": "last_trade_date",
                "lastPrice": "last_price",
                "openInterest": "open_interest",
                "impliedVolatility": "yahoo_implied_volatility",
                "inTheMoney": "in_the_money",
                "contractSize": "contract_size",
            }
        )

        normalized["underlying_symbol"] = symbol
        normalized["underlying_price"] = underlying_price
        normalized["expiration"] = expiration
        normalized["option_type"] = option_type
        normalized["snapshot_time_utc"] = snapshot_time.isoformat()

        normalized["mid"] = self._compute_mid_price(normalized)
        normalized["moneyness"] = normalized["strike"] / underlying_price
        normalized["spread"] = normalized["ask"] - normalized["bid"]
        normalized["spread_pct"] = normalized["spread"] / normalized["mid"]

        columns = [
            "snapshot_time_utc",
            "underlying_symbol",
            "underlying_price",
            "contract_symbol",
            "option_type",
            "expiration",
            "strike",
            "moneyness",
            "bid",
            "ask",
            "mid",
            "spread",
            "spread_pct",
            "last_price",
            "volume",
            "open_interest",
            "yahoo_implied_volatility",
            "in_the_money",
            "last_trade_date",
            "contract_size",
            "currency",
        ]

        existing_columns = [
            column
            for column in columns
            if column in normalized.columns
        ]

        return normalized[existing_columns]
    
    @staticmethod
    def _compute_mid_price(df: pd.DataFrame) -> pd.Series:
        bid = pd.to_numeric(df["bid"], errors="coerce")
        ask = pd.to_numeric(df["ask"], errors="coerce")

        valid_quote = (bid > 0) & (ask > 0)

        mid = (bid + ask) / 2.0

        return mid.where(valid_quote)

if __name__ == "__main__":
    client = YahooOptionsClient()

    chain, path = client.fetch_and_save("SPY")

    print(chain.head())
    print(f"Rows kept: {len(chain)}")
    print(f"Saved snapshot to: {path}")