"""Generate valid Freqtrade config.json using native Freqtrade patterns.

Leverages Freqtrade's built-in configuration schema so the bot boots without
validation errors.  All values are derived from the strategy DNA + Athena risk
settings so backtest → paper → live parity is guaranteed.
"""
import json
import secrets
from pathlib import Path
from typing import Any, Dict, Optional


# ── defaults aligned with Freqtrade constants ───────────────────
DEFAULT_WALLET = 50.0  # Starting capital for small-account growth
DEFAULT_FEE = 0.001
DEFAULT_TIMEFRAME = "1h"
DEFAULT_STAKE_CURRENCY = "USDT"


def build_config(
    strategy_name: str,
    strategy_path: str,
    pair: str,
    timeframe: str = DEFAULT_TIMEFRAME,
    mode: str = "paper",
    stake_amount: str = "unlimited",
    wallet_balance: float = DEFAULT_WALLET,
    max_open_trades: int = 1,
    fee: float = DEFAULT_FEE,
    dna: Optional[Dict[str, Any]] = None,
    risk: Optional[Dict[str, Any]] = None,
    db_url: Optional[str] = None,
    api_port: Optional[int] = None,
    exchange_key: str = "",
    exchange_secret: str = "",
    sandbox: bool = False,
) -> Dict[str, Any]:
    """Return a Freqtrade-compatible config dict.

    Parameters mirror Freqtrade's JSON schema so the bot validates cleanly.
    """
    dna = dna or {}
    risk = risk or {}

    # Derive stoploss from DNA if available
    atr_period = dna.get("atr_period", 14)
    atr_mult = dna.get("atr_multiplier", 2.0)
    # A rough heuristic: 1 ATR ≈ 1% for BTC, scale by multiplier
    stoploss = -abs(atr_mult * 0.01)

    # JWT secret for the embedded API server (Athena will query this)
    jwt_secret = secrets.token_hex(32)
    api_user = "athena"
    api_pass = secrets.token_urlsafe(16)

    cfg: Dict[str, Any] = {
        # ── strategy ──
        "strategy": strategy_name,
        "strategy_path": strategy_path,
        "timeframe": timeframe,
        "timeframe_detail": None,
        "max_open_trades": max_open_trades,
        "cancel_open_orders_on_exit": True,
        "amend_last_stake_amount": False,
        "position_adjustment_enable": False,
        "max_entry_position_adjustment": 0,
        "use_exit_signal": True,
        "exit_profit_only": False,
        "ignore_roi_if_entry_signal": False,

        # ── wallet / stake ──
        "stake_currency": DEFAULT_STAKE_CURRENCY,
        "stake_amount": stake_amount,
        "tradable_balance_ratio": 0.99,
        "fiat_display_currency": "USD",
        "dry_run": mode == "paper",
        "dry_run_wallet": wallet_balance,

        # ── pricing ──
        "entry_pricing": {
            "price_side": "other",
            "use_order_book": True,
            "order_book_top": 1,
            "price_last_balance": 0.0,
            "check_depth_of_market": {"enabled": False, "bids_to_ask_delta": 1},
        },
        "exit_pricing": {
            "price_side": "other",
            "use_order_book": True,
            "order_book_top": 1,
        },

        # ── order types ──
        "order_types": {
            "entry": "market",
            "exit": "market",
            "emergency_exit": "market",
            "force_exit": "market",
            "stoploss": "market",
            "take_profit": "limit",
            "stoploss_on_exchange": False,
            "stoploss_on_exchange_interval": 60,
        },

        # ── timeouts ──
        "unfilledtimeout": {
            "entry": 10,
            "exit": 10,
            "unit": "minutes",
        },

        # ── exchange ──
        "exchange": {
            "name": "binance",
            "key": exchange_key,
            "secret": exchange_secret,
            "password": "",
            "ccxt_config": {"enableRateLimit": True},
            "ccxt_async_config": {"enableRateLimit": True},
            "pair_whitelist": [pair],
            "pair_blacklist": [],
            "sandbox": sandbox,
        },

        # ── pairlist ──
        "pairlists": [{"method": "StaticPairList"}],

        # ── stoploss / risk ──
        "stoploss": stoploss,
        "trailing_stop": dna.get("use_trailing_stop", False),
        "trailing_stop_positive": dna.get("trailing_stop_positive", 0.01),
        "trailing_stop_positive_offset": dna.get("trailing_stop_offset", 0.02),
        "trailing_only_offset_is_reached": True,

        # ── fee ──
        "fee": fee,

        # ── trading mode ──
        "trading_mode": "futures",
        "margin_mode": "cross",

        # ── data ──
        "dataformat_ohlcv": "feather",
        "dataformat_trades": "feather",

        # ── internals ──
        "internals": {
            "process_throttle_secs": 5,
            "heartbeat_interval": 60,
        },

        # ── logging ──
        "logging": {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["console"],
            },
        },

        # ── database ──
        "db_url": db_url or "sqlite:///tradesv3.dryrun.sqlite",

        # ── API server (used by Athena for status queries) ──
        "api_server": {
            "enabled": True,
            "listen_ip_address": "127.0.0.1",
            "listen_port": api_port or 0,
            "username": api_user,
            "password": api_pass,
            "jwt_secret_key": jwt_secret,
            "ws_token": secrets.token_urlsafe(32),
            "CORS_origins": [],
            "verbosity": "info",
        },
    }

    return cfg


def write_config(config: Dict[str, Any], path: Path) -> Path:
    """Write config dict to JSON file."""
    path.write_text(json.dumps(config, indent=2))
    return path
