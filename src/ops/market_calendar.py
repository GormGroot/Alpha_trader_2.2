"""
Market Calendar — Multi-timezone market schedule for Alpha Trading Platform.

Handles open/close times for all markets in local CET time (Denmark).
Supports pre-market, regular and post-market sessions.

Market schedule (all times CET):
  New Zealand  22:00 – 03:00  (pre: 21:30)
  Australia    01:00 – 07:00  (pre: 00:30)
  Tokyo        01:00 – 07:30  (pre: 00:30, lunch 03:30-04:30)
  Hong Kong    02:00 – 08:00  (pre: 01:30)
  Mumbai       04:45 – 11:15  (pre: 04:15)
  EU / Nordic  09:00 – 17:30  (pre: 08:00)
  London       09:00 – 17:30  (pre: 08:00)
  US Pre       10:00 – 15:30  (extended pre-market)
  US Regular   15:30 – 22:00
  US Post      22:00 – 02:00  (after-hours)
  Crypto       00:00 – 24:00  (always open)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, date, timedelta
from enum import Enum
from zoneinfo import ZoneInfo
from loguru import logger


# ── Timezones ──────────────────────────────────────────────
TZ_CET    = ZoneInfo("Europe/Copenhagen")
TZ_ET     = ZoneInfo("America/New_York")
TZ_NZ     = ZoneInfo("Pacific/Auckland")
TZ_SYDNEY = ZoneInfo("Australia/Sydney")
TZ_TOKYO  = ZoneInfo("Asia/Tokyo")


def _now_cet() -> datetime:
    """Get CET time from web-synced time service, fallback to local clock."""
    try:
        from src.ops.time_service import now_cet
        return now_cet()
    except Exception:
        return _now_cet()
TZ_HK     = ZoneInfo("Asia/Hong_Kong")
TZ_INDIA  = ZoneInfo("Asia/Kolkata")
TZ_LONDON = ZoneInfo("Europe/London")
TZ_UTC    = ZoneInfo("UTC")


class SessionType(Enum):
    PRE_MARKET  = "pre_market"
    REGULAR     = "regular"
    POST_MARKET = "post_market"
    CLOSED      = "closed"


@dataclass
class MarketSession:
    """Represents one trading session for a market."""
    market:       str
    session_type: SessionType
    open_cet:     time    # Opening time in CET
    close_cet:    time    # Closing time in CET
    crosses_midnight: bool = False  # True if session spans midnight CET


@dataclass
class MarketStatus:
    """Current status of a market."""
    market:        str
    is_open:       bool
    session_type:  SessionType
    symbols:       list[str]
    opens_in_min:  int | None = None   # Minutes until next open
    closes_in_min: int | None = None   # Minutes until close
    description:   str = ""


# ── Symbol groups per market ───────────────────────────────

MARKET_SYMBOLS: dict[str, list[str]] = {
    "crypto": [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD",
        "XRP-USD", "ADA-USD", "AVAX-USD", "DOT-USD", "MATIC-USD",
    ],
    "new_zealand": [
        "SPK.NZ", "FBU.NZ", "AIA.NZ", "MFT.NZ", "EBO.NZ",
        "ATM.NZ", "PCT.NZ", "CEN.NZ", "MEL.NZ", "RYM.NZ",
    ],
    "australia": [
        "BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "WBC.AX",
        "ANZ.AX", "WES.AX", "WOW.AX", "MQG.AX", "RIO.AX",
        "FMG.AX", "NCM.AX", "TLS.AX", "WPL.AX", "STO.AX",
        "QBE.AX", "AMC.AX", "TCL.AX", "GMG.AX", "REA.AX",
    ],
    "japan": [
        "7203.T", "6758.T", "9984.T", "8306.T", "6501.T",
        "7974.T", "4519.T", "8316.T", "9432.T", "6954.T",
        "7267.T", "6861.T", "4502.T", "8035.T", "6902.T",
        "9433.T", "7751.T", "6301.T", "4661.T", "2914.T",
    ],
    "hong_kong": [
        "0700.HK", "9988.HK", "0941.HK", "1299.HK", "0005.HK",
        "0939.HK", "1398.HK", "2318.HK", "0388.HK", "1810.HK",
        "9999.HK", "0011.HK", "0883.HK", "2020.HK", "9618.HK",
        "3690.HK", "0175.HK", "0002.HK", "0003.HK", "0016.HK",
    ],
    "india": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        "LT.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS",
        "SUNPHARMA.NS", "WIPRO.NS", "HCLTECH.NS", "TITAN.NS", "NESTLEIND.NS",
    ],
    "eu_nordic": [
        # Nordic
        "NOVO-B.CO", "ORSTED.CO", "MAERSK-B.CO", "DSV.CO", "COLOB.CO",
        "DEMANT.CO", "GMAB.CO", "ROCKB.CO", "TRYG.CO", "CARL-B.CO",
        "VOLV-B.ST", "ERIC-B.ST", "ATCO-A.ST", "SEB-A.ST", "SWED-A.ST",
        "SAND.ST", "SKF-B.ST", "HM-B.ST", "INVE-B.ST", "ESSITY-B.ST",
        "DNB.OL", "EQNR.OL", "TEL.OL", "YAR.OL", "MOWI.OL",
        "SALM.OL", "NHY.OL", "NESTE.HE", "NOKIA.HE", "SAMPO.HE", "KNEBV.HE",
        # Germany
        "SAP.DE", "SIE.DE", "ALV.DE", "MUV2.DE", "DTE.DE",
        "BAYN.DE", "BMW.DE", "MBG.DE", "VOW3.DE", "BAS.DE",
        "DBK.DE", "ADS.DE", "AIR.DE", "HEN3.DE", "RWE.DE",
        # France
        "MC.PA", "OR.PA", "TTE.PA", "SAN.PA", "BNP.PA",
        "ACA.PA", "SGO.PA", "RI.PA", "KER.PA", "CAP.PA", "DSY.PA",
        # Netherlands / Switzerland / Spain / Italy
        "ASML.AS", "PHIA.AS", "HEIA.AS", "INGA.AS",
        "NESN.SW", "NOVN.SW", "ROG.SW", "UBSG.SW", "ABBN.SW", "ZURN.SW",
        "SAN.MC", "BBVA.MC", "ITX.MC", "IBE.MC",
        "ENI.MI", "ENEL.MI", "ISP.MI", "UCG.MI", "LDO.MI",
    ],
    "london": [
        "SHEL.L", "AZN.L", "HSBA.L", "BP.L", "GSK.L",
        "ULVR.L", "RIO.L", "BHP.L", "LSEG.L", "DGE.L",
    ],
    "us_stocks": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "BRK-B", "JPM", "V", "UNH", "JNJ", "XOM", "PG", "MA",
        "HD", "CVX", "MRK", "ABBV", "PEP", "KO", "AVGO", "COST",
        "WMT", "BAC", "DIS", "CSCO", "ADBE", "CRM", "AMD", "INTC",
        "NFLX", "QCOM", "TXN", "IBM", "GE", "CAT", "BA", "GS",
        "MS", "C", "WFC", "AXP", "LMT", "RTX", "UPS", "FDX",
        "MCD", "SBUX", "NKE", "PFE", "TMO", "ABT", "MDT", "AMGN",
        "GILD", "BMY",
    ],
    "chicago": [
        "GC=F", "SI=F", "CL=F", "BZ=F", "NG=F",
        "ZC=F", "ZW=F", "ZS=F", "ZL=F", "ZM=F",
        "HG=F", "PL=F", "PA=F", "LE=F", "HE=F",
        "ES=F", "NQ=F", "YM=F", "RTY=F",
        "ZN=F", "ZB=F", "ZT=F", "^VIX",
    ],
    "etfs": [
        "SPY", "QQQ", "IWM", "VTI", "VEA", "VWO", "EWJ", "EWH",
        "INDA", "EWG", "EWQ", "EWU", "GLD", "SLV", "USO", "TLT",
        "HYG", "XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY",
        "XLP", "XLB", "XLRE", "XLU",
    ],
}


# ── Market Sessions (all times in CET) ────────────────────

MARKET_SESSIONS: list[MarketSession] = [
    # New Zealand — opens 22:00 CET (crosses midnight)
    MarketSession("new_zealand", SessionType.PRE_MARKET,  time(21, 30), time(22,  0), crosses_midnight=False),
    MarketSession("new_zealand", SessionType.REGULAR,     time(22,  0), time( 3,  0), crosses_midnight=True),

    # Australia (ASX) — 01:00–07:00 CET
    MarketSession("australia",   SessionType.PRE_MARKET,  time( 0, 30), time( 1,  0)),
    MarketSession("australia",   SessionType.REGULAR,     time( 1,  0), time( 7,  0)),

    # Japan (TSE) — 01:00–07:30 CET (lunch 03:30–04:30)
    MarketSession("japan",       SessionType.PRE_MARKET,  time( 0, 30), time( 1,  0)),
    MarketSession("japan",       SessionType.REGULAR,     time( 1,  0), time( 7, 30)),

    # Hong Kong (HKEX) — 02:00–08:00 CET
    MarketSession("hong_kong",   SessionType.PRE_MARKET,  time( 1, 30), time( 2,  0)),
    MarketSession("hong_kong",   SessionType.REGULAR,     time( 2,  0), time( 8,  0)),

    # India (NSE) — 04:45–11:15 CET
    MarketSession("india",       SessionType.PRE_MARKET,  time( 4, 15), time( 4, 45)),
    MarketSession("india",       SessionType.REGULAR,     time( 4, 45), time(11, 15)),

    # EU + Nordic — 09:00–17:30 CET
    MarketSession("eu_nordic",   SessionType.PRE_MARKET,  time( 8,  0), time( 9,  0)),
    MarketSession("eu_nordic",   SessionType.REGULAR,     time( 9,  0), time(17, 30)),

    # London (LSE) — 09:00–17:30 CET
    MarketSession("london",      SessionType.PRE_MARKET,  time( 8,  0), time( 9,  0)),
    MarketSession("london",      SessionType.REGULAR,     time( 9,  0), time(17, 30)),

    # US Pre-market — 10:00–15:30 CET
    MarketSession("us_stocks",   SessionType.PRE_MARKET,  time(10,  0), time(15, 30)),
    MarketSession("chicago",     SessionType.PRE_MARKET,  time(10,  0), time(15, 30)),

    # US Regular — 15:30–22:00 CET
    MarketSession("us_stocks",   SessionType.REGULAR,     time(15, 30), time(22,  0)),
    MarketSession("chicago",     SessionType.REGULAR,     time(15, 30), time(22,  0)),

    # US Post-market — 22:00–02:00 CET (crosses midnight)
    MarketSession("us_stocks",   SessionType.POST_MARKET, time(22,  0), time( 2,  0), crosses_midnight=True),

    # ETFs follow US hours
    MarketSession("etfs",        SessionType.PRE_MARKET,  time(10,  0), time(15, 30)),
    MarketSession("etfs",        SessionType.REGULAR,     time(15, 30), time(22,  0)),
    MarketSession("etfs",        SessionType.POST_MARKET, time(22,  0), time( 2,  0), crosses_midnight=True),
]


# ── Holiday calendars ──────────────────────────────────────
#
# Uses the `holidays` package for comprehensive, government-maintained
# holiday lists per country.  Falls back to minimal hard-coded sets
# when the package is not installed.

def _is_weekend(d: date) -> bool:
    return d.weekday() >= 5


# Map each market key to its ISO country code(s).
# When multiple codes are listed the market is closed on the *union*
# of all holidays (conservative — avoids trading into a closed venue).
_MARKET_COUNTRY_CODES: dict[str, list[str]] = {
    "new_zealand": ["NZ"],
    "australia":   ["AU"],
    "japan":       ["JP"],
    "hong_kong":   ["HK"],
    "india":       ["IN"],
    "eu_nordic":   ["DK", "DE"],     # Danish + German holidays cover most EU/Nordic closures
    "london":      ["GB"],
    "us_stocks":   ["US"],
    "chicago":     ["US"],
    "etfs":        ["US"],
}

# Cache: (country_code, year) → set[date]
_holiday_cache: dict[tuple[str, int], set[date]] = {}


def _get_holidays(country_code: str, year: int) -> set[date]:
    """Return the set of public-holiday dates for *country_code* in *year*.

    Uses the ``holidays`` package when available, otherwise falls back to
    a minimal hard-coded set for DK and US.
    """
    key = (country_code, year)
    if key in _holiday_cache:
        return _holiday_cache[key]

    result: set[date] = set()
    try:
        import holidays as _hol
        result = set(_hol.country_holidays(country_code, years=year).keys())
    except Exception:
        # Fallback for DK and US when package is missing
        from datetime import date as dt
        if country_code == "DK":
            result = {
                dt(year, 1, 1), dt(year, 6, 5),
                dt(year, 12, 24), dt(year, 12, 25), dt(year, 12, 26), dt(year, 12, 31),
            }
            try:
                from dateutil.easter import easter
                e = easter(year)
                result.update({
                    e - timedelta(days=3), e - timedelta(days=2), e,
                    e + timedelta(days=1), e + timedelta(days=39), e + timedelta(days=50),
                })
            except ImportError:
                pass
        elif country_code == "US":
            result = {dt(year, 1, 1), dt(year, 7, 4), dt(year, 12, 25)}

    _holiday_cache[key] = result
    return result


def _market_holidays(market: str, year: int) -> set[date]:
    """Return all holiday dates for a *market* in *year* (union of its countries)."""
    codes = _MARKET_COUNTRY_CODES.get(market, [])
    combined: set[date] = set()
    for code in codes:
        combined |= _get_holidays(code, year)
    return combined


def is_trading_day(market: str, d: date | None = None) -> bool:
    """Check if a given date is a trading day for a market."""
    if d is None:
        d = _now_cet().date()

    # Crypto never closes
    if market == "crypto":
        return True

    # Weekends closed for all equity markets
    if _is_weekend(d):
        return False

    return d not in _market_holidays(market, d.year)


# ── Weekend Rotation Helpers ──────────────────────────────

def get_friday_close_schedule(ref_date: date | None = None) -> list[tuple[time, list[str]]]:
    """Return (close_time_cet, [market, ...]) pairs sorted by close time.

    Each entry represents a group of exchanges that share the same
    REGULAR close on the given date.  The caller should close positions
    in those markets at or shortly after the listed time.

    Only includes markets that are actually **trading** on *ref_date*
    (i.e. not on a national holiday).  Crypto is always excluded.
    """
    from collections import defaultdict

    if ref_date is None:
        ref_date = _now_cet().date()

    close_map: dict[time, set[str]] = defaultdict(set)
    for s in MARKET_SESSIONS:
        if s.session_type != SessionType.REGULAR:
            continue
        if s.market == "crypto":
            continue
        # Skip markets closed today due to holiday
        if not is_trading_day(s.market, ref_date):
            continue
        # For midnight-crossing sessions the "close" is on the next
        # calendar day — treat them as closing at 23:59.
        close_t = s.close_cet if not s.crosses_midnight else time(23, 59)
        close_map[close_t].add(s.market)

    return sorted(
        [(t, sorted(markets)) for t, markets in close_map.items()],
        key=lambda pair: pair[0],
    )


def next_trading_day(market: str, after: date | None = None) -> date:
    """Return the next date on which *market* is open, starting from *after* + 1.

    Skips weekends **and** national holidays for the exchange.
    """
    if after is None:
        after = _now_cet().date()
    d = after + timedelta(days=1)
    # Safety limit: 30 days (covers any realistic holiday stretch)
    for _ in range(30):
        if is_trading_day(market, d):
            return d
        d += timedelta(days=1)
    return d  # fallback


def is_last_trading_day_before_break(market: str, d: date | None = None) -> bool:
    """True when *d* is a trading day and the *next* calendar day is NOT a trading day.

    This detects:
      - Regular Fridays (next day = Saturday)
      - Thursdays before a Friday holiday (e.g. Good Friday)
      - Any day before a multi-day holiday stretch
    """
    if d is None:
        d = _now_cet().date()
    if not is_trading_day(market, d):
        return False
    return not is_trading_day(market, d + timedelta(days=1))


def get_earliest_reopen(after: date | None = None) -> tuple[date, time, str]:
    """Return (date, open_time_cet, market) for the first exchange to reopen.

    Scans all non-crypto markets to find which one resumes trading
    earliest after the given date.  Accounts for per-market holidays
    (e.g. NZ may reopen Monday while US stays closed, or vice versa).

    NZ opens on the *evening before* its trading date (Sunday 21:30 CET
    for a Monday trading day), so we check for Sunday-evening openers.
    """
    if after is None:
        after = _now_cet().date()

    SUNDAY_EVENING_CUTOFF = time(18, 0)

    # Collect the earliest session open per market
    earliest_session: dict[str, time] = {}
    for s in MARKET_SESSIONS:
        if s.market == "crypto":
            continue
        if s.market not in earliest_session or s.open_cet < earliest_session[s.market]:
            earliest_session[s.market] = s.open_cet

    best_dt: date | None = None
    best_time = time(23, 59)
    best_market = ""

    for market, open_t in earliest_session.items():
        ntd = next_trading_day(market, after)
        # Sunday-evening openers: NZ opens the evening *before* ntd
        if open_t >= SUNDAY_EVENING_CUTOFF:
            effective_date = ntd - timedelta(days=1)
        else:
            effective_date = ntd

        if best_dt is None or effective_date < best_dt or (
            effective_date == best_dt and open_t < best_time
        ):
            best_dt = effective_date
            best_time = open_t
            best_market = market

    return best_dt or (after + timedelta(days=1)), best_time, best_market


def get_market_close_time(market: str) -> time | None:
    """Return the REGULAR session close_cet for *market*, or None."""
    for s in MARKET_SESSIONS:
        if s.market == market and s.session_type == SessionType.REGULAR:
            return s.close_cet
    return None


def get_market_open_time(market: str, session: SessionType = SessionType.PRE_MARKET) -> time | None:
    """Return the earliest open_cet for *market* (defaults to PRE_MARKET)."""
    for s in MARKET_SESSIONS:
        if s.market == market and s.session_type == session:
            return s.open_cet
    # Fallback to REGULAR if no pre-market
    for s in MARKET_SESSIONS:
        if s.market == market and s.session_type == SessionType.REGULAR:
            return s.open_cet
    return None


# ── Core Calendar Engine ───────────────────────────────────

class MarketCalendar:
    """
    Determines which markets are open right now (CET)
    and returns the appropriate symbols to scan.

    Usage:
        cal = MarketCalendar()
        open_markets = cal.get_open_markets()
        symbols = cal.get_symbols_to_scan()
        status = cal.get_all_status()
    """

    def __init__(self, include_pre_market: bool = True,
                 include_post_market: bool = True):
        self.include_pre = include_pre_market
        self.include_post = include_post_market

    def _is_session_open(self, session: MarketSession, now: time) -> bool:
        """Check if a session is currently open based on CET time."""
        if session.crosses_midnight:
            # Session spans midnight: open if time >= open OR time < close
            return now >= session.open_cet or now < session.close_cet
        else:
            return session.open_cet <= now < session.close_cet

    def get_open_markets(self, now: datetime | None = None) -> list[str]:
        """Return list of currently open markets."""
        if now is None:
            now = _now_cet()

        current_time = now.time()
        today = now.date()
        open_markets = []

        # Crypto is always open
        open_markets.append("crypto")

        for session in MARKET_SESSIONS:
            market = session.market

            # Skip if already added
            if market in open_markets:
                continue

            # Check trading day
            if not is_trading_day(market, today):
                continue

            # Check session type filter
            if session.session_type == SessionType.PRE_MARKET and not self.include_pre:
                continue
            if session.session_type == SessionType.POST_MARKET and not self.include_post:
                continue

            if self._is_session_open(session, current_time):
                open_markets.append(market)

        return open_markets

    def get_symbols_to_scan(self, now: datetime | None = None) -> list[str]:
        """Return all symbols that should be scanned right now."""
        open_markets = self.get_open_markets(now)
        symbols = []
        for market in open_markets:
            for sym in MARKET_SYMBOLS.get(market, []):
                if sym not in symbols:
                    symbols.append(sym)
        return symbols

    def get_current_session(self, market: str, now: datetime | None = None) -> SessionType:
        """Get the current session type for a specific market."""
        if now is None:
            now = _now_cet()
        current_time = now.time()
        today = now.date()

        if market == "crypto":
            return SessionType.REGULAR

        if not is_trading_day(market, today):
            return SessionType.CLOSED

        for session in MARKET_SESSIONS:
            if session.market != market:
                continue
            if session.session_type == SessionType.PRE_MARKET and not self.include_pre:
                continue
            if session.session_type == SessionType.POST_MARKET and not self.include_post:
                continue
            if self._is_session_open(session, current_time):
                return session.session_type

        return SessionType.CLOSED

    def get_all_status(self, now: datetime | None = None) -> list[MarketStatus]:
        """Get status for all markets."""
        if now is None:
            now = _now_cet()

        statuses = []
        all_markets = list(MARKET_SYMBOLS.keys())

        for market in all_markets:
            session_type = self.get_current_session(market, now)
            is_open = session_type != SessionType.CLOSED

            status = MarketStatus(
                market=market,
                is_open=is_open,
                session_type=session_type,
                symbols=MARKET_SYMBOLS.get(market, []),
                description=self._describe(market, session_type, now),
            )
            statuses.append(status)

        return statuses

    def _describe(self, market: str, session: SessionType, now: datetime) -> str:
        names = {
            "crypto":      "Crypto (24/7)",
            "new_zealand": "New Zealand (NZX)",
            "australia":   "Australia (ASX)",
            "japan":       "Japan (TSE)",
            "hong_kong":   "Hong Kong (HKEX)",
            "india":       "India (NSE)",
            "eu_nordic":   "EU + Nordic",
            "london":      "London (LSE)",
            "us_stocks":   "US Stocks (NYSE/NASDAQ)",
            "chicago":     "Chicago (CME/CBOT)",
            "etfs":        "ETFs",
        }
        name = names.get(market, market)
        if session == SessionType.CLOSED:
            return f"{name} — CLOSED"
        elif session == SessionType.PRE_MARKET:
            return f"{name} — PRE-MARKET"
        elif session == SessionType.POST_MARKET:
            return f"{name} — POST-MARKET"
        else:
            return f"{name} — OPEN"

    def print_status(self) -> None:
        """Print a human-readable market status overview."""
        now = _now_cet()
        print(f"\n{'═'*55}")
        print(f"  Market Status — {now:%Y-%m-%d %H:%M CET}")
        print(f"{'═'*55}")
        for s in self.get_all_status(now):
            icon = "🟢" if s.is_open else "🔴"
            sym_count = len(s.symbols)
            print(f"  {icon}  {s.description:<35} ({sym_count} symbols)")
        symbols = self.get_symbols_to_scan(now)
        print(f"{'─'*55}")
        print(f"  Total symbols to scan now: {len(symbols)}")
        print(f"{'═'*55}\n")


# ── Convenience singleton ──────────────────────────────────
_calendar: MarketCalendar | None = None

def get_calendar() -> MarketCalendar:
    global _calendar
    if _calendar is None:
        _calendar = MarketCalendar(include_pre_market=True, include_post_market=True)
    return _calendar
