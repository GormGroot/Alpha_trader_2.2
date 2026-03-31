"""
Tests for InsiderTracker – SEC EDGAR integration og smart money signals.

Alle SEC API-kald mockes – ingen netværksforbindelse kræves.
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.data.insider_tracking import (
    InsiderTracker,
    SECEdgarClient,
    InsiderTrade,
    InsiderSentimentScore,
    InsiderSentiment,
    TransactionType,
    InstitutionalHolding,
    SmartMoneyFlow,
    ShortInterestData,
    SmartMoneyReport,
    MAJOR_FUNDS,
    INSIDER_ROLES,
    SEC_USER_AGENT,
)


# ── Helpers ──────────────────────────────────────────────────

def _make_trade(
    symbol: str = "AAPL",
    name: str = "Tim Cook",
    title: str = "CEO",
    tx_type: TransactionType = TransactionType.PURCHASE,
    shares: float = 10_000,
    price: float = 150.0,
    days_ago: int = 5,
) -> InsiderTrade:
    """Opret en test InsiderTrade."""
    return InsiderTrade(
        symbol=symbol,
        insider_name=name,
        insider_title=title,
        transaction_type=tx_type,
        shares=shares,
        price=price,
        value=shares * price,
        date=datetime.now() - timedelta(days=days_ago),
        filing_date=datetime.now() - timedelta(days=days_ago - 1),
        ownership_after=50_000,
        is_direct=True,
    )


def _make_trades_cluster(symbol: str = "AAPL", n: int = 4) -> list[InsiderTrade]:
    """Opret cluster af insider-køb (flere insidere inden for 14 dage)."""
    names = ["Tim Cook", "Luca Maestri", "Jeff Williams", "Deirdre O'Brien", "Craig Federighi"]
    titles = ["CEO", "CFO", "COO", "SVP", "SVP"]
    return [
        _make_trade(
            symbol=symbol,
            name=names[i],
            title=titles[i],
            days_ago=i * 3,  # Alle inden for ~12 dage
        )
        for i in range(min(n, len(names)))
    ]


def _tmp_cache_dir():
    """Returnér temp-mappe til cache."""
    return tempfile.mkdtemp()


# ── Test TransactionType ─────────────────────────────────────

class TestTransactionType:
    def test_purchase_value(self):
        assert TransactionType.PURCHASE.value == "P"

    def test_sale_value(self):
        assert TransactionType.SALE.value == "S"

    def test_all_types(self):
        assert len(TransactionType) == 6


# ── Test InsiderSentiment ────────────────────────────────────

class TestInsiderSentiment:
    def test_all_sentiments(self):
        assert len(InsiderSentiment) == 5

    def test_values(self):
        assert InsiderSentiment.VERY_BULLISH.value == "very_bullish"
        assert InsiderSentiment.BEARISH.value == "bearish"


# ── Test InsiderTrade ────────────────────────────────────────

class TestInsiderTrade:
    def test_is_purchase(self):
        t = _make_trade(tx_type=TransactionType.PURCHASE)
        assert t.is_purchase is True
        assert t.is_sale is False

    def test_is_sale(self):
        t = _make_trade(tx_type=TransactionType.SALE)
        assert t.is_purchase is False
        assert t.is_sale is True

    def test_is_c_suite_ceo(self):
        t = _make_trade(title="CEO")
        assert t.is_c_suite is True

    def test_is_c_suite_cfo(self):
        t = _make_trade(title="CFO")
        assert t.is_c_suite is True

    def test_is_c_suite_director(self):
        t = _make_trade(title="Director")
        assert t.is_c_suite is False

    def test_value_calculated(self):
        t = _make_trade(shares=100, price=50.0)
        assert t.value == 5_000.0

    def test_grant_not_purchase(self):
        t = _make_trade(tx_type=TransactionType.GRANT)
        assert t.is_purchase is False
        assert t.is_sale is False


# ── Test InsiderSentimentScore ───────────────────────────────

class TestInsiderSentimentScore:
    def test_confidence_boost_cluster_buying(self):
        score = InsiderSentimentScore(
            symbol="AAPL",
            sentiment=InsiderSentiment.VERY_BULLISH,
            score=50.0,
            net_purchases=5,
            net_sales=0,
            total_buy_value=1_000_000,
            total_sell_value=0,
            cluster_buying=True,
            cluster_selling=False,
            c_suite_buying=True,
        )
        assert score.confidence_boost == 15

    def test_confidence_boost_bearish(self):
        score = InsiderSentimentScore(
            symbol="AAPL",
            sentiment=InsiderSentiment.BEARISH,
            score=-40.0,
            net_purchases=0,
            net_sales=5,
            total_buy_value=0,
            total_sell_value=500_000,
            cluster_buying=False,
            cluster_selling=False,
            c_suite_buying=False,
        )
        assert score.confidence_boost == -8

    def test_confidence_boost_very_bearish(self):
        score = InsiderSentimentScore(
            symbol="AAPL",
            sentiment=InsiderSentiment.VERY_BEARISH,
            score=-80.0,
            net_purchases=0,
            net_sales=10,
            total_buy_value=0,
            total_sell_value=2_000_000,
            cluster_buying=False,
            cluster_selling=True,
            c_suite_buying=False,
        )
        assert score.confidence_boost == -15

    def test_confidence_boost_neutral(self):
        score = InsiderSentimentScore(
            symbol="AAPL",
            sentiment=InsiderSentiment.NEUTRAL,
            score=0.0,
            net_purchases=1,
            net_sales=1,
            total_buy_value=50_000,
            total_sell_value=50_000,
            cluster_buying=False,
            cluster_selling=False,
            c_suite_buying=False,
        )
        assert score.confidence_boost == 0

    def test_confidence_boost_bounded(self):
        score = InsiderSentimentScore(
            symbol="AAPL",
            sentiment=InsiderSentiment.VERY_BULLISH,
            score=100.0,
            net_purchases=20,
            net_sales=0,
            total_buy_value=10_000_000,
            total_sell_value=0,
            cluster_buying=True,
            cluster_selling=False,
            c_suite_buying=True,
        )
        assert -15 <= score.confidence_boost <= 15

    def test_c_suite_only_boost(self):
        score = InsiderSentimentScore(
            symbol="AAPL",
            sentiment=InsiderSentiment.BULLISH,
            score=35.0,
            net_purchases=2,
            net_sales=0,
            total_buy_value=200_000,
            total_sell_value=0,
            cluster_buying=False,
            cluster_selling=False,
            c_suite_buying=True,
        )
        # c_suite +5, bullish max(5, 8) = 8
        assert score.confidence_boost == 8


# ── Test ShortInterestData ───────────────────────────────────

class TestShortInterestData:
    def test_heavily_shorted(self):
        data = ShortInterestData(
            symbol="GME",
            short_interest=50_000_000,
            short_pct_float=25.0,
            short_ratio=8.0,
            avg_volume=5_000_000,
        )
        assert data.is_heavily_shorted is True

    def test_not_heavily_shorted(self):
        data = ShortInterestData(
            symbol="AAPL",
            short_interest=10_000,
            short_pct_float=1.0,
            short_ratio=0.5,
            avg_volume=50_000_000,
        )
        assert data.is_heavily_shorted is False

    def test_days_to_cover(self):
        data = ShortInterestData(
            symbol="GME",
            short_interest=10_000_000,
            short_pct_float=20.0,
            short_ratio=5.0,
            avg_volume=2_000_000,
        )
        assert data.days_to_cover == pytest.approx(5.0)

    def test_days_to_cover_zero_volume(self):
        data = ShortInterestData(
            symbol="TEST",
            short_interest=1000,
            short_pct_float=5.0,
            short_ratio=0,
            avg_volume=0,
        )
        assert data.days_to_cover == 0.0


# ── Test InstitutionalHolding ────────────────────────────────

class TestInstitutionalHolding:
    def test_creation(self):
        h = InstitutionalHolding(
            fund_name="Berkshire Hathaway",
            symbol="AAPL",
            shares=900_000_000,
            value_usd=135_000_000_000,
            pct_portfolio=48.5,
            quarter="2025Q4",
        )
        assert h.fund_name == "Berkshire Hathaway"
        assert h.is_new_position is False
        assert h.change_shares == 0


# ── Test SmartMoneyFlow ──────────────────────────────────────

class TestSmartMoneyFlow:
    def test_creation(self):
        flow = SmartMoneyFlow(
            symbol="AAPL",
            institutional_holders=5,
            total_institutional_value=200_000_000_000,
            net_institutional_change=5_000_000_000,
            new_positions=["ARK Invest"],
            closed_positions=[],
            increased=["Berkshire Hathaway"],
            decreased=["Citadel Advisors"],
        )
        assert flow.institutional_holders == 5
        assert len(flow.new_positions) == 1


# ── Test SmartMoneyReport ────────────────────────────────────

class TestSmartMoneyReport:
    def test_defaults(self):
        r = SmartMoneyReport(
            symbol="AAPL",
            insider_sentiment=None,
            smart_money_flow=None,
            short_interest=None,
        )
        assert r.overall_signal == "neutral"
        assert r.confidence_adjustment == 0
        assert r.insider_sentiment is None
        assert r.short_interest is None


# ── Test SECEdgarClient ──────────────────────────────────────

class TestSECEdgarClient:
    def test_init_creates_db(self):
        cache_dir = _tmp_cache_dir()
        client = SECEdgarClient(cache_dir=cache_dir)
        db_path = Path(cache_dir) / "insider_tracking.db"
        assert db_path.exists()

    def test_db_tables_exist(self):
        cache_dir = _tmp_cache_dir()
        client = SECEdgarClient(cache_dir=cache_dir)
        db_path = Path(cache_dir) / "insider_tracking.db"
        conn = sqlite3.connect(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "insider_trades" in table_names
        assert "institutional_holdings" in table_names
        assert "short_interest" in table_names
        assert "cik_lookup" in table_names
        conn.close()

    def test_squeeze_potential_extreme(self):
        result = SECEdgarClient._calc_squeeze_potential(45, 12, 25)
        assert result == "extreme"

    def test_squeeze_potential_high(self):
        result = SECEdgarClient._calc_squeeze_potential(30, 7, 15)
        assert result == "high"

    def test_squeeze_potential_medium(self):
        result = SECEdgarClient._calc_squeeze_potential(16, 4, 12)
        assert result == "medium"

    def test_squeeze_potential_low(self):
        result = SECEdgarClient._calc_squeeze_potential(3, 1, 0)
        assert result == "low"

    def test_insider_cache_write_read(self):
        cache_dir = _tmp_cache_dir()
        client = SECEdgarClient(cache_dir=cache_dir)
        trades = [_make_trade(days_ago=5)]
        client._write_insider_cache(trades)
        cached = client._read_insider_cache("AAPL", lookback_days=30)
        assert len(cached) == 1
        assert cached[0].symbol == "AAPL"
        assert cached[0].insider_name == "Tim Cook"

    def test_insider_cache_expired(self):
        """Cache ældre end 24 timer returnerer tom."""
        cache_dir = _tmp_cache_dir()
        client = SECEdgarClient(cache_dir=cache_dir)
        # Indsæt med gammel fetched_at
        old_time = (datetime.now() - timedelta(hours=25)).isoformat()
        with client._get_conn() as conn:
            conn.execute(
                """INSERT INTO insider_trades
                   (symbol, insider_name, insider_title, transaction_type,
                    shares, price, value, trade_date, filing_date,
                    ownership_after, is_direct, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("AAPL", "Test", "CEO", "P", 100, 150, 15000,
                 datetime.now().strftime("%Y-%m-%d"),
                 datetime.now().strftime("%Y-%m-%d"),
                 1000, 1, old_time),
            )
        cached = client._read_insider_cache("AAPL", lookback_days=30)
        assert len(cached) == 0

    def test_holdings_cache_write_read(self):
        cache_dir = _tmp_cache_dir()
        client = SECEdgarClient(cache_dir=cache_dir)
        holdings = [
            InstitutionalHolding(
                fund_name="Berkshire", symbol="AAPL",
                shares=100_000, value_usd=15_000_000,
                pct_portfolio=5.0, quarter="2025Q4",
            )
        ]
        client._write_holdings_cache(holdings)
        cached = client._read_holdings_cache("AAPL")
        assert len(cached) == 1
        assert cached[0].fund_name == "Berkshire"

    def test_short_cache_write_read(self):
        cache_dir = _tmp_cache_dir()
        client = SECEdgarClient(cache_dir=cache_dir)
        data = ShortInterestData(
            symbol="GME",
            short_interest=50_000_000,
            short_pct_float=25.0,
            short_ratio=8.0,
            avg_volume=5_000_000,
            date=datetime.now(),
        )
        client._write_short_cache(data)
        cached = client._read_short_cache("GME")
        assert cached is not None
        assert cached.symbol == "GME"
        assert cached.short_pct_float == pytest.approx(25.0)


# ── Test InsiderTracker ──────────────────────────────────────

class TestInsiderTracker:
    def test_init(self):
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir())
        assert tracker._cluster_window == 14
        assert tracker._cluster_min == 3
        assert tracker._sentiment_lookback == 90

    def test_custom_init(self):
        tracker = InsiderTracker(
            cache_dir=_tmp_cache_dir(),
            cluster_window_days=7,
            cluster_min_insiders=2,
            sentiment_lookback_days=60,
        )
        assert tracker._cluster_window == 7
        assert tracker._cluster_min == 2


# ── Test Cluster Detection ───────────────────────────────────

class TestClusterDetection:
    def test_cluster_detected(self):
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir(), cluster_min_insiders=3)
        trades = _make_trades_cluster(n=4)
        assert tracker._detect_cluster(trades, is_buy=True) is True

    def test_no_cluster_too_few(self):
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir(), cluster_min_insiders=3)
        trades = _make_trades_cluster(n=2)
        assert tracker._detect_cluster(trades, is_buy=True) is False

    def test_no_cluster_empty(self):
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir())
        assert tracker._detect_cluster([], is_buy=True) is False

    def test_cluster_window_respected(self):
        """Trades spredt over >14 dage bør ikke trigge cluster."""
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir(), cluster_min_insiders=3)
        trades = [
            _make_trade(name="Person A", days_ago=1),
            _make_trade(name="Person B", days_ago=20),
            _make_trade(name="Person C", days_ago=40),
        ]
        assert tracker._detect_cluster(trades, is_buy=True) is False

    def test_same_insider_not_counted_twice(self):
        """Samme insider flere gange tæller som 1."""
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir(), cluster_min_insiders=3)
        trades = [
            _make_trade(name="Tim Cook", days_ago=1),
            _make_trade(name="Tim Cook", days_ago=3),
            _make_trade(name="Tim Cook", days_ago=5),
        ]
        assert tracker._detect_cluster(trades, is_buy=True) is False


# ── Test Sentiment Score Calculation ─────────────────────────

class TestSentimentScore:
    def test_all_buys_positive(self):
        score = InsiderTracker._calc_sentiment_score(
            purchases=5, sales=0,
            buy_value=500_000, sell_value=0,
            cluster_buying=False, cluster_selling=False,
            c_suite_buying=False,
        )
        assert score > 0

    def test_all_sales_negative(self):
        score = InsiderTracker._calc_sentiment_score(
            purchases=0, sales=5,
            buy_value=0, sell_value=500_000,
            cluster_buying=False, cluster_selling=False,
            c_suite_buying=False,
        )
        assert score < 0

    def test_no_trades_zero(self):
        score = InsiderTracker._calc_sentiment_score(
            purchases=0, sales=0,
            buy_value=0, sell_value=0,
            cluster_buying=False, cluster_selling=False,
            c_suite_buying=False,
        )
        assert score == 0.0

    def test_cluster_buying_boosts(self):
        base = InsiderTracker._calc_sentiment_score(
            purchases=3, sales=1,
            buy_value=300_000, sell_value=100_000,
            cluster_buying=False, cluster_selling=False,
            c_suite_buying=False,
        )
        boosted = InsiderTracker._calc_sentiment_score(
            purchases=3, sales=1,
            buy_value=300_000, sell_value=100_000,
            cluster_buying=True, cluster_selling=False,
            c_suite_buying=False,
        )
        assert boosted > base

    def test_c_suite_boosts(self):
        base = InsiderTracker._calc_sentiment_score(
            purchases=2, sales=0,
            buy_value=200_000, sell_value=0,
            cluster_buying=False, cluster_selling=False,
            c_suite_buying=False,
        )
        boosted = InsiderTracker._calc_sentiment_score(
            purchases=2, sales=0,
            buy_value=200_000, sell_value=0,
            cluster_buying=False, cluster_selling=False,
            c_suite_buying=True,
        )
        assert boosted > base

    def test_bounded(self):
        score = InsiderTracker._calc_sentiment_score(
            purchases=100, sales=0,
            buy_value=100_000_000, sell_value=0,
            cluster_buying=True, cluster_selling=False,
            c_suite_buying=True,
        )
        assert -100 <= score <= 100


# ── Test Score to Sentiment ──────────────────────────────────

class TestScoreToSentiment:
    def test_very_bullish(self):
        s = InsiderTracker._score_to_sentiment(50, cluster_buying=True, cluster_selling=False)
        assert s == InsiderSentiment.VERY_BULLISH

    def test_very_bearish(self):
        s = InsiderTracker._score_to_sentiment(-50, cluster_buying=False, cluster_selling=True)
        assert s == InsiderSentiment.VERY_BEARISH

    def test_bullish(self):
        s = InsiderTracker._score_to_sentiment(40, cluster_buying=False, cluster_selling=False)
        assert s == InsiderSentiment.BULLISH

    def test_bearish(self):
        s = InsiderTracker._score_to_sentiment(-40, cluster_buying=False, cluster_selling=False)
        assert s == InsiderSentiment.BEARISH

    def test_neutral(self):
        s = InsiderTracker._score_to_sentiment(5, cluster_buying=False, cluster_selling=False)
        assert s == InsiderSentiment.NEUTRAL

    def test_cluster_buying_low_score_neutral(self):
        """Cluster buying med lav score (≤20) forbliver neutral."""
        s = InsiderTracker._score_to_sentiment(10, cluster_buying=True, cluster_selling=False)
        assert s == InsiderSentiment.NEUTRAL


# ── Test InsiderSentiment med Mocked Trades ──────────────────

class TestInsiderSentimentWithMocks:
    def test_bullish_sentiment(self):
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir())
        trades = [
            _make_trade(name="CEO A", title="CEO", tx_type=TransactionType.PURCHASE, days_ago=5),
            _make_trade(name="CFO B", title="CFO", tx_type=TransactionType.PURCHASE, days_ago=7),
            _make_trade(name="COO C", title="COO", tx_type=TransactionType.PURCHASE, days_ago=10),
        ]
        with patch.object(tracker._client, "get_insider_trades", return_value=trades):
            result = tracker.get_insider_sentiment("AAPL")
            assert result.sentiment in (InsiderSentiment.VERY_BULLISH, InsiderSentiment.BULLISH)
            assert result.net_purchases == 3
            assert result.net_sales == 0
            assert result.c_suite_buying is True

    def test_bearish_sentiment(self):
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir())
        trades = [
            _make_trade(name="Dir A", title="Director", tx_type=TransactionType.SALE, days_ago=3),
            _make_trade(name="Dir B", title="Director", tx_type=TransactionType.SALE, days_ago=5),
            _make_trade(name="Dir C", title="Director", tx_type=TransactionType.SALE, days_ago=7),
            _make_trade(name="Dir D", title="Director", tx_type=TransactionType.SALE, days_ago=9),
            _make_trade(name="Dir E", title="Director", tx_type=TransactionType.SALE, days_ago=11),
        ]
        with patch.object(tracker._client, "get_insider_trades", return_value=trades):
            result = tracker.get_insider_sentiment("AAPL")
            assert result.sentiment in (InsiderSentiment.VERY_BEARISH, InsiderSentiment.BEARISH)
            assert result.net_sales == 5
            assert result.c_suite_buying is False

    def test_no_trades_neutral(self):
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir())
        with patch.object(tracker._client, "get_insider_trades", return_value=[]):
            result = tracker.get_insider_sentiment("AAPL")
            assert result.sentiment == InsiderSentiment.NEUTRAL
            assert result.score == 0.0

    def test_grants_ignored(self):
        """Grants og exercises bør ikke tælle som køb/salg."""
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir())
        trades = [
            _make_trade(tx_type=TransactionType.GRANT, days_ago=5),
            _make_trade(tx_type=TransactionType.EXERCISE, days_ago=7),
        ]
        with patch.object(tracker._client, "get_insider_trades", return_value=trades):
            result = tracker.get_insider_sentiment("AAPL")
            assert result.net_purchases == 0
            assert result.net_sales == 0
            assert result.sentiment == InsiderSentiment.NEUTRAL


# ── Test Smart Money Report ──────────────────────────────────

class TestSmartMoneyReportGeneration:
    def test_full_report(self):
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir())

        mock_sentiment = InsiderSentimentScore(
            symbol="AAPL",
            sentiment=InsiderSentiment.BULLISH,
            score=40.0,
            net_purchases=3,
            net_sales=0,
            total_buy_value=300_000,
            total_sell_value=0,
            cluster_buying=True,
            cluster_selling=False,
            c_suite_buying=True,
        )
        mock_flow = SmartMoneyFlow(
            symbol="AAPL",
            institutional_holders=5,
            total_institutional_value=100_000_000,
            net_institutional_change=5_000_000,
            new_positions=["ARK Invest"],
            closed_positions=[],
            increased=["Berkshire"],
            decreased=[],
        )
        mock_short = ShortInterestData(
            symbol="AAPL",
            short_interest=10_000,
            short_pct_float=1.0,
            short_ratio=0.5,
            avg_volume=50_000_000,
        )

        with patch.object(tracker, "get_insider_sentiment", return_value=mock_sentiment), \
             patch.object(tracker, "get_smart_money_flow", return_value=mock_flow), \
             patch.object(tracker, "get_short_interest", return_value=mock_short):
            report = tracker.get_smart_money_report("AAPL")

        assert report.symbol == "AAPL"
        assert report.overall_signal == "bullish"
        assert report.confidence_adjustment > 0
        assert any("Cluster buying" in w for w in report.warnings)
        assert any("C-suite" in w for w in report.warnings)

    def test_report_with_high_short_interest(self):
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir())

        mock_sentiment = InsiderSentimentScore(
            symbol="GME",
            sentiment=InsiderSentiment.NEUTRAL,
            score=0.0,
            net_purchases=0,
            net_sales=0,
            total_buy_value=0,
            total_sell_value=0,
            cluster_buying=False,
            cluster_selling=False,
            c_suite_buying=False,
        )
        mock_short = ShortInterestData(
            symbol="GME",
            short_interest=50_000_000,
            short_pct_float=25.0,
            short_ratio=8.0,
            avg_volume=5_000_000,
            squeeze_potential="high",
        )

        with patch.object(tracker, "get_insider_sentiment", return_value=mock_sentiment), \
             patch.object(tracker, "get_smart_money_flow", return_value=SmartMoneyFlow(
                 symbol="GME", institutional_holders=0,
                 total_institutional_value=0, net_institutional_change=0,
                 new_positions=[], closed_positions=[], increased=[], decreased=[],
             )), \
             patch.object(tracker, "get_short_interest", return_value=mock_short):
            report = tracker.get_smart_money_report("GME")

        assert report.confidence_adjustment < 0
        assert any("tungt shortet" in w for w in report.warnings)
        assert any("Short squeeze" in w for w in report.warnings)

    def test_report_handles_errors_gracefully(self):
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir())

        with patch.object(tracker, "get_insider_sentiment", side_effect=Exception("test")), \
             patch.object(tracker, "get_smart_money_flow", side_effect=Exception("test")), \
             patch.object(tracker, "get_short_interest", side_effect=Exception("test")):
            report = tracker.get_smart_money_report("AAPL")

        assert report.symbol == "AAPL"
        assert report.overall_signal == "neutral"
        assert report.insider_sentiment is None
        assert report.short_interest is None


# ── Test Strategy Integration ────────────────────────────────

class TestStrategyIntegration:
    def test_confidence_adjustment(self):
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir())
        mock_report = SmartMoneyReport(
            symbol="AAPL",
            insider_sentiment=None,
            smart_money_flow=None,
            short_interest=None,
            confidence_adjustment=10,
        )
        with patch.object(tracker, "get_smart_money_report", return_value=mock_report):
            adj = tracker.get_confidence_adjustment("AAPL")
            assert adj == 10

    def test_short_interest_warning(self):
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir())
        mock_short = ShortInterestData(
            symbol="GME",
            short_interest=50_000_000,
            short_pct_float=25.0,
            short_ratio=8.0,
            avg_volume=5_000_000,
        )
        with patch.object(tracker._client, "get_short_interest", return_value=mock_short):
            warning = tracker.get_short_interest_warning("GME", threshold=20.0)
            assert warning is not None
            assert "25.0%" in warning

    def test_no_short_warning_below_threshold(self):
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir())
        mock_short = ShortInterestData(
            symbol="AAPL",
            short_interest=10_000,
            short_pct_float=1.0,
            short_ratio=0.5,
            avg_volume=50_000_000,
        )
        with patch.object(tracker._client, "get_short_interest", return_value=mock_short):
            warning = tracker.get_short_interest_warning("AAPL", threshold=20.0)
            assert warning is None


# ── Test Explain ─────────────────────────────────────────────

class TestExplain:
    def test_explain_contains_sections(self):
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir())

        mock_sentiment = InsiderSentimentScore(
            symbol="AAPL",
            sentiment=InsiderSentiment.BULLISH,
            score=40.0,
            net_purchases=3,
            net_sales=1,
            total_buy_value=300_000,
            total_sell_value=50_000,
            cluster_buying=False,
            cluster_selling=False,
            c_suite_buying=True,
        )
        mock_short = ShortInterestData(
            symbol="AAPL",
            short_interest=10_000,
            short_pct_float=2.0,
            short_ratio=0.5,
            avg_volume=50_000_000,
        )

        with patch.object(tracker, "get_smart_money_report", return_value=SmartMoneyReport(
            symbol="AAPL",
            insider_sentiment=mock_sentiment,
            smart_money_flow=SmartMoneyFlow(
                symbol="AAPL", institutional_holders=3,
                total_institutional_value=100_000_000,
                net_institutional_change=0,
                new_positions=[], closed_positions=[],
                increased=[], decreased=[],
            ),
            short_interest=mock_short,
            overall_signal="bullish",
            confidence_adjustment=8,
        )):
            text = tracker.explain("AAPL")

        assert "SMART MONEY RAPPORT" in text
        assert "INSIDER AKTIVITET" in text
        assert "SHORT INTEREST" in text
        assert "INSTITUTIONELLE INVESTORER" in text
        assert "SAMLET VURDERING" in text
        assert "AAPL" in text

    def test_print_report(self, capsys):
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir())

        with patch.object(tracker, "get_smart_money_report", return_value=SmartMoneyReport(
            symbol="AAPL",
            insider_sentiment=None,
            smart_money_flow=None,
            short_interest=None,
            overall_signal="neutral",
        )):
            tracker.print_report("AAPL")

        captured = capsys.readouterr()
        assert "SMART MONEY RAPPORT" in captured.out


# ── Test Top Insider Buys ────────────────────────────────────

class TestTopInsiderBuys:
    def test_filters_by_min_value(self):
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir())
        trades = [
            _make_trade(symbol="AAPL", shares=100, price=150.0, days_ago=5),      # $15,000
            _make_trade(symbol="MSFT", shares=1000, price=300.0, days_ago=3),     # $300,000
            _make_trade(symbol="GOOGL", shares=500, price=120.0, days_ago=7),     # $60,000
        ]

        with patch.object(tracker._client, "get_insider_trades", return_value=trades):
            top = tracker.get_top_insider_buys(
                symbols=["AAPL"], lookback_days=30, min_value=50_000,
            )
        # Kun $300k og $60k bør passe
        assert all(t.value >= 50_000 for t in top)

    def test_sorted_by_value(self):
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir())
        trades = [
            _make_trade(shares=100, price=100.0, days_ago=5),
            _make_trade(shares=1000, price=100.0, name="Person B", days_ago=3),
        ]

        with patch.object(tracker._client, "get_insider_trades", return_value=trades):
            top = tracker.get_top_insider_buys(
                symbols=["AAPL"], lookback_days=30, min_value=0,
            )
        if len(top) >= 2:
            assert top[0].value >= top[1].value


# ── Test Scan Symbols ────────────────────────────────────────

class TestScanSymbols:
    def test_scan_multiple(self):
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir())
        mock_report = SmartMoneyReport(
            symbol="AAPL", insider_sentiment=None,
            smart_money_flow=None, short_interest=None,
            overall_signal="bullish",
        )

        with patch.object(tracker, "get_smart_money_report", return_value=mock_report):
            results = tracker.scan_symbols(["AAPL", "MSFT"])

        assert len(results) == 2
        assert "AAPL" in results
        assert "MSFT" in results

    def test_scan_handles_errors(self):
        tracker = InsiderTracker(cache_dir=_tmp_cache_dir())

        with patch.object(tracker, "get_smart_money_report", side_effect=Exception("fail")):
            results = tracker.scan_symbols(["AAPL"])

        assert len(results) == 0


# ── Test Constants ───────────────────────────────────────────

class TestConstants:
    def test_major_funds_populated(self):
        assert len(MAJOR_FUNDS) >= 8
        assert "Berkshire Hathaway" in MAJOR_FUNDS
        assert "Renaissance Technologies" in MAJOR_FUNDS

    def test_insider_roles(self):
        assert "CEO" in INSIDER_ROLES
        assert "CFO" in INSIDER_ROLES
        assert "Director" in INSIDER_ROLES

    def test_user_agent(self):
        assert "AlphaTrading" in SEC_USER_AGENT or "alpha" in SEC_USER_AGENT.lower()


# ── Test Form 4 XML Parsing ─────────────────────────────────

class TestForm4XMLParsing:
    SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
    <ownershipDocument>
        <reportingOwner>
            <reportingOwnerId>
                <rptOwnerCik>0001234567</rptOwnerCik>
                <rptOwnerName>John Doe</rptOwnerName>
            </reportingOwnerId>
            <reportingOwnerRelationship>
                <isDirector>0</isDirector>
                <isOfficer>1</isOfficer>
                <officerTitle>Chief Executive Officer</officerTitle>
            </reportingOwnerRelationship>
        </reportingOwner>
        <nonDerivativeTable>
            <nonDerivativeTransaction>
                <transactionDate><value>2026-03-10</value></transactionDate>
                <transactionCoding>
                    <transactionCode>P</transactionCode>
                </transactionCoding>
                <transactionAmounts>
                    <transactionShares><value>5000</value></transactionShares>
                    <transactionPricePerShare><value>150.00</value></transactionPricePerShare>
                </transactionAmounts>
                <postTransactionAmounts>
                    <sharesOwnedFollowingTransaction><value>25000</value></sharesOwnedFollowingTransaction>
                </postTransactionAmounts>
                <ownershipNature>
                    <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
                </ownershipNature>
            </nonDerivativeTransaction>
        </nonDerivativeTable>
    </ownershipDocument>"""

    def test_parse_form4_xml(self):
        cache_dir = _tmp_cache_dir()
        client = SECEdgarClient(cache_dir=cache_dir)
        trades = client._parse_form4_xml("AAPL", self.SAMPLE_XML, "2026-03-11")
        assert len(trades) == 1
        t = trades[0]
        assert t.symbol == "AAPL"
        assert t.insider_name == "John Doe"
        assert t.insider_title == "Chief Executive Officer"
        assert t.transaction_type == TransactionType.PURCHASE
        assert t.shares == 5000
        assert t.price == 150.0
        assert t.value == 750_000.0
        assert t.ownership_after == 25000
        assert t.is_direct is True

    def test_parse_sale_xml(self):
        xml = self.SAMPLE_XML.replace(
            "<transactionCode>P</transactionCode>",
            "<transactionCode>S</transactionCode>",
        )
        cache_dir = _tmp_cache_dir()
        client = SECEdgarClient(cache_dir=cache_dir)
        trades = client._parse_form4_xml("AAPL", xml, "2026-03-11")
        assert len(trades) == 1
        assert trades[0].transaction_type == TransactionType.SALE

    def test_parse_invalid_xml(self):
        cache_dir = _tmp_cache_dir()
        client = SECEdgarClient(cache_dir=cache_dir)
        trades = client._parse_form4_xml("AAPL", "invalid xml", "2026-03-11")
        assert trades == []

    def test_parse_director_role(self):
        xml = self.SAMPLE_XML.replace(
            "<isDirector>0</isDirector>",
            "<isDirector>1</isDirector>",
        ).replace(
            "<isOfficer>1</isOfficer>",
            "<isOfficer>0</isOfficer>",
        ).replace(
            "<officerTitle>Chief Executive Officer</officerTitle>",
            "<officerTitle></officerTitle>",
        )
        cache_dir = _tmp_cache_dir()
        client = SECEdgarClient(cache_dir=cache_dir)
        trades = client._parse_form4_xml("AAPL", xml, "2026-03-11")
        assert len(trades) == 1
        assert trades[0].insider_title == "Director"
