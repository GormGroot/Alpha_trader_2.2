"""
Tests for AlternativeDataTracker – Google Trends, GitHub, USPTO, job data.

Alle API-kald mockes – ingen netværksforbindelse kræves.
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from src.data.alternative_data import (
    AlternativeDataTracker,
    GoogleTrendsResult,
    WebTrafficResult,
    JobPostingsResult,
    PatentResult,
    GitHubActivityResult,
    AppRankingResult,
    AltDataScore,
    TrendDirection,
    AltDataSignal,
    DEFAULT_SEARCH_TERMS,
    DEFAULT_GITHUB_ORGS,
    DEFAULT_PATENT_ASSIGNEES,
    DEFAULT_WEBSITES,
)


# ── Helpers ──────────────────────────────────────────────────

def _tmp_cache_dir() -> str:
    return tempfile.mkdtemp()


def _make_trends(
    symbol: str = "AAPL",
    current: float = 75.0,
    avg_30d: float = 70.0,
    avg_90d: float = 60.0,
    direction: TrendDirection = TrendDirection.RISING,
    change_pct: float = 16.7,
    spike: bool = False,
) -> GoogleTrendsResult:
    return GoogleTrendsResult(
        symbol=symbol,
        search_terms=["Apple", "iPhone"],
        current_interest=current,
        avg_interest_30d=avg_30d,
        avg_interest_90d=avg_90d,
        trend_direction=direction,
        change_pct_30d=change_pct,
        spike_detected=spike,
        related_rising=["buy iPhone 16", "Apple AI"],
    )


def _make_github(
    symbol: str = "MSFT",
    stars: int = 150_000,
    repos: int = 500,
    direction: TrendDirection = TrendDirection.RISING,
) -> GitHubActivityResult:
    return GitHubActivityResult(
        symbol=symbol,
        org_name="microsoft",
        public_repos=repos,
        total_stars=stars,
        total_forks=stars // 3,
        recent_commits_30d=1000,
        trend_direction=direction,
        top_repos=[{"name": "vscode", "stars": 160000, "language": "TypeScript"}],
    )


def _make_patent(
    symbol: str = "AAPL",
    ytd: int = 200,
    prev: int = 180,
    ai_count: int = 15,
) -> PatentResult:
    return PatentResult(
        symbol=symbol,
        assignee="Apple Inc",
        total_patents_ytd=ytd,
        total_patents_prev_year=prev,
        change_pct=((ytd - prev) / prev * 100) if prev > 0 else 0,
        recent_patents=["AI Processing Unit", "Neural Engine v3"],
        ai_related_count=ai_count,
    )


# ── Test TrendDirection ──────────────────────────────────────

class TestTrendDirection:
    def test_all_values(self):
        assert len(TrendDirection) == 4

    def test_values(self):
        assert TrendDirection.RISING.value == "rising"
        assert TrendDirection.SPIKE.value == "spike"
        assert TrendDirection.FALLING.value == "falling"
        assert TrendDirection.STABLE.value == "stable"


# ── Test AltDataSignal ───────────────────────────────────────

class TestAltDataSignal:
    def test_all_values(self):
        assert len(AltDataSignal) == 3
        assert AltDataSignal.BULLISH.value == "bullish"


# ── Test GoogleTrendsResult ──────────────────────────────────

class TestGoogleTrendsResult:
    def test_score_rising(self):
        gt = _make_trends(direction=TrendDirection.RISING, change_pct=30)
        assert gt.score > 60

    def test_score_falling(self):
        gt = _make_trends(direction=TrendDirection.FALLING, change_pct=-25)
        assert gt.score < 45

    def test_score_stable(self):
        gt = _make_trends(direction=TrendDirection.STABLE, change_pct=0)
        assert 40 <= gt.score <= 60

    def test_score_spike(self):
        gt = _make_trends(direction=TrendDirection.SPIKE, change_pct=60)
        assert gt.score > 65

    def test_score_bounded(self):
        gt = _make_trends(direction=TrendDirection.RISING, change_pct=100)
        assert 0 <= gt.score <= 100

        gt2 = _make_trends(direction=TrendDirection.FALLING, change_pct=-100)
        assert 0 <= gt2.score <= 100


# ── Test WebTrafficResult ────────────────────────────────────

class TestWebTrafficResult:
    def test_score_rising(self):
        wt = WebTrafficResult(
            symbol="AAPL", website="apple.com",
            estimated_visits=50_000_000,
            trend_direction=TrendDirection.RISING,
            change_pct=25.0,
        )
        assert wt.score > 60

    def test_score_falling(self):
        wt = WebTrafficResult(
            symbol="NFLX", website="netflix.com",
            estimated_visits=30_000_000,
            trend_direction=TrendDirection.FALLING,
            change_pct=-25.0,
        )
        assert wt.score < 40

    def test_score_bounded(self):
        wt = WebTrafficResult(
            symbol="X", website="x.com",
            estimated_visits=0,
            trend_direction=TrendDirection.FALLING,
            change_pct=-50.0,
        )
        assert 0 <= wt.score <= 100


# ── Test JobPostingsResult ───────────────────────────────────

class TestJobPostingsResult:
    def test_score_hiring(self):
        jp = JobPostingsResult(
            symbol="AMZN", company_name="Amazon",
            active_postings=5000,
            change_pct_30d=40.0,
            trend_direction=TrendDirection.RISING,
            top_categories=["Engineering"],
            hiring_signal=AltDataSignal.BULLISH,
        )
        assert jp.score > 60

    def test_score_cutting(self):
        jp = JobPostingsResult(
            symbol="META", company_name="Meta",
            active_postings=1000,
            change_pct_30d=-30.0,
            trend_direction=TrendDirection.FALLING,
            top_categories=["Sales"],
            hiring_signal=AltDataSignal.BEARISH,
        )
        assert jp.score < 40


# ── Test PatentResult ────────────────────────────────────────

class TestPatentResult:
    def test_score_innovation(self):
        pt = _make_patent(ytd=250, prev=180, ai_count=20)
        assert pt.score > 65

    def test_score_declining(self):
        pt = _make_patent(ytd=100, prev=200, ai_count=0)
        assert pt.score < 45

    def test_ai_bonus(self):
        no_ai = _make_patent(ytd=200, prev=200, ai_count=0)
        with_ai = _make_patent(ytd=200, prev=200, ai_count=15)
        assert with_ai.score > no_ai.score

    def test_score_bounded(self):
        pt = _make_patent(ytd=1000, prev=100, ai_count=50)
        assert 0 <= pt.score <= 100


# ── Test GitHubActivityResult ────────────────────────────────

class TestGitHubActivityResult:
    def test_score_active(self):
        gh = _make_github(stars=200_000, direction=TrendDirection.RISING)
        assert gh.score > 65

    def test_score_declining(self):
        gh = _make_github(stars=5_000, direction=TrendDirection.FALLING)
        assert gh.score < 50

    def test_high_stars_bonus(self):
        low = _make_github(stars=1_000)
        high = _make_github(stars=200_000)
        assert high.score > low.score


# ── Test AppRankingResult ────────────────────────────────────

class TestAppRankingResult:
    def test_top_5_score(self):
        ar = AppRankingResult(
            symbol="AAPL", app_name="Apple Store",
            category_rank=3, overall_rank=6,
            trend_direction=TrendDirection.RISING,
        )
        assert ar.score > 70

    def test_low_rank_score(self):
        ar = AppRankingResult(
            symbol="RBLX", app_name="Roblox",
            category_rank=80, overall_rank=160,
            trend_direction=TrendDirection.FALLING,
        )
        assert ar.score < 50

    def test_no_rank(self):
        ar = AppRankingResult(
            symbol="IBM", app_name="IBM",
            category_rank=None, overall_rank=None,
        )
        assert ar.score == 50


# ── Test AlternativeDataTracker Init ─────────────────────────

class TestAlternativeDataTrackerInit:
    def test_creates_db(self):
        cache_dir = _tmp_cache_dir()
        tracker = AlternativeDataTracker(cache_dir=cache_dir)
        db_path = Path(cache_dir) / "alternative_data.db"
        assert db_path.exists()

    def test_db_tables(self):
        cache_dir = _tmp_cache_dir()
        tracker = AlternativeDataTracker(cache_dir=cache_dir)
        db_path = Path(cache_dir) / "alternative_data.db"
        conn = sqlite3.connect(db_path)
        tables = {t[0] for t in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "google_trends" in tables
        assert "github_activity" in tables
        assert "patent_data" in tables
        assert "alt_data_scores" in tables
        conn.close()

    def test_custom_mappings(self):
        tracker = AlternativeDataTracker(
            cache_dir=_tmp_cache_dir(),
            search_terms={"TEST": ["Test Corp"]},
            github_orgs={"TEST": "testorg"},
            patent_assignees={"TEST": "Test Corp"},
        )
        assert "TEST" in tracker._search_terms
        assert "TEST" in tracker._github_orgs
        assert "TEST" in tracker._patent_assignees


# ── Test Google Trends ───────────────────────────────────────

class TestGoogleTrends:
    def test_trends_with_mock_pytrends(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())

        # Mock pytrends
        mock_interest = pd.DataFrame({
            "Apple": [50, 55, 60, 58, 62, 65, 70, 72, 75, 78, 80, 82],
            "isPartial": [False] * 12,
        }, index=pd.date_range("2026-01-01", periods=12, freq="W"))

        mock_pytrends = MagicMock()
        mock_pytrends.interest_over_time.return_value = mock_interest
        mock_pytrends.related_queries.return_value = {
            "Apple": {
                "rising": pd.DataFrame({"query": ["buy iPhone", "Apple AI"], "value": [100, 80]}),
                "top": pd.DataFrame({"query": ["Apple"], "value": [100]}),
            }
        }

        with patch("src.data.alternative_data.TrendReq", return_value=mock_pytrends, create=True), \
             patch("src.data.alternative_data._HAS_PYTRENDS", True):
            result = tracker.get_google_trends("AAPL", use_cache=False)

        assert result.symbol == "AAPL"
        assert result.current_interest > 0
        assert result.trend_direction in (TrendDirection.RISING, TrendDirection.STABLE, TrendDirection.SPIKE)
        assert len(result.related_rising) > 0

    def test_trends_no_pytrends(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())

        with patch("src.data.alternative_data._HAS_PYTRENDS", False):
            result = tracker.get_google_trends("AAPL", use_cache=False)

        assert result.current_interest == 0
        assert result.trend_direction == TrendDirection.STABLE

    def test_trends_cache_write_read(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())
        trends = _make_trends()
        tracker._write_trends_cache(trends)
        cached = tracker._read_trends_cache("AAPL")
        assert cached is not None
        assert cached.symbol == "AAPL"
        assert cached.current_interest == 75.0

    def test_trends_cache_expired(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())
        # Insert med gammel timestamp
        old_time = (datetime.now() - timedelta(hours=13)).isoformat()
        with tracker._get_conn() as conn:
            conn.execute(
                """INSERT INTO google_trends
                   (symbol, search_term, current_interest, avg_30d, avg_90d,
                    change_pct_30d, trend_direction, spike_detected, date, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("AAPL", "Apple", 50, 45, 40, 10, "rising", 0,
                 datetime.now().strftime("%Y-%m-%d"), old_time),
            )
        cached = tracker._read_trends_cache("AAPL")
        assert cached is None

    def test_empty_trends_result(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())
        result = tracker._empty_trends_result("AAPL", ["Apple"])
        assert result.current_interest == 0
        assert result.symbol == "AAPL"


# ── Test Web Traffic ─────────────────────────────────────────

class TestWebTraffic:
    def test_estimate_uses_trends(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())
        mock_trends = _make_trends(current=80, change_pct=20)

        with patch.object(tracker, "get_google_trends", return_value=mock_trends):
            result = tracker.estimate_web_traffic("AAPL")

        assert result.symbol == "AAPL"
        assert result.website == "apple.com"
        assert result.estimated_visits > 0

    def test_unknown_website(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())
        result = tracker.estimate_web_traffic("UNKNOWN_TICKER")
        assert result.website == "unknown"
        assert result.estimated_visits == 0


# ── Test Job Postings ────────────────────────────────────────

class TestJobPostings:
    def test_analyze_jobs(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())
        mock_trends = _make_trends(change_pct=30)

        with patch.object(tracker, "get_google_trends", return_value=mock_trends):
            result = tracker.analyze_job_postings("AAPL")

        assert result.symbol == "AAPL"
        assert result.hiring_signal == AltDataSignal.BULLISH

    def test_firing_signal(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())
        mock_trends = _make_trends(change_pct=-30)

        with patch.object(tracker, "get_google_trends", return_value=mock_trends):
            result = tracker.analyze_job_postings("META")

        assert result.hiring_signal == AltDataSignal.BEARISH


# ── Test Patent Activity ─────────────────────────────────────

class TestPatentActivity:
    def test_patent_fetch(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())

        with patch.object(
            tracker, "_fetch_patents_from_uspto",
            side_effect=[(150, ["AI Chip", "Neural Engine"], 5), (120, [], 0)],
        ):
            result = tracker.get_patent_activity("AAPL", use_cache=False)

        assert result.symbol == "AAPL"
        assert result.total_patents_ytd == 150
        assert result.total_patents_prev_year == 120
        assert result.ai_related_count == 5

    def test_unknown_assignee(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())
        result = tracker.get_patent_activity("UNKNOWNTICKER")
        assert result.total_patents_ytd == 0

    def test_patent_cache(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())
        patent = _make_patent()
        tracker._write_patent_cache(patent)
        cached = tracker._read_patent_cache("AAPL")
        assert cached is not None
        assert cached.total_patents_ytd == 200


# ── Test GitHub Activity ─────────────────────────────────────

class TestGitHubActivity:
    def test_github_fetch(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())

        mock_org_resp = MagicMock()
        mock_org_resp.status_code = 200
        mock_org_resp.json.return_value = {"public_repos": 500}

        mock_repos_resp = MagicMock()
        mock_repos_resp.status_code = 200
        mock_repos_resp.json.return_value = [
            {
                "name": "vscode",
                "stargazers_count": 160000,
                "forks_count": 28000,
                "language": "TypeScript",
                "updated_at": datetime.now().isoformat(),
            },
            {
                "name": "typescript",
                "stargazers_count": 95000,
                "forks_count": 12000,
                "language": "TypeScript",
                "updated_at": datetime.now().isoformat(),
            },
        ]

        with patch("src.data.alternative_data._requests") as mock_requests, \
             patch("src.data.alternative_data._HAS_REQUESTS", True):
            mock_requests.get.side_effect = [mock_org_resp, mock_repos_resp]
            result = tracker.get_github_activity("MSFT", use_cache=False)

        assert result.symbol == "MSFT"
        assert result.org_name == "microsoft"
        assert result.public_repos == 500
        assert result.total_stars > 0
        assert len(result.top_repos) > 0

    def test_github_unknown_org(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())
        result = tracker.get_github_activity("UNKNOWNTICKER")
        assert result.org_name == "unknown"
        assert result.total_stars == 0

    def test_github_cache(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())
        gh = _make_github()
        tracker._write_github_cache(gh)
        cached = tracker._read_github_cache("MSFT")
        assert cached is not None
        assert cached.total_stars == 150_000

    def test_github_api_error(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())

        mock_resp = MagicMock()
        mock_resp.status_code = 403

        with patch("src.data.alternative_data._requests") as mock_requests, \
             patch("src.data.alternative_data._HAS_REQUESTS", True):
            mock_requests.get.return_value = mock_resp
            result = tracker.get_github_activity("MSFT", use_cache=False)

        assert result.total_stars == 0


# ── Test App Ranking ─────────────────────────────────────────

class TestAppRanking:
    def test_app_ranking(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())
        mock_trends = _make_trends(current=90, change_pct=25)

        with patch.object(tracker, "get_google_trends", return_value=mock_trends):
            result = tracker.get_app_ranking("AAPL")

        assert result.symbol == "AAPL"
        assert result.category_rank is not None
        assert result.category_rank <= 10  # High interest → low rank number


# ── Test Alt Data Score ──────────────────────────────────────

class TestAltDataScore:
    def test_full_score(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())

        mock_trends = _make_trends(current=80, change_pct=20)
        mock_traffic = WebTrafficResult(
            symbol="AAPL", website="apple.com",
            estimated_visits=80_000_000,
            trend_direction=TrendDirection.RISING,
            change_pct=15.0, signal=AltDataSignal.BULLISH,
        )
        mock_jobs = JobPostingsResult(
            symbol="AAPL", company_name="Apple",
            active_postings=3000, change_pct_30d=15.0,
            trend_direction=TrendDirection.STABLE,
            top_categories=["Engineering"],
            hiring_signal=AltDataSignal.NEUTRAL,
        )
        mock_patent = _make_patent(ytd=200, prev=180, ai_count=10)
        mock_github = _make_github(stars=100_000)
        mock_app = AppRankingResult(
            symbol="AAPL", app_name="Apple Store",
            category_rank=3, overall_rank=6,
            trend_direction=TrendDirection.RISING,
        )

        with patch.object(tracker, "get_google_trends", return_value=mock_trends), \
             patch.object(tracker, "estimate_web_traffic", return_value=mock_traffic), \
             patch.object(tracker, "analyze_job_postings", return_value=mock_jobs), \
             patch.object(tracker, "get_patent_activity", return_value=mock_patent), \
             patch.object(tracker, "get_github_activity", return_value=mock_github), \
             patch.object(tracker, "get_app_ranking", return_value=mock_app):
            score = tracker.calculate_alt_data_score("AAPL")

        assert score.symbol == "AAPL"
        assert 0 <= score.overall_score <= 100
        assert score.signal in (AltDataSignal.BULLISH, AltDataSignal.NEUTRAL, AltDataSignal.BEARISH)
        assert -10 <= score.confidence_adjustment <= 10
        assert len(score.components) > 0

    def test_score_handles_errors(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())

        with patch.object(tracker, "get_google_trends", side_effect=Exception("fail")), \
             patch.object(tracker, "estimate_web_traffic", side_effect=Exception("fail")), \
             patch.object(tracker, "analyze_job_postings", side_effect=Exception("fail")), \
             patch.object(tracker, "get_patent_activity", side_effect=Exception("fail")), \
             patch.object(tracker, "get_github_activity", side_effect=Exception("fail")), \
             patch.object(tracker, "get_app_ranking", side_effect=Exception("fail")):
            score = tracker.calculate_alt_data_score("AAPL")

        assert score.symbol == "AAPL"
        assert score.overall_score == 50.0  # Default neutral
        assert score.signal == AltDataSignal.NEUTRAL

    def test_score_bullish(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())

        # All bullish signals
        mock_trends = _make_trends(
            current=90, change_pct=50,
            direction=TrendDirection.SPIKE, spike=True,
        )

        with patch.object(tracker, "get_google_trends", return_value=mock_trends), \
             patch.object(tracker, "estimate_web_traffic", return_value=WebTrafficResult(
                 symbol="AAPL", website="apple.com", estimated_visits=90_000_000,
                 trend_direction=TrendDirection.RISING, change_pct=30.0,
             )), \
             patch.object(tracker, "analyze_job_postings", return_value=JobPostingsResult(
                 symbol="AAPL", company_name="Apple", active_postings=5000,
                 change_pct_30d=40.0, trend_direction=TrendDirection.RISING,
                 top_categories=["Eng"], hiring_signal=AltDataSignal.BULLISH,
             )), \
             patch.object(tracker, "get_patent_activity", return_value=_make_patent(
                 ytd=300, prev=200, ai_count=20,
             )), \
             patch.object(tracker, "get_github_activity", return_value=_make_github(
                 stars=200_000, direction=TrendDirection.RISING,
             )), \
             patch.object(tracker, "get_app_ranking", return_value=AppRankingResult(
                 symbol="AAPL", app_name="Apple", category_rank=1, overall_rank=2,
                 trend_direction=TrendDirection.RISING,
             )):
            score = tracker.calculate_alt_data_score("AAPL")

        assert score.overall_score > 60
        assert score.signal == AltDataSignal.BULLISH
        assert score.confidence_adjustment > 0

    def test_confidence_bounded(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())

        with patch.object(tracker, "get_google_trends", side_effect=Exception), \
             patch.object(tracker, "estimate_web_traffic", side_effect=Exception), \
             patch.object(tracker, "analyze_job_postings", side_effect=Exception), \
             patch.object(tracker, "get_patent_activity", side_effect=Exception), \
             patch.object(tracker, "get_github_activity", side_effect=Exception), \
             patch.object(tracker, "get_app_ranking", side_effect=Exception):
            score = tracker.calculate_alt_data_score("AAPL")

        assert -10 <= score.confidence_adjustment <= 10

    def test_score_cache(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())
        score = AltDataScore(
            symbol="AAPL", overall_score=72.0,
            signal=AltDataSignal.BULLISH,
            components={"google_trends": 80, "github": 65},
        )
        tracker._write_score_cache(score)
        # Verify in DB
        with tracker._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM alt_data_scores WHERE symbol = 'AAPL'"
            ).fetchall()
        assert len(rows) == 1


# ── Test Scan Symbols ────────────────────────────────────────

class TestScanSymbols:
    def test_scan(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())
        mock_score = AltDataScore(
            symbol="AAPL", overall_score=65,
            signal=AltDataSignal.BULLISH,
        )
        with patch.object(tracker, "calculate_alt_data_score", return_value=mock_score):
            results = tracker.scan_symbols(["AAPL", "MSFT"])
        assert len(results) == 2

    def test_scan_handles_errors(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())
        with patch.object(tracker, "calculate_alt_data_score", side_effect=Exception("fail")):
            results = tracker.scan_symbols(["AAPL"])
        assert len(results) == 0


# ── Test Strategy Integration ────────────────────────────────

class TestStrategyIntegration:
    def test_confidence_adjustment(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())
        mock_score = AltDataScore(
            symbol="AAPL", overall_score=70,
            signal=AltDataSignal.BULLISH,
            confidence_adjustment=4,
        )
        with patch.object(tracker, "calculate_alt_data_score", return_value=mock_score):
            adj = tracker.get_confidence_adjustment("AAPL")
        assert adj == 4


# ── Test Explain ─────────────────────────────────────────────

class TestExplain:
    def test_explain_contains_sections(self):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())

        mock_score = AltDataScore(
            symbol="AAPL",
            overall_score=68.0,
            signal=AltDataSignal.BULLISH,
            google_trends=_make_trends(),
            job_postings=JobPostingsResult(
                symbol="AAPL", company_name="Apple",
                active_postings=3000, change_pct_30d=15,
                trend_direction=TrendDirection.STABLE,
                top_categories=["Eng"], hiring_signal=AltDataSignal.NEUTRAL,
            ),
            patents=_make_patent(),
            github_activity=_make_github(),
            components={"google_trends": 75, "github": 70, "patents": 65},
            confidence_adjustment=4,
        )

        with patch.object(tracker, "calculate_alt_data_score", return_value=mock_score):
            text = tracker.explain("AAPL")

        assert "ALTERNATIV DATA RAPPORT" in text
        assert "GOOGLE TRENDS" in text
        assert "JOBOPSLAG" in text
        assert "PATENT" in text
        assert "GITHUB" in text
        assert "SAMLET ALT DATA SCORE" in text

    def test_print_report(self, capsys):
        tracker = AlternativeDataTracker(cache_dir=_tmp_cache_dir())
        mock_score = AltDataScore(
            symbol="AAPL", overall_score=50,
            signal=AltDataSignal.NEUTRAL,
            components={},
        )
        with patch.object(tracker, "calculate_alt_data_score", return_value=mock_score):
            tracker.print_report("AAPL")

        captured = capsys.readouterr()
        assert "ALTERNATIV DATA RAPPORT" in captured.out


# ── Test Constants / Mappings ────────────────────────────────

class TestConstants:
    def test_search_terms(self):
        assert "AAPL" in DEFAULT_SEARCH_TERMS
        assert "TSLA" in DEFAULT_SEARCH_TERMS
        assert len(DEFAULT_SEARCH_TERMS["AAPL"]) >= 2

    def test_github_orgs(self):
        assert "MSFT" in DEFAULT_GITHUB_ORGS
        assert DEFAULT_GITHUB_ORGS["MSFT"] == "microsoft"

    def test_patent_assignees(self):
        assert "AAPL" in DEFAULT_PATENT_ASSIGNEES
        assert "Apple" in DEFAULT_PATENT_ASSIGNEES["AAPL"]

    def test_websites(self):
        assert "AAPL" in DEFAULT_WEBSITES
        assert DEFAULT_WEBSITES["AAPL"] == "apple.com"
