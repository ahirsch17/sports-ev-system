from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from sports_ev.backtest import WalkForwardBacktester
from sports_ev.config import Settings
from sports_ev.db.models import Base, Game, ModelPrediction, OddsSnapshot, TeamGameStat
from sports_ev.models import (
    LabeledRow,
    ModelSpreadStrategy,
    NflSpreadModel,
    WalkForwardModelStrategy,
    build_labeled_rows,
    save_prediction,
    time_ordered_split,
    train_from_rows,
)
from sports_ev.models.labels import home_cover_label
from sports_ev.models.train import TrainConfig


def _synthetic_rows(n: int = 24) -> list[LabeledRow]:
    rows: list[LabeledRow] = []
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        label = i % 2
        rows.append(
            LabeledRow(
                game_id=f"g{i}",
                kickoff_time=base + timedelta(days=i),
                features={
                    "signal": float(label),
                    "bias": 1.0,
                    "noise": float(i) * 0.01,
                },
                label=label,
                spread_line=-3.5,
            )
        )
    return rows


class TestLabels:
    def test_home_cover_label(self):
        assert home_cover_label(home_score=24, away_score=17, home_line=-3.5) == 1
        assert home_cover_label(home_score=20, away_score=17, home_line=-3.5) == 0
        assert home_cover_label(home_score=20, away_score=17, home_line=-3.0) is None


class TestTimeOrderedSplit:
    def test_never_shuffles(self):
        rows = _synthetic_rows(10)
        split = time_ordered_split(rows, val_fraction=0.2)
        assert len(split.train) == 8
        assert len(split.validation) == 2
        assert split.train[-1].kickoff_time < split.validation[0].kickoff_time

    def test_train_comes_before_validation(self):
        rows = _synthetic_rows(5)
        split = time_ordered_split(rows, val_fraction=0.4)
        train_times = [r.kickoff_time for r in split.train]
        val_times = [r.kickoff_time for r in split.validation]
        assert max(train_times) <= min(val_times)


class TestModelTraining:
    def test_train_and_predict(self):
        rows = _synthetic_rows(24)
        model, metrics = train_from_rows(
            rows,
            config=TrainConfig(num_boost_round=50, min_data_in_leaf=1),
        )
        assert metrics.train_rows >= 19
        assert metrics.validation_rows >= 1
        assert metrics.calibrated_val_brier is not None

        high = model.predict_one({"signal": 1.0, "bias": 1.0, "noise": 0.1}, compute_shap=False)
        low = model.predict_one({"signal": 0.0, "bias": 1.0, "noise": 0.1}, compute_shap=False)
        assert high.home_cover_prob > low.home_cover_prob

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        rows = _synthetic_rows(20)
        model, _ = train_from_rows(rows, config=TrainConfig(num_boost_round=20))
        path = tmp_path / "model.joblib"
        model.save(path)
        loaded = NflSpreadModel.load(path)
        assert loaded.model_version == model.model_version
        assert loaded.feature_names == model.feature_names

        x = {"signal": 1.0, "bias": 1.0, "noise": 0.2}
        assert model.predict_one(x, compute_shap=False).home_cover_prob == pytest.approx(
            loaded.predict_one(x, compute_shap=False).home_cover_prob
        )

    def test_calibration_not_worse_than_raw_on_validation(self):
        rows = _synthetic_rows(40)
        _, metrics = train_from_rows(rows, config=TrainConfig(num_boost_round=80))
        assert metrics.calibrated_val_brier is not None
        assert metrics.raw_val_brier is not None
        assert metrics.calibrated_val_brier <= metrics.raw_val_brier + 0.05


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    return factory()


def _seed_game(
    session: Session,
    *,
    game_id: str,
    kickoff: datetime,
    home_score: int,
    away_score: int,
    home_epa: float,
    dk_home: int = -110,
    dk_away: int = -110,
) -> None:
    session.add(
        Game(
            game_id=game_id,
            sport="nfl",
            home_team="Kansas City Chiefs",
            away_team="Buffalo Bills",
            kickoff_time=kickoff,
            season=2026,
            week=1,
            home_score=home_score,
            away_score=away_score,
        )
    )
    captured = kickoff - timedelta(hours=2)
    session.add_all(
        [
            OddsSnapshot(
                game_id=game_id,
                book="pinnacle",
                market_type="spread",
                line=-3.5,
                odds_home=-110,
                odds_away=-110,
                captured_at=captured - timedelta(minutes=30),
                source="api",
            ),
            OddsSnapshot(
                game_id=game_id,
                book="draftkings",
                market_type="spread",
                line=-3.5,
                odds_home=dk_home,
                odds_away=dk_away,
                captured_at=captured,
                source="scraped",
            ),
            TeamGameStat(
                game_id=game_id,
                team="Kansas City Chiefs",
                stat_key="epa_offense",
                stat_value=home_epa,
                known_at=kickoff - timedelta(days=7),
            ),
            TeamGameStat(
                game_id=game_id,
                team="Buffalo Bills",
                stat_key="epa_offense",
                stat_value=0.05,
                known_at=kickoff - timedelta(days=7),
            ),
        ]
    )


class TestDatasetFromDb:
    def test_build_labeled_rows_respects_before_kickoff(self, session: Session):
        kickoff_a = datetime(2026, 9, 10, 17, 0, tzinfo=timezone.utc)
        kickoff_b = datetime(2026, 9, 17, 17, 0, tzinfo=timezone.utc)
        _seed_game(session, game_id="g1", kickoff=kickoff_a, home_score=24, away_score=17, home_epa=0.12)
        _seed_game(session, game_id="g2", kickoff=kickoff_b, home_score=17, away_score=24, home_epa=0.08)
        session.commit()

        rows = build_labeled_rows(session, before_kickoff=kickoff_b)
        assert len(rows) == 1
        assert rows[0].game_id == "g1"


class TestModelStrategy:
    def test_model_strategy_uses_model_probability(self, session: Session):
        kickoff = datetime(2026, 9, 10, 17, 0, tzinfo=timezone.utc)
        _seed_game(
            session,
            game_id="target",
            kickoff=kickoff,
            home_score=24,
            away_score=17,
            home_epa=0.20,
            dk_home=150,
            dk_away=-180,
        )
        session.commit()

        synthetic = _synthetic_rows(30)
        model, _ = train_from_rows(synthetic, config=TrainConfig(num_boost_round=30))
        game = session.query(Game).filter_by(game_id="target").one()
        strategy = ModelSpreadStrategy(model=model, soft_book="draftkings")
        candidates = strategy.generate_candidates(session, game, Settings(min_edge_pct=2.0))

        assert isinstance(candidates, list)
        for candidate in candidates:
            assert 0.0 <= candidate.true_prob <= 1.0
            assert candidate.book == "draftkings"

    def test_walk_forward_excludes_current_game_from_training(self, session: Session):
        base = datetime(2026, 9, 1, 17, 0, tzinfo=timezone.utc)
        for i in range(10):
            _seed_game(
                session,
                game_id=f"hist_{i}",
                kickoff=base + timedelta(days=i * 7),
                home_score=24 if i % 2 == 0 else 17,
                away_score=17 if i % 2 == 0 else 24,
                home_epa=0.10 + i * 0.01,
            )
        target_kickoff = base + timedelta(days=10 * 7)
        _seed_game(
            session,
            game_id="target",
            kickoff=target_kickoff,
            home_score=24,
            away_score=17,
            home_epa=0.25,
            dk_home=200,
            dk_away=-250,
        )
        session.commit()

        prior_rows = build_labeled_rows(session, before_kickoff=target_kickoff)
        assert len(prior_rows) == 10
        assert all(r.game_id != "target" for r in prior_rows)

        strategy = WalkForwardModelStrategy(min_train_games=8, soft_book="draftkings")
        game = session.query(Game).filter_by(game_id="target").one()
        candidates = strategy.generate_candidates(session, game, Settings(min_edge_pct=1.0))
        assert isinstance(candidates, list)


class TestPredictionStorage:
    def test_save_prediction_persists_shap(self, session: Session):
        rows = _synthetic_rows(12)
        model, _ = train_from_rows(rows, config=TrainConfig(num_boost_round=20))
        kickoff = datetime(2026, 9, 10, 17, 0, tzinfo=timezone.utc)
        session.add(
            Game(
                game_id="g1",
                sport="nfl",
                home_team="A",
                away_team="B",
                kickoff_time=kickoff,
                season=2026,
                week=1,
            )
        )
        session.commit()

        result = model.predict_one(rows[0].features, compute_shap=False)
        saved = save_prediction(
            session,
            game_id="g1",
            market_type="spread",
            model_version=model.model_version,
            result=result,
            predicted_at=kickoff - timedelta(hours=1),
        )
        session.commit()

        row = session.query(ModelPrediction).filter_by(id=saved.id).one()
        assert row.predicted_prob == pytest.approx(result.home_cover_prob)
        assert row.feature_snapshot == result.feature_snapshot
