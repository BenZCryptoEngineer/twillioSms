"""
Post-load data quality validation.

Runs integrity checks on loaded Match data before it enters
the feature engineering pipeline.
"""

import logging
from collections import Counter

from tennis_miner.core.schema import Match, ShotOutcome

log = logging.getLogger(__name__)


class DataValidator:
    """Validates a collection of Match objects for quality issues."""

    def validate(self, matches: list[Match]) -> dict:
        """Run all checks. Returns summary dict + logs warnings."""
        issues = []
        stats = Counter()

        for m in matches:
            stats["matches"] += 1
            stats["points"] += m.n_points
            stats["points_with_shots"] += m.n_points_with_shots

            if m.n_points == 0:
                issues.append(("error", f"{m.match_id}: zero points"))
                continue

            for p in m.points:
                if p.has_shots:
                    stats["total_shots"] += len(p.shots)

                    # Check rally consistency
                    if p.rally_length != len(p.shots):
                        issues.append((
                            "warn",
                            f"{p.point_id}: rally_length={p.rally_length} "
                            f"but {len(p.shots)} shots"
                        ))

                    # Check last shot has point-ending outcome
                    last = p.shots[-1]
                    if not last.is_point_ending:
                        issues.append((
                            "warn",
                            f"{p.point_id}: last shot outcome is "
                            f"{last.outcome.value}, expected point-ending"
                        ))

                    # Check alternating server/returner
                    for i, shot in enumerate(p.shots):
                        expected = "server" if i % 2 == 0 else "returner"
                        if shot.player != expected:
                            issues.append((
                                "warn",
                                f"{p.point_id}: shot {i+1} player is "
                                f"'{shot.player}', expected '{expected}'"
                            ))
                            break

                # Score sanity
                valid_scores = {"0", "15", "30", "40", "AD", "0.0", "15.0", "30.0", "40.0"}
                if p.server_score not in valid_scores and not p.is_tiebreak:
                    stats["unusual_scores"] += 1

        # Label balance
        all_labels = [p.server_won for m in matches for p in m.points]
        if all_labels:
            win_rate = sum(all_labels) / len(all_labels)
            if win_rate < 0.3 or win_rate > 0.8:
                issues.append((
                    "warn",
                    f"Server win rate is {win_rate:.2%} — "
                    f"possible label encoding issue"
                ))
            stats["server_win_rate"] = round(win_rate, 4)

        # Log issues
        errors = [msg for level, msg in issues if level == "error"]
        warns = [msg for level, msg in issues if level == "warn"]

        if errors:
            log.error(f"Data validation: {len(errors)} errors")
            for e in errors[:10]:
                log.error(f"  {e}")
        if warns:
            log.warning(f"Data validation: {len(warns)} warnings")
            for w in warns[:10]:
                log.warning(f"  {w}")

        return {
            "stats": dict(stats),
            "errors": errors,
            "warnings": warns,
            "is_clean": len(errors) == 0,
        }
