"""
Fraud Transaction Detection Agent
----------------------------------
Core detection engine using rule-based heuristics + statistical anomaly detection.
No external ML libraries required — works with standard Python + pandas + numpy.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
import re


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class FraudResult:
    transaction_id: str
    is_fraud: bool
    risk_score: float          # 0.0 (safe) → 1.0 (certain fraud)
    risk_level: str            # LOW / MEDIUM / HIGH / CRITICAL
    triggered_rules: List[str]
    explanation: str


@dataclass
class AgentReport:
    total_transactions: int
    flagged_count: int
    fraud_rate: float
    results: List[FraudResult]
    summary: str
    stats: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# Detection Rules
# ─────────────────────────────────────────────

class FraudDetectionAgent:
    """
    Multi-layer fraud detection using:
    1. Rule-based checks (thresholds, velocity, patterns)
    2. Statistical anomaly detection (Z-score, IQR)
    3. Behavioral pattern analysis
    """

    def __init__(self, config: dict = None): # type: ignore
        self.config = config or {}
        self.high_amount_threshold = self.config.get("high_amount_threshold", 5000)
        self.z_score_threshold = self.config.get("z_score_threshold", 3.0)
        self.velocity_window = self.config.get("velocity_window", 5)   # txns per hour
        self.round_amount_threshold = self.config.get("round_amount_threshold", 1000)

    # ── Column normalizer ─────────────────────

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map common column name variants to standard names."""
        col_map = {}
        lower_cols = {c.lower().strip(): c for c in df.columns}

        aliases = {
            "transaction_id": ["transaction_id", "txn_id", "id", "trans_id", "transactionid"],
            "amount":         ["amount", "amt", "transaction_amount", "value", "sum"],
            "timestamp":      ["timestamp", "time", "date", "datetime", "transaction_date", "trans_date"],
            "merchant":       ["merchant", "merchant_name", "vendor", "payee", "store"],
            "location":       ["location", "city", "country", "region", "place"],
            "card_number":    ["card_number", "card", "card_no", "cardnumber", "pan"],
            "category":       ["category", "type", "txn_type", "transaction_type"],
        }

        for standard, options in aliases.items():
            for opt in options:
                if opt in lower_cols:
                    col_map[lower_cols[opt]] = standard
                    break

        df = df.rename(columns=col_map)

        # Ensure transaction_id exists
        if "transaction_id" not in df.columns:
            df["transaction_id"] = [f"TXN-{i+1:04d}" for i in range(len(df))]

        return df

    # ── Rule 1: High Amount ───────────────────

    def _rule_high_amount(self, row, mean_amt, std_amt) -> Tuple[bool, str, float]:
        amt = row.get("amount", 0)
        if pd.isna(amt):
            return False, "", 0.0
        amt = float(amt)
        score = 0.0
        if amt > self.high_amount_threshold:
            score = min(0.4, (amt - self.high_amount_threshold) / self.high_amount_threshold * 0.4)
            return True, f"Amount ${amt:,.2f} exceeds threshold ${self.high_amount_threshold:,}", score
        return False, "", 0.0

    # ── Rule 2: Statistical Anomaly (Z-score) ─

    def _rule_zscore(self, row, mean_amt, std_amt) -> Tuple[bool, str, float]:
        amt = row.get("amount", 0)
        if pd.isna(amt) or std_amt == 0:
            return False, "", 0.0
        amt = float(amt)
        z = abs((amt - mean_amt) / std_amt)
        if z > self.z_score_threshold:
            score = min(0.35, (z - self.z_score_threshold) / self.z_score_threshold * 0.35)
            return True, f"Statistical anomaly: Z-score {z:.2f} (threshold {self.z_score_threshold})", score
        return False, "", 0.0

    # ── Rule 3: Round-Number Pattern ──────────

    def _rule_round_amount(self, row) -> Tuple[bool, str, float]:
        amt = row.get("amount", 0)
        if pd.isna(amt):
            return False, "", 0.0
        amt = float(amt)
        if amt >= self.round_amount_threshold and amt % 100 == 0:
            score = 0.15
            return True, f"Suspiciously round amount: ${amt:,.0f}", score
        return False, "", 0.0

    # ── Rule 4: Rapid Velocity (same card) ────

    def _rule_velocity(self, row, df) -> Tuple[bool, str, float]:
        if "card_number" not in df.columns or "timestamp" not in df.columns:
            return False, "", 0.0
        card = row.get("card_number")
        ts = row.get("timestamp")
        if pd.isna(card) or pd.isna(ts):
            return False, "", 0.0
        try:
            ts = pd.to_datetime(ts)
            window_start = ts - pd.Timedelta(hours=1)
            same_card = df[df["card_number"] == card]
            same_card_ts = pd.to_datetime(same_card["timestamp"], errors="coerce")
            count = ((same_card_ts >= window_start) & (same_card_ts <= ts)).sum()
            if count > self.velocity_window:
                score = min(0.4, (count - self.velocity_window) / self.velocity_window * 0.3)
                return True, f"High velocity: {count} transactions in 1 hour for same card", score
        except Exception:
            pass
        return False, "", 0.0

    # ── Rule 5: Odd Hours ─────────────────────

    def _rule_odd_hours(self, row) -> Tuple[bool, str, float]:
        ts = row.get("timestamp")
        if pd.isna(ts):
            return False, "", 0.0
        try:
            hour = pd.to_datetime(ts).hour
            if 1 <= hour <= 4:
                return True, f"Transaction at unusual hour: {hour:02d}:00", 0.1
        except Exception:
            pass
        return False, "", 0.0

    # ── Rule 6: Duplicate Detection ───────────

    def _rule_duplicate(self, row, df) -> Tuple[bool, str, float]:
        amt = row.get("amount")
        card = row.get("card_number")
        ts = row.get("timestamp")
        if pd.isna(amt) or pd.isna(card):
            return False, "", 0.0
        try:
            ts = pd.to_datetime(ts)
            window = ts - pd.Timedelta(minutes=10)
            same = df[
                (df["card_number"] == card) &
                (df["amount"] == amt)
            ]
            same_ts = pd.to_datetime(same["timestamp"], errors="coerce")
            dupes = ((same_ts >= window) & (same_ts < ts)).sum()
            if dupes > 0:
                return True, f"Possible duplicate: same card & amount within 10 minutes", 0.45
        except Exception:
            pass
        return False, "", 0.0

    # ── Rule 7: IQR Outlier ───────────────────

    def _rule_iqr_outlier(self, row, q1, q3) -> Tuple[bool, str, float]:
        amt = row.get("amount", 0)
        if pd.isna(amt):
            return False, "", 0.0
        amt = float(amt)
        iqr = q3 - q1
        upper_fence = q3 + 3.0 * iqr
        if amt > upper_fence:
            score = min(0.25, (amt - upper_fence) / upper_fence * 0.25)
            return True, f"IQR outlier: ${amt:,.2f} exceeds upper fence ${upper_fence:,.2f}", score
        return False, "", 0.0

    # ─────────────────────────────────────────────
    # Main Analysis
    # ─────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> AgentReport:
        df = self._normalize_columns(df.copy())

        # Pre-compute stats
        amounts = pd.to_numeric(df.get("amount", pd.Series(dtype=float)), errors="coerce").dropna()
        mean_amt = float(amounts.mean()) if len(amounts) > 0 else 0
        std_amt  = float(amounts.std())  if len(amounts) > 1 else 1
        q1       = float(amounts.quantile(0.25)) if len(amounts) > 0 else 0
        q3       = float(amounts.quantile(0.75)) if len(amounts) > 0 else 0

        results: List[FraudResult] = []

        for _, row in df.iterrows():
            txn_id = str(row.get("transaction_id", f"TXN-{_}"))
            triggered = []
            total_score = 0.0

            rules = [
                self._rule_high_amount(row, mean_amt, std_amt),
                self._rule_zscore(row, mean_amt, std_amt),
                self._rule_round_amount(row),
                self._rule_velocity(row, df),
                self._rule_odd_hours(row),
                self._rule_duplicate(row, df),
                self._rule_iqr_outlier(row, q1, q3),
            ]

            for fired, msg, score in rules:
                if fired:
                    triggered.append(msg)
                    total_score += score

            total_score = min(1.0, total_score)

            # Risk level
            if total_score >= 0.75:
                risk_level = "CRITICAL"
            elif total_score >= 0.50:
                risk_level = "HIGH"
            elif total_score >= 0.25:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            is_fraud = total_score >= 0.40

            # Explanation
            if triggered:
                explanation = f"Flagged by {len(triggered)} rule(s): " + "; ".join(triggered[:2])
                if len(triggered) > 2:
                    explanation += f" (+{len(triggered)-2} more)"
            else:
                explanation = "No suspicious patterns detected."

            results.append(FraudResult(
                transaction_id=txn_id,
                is_fraud=is_fraud,
                risk_score=round(total_score, 4),
                risk_level=risk_level,
                triggered_rules=triggered,
                explanation=explanation,
            ))

        flagged = [r for r in results if r.is_fraud]
        fraud_rate = len(flagged) / len(results) if results else 0

        # Build summary
        summary = (
            f"Analyzed {len(results)} transactions. "
            f"Flagged {len(flagged)} as potentially fraudulent ({fraud_rate*100:.1f}%). "
            f"Mean transaction amount: ${mean_amt:,.2f}. "
            f"Std deviation: ${std_amt:,.2f}."
        )

        stats = {
            "mean_amount": round(mean_amt, 2),
            "std_amount":  round(std_amt, 2),
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "high_amount_threshold": self.high_amount_threshold,
            "z_score_threshold": self.z_score_threshold,
        }

        return AgentReport(
            total_transactions=len(results),
            flagged_count=len(flagged),
            fraud_rate=round(fraud_rate, 4),
            results=results,
            summary=summary,
            stats=stats,
        )

    def results_to_dataframe(self, report: AgentReport) -> pd.DataFrame:
        rows = []
        for r in report.results:
            rows.append({
                "Transaction ID": r.transaction_id,
                "Fraud?": "🚨 YES" if r.is_fraud else "✅ NO",
                "Risk Score": r.risk_score,
                "Risk Level": r.risk_level,
                "Rules Triggered": len(r.triggered_rules),
                "Explanation": r.explanation,
            })
        return pd.DataFrame(rows)
