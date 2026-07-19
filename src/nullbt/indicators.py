"""기술적 분석 모듈"""

from typing import Optional

import pandas as pd
import logging

logger = logging.getLogger(__name__)

try:
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
    from ta.volatility import BollingerBands
except ImportError:
    logger.warning("ta library not available, using fallback implementations")
    RSIIndicator = None
    MACD = None
    BollingerBands = None


class TechnicalAnalyzer:
    """기술적 지표 분석기"""

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """RSI 계산

        Args:
            df: OHLCV 데이터프레임
            period: RSI 기간

        Returns:
            RSI 값 (0-100) 또는 None
        """
        if len(df) < period:
            return None

        try:
            if RSIIndicator:
                rsi = RSIIndicator(close=df["close"], window=period)
                return round(rsi.rsi().iloc[-1], 2)
            else:
                # Fallback implementation
                delta = df["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return round(rsi.iloc[-1], 2)
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            return None

    @staticmethod
    def calculate_macd(df: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
        """MACD 계산

        Args:
            df: OHLCV 데이터프레임

        Returns:
            (MACD 값, Signal 값) 또는 (None, None)
        """
        if len(df) < 26:
            return None, None

        try:
            if MACD:
                macd = MACD(close=df["close"])
                return (
                    round(macd.macd().iloc[-1], 2),
                    round(macd.macd_signal().iloc[-1], 2),
                )
            else:
                # Fallback implementation
                exp1 = df["close"].ewm(span=12, adjust=False).mean()
                exp2 = df["close"].ewm(span=26, adjust=False).mean()
                macd_line = exp1 - exp2
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                return round(macd_line.iloc[-1], 2), round(signal_line.iloc[-1], 2)
        except Exception as e:
            logger.error(f"MACD calculation error: {e}")
            return None, None

    @staticmethod
    def calculate_moving_averages(
        df: pd.DataFrame, periods: list[int] = None
    ) -> dict[str, Optional[float]]:
        """이동평균선 계산

        Args:
            df: OHLCV 데이터프레임
            periods: 이동평균 기간 리스트

        Returns:
            이동평균값 딕셔너리
        """
        if periods is None:
            periods = [5, 20, 60, 120]

        mas = {}
        for period in periods:
            if len(df) >= period:
                ma = df["close"].rolling(period).mean().iloc[-1]
                mas[f"ma{period}"] = round(ma, 2)
            else:
                mas[f"ma{period}"] = None

        return mas

    @staticmethod
    def is_golden_cross(df: pd.DataFrame) -> bool:
        """골든크로스 확인 (단기 MA가 장기 MA 상향 돌파)

        Args:
            df: OHLCV 데이터프레임

        Returns:
            골든크로스 여부
        """
        if len(df) < 60:
            return False

        try:
            ma20 = df["close"].rolling(20).mean()
            ma60 = df["close"].rolling(60).mean()

            # 최근에 교차했는지 (오늘 위, 어제 아래 또는 같음)
            return bool(ma20.iloc[-1] > ma60.iloc[-1] and ma20.iloc[-2] <= ma60.iloc[-2])
        except Exception as e:
            logger.error(f"Golden cross check error: {e}")
            return False

    @staticmethod
    def is_dead_cross(df: pd.DataFrame) -> bool:
        """데드크로스 확인 (단기 MA가 장기 MA 하향 돌파)

        Args:
            df: OHLCV 데이터프레임

        Returns:
            데드크로스 여부
        """
        if len(df) < 60:
            return False

        try:
            ma20 = df["close"].rolling(20).mean()
            ma60 = df["close"].rolling(60).mean()

            return bool(ma20.iloc[-1] < ma60.iloc[-1] and ma20.iloc[-2] >= ma60.iloc[-2])
        except Exception as e:
            logger.error(f"Dead cross check error: {e}")
            return False

    @staticmethod
    def calculate_momentum(df: pd.DataFrame, period: int = 20) -> float:
        """모멘텀 계산 (n일 수익률)

        Args:
            df: OHLCV 데이터프레임
            period: 기간

        Returns:
            모멘텀 (%)
        """
        if len(df) < period:
            return 0.0

        try:
            return round(df["close"].pct_change(period).iloc[-1] * 100, 2)
        except Exception as e:
            logger.error(f"Momentum calculation error: {e}")
            return 0.0

    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
    ) -> dict[str, Optional[float]]:
        """볼린저 밴드 계산

        Args:
            df: OHLCV 데이터프레임
            period: 기간
            std_dev: 표준편차 배수

        Returns:
            볼린저 밴드 값 딕셔너리
        """
        if len(df) < period:
            return {"upper": None, "middle": None, "lower": None, "percent_b": None}

        try:
            if BollingerBands:
                bb = BollingerBands(close=df["close"], window=period, window_dev=std_dev)
                return {
                    "upper": round(bb.bollinger_hband().iloc[-1], 2),
                    "middle": round(bb.bollinger_mavg().iloc[-1], 2),
                    "lower": round(bb.bollinger_lband().iloc[-1], 2),
                    "percent_b": round(bb.bollinger_pband().iloc[-1], 4),
                }
            else:
                # Fallback
                middle = df["close"].rolling(period).mean()
                std = df["close"].rolling(period).std()
                upper = middle + (std * std_dev)
                lower = middle - (std * std_dev)
                current_price = df["close"].iloc[-1]
                percent_b = (current_price - lower.iloc[-1]) / (
                    upper.iloc[-1] - lower.iloc[-1]
                )
                return {
                    "upper": round(upper.iloc[-1], 2),
                    "middle": round(middle.iloc[-1], 2),
                    "lower": round(lower.iloc[-1], 2),
                    "percent_b": round(percent_b, 4),
                }
        except Exception as e:
            logger.error(f"Bollinger Bands calculation error: {e}")
            return {"upper": None, "middle": None, "lower": None, "percent_b": None}

    @staticmethod
    def get_trend_strength(df: pd.DataFrame) -> dict[str, any]:
        """추세 강도 분석

        Args:
            df: OHLCV 데이터프레임

        Returns:
            추세 정보 딕셔너리
        """
        if len(df) < 60:
            return {"trend": "unknown", "strength": 0}

        try:
            ma20 = df["close"].rolling(20).mean().iloc[-1]
            ma60 = df["close"].rolling(60).mean().iloc[-1]
            current_price = df["close"].iloc[-1]

            # 추세 판단
            if current_price > ma20 > ma60:
                trend = "uptrend"
                strength = min(100, int((current_price / ma60 - 1) * 100))
            elif current_price < ma20 < ma60:
                trend = "downtrend"
                strength = min(100, int((1 - current_price / ma60) * 100))
            else:
                trend = "sideways"
                strength = 0

            return {"trend": trend, "strength": strength}
        except Exception as e:
            logger.error(f"Trend analysis error: {e}")
            return {"trend": "unknown", "strength": 0}

    @staticmethod
    def calculate_envelope(
        df: pd.DataFrame, period: int = 20, percent: float = 10.0
    ) -> dict[str, any]:
        """Envelope 계산 (이동평균 ± N%)

        Args:
            df: OHLCV 데이터프레임
            period: 이동평균 기간 (기본 20)
            percent: 상하 폭 % (기본 10)

        Returns:
            Envelope 정보 딕셔너리
        """
        if len(df) < period:
            return {
                "upper": None,
                "middle": None,
                "lower": None,
                "breakout": False,
                "position": "unknown",
            }

        try:
            middle = df["close"].rolling(period).mean()
            upper = middle * (1 + percent / 100)
            lower = middle * (1 - percent / 100)

            current_price = df["close"].iloc[-1]
            prev_price = df["close"].iloc[-2] if len(df) > 1 else current_price

            upper_val = upper.iloc[-1]
            prev_upper = upper.iloc[-2] if len(df) > 1 else upper_val

            # 상단 돌파 체크 (오늘 위, 어제 아래)
            breakout = bool(current_price > upper_val and prev_price <= prev_upper)

            # 현재 위치
            if current_price > upper_val:
                position = "above_upper"
            elif current_price < lower.iloc[-1]:
                position = "below_lower"
            else:
                position = "inside"

            return {
                "upper": round(upper_val, 0),
                "middle": round(middle.iloc[-1], 0),
                "lower": round(lower.iloc[-1], 0),
                "breakout": breakout,
                "position": position,
            }
        except Exception as e:
            logger.error(f"Envelope calculation error: {e}")
            return {
                "upper": None,
                "middle": None,
                "lower": None,
                "breakout": False,
                "position": "unknown",
            }

    @staticmethod
    def check_52week_high(df: pd.DataFrame, current_price: float) -> dict[str, any]:
        """52주 신고가 체크

        Args:
            df: 일봉 OHLCV 데이터프레임 (최소 250일)
            current_price: 현재가

        Returns:
            52주 신고가 정보
        """
        try:
            # 52주 = 약 250 거래일
            lookback = min(250, len(df))
            if lookback < 20:
                return {
                    "is_52week_high": False,
                    "high_52week": None,
                    "distance_pct": None,
                }

            high_52week = df["high"].tail(lookback).max()
            distance_pct = round((current_price / high_52week - 1) * 100, 2)

            # 신고가 = 현재가가 52주 최고가의 99% 이상
            is_high = current_price >= high_52week * 0.99

            return {
                "is_52week_high": is_high,
                "high_52week": round(high_52week, 0),
                "distance_pct": distance_pct,
            }
        except Exception as e:
            logger.error(f"52week high check error: {e}")
            return {
                "is_52week_high": False,
                "high_52week": None,
                "distance_pct": None,
            }

    def analyze(self, df: pd.DataFrame) -> dict:
        """종합 기술적 분석 (분봉용)

        Args:
            df: OHLCV 데이터프레임

        Returns:
            분석 결과 딕셔너리
        """
        return {
            "rsi": self.calculate_rsi(df),
            "mas": self.calculate_moving_averages(df),
            "golden_cross": self.is_golden_cross(df),
            "dead_cross": self.is_dead_cross(df),
            "momentum": self.calculate_momentum(df),
            "bollinger": self.calculate_bollinger_bands(df),
            "trend": self.get_trend_strength(df),
        }

    def analyze_daily(self, df: pd.DataFrame, current_price: float = None) -> dict:
        """일봉 기술적 분석 (추세 판단용)

        Args:
            df: 일봉 OHLCV 데이터프레임
            current_price: 현재가 (52주 신고가 체크용)

        Returns:
            일봉 분석 결과 딕셔너리
        """
        if current_price is None:
            current_price = df["close"].iloc[-1] if not df.empty else 0

        envelope = self.calculate_envelope(df, period=20, percent=10)
        week52 = self.check_52week_high(df, current_price)

        return {
            "rsi": self.calculate_rsi(df),
            "mas": self.calculate_moving_averages(df),
            "envelope": envelope,
            "envelope_breakout": envelope.get("breakout", False),
            "is_52week_high": week52.get("is_52week_high", False),
            "high_52week": week52.get("high_52week"),
            "distance_from_high_pct": week52.get("distance_pct"),
            "trend": self.get_trend_strength(df),
            "golden_cross": self.is_golden_cross(df),
        }
