"""매물대 돌파 + 모멘텀 전략"""

from datetime import datetime

import logging

logger = logging.getLogger(__name__)

from .base_strategy import BaseStrategy


class BreakoutMomentumStrategy(BaseStrategy):
    """매물대 돌파 + 모멘텀 복합 전략"""

    def __init__(self, config: dict = None, survival_filter=None):
        """초기화

        Args:
            config: 전략 설정
            survival_filter: SurvivalFilter 인스턴스 (선택). 설정 시 BUY 신호에 3중 필터 적용.
        """
        super().__init__("BreakoutMomentum", config)
        self.volume_threshold = self.get_config_value("volume_threshold", 1.5)
        self.confidence_threshold = self.get_config_value("confidence_threshold", 70)
        self.survival_filter = survival_filter  # 3중 필터 (수급/섹터/재료)

    def generate_signal(self, data: dict) -> dict:
        """종합 매매 신호 생성 (멀티 타임프레임)

        Args:
            data: 분석 데이터
                - stock_code: 종목 코드
                - stock_name: 종목명
                - current_price: 현재가 정보
                - daily_technical: 일봉 기술적 분석
                - daily_volume_profile: 일봉 매물대
                - ohlcv: 분봉 OHLCV 데이터프레임
                - technical: 분봉 기술적 분석 결과
                - volume_profile: 분봉 매물대 분석 결과
                - news_sentiment: 뉴스 분석 결과

        Returns:
            신호 정보 딕셔너리
        """
        score = 0
        reasons = []

        try:
            if not self.validate_data(data):
                return self._create_signal(data, "HOLD", 0, ["데이터 불완전"])

            current_price = data["current_price"]
            ohlcv = data.get("ohlcv")
            daily_technical = data.get("daily_technical", {})
            minute_technical = data.get("technical", {})

            # 거래량 급증 비율 계산
            avg_volume = ohlcv["volume"].mean() if ohlcv is not None and not ohlcv.empty else 1
            volume_surge = current_price.get("volume", 0) / max(avg_volume, 1)

            # ========== 일봉 기준 평가 (50점) ==========

            # 1. Envelope 상단 돌파 - 추세 전환 (25점)
            envelope_score, envelope_reason = self._evaluate_envelope(daily_technical)
            if envelope_score > 0:
                score += envelope_score
                reasons.append(envelope_reason)

            # 2. 52주 신고가 (15점)
            high52_score, high52_reason = self._evaluate_52week_high(daily_technical)
            if high52_score > 0:
                score += high52_score
                reasons.append(high52_reason)

            # 3. 일봉 매물대 돌파 (10점)
            daily_breakout_score, daily_breakout_reason = self._evaluate_breakout(
                data.get("daily_volume_profile"), current_price, volume_surge, "일봉"
            )
            if daily_breakout_score > 0:
                score += min(10, daily_breakout_score // 3)  # 최대 10점
                reasons.append(daily_breakout_reason)

            # ========== 분봉 기준 평가 (30점) ==========

            # 4. 분봉 매물대 돌파 (15점)
            minute_breakout_score, minute_breakout_reason = self._evaluate_breakout(
                data.get("volume_profile"), current_price, volume_surge, "분봉"
            )
            if minute_breakout_score > 0:
                score += min(15, minute_breakout_score // 2)
                reasons.append(minute_breakout_reason)

            # 5. 거래량 평가 (15점)
            volume_score, volume_reason = self._evaluate_volume(volume_surge)
            if volume_score > 0:
                score += min(15, volume_score)
                reasons.append(volume_reason)

            # ========== 기술적 지표 (10점) ==========

            # 6. RSI만 평가 (10점)
            rsi_score, rsi_reason = self._evaluate_rsi(minute_technical)
            if rsi_score != 0:
                score += rsi_score
                if rsi_reason:
                    reasons.append(rsi_reason)

            # ========== 뉴스/공시 (10점) ==========

            # 7. AI 뉴스 분석 (10점)
            news_score, news_reason = self._evaluate_news(data.get("news_sentiment"))
            if news_score != 0:
                score += min(10, abs(news_score)) * (1 if news_score > 0 else -1)
                if news_reason:
                    reasons.append(news_reason)

            # 신호 결정
            action = self._determine_action(score)

            signal = self._create_signal(data, action, score, reasons)

            # 3중 필터 적용 (BUY/STRONG_BUY인 경우만)
            if self.survival_filter and action in ("BUY", "STRONG_BUY"):
                stock_code = data.get("stock_code", "")
                stock_name = data.get("stock_name", "")
                try:
                    filter_result = self.survival_filter.filter(stock_code, stock_name)
                    signal["filter_result"] = filter_result
                    signal["survival_passed"] = filter_result["통과"]
                    signal["survival_score"] = filter_result["점수"]

                    if not filter_result["통과"]:
                        # 필터 탈락 시 HOLD로 다운그레이드
                        logger.info(
                            f"[{stock_code}] {stock_name} 3중 필터 탈락 "
                            f"(점수 {filter_result['점수']}/3) → HOLD 전환"
                        )
                        signal["action"] = "HOLD"
                        signal["reasons"].append(
                            f"3중 필터 탈락: 수급={'✅' if filter_result['수급'] else '❌'} "
                            f"섹터={filter_result['섹터']} 재료={filter_result['재료']}"
                        )
                    else:
                        logger.info(
                            f"[{stock_code}] {stock_name} 3중 필터 통과 "
                            f"(점수 {filter_result['점수']}/3)"
                        )
                        signal["reasons"].append(
                            f"3중 필터 통과: 수급={'✅' if filter_result['수급'] else '❌'} "
                            f"섹터={filter_result['섹터']} 재료={filter_result['재료']}"
                        )
                except Exception as fe:
                    logger.warning(f"3중 필터 오류 ({stock_code}): {fe}")
                    signal["filter_result"] = None
                    signal["survival_passed"] = None

            return signal

        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return self._create_signal(data, "HOLD", 0, [f"오류: {str(e)}"])

    def _evaluate_envelope(self, daily_technical: dict) -> tuple[int, str]:
        """Envelope 상단 돌파 평가 (추세 전환)

        Returns:
            (점수, 이유)
        """
        if not daily_technical:
            return 0, ""

        envelope = daily_technical.get("envelope", {})

        # 상단 돌파 = 추세 전환
        if envelope.get("breakout"):
            return 25, "📈 Envelope 상단 돌파 (추세전환)"

        # 상단 위에 있음 = 상승 추세 유지
        if envelope.get("position") == "above_upper":
            return 10, "Envelope 상단 유지"

        return 0, ""

    def _evaluate_52week_high(self, daily_technical: dict) -> tuple[int, str]:
        """52주 신고가 평가

        Returns:
            (점수, 이유)
        """
        if not daily_technical:
            return 0, ""

        if daily_technical.get("is_52week_high"):
            return 15, "🚀 52주 신고가"

        # 신고가 근접 (5% 이내)
        distance = daily_technical.get("distance_from_high_pct")
        if distance is not None and distance >= -5:
            return 5, f"52주 신고가 근접 ({distance:.1f}%)"

        return 0, ""

    def _evaluate_breakout(
        self, volume_profile: dict, current_price: dict, volume_surge: float, timeframe: str = ""
    ) -> tuple[int, str]:
        """매물대 돌파 평가

        Returns:
            (점수, 이유)
        """
        if not volume_profile:
            return 0, ""

        resistance_zones = volume_profile.get("resistance_zones", [])
        price = current_price.get("price", 0)
        prefix = f"{timeframe} " if timeframe else ""

        for zone in resistance_zones:
            if price > zone.get("price_high", 0):
                # 돌파 확인, 거래량에 따른 점수
                if volume_surge > 3.0:
                    return 30, f"{prefix}매물대 강력 돌파 (거래량 {volume_surge:.1f}배)"
                elif volume_surge > 2.0:
                    return 25, f"{prefix}매물대 돌파 (거래량 {volume_surge:.1f}배)"
                elif volume_surge > self.volume_threshold:
                    return 20, f"{prefix}매물대 돌파 시도 (거래량 {volume_surge:.1f}배)"
                else:
                    return 10, f"{prefix}매물대 돌파 (거래량 부족)"

        return 0, ""

    def _evaluate_rsi(self, technical: dict) -> tuple[int, str]:
        """RSI 평가

        Returns:
            (점수, 이유)
        """
        if not technical:
            return 0, ""

        rsi = technical.get("rsi")
        if rsi is None:
            return 0, ""

        if 40 < rsi < 60:
            return 10, f"RSI {rsi:.0f} (적정)"
        elif 30 < rsi <= 40:
            return 5, f"RSI {rsi:.0f} (반등 기대)"
        elif rsi <= 30:
            return 3, f"RSI {rsi:.0f} (과매도)"
        elif rsi >= 70:
            return -5, f"RSI {rsi:.0f} (과매수 주의)"

        return 0, ""

    def _evaluate_volume(self, volume_surge: float) -> tuple[int, str]:
        """거래량 평가

        Returns:
            (점수, 이유)
        """
        if volume_surge > self.volume_threshold * 2:
            return 20, f"거래량 폭증 {volume_surge:.1f}배"
        elif volume_surge > self.volume_threshold * 1.5:
            return 15, f"거래량 급증 {volume_surge:.1f}배"
        elif volume_surge > self.volume_threshold:
            return 10, f"거래량 증가 {volume_surge:.1f}배"
        return 0, ""

    def _evaluate_technical(self, technical: dict) -> tuple[int, list]:
        """기술적 지표 평가 (레거시 - 분봉용)

        Returns:
            (점수, 이유 리스트)
        """
        score = 0
        reasons = []

        # 이동평균 골든크로스 (0-10점)
        if technical.get("golden_cross"):
            score += 10
            reasons.append("MA 골든크로스")

        return score, reasons

    def _evaluate_news(self, news_sentiment: dict) -> tuple[int, str]:
        """뉴스 감성 평가

        Returns:
            (점수, 이유)
        """
        if not news_sentiment:
            return 0, ""

        sentiment = news_sentiment.get("감성", "중립")
        impact = news_sentiment.get("impact_score", 5)

        if sentiment == "호재":
            score = min(20, impact * 2)
            return score, f"뉴스 호재 (+{score})"
        elif sentiment == "악재":
            score = min(20, impact * 2)
            return -score, f"뉴스 악재 (-{score})"

        return 0, ""

    def _evaluate_disclosure(self, disclosure: dict) -> tuple[int, str]:
        """공시 평가

        Returns:
            (점수, 이유)
        """
        if not disclosure:
            return 0, ""

        impact = disclosure.get("영향", "중립")
        importance = disclosure.get("중요도", "하")

        if impact == "긍정":
            if importance == "상":
                return 10, "중요 긍정 공시"
            elif importance == "중":
                return 7, "긍정 공시"
            return 5, "소폭 긍정 공시"
        elif impact == "부정":
            if importance == "상":
                return -10, "중요 부정 공시"
            elif importance == "중":
                return -7, "부정 공시"
            return -5, "소폭 부정 공시"

        return 0, ""

    def _evaluate_trend(self, technical: dict) -> tuple[int, str]:
        """추세 평가

        Returns:
            (점수, 이유)
        """
        trend_info = technical.get("trend", {})
        trend = trend_info.get("trend", "unknown")
        strength = trend_info.get("strength", 0)

        if trend == "uptrend":
            if strength > 10:
                return 10, f"강한 상승추세 ({strength}%)"
            elif strength > 5:
                return 5, f"상승추세 ({strength}%)"
        elif trend == "downtrend":
            if strength > 10:
                return -5, "강한 하락추세"

        return 0, ""

    def _determine_action(self, score: int) -> str:
        """점수 기반 행동 결정

        Args:
            score: 종합 점수

        Returns:
            행동 문자열
        """
        if score >= 80:
            return "STRONG_BUY"
        elif score >= self.confidence_threshold:
            return "BUY"
        elif score <= 20:
            return "SELL"
        else:
            return "HOLD"

    def _create_signal(
        self, data: dict, action: str, confidence: int, reasons: list
    ) -> dict:
        """신호 딕셔너리 생성

        Args:
            data: 원본 데이터
            action: 행동
            confidence: 신뢰도
            reasons: 이유 리스트

        Returns:
            신호 딕셔너리
        """
        current_price = data.get("current_price", {})

        return {
            "stock_code": data.get("stock_code", ""),
            "stock_name": data.get("stock_name", "Unknown"),
            "action": action,
            "confidence": max(0, min(100, confidence)),
            "reasons": [r for r in reasons if r],  # 빈 문자열 제거
            "price": current_price.get("price", 0),
            "timestamp": datetime.now(),
        }
