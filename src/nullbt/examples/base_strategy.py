"""전략 기본 클래스"""

from abc import ABC, abstractmethod
from typing import Any


class BaseStrategy(ABC):
    """매매 전략 기본 클래스"""

    def __init__(self, name: str, config: dict = None):
        """초기화

        Args:
            name: 전략 이름
            config: 전략 설정
        """
        self.name = name
        self.config = config or {}

    @abstractmethod
    def generate_signal(self, data: dict) -> dict:
        """매매 신호 생성

        Args:
            data: 분석 데이터

        Returns:
            신호 정보 딕셔너리
            {
                'action': str,      # BUY, SELL, HOLD, STRONG_BUY
                'confidence': int,  # 0-100
                'reasons': list,    # 신호 이유 리스트
                'price': float,     # 현재가
                'timestamp': datetime
            }
        """
        pass

    def validate_data(self, data: dict) -> bool:
        """데이터 유효성 검증

        Args:
            data: 검증할 데이터

        Returns:
            유효성 여부
        """
        required_fields = ["stock_code", "current_price", "ohlcv"]
        return all(field in data for field in required_fields)

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """설정 값 조회

        Args:
            key: 설정 키
            default: 기본값

        Returns:
            설정 값
        """
        return self.config.get(key, default)
