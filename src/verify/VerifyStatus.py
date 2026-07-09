# 정성우

from enum import Enum

# 사진 검증 결과를 가진 enum 클래스이다.
class VerifyStatus(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"  # 사진이 애매함
    FAIL = "FAIL"
    UNSUPPORTED = "UNSUPPORTED"  # 검증 불가능한 챌린지 제목 또는 카테고리