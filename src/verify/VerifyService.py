# 정성우

import re
from typing import Dict, List, Tuple, Any

import torch
from src.verify.clip import clip
from PIL import Image

from typing import Optional
from src.verify.prompts import CATEGORY_PROMPTS
from src.verify.prompts import (map_title_to_categories, get_negative_prompts, is_yolo_supported, get_expected_objects)
from src.verify.YoloVerifier import YoloVerifier  # YOLO 검사 py파일

from src.verify.VerifyStatus import VerifyStatus

# 유사도 통과 정책 값이다.
PASS_THRESHOLD = 0.20  # 카테고리 점수가 이 값 이상이어야 통과한다.
REVIEW_THRESHOLD = 0.18  # 카테고리 점수가 이 값 이하면 실패한다.
MARGIN = 0.04  # 카테고리 점수가 generic 점수보다 얼마나 더 커야 하는지의 차이이다.
READING_MARGIN = 0.02  # 독서는 다른 카테고리에 비해 책 사진만 올렸을 때 실패하는 경우가 많으므로 margin값을 낮춘다.

"""
텍스트와 이미지를 받아서 유사도를 검사해주는 클래스이다.
"""
class ClipVerifier:
    # ClipVerifier를 초기화 한다. 서버 시작 시 한 번만 로딩된다.
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        # YOLO 객체 검출기이다.
        self.yolo_verifier = YoloVerifier()

    """
    이미지와 프롬프트의 점수를 계산하는 함수이다.
    소프트맥스 없이 정규화된 임베딩의 내적(= 코사인 유사도) 점수를 반환한다.
    """
    @torch.no_grad()  # 현재는 추론만 하고 학습하지 않으므로 그래디언트 계산을 끈다.
    def score_image_against_prompts(self, image: Image.Image, prompts: List[str]) -> Dict[str, float]:  # 이미지와 앞서 선별한 프롬프트를 매개변수로 받는다.
        image_in = self.preprocess(image).unsqueeze(0).to(self.device)  # 이미지를 CLIP 입력 규격에 맞게 변환한다.

        text_tokens = clip.tokenize(prompts).to(self.device)  # 프롬프트 리스트를 CLIP가 받을 수 있게 변환한다.

        # 이미지와 프롬프트의 임베딩(특징 벡터)을 추출한다.
        image_features = self.model.encode_image(image_in)
        text_features = self.model.encode_text(text_tokens)

        # 이미지와 프롬프트에 L2 정규화를 수행하여 코사인 유사도 계산 준비를 한다.
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity scores: shape (1, N) -> 유사도 계산
        sims = (image_features @ text_features.T).squeeze(0)  # (N,)

        # 점수를 float로 변환하고, 최종적으로 (프롬프트, 점수) 형태로 파이썬 dict으로 변환한다.
        out: Dict[str, float] = {}
        for p, v in zip(prompts, sims.tolist()):
            out[p] = float(v)
        return out

    """
    이미지와 챌린지 이름을 받아서 둘 사이의 유사도를 판정하는 함수이다.
    단순히 한 카테고리 프롬프트의 유사도가 높다고 통과하는 것이 아니라, Generic Prompt 및 Negative Prompt와 비교하여 최종 판정한다.
    """
    def verify(self, title: str, image: Image.Image) -> Dict[str, Any]:
        categories = map_title_to_categories(title)  # 챌린지 이름에 맞는 카테고리들을 매칭한다.

        # 매칭된 카테고리가 없는 경우, CLIP 결과를 unsupported로 반환한다.
        if not categories:
            return {
                    "title": title,
                    "categories": [],
                    "bestPrompt": "",
                    "score": 0.0,
                    "bestGenericPrompt": "",
                    "genericScore": 0.0,
                    "passed": VerifyStatus.UNSUPPORTED.name,
                    "threshold": PASS_THRESHOLD,
                    "margin": MARGIN,
                    "promptScores": {},
                    "genericPromptScores": {},
            }

        # 최종 비교대상이 될 카테고리 프롬프트를 구성한다. ex) 카테고리에 fitness가 있다면 "a person exercising"등의 문장을 category_prompts 리스트에 넣는다.
        category_prompts: List[str] = []
        for c in categories:
            # extend로 리스트에 여러 문장을 한 번에 추가한다. 여러 카테고리가 선택된 경우 모든 Prompt를 합쳐 비교한다.
            category_prompts.extend(CATEGORY_PROMPTS[c])
        category_prompts = list(dict.fromkeys(category_prompts))  # 중복을 제거하고 등장 순서를 유지한다.

        # margin 계산을 위한 generic 프롬프트를 가져온다.
        generic_prompts = CATEGORY_PROMPTS["generic"]

        # 분류한 카테고리에 맞는 네거티브 프롬프트를 가져온다.
        negative_prompts = get_negative_prompts(categories)

        # 유사도 점수를 계산한다.
        category_scores = self.score_image_against_prompts(image, category_prompts)  # 카테고리 관련 프롬프트에 관한 점수를 계산한다.
        generic_scores = self.score_image_against_prompts(image, generic_prompts)  # generic 프롬프트에 대한 점수를 계산한다.
        negative_scores = self.score_image_against_prompts(image,negative_prompts)  # 네거티브 프롬프트에 대한 점수를 계산한다.

        # 각 그룹에서 제일 높은 유사도 점수를 가진 프롬프트를 뽑는다. 즉 여러 프롬프트 중 하나라도 강하게 맞으면 그 카테고리로 본다는 의미이다.
        best_prompt, best_score = max(category_scores.items(), key=lambda kv: kv[1]) 
        best_generic_prompt, best_generic_score = max(generic_scores.items(), key=lambda kv: kv[1]) 
        best_negative_prompt, best_negative_score = max(negative_scores.items(),key=lambda kv: kv[1])

        # 기본적으로 YOLO 검사는 수행하지 않는다.
        yolo_checked = False
        yolo_pass = True
        detected_objects = []
        expected_objects = set()

        # 현재 카테고리가 YOLO를 지원하는 경우에만 수행한다.
        if is_yolo_supported(categories):
            yolo_checked = True

            # YOLO 검사를 수행하고 검출된 객체들을 저장한다.
            detected_objects = self.yolo_verifier.detect_objects(image)
            # 해당 카테고리에 기대되는 객체 목록을 가져온다.
            expected_objects = get_expected_objects(categories)

            # 기대 객체가 하나라도 검출되면 통과한다. 없다면 FALSE
            yolo_pass = any(
                obj in expected_objects
                for obj in detected_objects
            )

        # 카테고리가 독서라면 margin값을 변경한다.
        margin = READING_MARGIN if "reading" in categories else MARGIN

        # 그냥 사람/실내/야외 사진같은 generic + margin보다 카테고리와의 유사도가 낮은 경우 실패한다.
        if best_score < best_generic_score + margin:
            passed = VerifyStatus.FAIL

        # 네거티브 프롬프트가 더 높은 경우에도 실패한다.
        elif best_score <= best_negative_score:
            passed = VerifyStatus.FAIL

        # 이미지와의 유사도가 높은 경우이다.
        elif best_score >= PASS_THRESHOLD:  
                if yolo_checked and not yolo_pass:  # 이미지와 유사도는 높지만 YOLO 검사는 통과하지 못한 경우이다.
                    passed = VerifyStatus.REVIEW
                else:  # 두 검사 모두 통과한 경우이다.
                    passed = VerifyStatus.PASS

        # 이미지의 유사도가 애매하여 리뷰가 필요한 경우이다.
        elif best_score >= REVIEW_THRESHOLD:
            passed = VerifyStatus.REVIEW

        # 이미지의 유사도가 너무 낮아 실패한 경우이다.
        else:  
            passed = VerifyStatus.FAIL

        return {
            "title": title,
            "categories": categories,
            "bestPrompt": best_prompt,
            "score": best_score,
            "bestGenericPrompt": best_generic_prompt,
            "genericScore": best_generic_score,
            "passed": passed.name,
            "threshold": PASS_THRESHOLD,
            "margin": margin,
            "promptScores": category_scores,  # 카테고리 프롬프트 별 점수 전체이다.
            "genericPromptScores": generic_scores,  # generic 프롬프트 별 점수 전체이다.
        }