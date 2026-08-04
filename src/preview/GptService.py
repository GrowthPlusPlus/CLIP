import os

from openai import OpenAI

# OpenAI 클라이언트 생성
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

SYSTEM_PROMPT = """
너는 챌린지 제목을 분류하는 AI이다.

가능한 카테고리:
fitness: 헬스, 운동, 근력운동, 유산소 운동
study: 일반적인 공부, 시험 준비, 학습
coding: 프로그래밍, 개발, 알고리즘, 코딩
reading: 독서, 책 읽기
sports: 축구, 농구, 야구, 수영 등 특정 스포츠
food: 음식 먹기, 식사
cooking: 요리, 베이킹
cleaning: 청소, 정리정돈
outdoor: 캠핑, 등산, 산책 등 야외 활동
art: 그림, 미술, 공예
music: 악기 연주, 노래, 작곡
gardening: 식물 키우기, 원예

규칙
- 가장 적합한 카테고리 하나만 선택한다.
- 적절한 카테고리가 없으면 unsupported를 출력한다.
- 확신이 없으면 unsupported를 출력한다.
- 반드시 카테고리 이름 하나만 출력한다.
- 다른 설명은 절대 출력하지 않는다.
"""

def gpt_classify_title(title: str) -> str:
    response = client.chat.completions.create(  # openai api를 호출하는 부분이다.
        model="gpt-5-mini",
        temperature=1,  # 생성 다양성을 조절한다. (0 ~ 2에서 0으로 갈수록 일관성있게 답변한다)
        messages=[
            {
                "role": "system",  
                "content": SYSTEM_PROMPT  # GPT의 역할을 정의한다.
            },
            {
                "role": "user",
                "content": title  # GPT에게 할 질문이다.
            }
        ]
    )

    # GPT의 답변을 반환한다.
    return response.choices[0].message.content.strip().lower()