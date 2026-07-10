# 정성우

import re
from typing import List
from typing import Set

"""
텍스트를 정규화 하기 위한 함수이다.
대소문자, 공백 차이를 제거하여 키워드 매칭을 안정적으로 하기 위한 전처리이다.
"""
def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

"""
규칙 기반으로 카테고리를 분류하는 함수이다. 
매개변수로 챌린지 제목을 받아서 챌린지 제목에 특정 키워드가 포함되면 카테고리를 부여한다. 여러 카테고리가 매칭되면 전부 반환하며, 상위 1~2개로 제한하는 것도 가능하다.
"""
def map_title_to_categories(title: str) -> List[str]:
    t = _normalize_text(title)  # 챌린지 제목을 정규화 한다.

    # 매칭된 카테고리를 담는 리스트이다.
    matched = [] 
    # 각 카테고리의 키워드를 순회하며 하나라도 포함되면 카테고리를 매칭한다. 다중 카테고리를 허용한다.
    for cat, kws in CATEGORY_KEYWORDS.items():  # cat: 키값 (fitness), kws: 키워드 값 (운동, 헬스 ...)
        for kw in kws:  # 카테고리의 키워드 리스트를 하나씩 꺼낸다. (운동, 헬스, 다이어트 등)
            if _normalize_text(kw) in t:  # 챌린지 제목에 키워드 값이 포함되는지 검사한다.
                matched.append(cat)
                break

    if not matched:  # 매칭되는 카테고리가 하나도 없다면 빈 칸으로 리턴한다.
        return []

    # 너무 많은 카테고리가 걸리면 우선순위에 따라 개수를 제한한다. 카테고리 순서를 다르게 하므로써 우선순위를 바꿀 수 있다.
    return matched[:2]  # 2개로 제한한다.

"""
위 함수와 다르게 카테고리 목록을 받아서 해당 카테고리 별 negative 프롬프트를 얻어서 반환하는 함수이다.
"""
def get_negative_prompts(categories: List[str]) -> List[str]:
    prompts = []  # 네거티브 프롬프트를 저장할 빈 리스트이다.

    for c in categories:  # 카테고리를 하나씩 꺼낸다.
        prompts.extend(NEGATIVE_PROMPTS[c])  # 카테고리 내부에 있는 프롬프트를 하나씩 저장한다.

    return list(dict.fromkeys(prompts))  # 중복을 제거하여 반환한다.

"""
YOLO를 사용할 카테고리인지 확인한다. 카테고리에 해당한다면 YOLO 검사를 수행한다.
"""
def is_yolo_supported(categories: List[str]) -> bool:
    return any(category in YOLO_SUPPORTED for category in categories)

"""
카테고리별 기대 객체를 반환한다. 
ex) 카테고리가 study, coding이라면 해당 카테고리에 존재하는 모든 객체 이름을 반환한다.
"""
def get_expected_objects(categories: List[str]) -> Set[str]:
    expected = set()

    for category in categories:
        if category in EXPECTED_OBJECTS:
            expected.update(EXPECTED_OBJECTS[category])

    return expected

"""
프롬프트 파일의 목적은 한국어로 들어오는 챌린지 제목을, CLIP이 잘 비교할 수 있는 ‘시각적 영어 문장 세트’로 변환하는 역할이다.

카테고리 프롬프트는 CLIP용 텍스트 후보 집합이다.
CLIP는 이미지 임베딩과 텍스트 임베딩을 통해 유사도를 비교하는데, 추상 키워드보다 시각적으로 관찰 가능한 문장을 더 안정적으로 생성한다.
따라서 추상적인 키워드가 들어오면 해당 키워드를 가장 관찰 가능한 문장으로 바꾸는 역할이다.
"""
CATEGORY_PROMPTS = {
    "fitness": [
        "a person exercising",
        "a person lifting weights",
        "a person working out in a gym",
        "a person running outdoors",
        "a person jogging in a park",
        "a person doing push-ups",
        "a person doing squats",
        "a person doing planks",
        "a person stretching",
        "a person doing yoga",
        "a person doing pilates",
        "a person using a treadmill",
        "a person riding a stationary bike",
        "a gym with exercise equipment",
        "dumbbells on a gym floor",
        "a barbell on a weight rack",
        "a fitness center",
        "sportswear in a gym",
        "a person exercising at home",
        "fitness training scene"
    ],

    "study": [
        "a person studying at a desk",
        "a student writing notes",
        "a student reading a textbook",
        "an open textbook on a desk",
        "a notebook with handwritten notes",
        "study materials on a desk",
        "books and notebooks on a study desk",
        "a desk with stationery",
        "a student solving math problems",
        "a person reviewing study notes",
        "a person preparing for an exam",
        "lecture notes on a desk",
        "a study environment indoors",
        "a quiet study room",
        "a student studying in a library",
        "a library study desk",
        "a student taking notes during class",
        "a laptop used for studying",
        "a computer displaying lecture slides",
        "a classroom study environment"
    ],

    "coding": [
        "a computer displaying programming code",
        "a laptop showing source code",
        "a software developer working",
        "a programmer typing on a keyboard",
        "a programming IDE on a monitor",
        "a coding workspace",
        "multiple monitors displaying code",
        "a laptop with code editor open",
        "a software development environment",
        "a computer screen with terminal commands",
        "a person debugging code",
        "a person writing software",
        "a keyboard and monitor on a desk",
        "a programming desk setup",
        "an integrated development environment",
        "a developer using a computer",
        "a workstation for programming",
        "a monitor displaying software development",
        "a coding session at a desk",
        "a computer workstation indoors"
    ],

    "reading": [
        "a person reading a book",
        "a person reading a novel",
        "a person reading at a desk",
        "a person reading in a library",
        "an open book on a desk",
        "a bookshelf filled with books",
        "a stack of books on a table",
        "a library study table",
        "a person holding a book",
        "a book with bookmarks",
        "a person reading indoors",
        "a person reading outdoors",
        "an ebook displayed on a tablet",
        "a person reading an ebook",
        "books neatly arranged on a shelf",
        "a reading corner with books",
        "a person focused on reading",
        "an open novel on a wooden desk",
        "a person turning the pages of a book",
        "a quiet reading environment"
    ],

    "sports": [
        "people playing soccer",
        "people playing basketball",
        "people playing baseball",
        "people playing volleyball",
        "people playing badminton",
        "people playing tennis",
        "people playing table tennis",
        "a person swimming in a pool",
        "a person climbing an indoor climbing wall",
        "a person skiing on snow",
        "a person snowboarding",
        "a person boxing",
        "a martial arts training session",
        "a golf course with a golfer",
        "a bowling alley",
        "sports equipment on a field",
        "an indoor sports court",
        "an outdoor sports field",
        "athletes playing a team sport",
        "a sports training session"
    ],

    "food": [
        "a healthy meal on a table",
        "a balanced meal",
        "a home cooked meal",
        "a healthy breakfast",
        "a healthy lunch",
        "a healthy dinner",
        "a salad in a bowl",
        "fresh vegetables on a plate",
        "fresh fruits on a table",
        "grilled chicken with vegetables",
        "a nutritious meal",
        "a lunch box with healthy food",
        "a meal prepared at home",
        "food served on a dining table",
        "a plate of healthy food",
        "a bowl of vegetables",
        "a high protein meal",
        "a healthy diet meal",
        "a colorful healthy dish",
        "a well prepared meal"
    ],

    "cooking": [
        "a person cooking in a kitchen",
        "a person preparing ingredients",
        "a person chopping vegetables",
        "a person slicing vegetables",
        "a person using a cutting board",
        "a person cooking with a frying pan",
        "a pot cooking on a stove",
        "ingredients on a kitchen counter",
        "a person stirring food in a pan",
        "a person seasoning food",
        "a person preparing a home cooked meal",
        "a person baking in an oven",
        "a tray of freshly baked cookies",
        "a homemade cake on a table",
        "a kitchen with cooking utensils",
        "a person making a sandwich",
        "a person preparing a salad",
        "a clean kitchen while cooking",
        "a person following a recipe",
        "a home cooking scene"
    ],

    "cleaning": [
        "a person cleaning a room",
        "a person using a vacuum cleaner",
        "a person mopping the floor",
        "a person sweeping the floor",
        "a person wiping a desk",
        "a person cleaning a kitchen",
        "a person washing dishes",
        "a sink full of clean dishes",
        "a person organizing a bookshelf",
        "a person organizing a closet",
        "a tidy desk with organized items",
        "a clean and organized room",
        "a clean bedroom",
        "a clean living room",
        "cleaning supplies on the floor",
        "a person taking out the trash",
        "a person cleaning a bathroom",
        "a person dusting furniture",
        "a person folding laundry",
        "a neat and tidy home"
    ],

    "outdoor": [
        "a person walking outdoors",
        "a person hiking on a mountain trail",
        "a person walking in a park",
        "people walking outdoors",
        "a person walking along a riverside",
        "a person walking on a forest trail",
        "a person camping outdoors",
        "a camping tent in nature",
        "a picnic in a park",
        "a person enjoying nature",
        "a mountain hiking trail",
        "a scenic park with walking paths",
        "a person walking near the beach",
        "a person taking a walk at sunset",
        "a person exploring a forest",
        "a nature trail surrounded by trees",
        "an outdoor recreation area",
        "a peaceful outdoor landscape",
        "a person spending time outdoors",
        "an outdoor walking scene"
    ],

    "art": [
        "a person drawing",
        "a person sketching in a sketchbook",
        "a person painting on a canvas",
        "a person coloring a drawing",
        "a sketchbook on a desk",
        "colored pencils on a table",
        "paint brushes and watercolor paints",
        "an artist working on a painting",
        "a canvas with colorful artwork",
        "a drawing tablet on a desk",
        "an illustration in progress",
        "an art studio with painting supplies",
        "markers and drawing tools",
        "a person creating digital art",
        "a finished painting on an easel",
        "a person making a watercolor painting",
        "a person drawing with a pencil",
        "an open sketchbook with drawings",
        "art supplies on a desk",
        "a creative art workspace"
    ],

    "music": [
        "a person playing the piano",
        "a person playing the guitar",
        "a person playing the drums",
        "a person playing the violin",
        "a person singing into a microphone",
        "a musician practicing an instrument",
        "sheet music on a music stand",
        "a piano in a practice room",
        "a guitar on a stand",
        "a drum set in a studio",
        "a violin with a music stand",
        "a keyboard instrument on a desk",
        "a person practicing music indoors",
        "musical instruments in a practice room",
        "a person rehearsing music",
        "a recording studio with instruments",
        "a person reading sheet music",
        "a music classroom",
        "a music practice session",
        "a musician performing"
    ],

    "gardening": [
        "a person watering plants",
        "a person taking care of plants",
        "a flower pot with green plants",
        "a person planting flowers",
        "gardening tools on the ground",
        "a home garden with flowers",
        "a person planting seeds",
        "a person repotting a plant",
        "a person trimming plants",
        "a person working in a garden",
        "green plants in flower pots",
        "a vegetable garden",
        "flowers blooming in a garden",
        "a person growing herbs",
        "a person caring for indoor plants",
        "a watering can next to plants",
        "a balcony garden",
        "a garden with healthy plants",
        "a person maintaining a garden",
        "a gardening scene outdoors"
    ],

    # 매핑 실패 시 fallback으로 키워드 매칭이 실패했을 때, 서버가 죽지 않도록 해준다.
    "generic": [
        "a photo of a person",  # 그냥 사람이 찍힌 사진
        "a photo taken indoors",  # 실내 사진
        "a photo taken outdoors",  # 야외 사진
        "a photo of an object",  # 사람없이 사물만 있는 사진
        "a close-up photo",
        "a picture of everyday life",
        "a casual snapshot",
        "a normal daily scene",
        "an indoor environment",
        "an outdoor environment",
        "a room inside a house",
        "a public place",
        "a desk with everyday objects",
        "a random object",
        "a household item",
        "a natural outdoor scene",
        "a city street",
        "a simple photograph",
        "a real world scene",
        "an ordinary photograph"
    ],
}
"""
한국어 기준 규칙 기반 키워드 매핑이다. 서비스 특성에 맞게 계속 보강하는 데이터(룰)이다.
해당 딕셔너리는 텍스트 분류기 역할을 수행한다.
사용자 챌린지 제목이 한국어로 들어오면, 출력으로 위 코드인 카테고리 레벨의 fitness, study 등을 내보낸다.
즉 자유 텍스트인 챌린지 제목을 소수의 고정된 카테고리로 매핑하기 위해 사용한다.
"""
CATEGORY_KEYWORDS = {
    "fitness": [
        "운동", "헬스", "웨이트",
        "웨이트트레이닝", "러닝", "달리기",
        "조깅", "마라톤", "걷기",
        "홈트", "홈트레이닝", "다이어트",
        "체중감량", "몸무게", "근력운동",
        "유산소", "스쿼트", "푸쉬업",
        "팔굽혀펴기", "플랭크", "버피",
        "필라테스", "요가", "스트레칭",
        "pt", "퍼스널트레이닝", "헬스장",
        "운동장", "크로스핏", "줄넘기"
    ],

    "study": [
        "공부", "학습", "스터디",
        "시험", "과제", "복습",
        "예습", "인강", "강의",
        "수업", "문제집", "교재",
        "필기", "노트", "암기",
        "오답노트", "수능", "모의고사",
        "편입", "편준", "공무원",
        "임용", "자격증", "정처기",
        "정보처리기사", "토익", "토플",
        "오픽", "jlpt", "hsk"
    ],

    "coding": [
        "코딩", "프로그래밍", "개발",
        "알고리즘", "백준", "리트코드",
        "코테", "코딩테스트", "깃허브",
        "git", "github", "spring",
        "springboot", "flutter", "react",
        "javascript", "typescript", "python",
        "java", "c++", "c언어",
        "kotlin", "sql", "mysql",
        "postgresql", "api", "백엔드",
        "프론트엔드", "앱개발", "웹개발"
    ],

    "reading": [
        "독서", "책", "도서",
        "소설", "에세이", "문학",
        "전자책", "ebook", "리딩",
        "도서관", "독서실", "베스트셀러",
        "자기계발서", "만화책", "책읽기"
    ],

    "sports": [
        "축구", "풋살", "농구",
        "야구", "배구", "탁구",
        "배드민턴", "테니스", "골프",
        "볼링", "당구", "수영",
        "클라이밍", "스키", "스노보드",
        "복싱", "주짓수", "태권도",
        "유도", "검도", "양궁"
    ],

    "food": [
        "식단", "건강식", "식사",
        "아침", "점심", "저녁",
        "샐러드", "닭가슴살", "단백질",
        "과일", "채소", "도시락",
        "현미", "고구마", "계란",
        "음식", "간식", "칼로리",
        "식단관리", "영양", "건강식단"
    ],

    "cooking": [
        "요리", "쿠킹", "베이킹",
        "제빵", "디저트", "반찬",
        "집밥", "홈쿡", "레시피",
        "조리", "요리연습", "빵만들기",
        "케이크", "쿠키", "김밥"
    ],

    "cleaning": [
        "청소", "방청소", "화장실청소",
        "설거지", "빨래", "정리",
        "정돈", "정리정돈", "분리수거",
        "청소기", "걸레질", "물걸레",
        "쓰레기", "청결", "대청소"
    ],

    "outdoor": [
        "산책", "걷기", "등산",
        "하이킹", "캠핑", "여행",
        "외출", "공원", "피크닉",
        "트레킹", "자연", "숲",
        "바다", "강", "호수",
        "나들이", "드라이브", "야외활동"
    ],

    "art": [
        "그림", "드로잉", "스케치",
        "채색", "색칠", "수채화",
        "유화", "캘리그라피", "일러스트",
        "디자인", "미술", "크로키",
        "손그림", "캔버스", "색연필"
    ],

    "music": [
        "피아노", "기타", "드럼",
        "바이올린", "첼로", "플루트",
        "색소폰", "악기", "연주",
        "노래", "보컬", "합창",
        "작곡", "편곡", "음악",
        "우쿨렐레", "베이스", "연습"
    ],

    "gardening": [
        "식물", "화분", "원예",
        "꽃", "나무", "정원",
        "물주기", "식집사", "다육이",
        "허브", "씨앗", "모종",
        "분갈이", "텃밭", "재배"
    ]
}

"""
네거티브 카테고리 프롬프트를 12개마다 저장한 목록이다.
해당 카테고리에 어울리지 않는 다른 카테고리의 대표 장면들이 들어간다.
"""
NEGATIVE_PROMPTS = {

    "fitness": [
        "a person studying at a desk",
        "a computer displaying programming code",
        "a person reading a book",
        "a person cooking in a kitchen",
        "a person cleaning a room",
        "a person drawing",
        "a person playing the piano",
        "a person watering plants"
    ],

    "study": [
        "a person lifting weights",
        "a person exercising in a gym",
        "people playing soccer",
        "a person cooking in a kitchen",
        "a person cleaning a room",
        "a person drawing",
        "a person playing the guitar",
        "a person watering plants"
    ],

    "coding": [
        "a person lifting weights",
        "people playing basketball",
        "a person reading a novel",
        "a person cooking in a kitchen",
        "a person cleaning a room",
        "a person drawing",
        "a musician performing",
        "a flower garden"
    ],

    "reading": [
        "a person lifting weights",
        "a computer displaying programming code",
        "people playing tennis",
        "a person cooking in a kitchen",
        "a person cleaning a room",
        "a person drawing",
        "a person playing the piano",
        "a person watering plants"
    ],

    "sports": [
        "a student studying at a desk",
        "a computer displaying programming code",
        "a person reading a book",
        "a person cooking in a kitchen",
        "a person cleaning a room",
        "a person drawing",
        "a person playing the violin",
        "a person watering plants"
    ],

    "food": [
        "a person lifting weights",
        "a student studying at a desk",
        "a computer displaying programming code",
        "people playing soccer",
        "a person cleaning a room",
        "a person drawing",
        "a musician practicing an instrument",
        "a flower garden"
    ],

    "cooking": [
        "a person lifting weights",
        "a student studying at a desk",
        "a computer displaying programming code",
        "people playing basketball",
        "a person cleaning a room",
        "a person drawing",
        "a person playing the piano",
        "a person watering plants"
    ],

    "cleaning": [
        "a person lifting weights",
        "a student studying at a desk",
        "a computer displaying programming code",
        "people playing soccer",
        "a healthy meal on a table",
        "a person drawing",
        "a person playing the guitar",
        "a flower garden"
    ],

    "outdoor": [
        "a person studying at a desk",
        "a computer displaying programming code",
        "a person reading a book",
        "a person lifting weights",
        "a person cooking in a kitchen",
        "a person cleaning a room",
        "a person playing the piano",
        "a person watering indoor plants"
    ],

    "art": [
        "a person lifting weights",
        "a student studying at a desk",
        "a computer displaying programming code",
        "people playing soccer",
        "a healthy meal on a table",
        "a person cleaning a room",
        "a musician performing",
        "a person watering plants"
    ],

    "music": [
        "a person lifting weights",
        "a student studying at a desk",
        "a computer displaying programming code",
        "people playing basketball",
        "a person cooking in a kitchen",
        "a person cleaning a room",
        "a person drawing",
        "a flower garden"
    ],

    "gardening": [
        "a person lifting weights",
        "a student studying at a desk",
        "a computer displaying programming code",
        "people playing soccer",
        "a healthy meal on a table",
        "a person cleaning a room",
        "a person drawing",
        "a person playing the piano"
    ]
}


"""
YOLO가 보조 검증을 수행할 카테고리이다.
해당 카테고리에 포함되는 챌린지 제목의 경우, YOLO 검사에서 최소 하나의 객체가 검출 되어야만 통과할 수 있다.
"""
YOLO_SUPPORTED = {
    "study",
    "coding",
    "reading",
    "food",
    "cooking",
}

"""
카테고리별 YOLO가 검출해야 하는 객체이다. COCO 데이터셋에서 지원하는 객체만 사용한다. (총 80개)
객체가 하나 이상 검출되면 해당 카테고리와 관련 있는 사진으로 판단한다.
"""
EXPECTED_OBJECTS = {
    "study": [
        "book", "laptop", "keyboard",
        "mouse", "cell phone", "backpack",
        "chair", "dining table",
    ],

    "coding": [
        "laptop", "keyboard", "mouse",
        "cell phone", "chair", "tv",
        "dining table",
    ],

    "reading": [
        "book", "chair", "couch",
        "bed",
    ],

    "food": [
        "banana", "apple", "orange",
        "broccoli", "carrot", "pizza",
        "cake", "sandwich", "bowl",
        "cup", "bottle", "dining table",
    ],

    "cooking": [
        "oven", "microwave", "sink",
        "refrigerator", "knife", "fork",
        "spoon", "bowl", "cup",
        "dining table",
    ],
}