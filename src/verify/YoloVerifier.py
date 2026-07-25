# 정성우

from typing import List, Optional

import torch
from PIL import Image
from ultralytics import YOLO

"""
YOLO를 이용하여 이미지 속 객체를 검출하는 클래스이다.
"""
class YoloVerifier:
    """
    서버 시작 시 YOLO 모델을 한 번만 로드한다.
    """
    def __init__(self,
                 model_name: str = "yolov8s.pt",  # YOLO 모델이다.
                 device: Optional[str] = None):

        self.device = device or (  # GPU가 있으면 cude, 없으면 cpu를 사용한다.
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = YOLO(model_name)  # 위에서 로드한 YOLO 모델을 메모리에 올린다.

    """
    이미지에서 검출된 객체 목록을 반환한다.
    """
    @torch.no_grad()  # 학습이 아닌 추론만 하므로 그래디언트 계산을 하지 않는다.
    def detect_objects(self, image: Image.Image) -> List[str]:

        # YOLO에서 객체를 찾는다.
        results = self.model.predict(
            source=image,  # 검사할 이미지
            device=self.device,  # GPU or CPU 사용
            verbose=False  # YOLO가 로그를 출력하지 않는다.
        )

        detected_objects = set()  # 검출된 객체를 set에 저장한다.

        # YOLO가 처리한 모든 사진이 results에 포함되어있다. 여기선 사진 한 장만 검사하므로 results 내부엔 객체 하나만 들어있다.
        for result in results:
            # 각 사진에서 모든 객체를 하나씩 꺼낸다. ex) 사람, 책, 노트북이 검출 됐다면 해당 값을을 꺼낸다.
            for box in result.boxes:
                # YOLO가 검출한 객체의 번호이다.
                class_id = int(box.cls.item())
                # 객체의 번호를 실제 객체 이름으로 변환한다.
                class_name = self.model.names[class_id]
                # set에 객체 이름을 저장한다.
                detected_objects.add(class_name)

        results = self.model.predict(
            source=image,
            device=self.device,
            verbose=False
        )

        # YOLO가 찾은 객체를 사각형과 확률로 사진에 그려주는 코드로 현재는 사용하지 않음
        # for i, result in enumerate(results):
        #     plotted = result.plot()  # 박스가 그려진 이미지 (numpy 배열)

        #     from PIL import Image
        #     Image.fromarray(plotted).save(f"yolo_result_{i}.jpg")

        return sorted(detected_objects)  # 검출된 객체의 순서를 일정하게 한다.