import math
from typing import Tuple, List, Generator

import cv2
from ultralytics import YOLO

import aflutils.video_utils as VU
from aflutils.logger import get_logger

logger = get_logger(__name__)


class YOLODetector:
    """YOLOv8 用于单人检测（支持 GPU 加速 + 流式推理）"""

    def __init__(self, model_path: str = 'yolov8n.pt', device: str = 'cpu'):
        self.device = device
        self.model = YOLO(model_path)
        if device == 'cuda':
            self.model.to('cuda')
            logger.info("YOLO model loaded on GPU")
        else:
            logger.info("YOLO model loaded on CPU")

    def detect_single_person_segments(self, video_path: str, stream=True) -> Tuple[List[Tuple[int, int]], float]:
        """
        检测视频中仅出现单人的时间段。
        使用流式 YOLO 推理，逐秒分析人数。
        :return: (segments列表，每个元素为(start_sec, end_sec), 总单人时长(秒))
        """
        if stream:
            segments = []
            start_sec = None
            sec_idx = 0

            # 逐帧获取人数
            for num_person in self._detect_video_stream_per_sec(video_path):
                if num_person == 1 and start_sec is None:
                    # 进入单人区间
                    start_sec = sec_idx
                elif num_person != 1 and start_sec is not None:
                    # 退出单人区间
                    end_sec = sec_idx
                    if end_sec > start_sec:
                        segments.append((start_sec, end_sec))
                    start_sec = None
                sec_idx += 1

            # 处理视频结尾仍在单人区间的情况
            if start_sec is not None and sec_idx is not None:
                segments.append((start_sec, sec_idx))

            total_sec_counts = sum(end - start for start, end in segments)
            return segments, total_sec_counts
        else:
            return self._detect_single_person_segments(video_path)

    def _detect_video_stream_per_sec(self, video_path: str, conf_threshold: float = 0.5) -> Generator[
        int, None, None]:
        """
        流式处理视频，逐帧生成 (timestamp_sec, num_person)。
        :param video_path: 视频文件路径
        :param conf_threshold: YOLO 置信度阈值
        :yield: 该帧中的人数
        """

        sampled_path = VU.sample_video_per_sec(video_path)
        try:
            # 使用 stream=True 进行流式推理
            results = self.model(
                source=sampled_path,
                stream=True,
                conf=conf_threshold,
                verbose=True,
                device=self.device
            )

            for result in results:
                num_person = 0
                if result.boxes is not None:
                    cls_ids = result.boxes.cls.int().tolist()
                    num_person = cls_ids.count(0)  # COCO class 0 = person
                yield num_person
        finally:
            VU.remove_videos([sampled_path])

    def _detect_single_person_segments(self, video_path: str, conf_threshold: float = 0.5) -> Tuple[
        List[Tuple[int, int]], float]:
        """
        检测视频中仅出现单人的时间段（逐秒检测，每秒取首帧）。

        :param video_path: 视频文件路径
        :param conf_threshold: YOLO 置信度阈值
        :return: (segments列表，每个元素为(start_sec, end_sec), 总单人时长(秒))
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or total_frames <= 0:
                raise ValueError(f"无效的帧率({fps})或总帧数({total_frames})")

            # 总秒数（向上取整，确保包含最后一秒不完整部分）
            total_seconds = math.ceil(total_frames / fps)

            segments = []
            start_sec = None

            for s in range(total_seconds):
                # 跳转到第 s 秒的第一帧
                frame_index = int(s * fps)  # 第 s 秒起始帧索引
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    # 若跳转后读取失败，可能视频结束，跳出循环
                    break

                # 推理
                # logger.info(f"对于{video_path}，yolo正在推理第{s}/{total_seconds}秒的帧")
                results = self.model(frame, conf=conf_threshold, verbose=True)
                num_person = 0
                if results[0].boxes is not None:
                    cls_ids = results[0].boxes.cls.int().tolist()
                    num_person = cls_ids.count(0)  # COCO class 0 = person

                # 单人区间逻辑
                if num_person == 1 and start_sec is None:
                    start_sec = s
                elif num_person != 1 and start_sec is not None:
                    end_sec = s
                    if end_sec > start_sec:
                        segments.append((start_sec, end_sec))
                    start_sec = None

            # 处理视频结尾仍在单人区间的情况
            if start_sec is not None:
                end_sec = total_seconds  # 区间右开，包含最后一整秒
                if end_sec > start_sec:
                    segments.append((start_sec, end_sec))

            total_sec_counts = sum(end - start for start, end in segments)
            return segments, total_sec_counts

        finally:
            cap.release()
