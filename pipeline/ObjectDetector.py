# Copyright (c) 2025 FRC 6328
# http://github.com/Mechanical-Advantage
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import cv2
import numpy as np
from config.config import ConfigStore
from PIL import Image
from vision_types import ObjDetectObservation
from ai_edge_litert.interpreter import Interpreter, load_delegate
from typing import List, Union


class ObjectDetector:
    def __init__(self) -> None:
        raise NotImplementedError

    def detect(self, image: cv2.Mat, config: ConfigStore) -> List[ObjDetectObservation]:
        raise NotImplementedError


# class CoreMLObjectDetector(ObjectDetector):
#     _model: Union[coremltools.models.MLModel, None] = None

#     def __init__(self) -> None:
#         pass

#     def detect(self, image: cv2.Mat, config: ConfigStore) -> List[ObjDetectObservation]:
#         # Load CoreML model
#         if self._model == None:
#             print("Loading object detection model")
#             self._model = coremltools.models.MLModel(config.local_config.obj_detect_model)
#             print("Finished loading object detection model")

#         # Create scaled frame for model
#         if len(image.shape) == 2:
#             image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#         image_scaled = np.zeros((640, 640, 3), dtype=np.uint8)
#         scaled_height = int(640 / (image.shape[1] / image.shape[0]))
#         bar_height = int((640 - scaled_height) / 2)
#         image_scaled[bar_height : bar_height + scaled_height, 0:640] = cv2.resize(image, (640, scaled_height))

#         # Run CoreML model
#         image_coreml = Image.fromarray(image_scaled)
#         prediction = self._model.predict({"image": image_coreml})

#         observations: List[ObjDetectObservation] = []
#         for coordinates, confidence in zip(prediction["coordinates"], prediction["confidence"]):
#             obj_class = max(range(len(confidence)), key=confidence.__getitem__)
#             confidence = float(confidence[obj_class])
#             x = coordinates[0] * image.shape[1]
#             y = ((coordinates[1] * 640 - bar_height) / scaled_height) * image.shape[0]
#             width = coordinates[2] * image.shape[1]
#             height = coordinates[3] / (scaled_height / 640) * image.shape[0]

#             corners = np.array(
#                 [
#                     [x - width / 2, y - height / 2],
#                     [x + width / 2, y - height / 2],
#                     [x - width / 2, y + height / 2],
#                     [x + width / 2, y + height / 2],
#                 ]
#             )
#             corners_undistorted = cv2.undistortPoints(
#                 corners,
#                 config.local_config.camera_matrix,
#                 config.local_config.distortion_coefficients,
#                 None,
#                 config.local_config.camera_matrix,
#             )

#             corner_angles = np.zeros((4, 2))
#             for index, corner in enumerate(corners_undistorted):
#                 vec = np.linalg.inv(config.local_config.camera_matrix).dot(np.array([corner[0][0], corner[0][1], 1]).T)
#                 corner_angles[index][0] = math.atan(vec[0])
#                 corner_angles[index][1] = math.atan(vec[1])

#             observations.append(ObjDetectObservation(obj_class, confidence, corner_angles, corners))

#         return observations

class LiteRtObjectDetector(ObjectDetector):
    _interpreter: Union[Interpreter, None] = None
    _labels: Union[List[str], None] = None    
    _in_det = None
    _out_det: Union[List[dict], None] = None
    _input_tensor = None
    _in_h = None
    _in_w = None

    def __init__(self) -> None:
        pass

    def _get_quant_params(self, det: dict):
        """Return (scale, zero_point) or (None, None) if not quantized."""
        qp = det.get('quantization_parameters')
        if qp:
            scales = qp.get('scales') or []
            zero_points = qp.get('zero_points') or []
            if len(scales) > 0:
                scale = float(scales[0])
                zp = int(zero_points[0]) if len(zero_points) > 0 else 0
                return scale, zp

        q = det.get('quantization')
        if q and isinstance(q, (list, tuple)) and len(q) >= 2:
            try:
                return float(q[0]), int(q[1])
            except Exception:
                pass

        return None, None

    def _dequantize_tensor(self, det: dict, tensor: np.ndarray) -> np.ndarray:
        """Dequantize a tensor using its tensor detail dict. Returns float32 array."""
        scale, zp = self._get_quant_params(det)
        if scale is None or scale == 0.0:
            return tensor.astype(np.float32)
        return scale * (tensor.astype(np.float32) - (zp or 0))

    def detect(self, image: cv2.Mat, config: ConfigStore) -> List[ObjDetectObservation]:
        if (len(image.shape) == 2) and config.local_config.obj_detect_greyscale:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        _, height, width, channels = image.shape

        if self._interpreter is None:
            delegate_options = { 'backend_type': 'htp' }
            qnn_delegate = load_delegate("libQnnTFLiteDelegate.so", options=delegate_options)
            interpreter = Interpreter(
                model_path=config.local_config.obj_detect_model,
                experimental_delegates=[qnn_delegate]
            )
            self._interpreter = interpreter
            self._interpreter.allocate_tensors()
            self._labels = [l.strip() for l in open(config.local_config.obj_detect_labels).readlines()]
            self._in_det = self._interpreter.get_input_details()
            self._out_det = self._interpreter.get_output_details()
            self._in_h, self._in_w = self._in_det[0]['shape'][1:3] 
            self._input_tensor = np.zeros((1, self._in_h, self._in_w, channels), dtype=np.uint8)
            
        cv2.resize(image, (self._in_w, self._in_h), dst=self._input_tensor[0])
        self._interpreter.set_tensor(self._in_det[0]['index'], self._input_tensor)
        self._interpreter.invoke()

        boxes_q = self._interpreter.get_tensor(self._out_det[0]['index'])[0]
        scores_q = self._interpreter.get_tensor(self._out_det[1]['index'])[0]
        classes_q = self._interpreter.get_tensor(self._out_det[2]['index'])[0]

        # Dequantize boxes and scores using class helpers (works for quantized or float tensors)
        boxes = self._dequantize_tensor(self._out_det[0], boxes_q)
        scores = self._dequantize_tensor(self._out_det[1], scores_q)
        classes = classes_q.astype(np.int32)[0]

        observations: List[ObjDetectObservation] = []

        mask = scores >= config.local_config.obj_detect_confidence_threshold 
        if np.any(mask):
            boxes = boxes[mask]
            scores = scores[mask]
            classes = classes[mask]

            x1, y1, x2, y2 = boxes.T
            boxes_cv2 = np.column_stack((x1, y1, x2 - x1, y2 - y1))

            idx_cv2 = cv2.dnn.NMSBoxes(
                bboxes=boxes_cv2.tolist(),
                scores=scores.tolist(),
                score_threshold=config.local_config.obj_detect_confidence_threshold,
                nms_threshold=0.5
            )

            if len(idx_cv2) > 0:
                for i in idx_cv2.flatten():
                    obj_class = classes[i]
                    confidence = float(scores[i])
                    x = boxes[i][0] * width
                    y = boxes[i][1] * height
                    box_width = (boxes[i][2] - boxes[i][0]) * width
                    box_height = (boxes[i][3] - boxes[i][1]) * height

                    corners = np.array(
                        [
                            [x, y],
                            [x + box_width, y],
                            [x, y + box_height],
                            [x + box_width, y + box_height],
                        ]
                    )
                    corners_undistorted = cv2.undistortPoints(
                        corners,
                        config.local_config.camera_matrix,
                        config.local_config.distortion_coefficients,
                        None,
                        config.local_config.camera_matrix,
                    )

                    corner_angles = np.zeros((4, 2))
                    for index, corner in enumerate(corners_undistorted):
                        vec = np.linalg.inv(config.local_config.camera_matrix).dot(np.array([corner[0][0], corner[0][1], 1]).T)
                        corner_angles[index][0] = np.arctan(vec[0])
                        corner_angles[index][1] = np.arctan(vec[1])

                    observations.append(ObjDetectObservation(obj_class, confidence, corner_angles, corners))


        return observations