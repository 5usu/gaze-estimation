import cv2
import numpy as np
from models import (
    resnet18,
    resnet34,
    resnet50,
    mobilenet_v2,
    mobileone_s0,
    mobileone_s1,
    mobileone_s2,
    mobileone_s3,
    mobileone_s4
)


def get_model(arch, bins, pretrained=True, inference_mode=False):
    """Return the model based on the specified architecture."""
    if arch == 'resnet18':
        model = resnet18(pretrained=pretrained, num_classes=bins)
    elif arch == 'resnet34':
        model = resnet34(pretrained=pretrained, num_classes=bins)
    elif arch == 'resnet50':
        model = resnet50(pretrained=pretrained, num_classes=bins)
    elif arch == "mobilenetv2":
        model = mobilenet_v2(pretrained=pretrained, num_classes=bins)
    elif arch == "mobileone_s0":
        model = mobileone_s0(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    elif arch == "mobileone_s1":
        model = mobileone_s1(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    elif arch == "mobileone_s2":
        model = mobileone_s2(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    elif arch == "mobileone_s3":
        model = mobileone_s3(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    elif arch == "mobileone_s4":
        model = mobileone_s4(pretrained=pretrained, num_classes=bins, inference_mode=inference_mode)
    else:
        raise ValueError(f"Please choose available model architecture, currently chosen: {arch}")
    return model


def draw_gaze(frame, bbox, pitch, yaw, thickness=2, color=(0, 0, 255)):
    """Draws gaze direction on a frame given bounding box and gaze angles."""
    x_min, y_min, x_max, y_max = map(int, bbox[:4])
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2

    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    length = x_max - x_min
    dx = int(-length * np.sin(pitch) * np.cos(yaw))
    dy = int(-length * np.sin(yaw))

    arrow_length = np.sqrt(dx**2 + dy**2)
    max_possible_length = length
    confidence_score = max(0, min(100, (1 - arrow_length/max_possible_length) * 100))

    point1 = (x_center, y_center)
    point2 = (x_center + dx, y_center + dy)
    cv2.circle(frame, (x_center, y_center), radius=4, color=color, thickness=-1)
    cv2.arrowedLine(
        frame,
        point1,
        point2,
        color=color,
        thickness=thickness,
        line_type=cv2.LINE_AA,
        tipLength=0.25
    )

    confidence_text = f"Confidence: {confidence_score:.1f}%"
    cv2.putText(
        frame,
        confidence_text,
        (x_min, y_min - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )

    return confidence_score


def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2, proportion=0.2):
    """Draws a bounding box with corners on the image."""
    x_min, y_min, x_max, y_max = map(int, bbox[:4])
    width = x_max - x_min
    height = y_max - y_min
    corner_length = int(proportion * min(width, height))

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 1)

    # Top-left corner
    cv2.line(image, (x_min, y_min), (x_min + corner_length, y_min), color, thickness)
    cv2.line(image, (x_min, y_min), (x_min, y_min + corner_length), color, thickness)

    # Top-right corner
    cv2.line(image, (x_max, y_min), (x_max - corner_length, y_min), color, thickness)
    cv2.line(image, (x_max, y_min), (x_max, y_min + corner_length), color, thickness)

    # Bottom-left corner
    cv2.line(image, (x_min, y_max), (x_min, y_max - corner_length), color, thickness)
    cv2.line(image, (x_min, y_max), (x_min + corner_length, y_max), color, thickness)

    # Bottom-right corner
    cv2.line(image, (x_max, y_max), (x_max, y_max - corner_length), color, thickness)
    cv2.line(image, (x_max, y_max), (x_max - corner_length, y_max), color, thickness)


def draw_bbox_gaze(frame: np.ndarray, bbox, pitch, yaw):
    """Draws bounding box and gaze direction on the frame."""
    draw_bbox(frame, bbox)
    confidence_score = draw_gaze(frame, bbox, pitch, yaw)
    return confidence_score
