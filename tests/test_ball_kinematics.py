import json

import cv2
import numpy as np

from utils.ball_kinematics import load_calibration


def _pongeye_calibration():
    return {
        "calibration": {
            "corners": {
                "bottomLeft": {"x": 299.3442515591005, "y": 440.15936076718197},
                "bottomRight": {"x": 968.7082009334476, "y": 443.4024909087516},
                "topLeft": {"x": 439.6573620709474, "y": 318.9177730217615},
                "topRight": {"x": 831.1191811144034, "y": 324.1409782215941},
            },
            "tableDimensions": {"lengthMeters": 2.74, "widthMeters": 1.525},
        },
        "exposure": {
            "width": 1280, "height": 720,
            "pixelsPerMetreAtTableCentre": 115.19961122740905,
            "extrinsicsIntrinsics": {
                "focalLengthXPx": 480.5194889231313, "focalLengthYPx": 480.5194889231313,
                "principalPointXPx": 640, "principalPointYPx": 360,
            },
            "extrinsics": {
                "rotation": [0.9993672899639305, -0.026937538860405626, 0.023224744353673638,
                             0.010621133998027336, -0.39716380109741656, -0.9176862789703505,
                             0.03394423754844563, 0.9173523227735328, -0.3966264043646853],
                "translationMetres": [-1.3630223237387427, 0.32073140900263425, 1.9226412394076728],
                "cameraPositionMetres": [1.2934908036118415, -1.6730729681133092, 1.0885569399632549],
            },
        },
    }


def test_loads_pongeye_schema_with_consistent_homography_and_pose(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(_pongeye_calibration()))
    cal = load_calibration(path)

    assert cal.image_size == (1280, 720)
    assert cal.has_pose
    np.testing.assert_allclose(cal.K, [[480.5194889231313, 0, 640], [0, 480.5194889231313, 360], [0, 0, 1]])
    np.testing.assert_allclose(cal.cam_centre, [1.2934908036118415, -1.6730729681133092, 1.0885569399632549])
    table_corners = np.array([[0, 0], [2.74, 0], [2.74, 1.525], [0, 1.525]], dtype=np.float32)
    projected = cv2.perspectiveTransform(table_corners[None, :, :], cal.H.astype(np.float64))[0]
    np.testing.assert_allclose(projected, [[299.34425, 440.15936], [968.7082, 443.40249], [831.11918, 324.14098], [439.65736, 318.91777]], atol=1e-3)


def test_loads_legacy_homography_schema(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "calibration": {"homographyTableToImage": [100, 0, 5, 0, 100, 7, 0, 0, 1]},
        "imageSize": {"width": 640, "height": 480},
        "solvedIntrinsics": {"focalLengthPx": 500},
    }))
    cal = load_calibration(path)
    assert cal.image_size == (640, 480)
    assert cal.has_pose


def test_accepts_device_nested_exposure_schema(tmp_path):
    raw = _pongeye_calibration()
    raw["device"] = {"exposure": raw.pop("exposure")}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(raw))
    cal = load_calibration(path)
    assert cal.has_pose
    assert cal.image_size == (1280, 720)
