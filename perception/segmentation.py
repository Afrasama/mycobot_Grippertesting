import pybullet as p
import numpy as np


def _get_camera_image_safe(width, height, view_matrix, proj_matrix):
    # Hardware OpenGL is best in GUI; TinyRenderer is safer in DIRECT.
    try:
        return p.getCameraImage(
            width,
            height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
    except Exception:
        return p.getCameraImage(
            width,
            height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )


def get_overhead_camera_image(width=640, height=480):
    """
    Capture RGB + depth + segmentation from a fixed overhead camera.
    """
    camera_pos = [0.35, 0.0, 0.6]  # above workspace center
    camera_target = [0.35, 0.0, 0.0]  # look at table center
    up_vector = [0, 1, 0]

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_pos,
        cameraTargetPosition=camera_target,
        cameraUpVector=up_vector,
    )

    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width / height,
        nearVal=0.01,
        farVal=2.0,
    )

    _, _, rgb_img, depth_img, seg_img = _get_camera_image_safe(
        width, height, view_matrix, proj_matrix
    )

    rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    depth_array = np.array(depth_img).reshape(height, width)
    seg_array = np.array(seg_img).reshape(height, width)
    return rgb_array, depth_array, seg_array


def compute_body_centroid(segmentation_mask, body_id, min_pixels=20):
    mask = segmentation_mask == body_id
    if int(np.sum(mask)) < int(min_pixels):
        return None
    ys, xs = np.where(mask)
    return float(np.mean(xs)), float(np.mean(ys))


def compute_cube_centroid(segmentation_mask, cube_id, min_pixels=20):
    # backward compatible alias
    return compute_body_centroid(segmentation_mask, cube_id, min_pixels=min_pixels)


def compute_pixel_error(centroid, width=640, height=480):
    center_x = width / 2
    center_y = height / 2
    return float(centroid[0] - center_x), float(centroid[1] - center_y)


def get_relative_pixel_error_overhead(
    target_body_id,
    reference_body_id,
    width=640,
    height=480,
    verbose=False,
):
    """
    Pixel error of target relative to reference, using overhead camera.
    Returns (dx, dy) where:
      dx > 0 => target is to the RIGHT of reference
      dy > 0 => target is BELOW reference (image y axis points down)
    """
    _, _, seg_mask = get_overhead_camera_image(width=width, height=height)
    target_centroid = compute_body_centroid(seg_mask, target_body_id)
    ref_centroid = compute_body_centroid(seg_mask, reference_body_id)

    if target_centroid is None or ref_centroid is None:
        if verbose:
            print("Target or reference not visible in overhead camera")
        return None

    dx = float(target_centroid[0] - ref_centroid[0])
    dy = float(target_centroid[1] - ref_centroid[1])

    if verbose:
        print("Target centroid:", target_centroid, "Ref centroid:", ref_centroid)
        print("Relative Pixel Error:", dx, dy)

    return dx, dy


def get_relative_pixel_error_overhead_and_rgb(
    target_body_id,
    reference_body_id,
    width=640,
    height=480,
    verbose=False,
):
    rgb, _, seg_mask = get_overhead_camera_image(width=width, height=height)
    target_centroid = compute_body_centroid(seg_mask, target_body_id)
    ref_centroid = compute_body_centroid(seg_mask, reference_body_id)

    if target_centroid is None or ref_centroid is None:
        if verbose:
            print("Target or reference not visible in overhead camera")
        return None, rgb

    dx = float(target_centroid[0] - ref_centroid[0])
    dy = float(target_centroid[1] - ref_centroid[1])

    if verbose:
        print("Target centroid:", target_centroid, "Ref centroid:", ref_centroid)
        print("Relative Pixel Error:", dx, dy)

    return (dx, dy), rgb


def get_cube_pixel_error(robot, ee_index, cube_id, verbose=False):
    """
    Backwards-compatible helper.
    Returns cube pixel error relative to image center (overhead camera).
    """
    width = 640
    height = 480
    _, _, seg_mask = get_overhead_camera_image(width=width, height=height)
    centroid = compute_cube_centroid(seg_mask, cube_id)
    if centroid is None:
        if verbose:
            print("Cube not visible in overhead camera")
        return None
    pixel_error_x, pixel_error_y = compute_pixel_error(
        centroid, width=width, height=height
    )
    if verbose:
        print("Centroid:", centroid)
        print("Pixel Error:", pixel_error_x, pixel_error_y)
    return pixel_error_x, pixel_error_y


def get_cube_pixel_error_and_rgb(robot, ee_index, cube_id, verbose=False):
    """
    Convenience helper: returns (pixel_error_x, pixel_error_y) OR None, plus RGB image.
    Overhead camera, relative to image center.
    """
    width = 640
    height = 480
    rgb, _, seg_mask = get_overhead_camera_image(width=width, height=height)
    centroid = compute_cube_centroid(seg_mask, cube_id)
    if centroid is None:
        if verbose:
            print("Cube not visible in overhead camera")
        return None, rgb
    err = compute_pixel_error(centroid, width=width, height=height)
    if verbose:
        print("Centroid:", centroid)
        print("Pixel Error:", err[0], err[1])
    return err, rgb