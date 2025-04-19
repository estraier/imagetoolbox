#! /usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
import logging
import io
import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile

import numpy as np
import cv2


from itb_stack import (
  main, set_logging_level,
  generate_colorbar, show_image, load_image, save_image, load_video, save_video,
  srgb_to_linear, linear_to_srgb, prophoto_rgb_to_linear, linear_to_prophoto_rgb,
  adobe_rgb_to_linear, linear_to_adobe_rgb, bt2020_to_linear, linear_to_bt2020,
  compute_brightness,
  adjust_white_balance_image, adjust_exposure_image,
  align_images_orb, align_images_sift, align_images_ecc,
  merge_images_average, merge_images_median, merge_images_geometric_mean,
  merge_images_minimum, merge_images_maximum,
  merge_images_weighted_average,
  merge_images_debevec, merge_images_robertson, merge_images_mertens,
  merge_images_focus_stacking, merge_images_grid, merge_images_stitch,
  tone_map_image_linear, tone_map_image_reinhard, tone_map_image_drago, tone_map_image_mantiuk,
  fill_black_margin_image, apply_preset_image,
  apply_global_histeq_image, apply_clahe_image, apply_artistic_filter_image,
  stretch_contrast_image,
  adjust_level_image, apply_linear_image,
  apply_gamma_image, apply_scaled_log_image, apply_sigmoid_image,
  saturate_image_linear, saturate_image_scaled_log,
  optimize_exposure_image, convert_grayscale_image,
  bilateral_denoise_image, masked_denoise_image,
  blur_image_gaussian, pyramid_down_naive, pyramid_up_naive,
  blur_image_pyramid, enhance_texture_image, unsharp_image_gaussian,
  perspective_correct_image, trim_image, scale_image, apply_vignetting_image, write_caption,
)


def generate_test_image(width=1000, height=1000, shift=(0, 0)):
  """Generate a test image with shapes for alignment testing."""
  image = np.full((height, width, 3), (128, 128, 128), dtype=np.float32) / 255
  image = (image * 255).astype(np.uint8)
  circle_center = (width // 4 + shift[0], height // 4 + shift[1])
  circle_center1 = circle_center[0] * 1 + shift[0], circle_center[1] * 1 + shift[1]
  circle_center2 = circle_center[0] * 2 + shift[0], circle_center[1] * 2 + shift[1]
  circle_center3 = circle_center[0] * 3 + shift[0], circle_center[1] * 3 + shift[1]
  cv2.circle(image, circle_center2, width // 3, (255, 255, 255), -1)
  cv2.circle(image, circle_center2, width // 4, (0, 0, 0), -1)
  cv2.circle(image, circle_center1, width // 6, (0, 0, 255), -1)
  cv2.circle(image, circle_center1, width // 10, (0, 255, 255), -1)
  cv2.circle(image, circle_center2, width // 6, (0, 255, 0), -1)
  cv2.circle(image, circle_center2, width // 10, (255, 255, 0), -1)
  cv2.circle(image, circle_center3, width // 6, (255, 0, 0), -1)
  cv2.circle(image, circle_center3, width // 10, (255, 0, 255), -1)
  n1, n2 = 0, 128
  for x in range(0, width, width // 20):
    for y in range(0, height, height // 20):
      sx = x + shift[0]
      sy = y + shift[1]
      th = (n1 + n2) % width // 200 + 1
      cv2.circle(image, (sx, sy), width // 50, (n1 % 256, n2 % 256, (n1 + n2) % 256), th)
      n1 += 48
      n2 += 24
  return image.astype(np.float32) / 255


def show_test_image():
  image = generate_test_image()
  show_image(image)


def split_image_vertically(image):
  height, width = image.shape[:2]
  left_half = image[:, :width // 2]
  right_half = image[:, width // 2:]
  return left_half, right_half


class TestItbStack(unittest.TestCase):

  def setUp(self):
    self.test_dir = tempfile.TemporaryDirectory()
    self.temp_path = self.test_dir.name

  def tearDown(self):
    self.test_dir.cleanup()

  @patch("cv2.imread", return_value=generate_test_image())
  def test_load_image_mock(self, mock_imread):
    image, bits, icc_name = load_image("test_image.jpg")
    self.assertEqual(image.shape, (1000, 1000, 3))
    self.assertIn(bits, [8, 16, 32])

  @patch("cv2.imwrite", return_value=True)
  def test_save_image_mock(self, mock_imwrite):
    image = generate_test_image()
    save_image("test_image.jpg", image, 8, "srgb")
    mock_imwrite.assert_called_once()

  def test_save_and_load_image_jpeg(self):
    image = generate_test_image()
    path = os.path.join(self.temp_path, "test_image.jpg")
    save_image(path, image, 8, "srgb")
    loaded_image, bits, icc_name = load_image(path)
    self.assertEqual(image.shape, loaded_image.shape)
    self.assertEqual(bits, 8)
    self.assertEqual(icc_name, "srgb")

  def test_save_and_load_image_tiff(self):
    image = generate_test_image()
    path = os.path.join(self.temp_path, "test_image.tif")
    save_image(path, image, 16, "srgb")
    loaded_image, bits, icc_name = load_image(path)
    self.assertEqual(image.shape, loaded_image.shape)
    self.assertEqual(bits, 16)
    self.assertEqual(icc_name, "srgb")

  def test_save_and_load_image_png(self):
    image = generate_test_image()
    path = os.path.join(self.temp_path, "test_image.png")
    save_image(path, image, 16, "srgb")
    loaded_image, bits, icc_name = load_image(path)
    self.assertEqual(image.shape, loaded_image.shape)
    self.assertEqual(bits, 16)
    self.assertEqual(icc_name, "srgb")

  def test_save_and_load_image_webp(self):
    image = generate_test_image()
    path = os.path.join(self.temp_path, "test_image.webp")
    save_image(path, image, 8, "srgb")
    loaded_image, bits, icc_name = load_image(path)
    self.assertEqual(image.shape, loaded_image.shape)
    self.assertEqual(bits, 8)
    self.assertEqual(icc_name, "srgb")

  @patch("cv2.VideoCapture")
  def test_load_video_mock(self, mock_video_capture):
    mock_cap = MagicMock()
    mock_video_capture.return_value = mock_cap
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda x: {
      cv2.CAP_PROP_FPS: 10,
      cv2.CAP_PROP_FRAME_COUNT: 30
    }.get(x, None)
    fake_frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
    mock_cap.read.side_effect = [(True, fake_frame)] * 30 + [(False, None)]
    frames = load_video("test_video.mp4", mem_allowance=1<<30, input_fps=5)
    self.assertEqual(len(frames), 15)
    self.assertTrue(all(isinstance(f[0], np.ndarray) for f in frames))
    self.assertEqual(frames[0][0].shape, (100, 100, 3))
    self.assertEqual(frames[0][1], 8)
    mock_video_capture.assert_called_once_with("test_video.mp4")
    mock_cap.release.assert_called_once()

  @patch("cv2.VideoWriter")
  def test_save_video_mock(self, mock_video_writer):
    images = [generate_test_image() for _ in range(3)]
    path = os.path.join(self.temp_path, "test_video.mp4")
    mock_writer_instance = mock_video_writer.return_value
    save_video(path, images, 1)
    mock_video_writer.assert_called_once_with(
      path, cv2.VideoWriter_fourcc(*"mp4v"), 1, (images[0].shape[1], images[0].shape[0]))
    self.assertEqual(mock_writer_instance.write.call_count, len(images))
    mock_writer_instance.release.assert_called_once()

  def test_save_and_load_video(self):
    images = [generate_test_image() for _ in range(3)]
    path = os.path.join(self.temp_path, "test_video.mp4")
    save_video(path, images, 1)
    image_data = load_video(path, 2 * (1<<30), 0)
    self.assertEqual(len(image_data), 3)
    for image, bits in image_data:
      self.assertEqual(bits, 8)
      self.assertEqual(image.shape, images[0].shape)

  def test_srgb_to_linear(self):
    image = generate_test_image()
    adjusted_image = srgb_to_linear(image)
    self.assertEqual(adjusted_image.shape, image.shape)

  def test_linear_to_srgb(self):
    image = generate_test_image()
    adjusted_image = linear_to_srgb(image)
    self.assertEqual(adjusted_image.shape, image.shape)

  def test_prophoto_rgb_to_linear(self):
    image = generate_test_image()
    adjusted_image = prophoto_rgb_to_linear(image)
    self.assertEqual(adjusted_image.shape, image.shape)

  def test_linear_to_prophoto_rgb(self):
    image = generate_test_image()
    adjusted_image = linear_to_prophoto_rgb(image)
    self.assertEqual(adjusted_image.shape, image.shape)

  def test_adobe_rgb_to_linear(self):
    image = generate_test_image()
    adjusted_image = adobe_rgb_to_linear(image)
    self.assertEqual(adjusted_image.shape, image.shape)

  def test_linear_to_adobe_rgb(self):
    image = generate_test_image()
    adjusted_image = linear_to_adobe_rgb(image)
    self.assertEqual(adjusted_image.shape, image.shape)

  def test_bt2020_to_linear(self):
    image = generate_test_image()
    adjusted_image = bt2020_to_linear(image)
    self.assertEqual(adjusted_image.shape, image.shape)

  def test_linear_to_bt2020(self):
    image = generate_test_image()
    adjusted_image = linear_to_bt2020(image)
    self.assertEqual(adjusted_image.shape, image.shape)

  def test_compute_brightness(self):
    image = generate_test_image()
    brightness = compute_brightness(image)
    self.assertTrue(0 <= brightness <= 1)

  def test_adjust_white_balance(self):
    image = generate_test_image()
    for name in ["auto", "auto-temp", "auto-scene", "10,12,11", "5500", "tungsten"]:
      processed = adjust_white_balance_image(image, name)
      self.assertEqual(processed.shape, image.shape)

  def test_adjust_exposure(self):
    image = generate_test_image()
    adjusted_image = adjust_exposure_image(image, 0.65)
    self.assertEqual(adjusted_image.shape, image.shape)

  def test_align_images_orb(self):
    images = [generate_test_image(shift=(i * 5, i * 5)) for i in range(3)]
    aligned = align_images_orb(images, set())
    self.assertEqual(len(aligned), len(images))

  def test_align_images_sift(self):
    images = [generate_test_image(shift=(i * 5, i * 5)) for i in range(3)]
    aligned = align_images_sift(images, set())
    self.assertEqual(len(aligned), len(images))

  def test_align_images_ecc(self):
    images = [generate_test_image(shift=(i * 5, i * 5)) for i in range(3)]
    aligned = align_images_ecc(images, set())
    self.assertEqual(len(aligned), len(images))

  def test_merge_images_average(self):
    images = [generate_test_image(shift=(i, i)) for i in range(10)]
    merged = merge_images_average(images)
    self.assertEqual(merged.shape, images[0].shape)

  def test_merge_images_median(self):
    images = [generate_test_image(shift=(i, i)) for i in range(10)]
    merged = merge_images_median(images)
    self.assertEqual(merged.shape, images[0].shape)

  def test_merge_images_geometric_mean(self):
    images = [generate_test_image(shift=(i, i)) for i in range(10)]
    merged = merge_images_geometric_mean(images)
    self.assertEqual(merged.shape, images[0].shape)

  def test_merge_images_minimum(self):
    images = [generate_test_image(shift=(i, i)) for i in range(10)]
    merged = merge_images_minimum(images)
    self.assertEqual(merged.shape, images[0].shape)

  def test_merge_images_maximum(self):
    images = [generate_test_image(shift=(i, i)) for i in range(10)]
    merged = merge_images_maximum(images)
    self.assertEqual(merged.shape, images[0].shape)

  def test_merge_images_weighted_average(self):
    images = [generate_test_image() for _ in range(3)]
    meta_list = [{"_fv_": 2.0}, {"_fv_": 2.8}, {"_fv_": 4.0}]
    merged = merge_images_weighted_average(images, meta_list)
    self.assertEqual(merged.shape, images[0].shape)

  def test_merge_images_weighted_average_fallback(self):
    images = [generate_test_image() for _ in range(3)]
    meta_list = [{}, {}, {}]
    merged = merge_images_weighted_average(images, meta_list)
    self.assertEqual(merged.shape, images[0].shape)

  def test_merge_images_debevec(self):
    images = [generate_test_image() for _ in range(3)]
    meta_list = [{"_fv_": 2.0, "_tv_": 1.0, "_sv_": 100.0},
                 {"_fv_": 2.8, "_tv_": 1.0, "_sv_": 100.0},
                 {"_fv_": 4.0, "_tv_": 1.0, "_sv_": 100.0}]
    merged = merge_images_debevec(images, meta_list)
    self.assertEqual(merged.shape, images[0].shape)

  def test_merge_images_debevec_fallback(self):
    images = [generate_test_image() for _ in range(3)]
    meta_list = [{}, {}, {}]
    merged = merge_images_debevec(images, meta_list)
    self.assertEqual(merged.shape, images[0].shape)

  def test_merge_images_robertson(self):
    images = [generate_test_image() for _ in range(3)]
    meta_list = [{"_fv_": 2.0, "_tv_": 1.0, "_sv_": 100.0},
                 {"_fv_": 2.8, "_tv_": 1.0, "_sv_": 100.0},
                 {"_fv_": 4.0, "_tv_": 1.0, "_sv_": 100.0}]
    merged = merge_images_robertson(images, meta_list)
    self.assertEqual(merged.shape, images[0].shape)

  def test_merge_images_robertson_fallback(self):
    images = [generate_test_image() for _ in range(3)]
    meta_list = [{}, {}, {}]
    merged = merge_images_robertson(images, meta_list)
    self.assertEqual(merged.shape, images[0].shape)

  def test_merge_images_mertens(self):
    images = [generate_test_image() for _ in range(3)]
    merged = merge_images_mertens(images)
    self.assertEqual(merged.shape, images[0].shape)

  def test_merge_images_focus_stacking(self):
    images = [generate_test_image() for _ in range(3)]
    merged = merge_images_focus_stacking(images)
    self.assertEqual(merged.shape, images[0].shape)

  def test_merge_images_grid(self):
    image = generate_test_image()
    left, right = split_image_vertically(image)
    l1, l2 = split_image_vertically(left)
    r1, r2 = split_image_vertically(right)
    images = [l1, l2, r1, r2]
    merged = merge_images_grid(images, 4)
    self.assertEqual(merged.shape[2], images[0].shape[2])
    merged = merge_images_grid(images, 2, 10, (0.5, 0.8, 0.2))
    self.assertEqual(merged.shape[2], images[0].shape[2])

  def test_merge_images_stitch(self):
    image = generate_test_image()
    left, right = split_image_vertically(image)
    l1, l2 = split_image_vertically(left)
    r1, r2 = split_image_vertically(right)
    mid = np.hstack((l2, r1))
    images = (left, mid, right)
    merged = merge_images_stitch(images)
    self.assertEqual(merged.shape[2], images[0].shape[2])

  def test_tone_map_image_linear(self):
    image = generate_test_image() * 2
    processed = tone_map_image_linear(image)
    self.assertEqual(processed.shape, image.shape)

  def test_tone_map_image_reinhard(self):
    image = generate_test_image() * 2
    processed = tone_map_image_reinhard(image)
    self.assertEqual(processed.shape, image.shape)

  def test_tone_map_image_drago(self):
    image = generate_test_image() * 2
    processed = tone_map_image_drago(image)
    self.assertEqual(processed.shape, image.shape)

  def test_tone_map_image_mantiuk(self):
    image = generate_test_image() * 2
    processed = tone_map_image_mantiuk(image)
    self.assertEqual(processed.shape, image.shape)

  def test_apply_preset_image(self):
    image = generate_test_image()
    for name in ["raw-std", "light", "vivid"]:
      meta = {"_sv_": 10000.0}
      processed = apply_preset_image(image, name, meta, True)
      self.assertEqual(processed.shape, image.shape)

  def test_fill_black_margin_image(self):
    image = generate_test_image()
    processed = fill_black_margin_image(image)
    self.assertEqual(processed.shape, image.shape)

  def test_apply_global_histeq_image(self):
    image = generate_test_image()
    processed = apply_global_histeq_image(image)
    self.assertEqual(processed.shape, image.shape)

  def test_apply_clahe_image(self):
    image = generate_test_image()
    processed = apply_clahe_image(image, 2)
    self.assertEqual(processed.shape, image.shape)

  def test_artistic_filter(self):
    image = generate_test_image()
    for name in ["pencil", "stylized"]:
      processed = apply_artistic_filter_image(image, name)
      self.assertEqual(processed.shape, image.shape)

  def test_stretch_contrast_image_upper(self):
    image = generate_test_image()
    processed = stretch_contrast_image(image, 0.9, 99, 0, -1)
    self.assertEqual(processed.shape, image.shape)

  def test_stretch_contrast_image_lower(self):
    image = generate_test_image()
    processed = stretch_contrast_image(image, 1.0, -1, 0, 1)
    self.assertEqual(processed.shape, image.shape)

  def test_adjust_level_image_auto(self):
    image = generate_test_image()
    processed = adjust_level_image(image)
    self.assertEqual(processed.shape, image.shape)

  def test_adjust_level_image_narrow(self):
    image = generate_test_image()
    processed = adjust_level_image(image, 0.2, 0.8, 0.3, 0.7)
    self.assertEqual(processed.shape, image.shape)

  def test_adjust_level_image_wide(self):
    image = generate_test_image()
    processed = adjust_level_image(image, -0.2, 1.1)
    self.assertEqual(processed.shape, image.shape)

  def test_apply_linear_image_lighten(self):
    image = generate_test_image()
    processed = apply_linear_image(image, 2.2)
    self.assertEqual(processed.shape, image.shape)

  def test_apply_linear_image_lighten(self):
    image = generate_test_image()
    processed = apply_linear_image(image, 1 / 2.2)
    self.assertEqual(processed.shape, image.shape)

  def test_apply_gamma_image_lighten(self):
    image = generate_test_image()
    processed = apply_gamma_image(image, 2.2)
    self.assertEqual(processed.shape, image.shape)

  def test_apply_gamma_image_lighten(self):
    image = generate_test_image()
    processed = apply_gamma_image(image, 1 / 2.2)
    self.assertEqual(processed.shape, image.shape)

  def test_apply_scaled_log_image_lighten(self):
    image = generate_test_image()
    processed = apply_scaled_log_image(image, 1)
    self.assertEqual(processed.shape, image.shape)

  def test_apply_scaled_log_image_darken(self):
    image = generate_test_image()
    processed = apply_scaled_log_image(image, -1)
    self.assertEqual(processed.shape, image.shape)

  def test_apply_sigmoidal_image_lighten(self):
    image = generate_test_image()
    processed = apply_sigmoid_image(image, 1, 0.5)
    self.assertEqual(processed.shape, image.shape)

  def test_apply_sigmoidal_image_darken(self):
    image = generate_test_image()
    processed = apply_sigmoid_image(image, -1, 0.5)
    self.assertEqual(processed.shape, image.shape)

  def test_saturate_image_linear(self):
    image = generate_test_image()
    processed = saturate_image_linear(image, 2)
    self.assertEqual(processed.shape, image.shape)

  def test_saturate_image_scaled_log_thicker(self):
    image = generate_test_image()
    processed = saturate_image_scaled_log(image, 3)
    self.assertEqual(processed.shape, image.shape)

  def test_saturate_image_scaled_log_thinner(self):
    image = generate_test_image()
    processed = saturate_image_scaled_log(image, -3)
    self.assertEqual(processed.shape, image.shape)

  def test_optimize_exposure_face_gamma(self):
    image = generate_test_image()
    processed = optimize_exposure_image(image, 0.5, mask="face", gamma_scale=3.2)
    self.assertEqual(processed.shape, image.shape)

  def test_optimize_exposure_oval_log(self):
    image = generate_test_image()
    processed = optimize_exposure_image(image, 0.5, mask="oval", log_scale=29)
    self.assertEqual(processed.shape, image.shape)

  def test_convert_grayscale_image(self):
    image = generate_test_image()
    for name in ["bt601", "lab", "hsv", "hsl", "laplacian", "sobel",
                 "stddev", "sharpness", "face", "focus", "lcs", "lcs:tricolor", "grabcut"]:
      processed = convert_grayscale_image(image, name)
      self.assertEqual(processed.shape, image.shape)

  def test_bilateral_denoise_image(self):
    image = generate_test_image()
    processed = bilateral_denoise_image(image, 3)
    self.assertEqual(processed.shape, image.shape)

  def test_masked_denoise_image(self):
    image = generate_test_image()
    processed = masked_denoise_image(image, 2, 2)
    self.assertEqual(processed.shape, image.shape)

  def test_blur_image_gaussian(self):
    image = generate_test_image()
    processed = blur_image_gaussian(image, 3)
    self.assertEqual(processed.shape, image.shape)

  def test_blur_image_pyramid_up_down_naive(self):
    image = generate_test_image()
    processed = pyramid_down_naive(image)
    processed = pyramid_up_naive(processed)
    self.assertEqual(processed.shape, image.shape)

  def test_blur_image_pyramid(self):
    image = generate_test_image()
    processed = blur_image_pyramid(image, 3)
    self.assertEqual(processed.shape, image.shape)

  def test_enhance_texture_image(self):
    image = generate_test_image()
    processed = enhance_texture_image(image, 3)
    self.assertEqual(processed.shape, image.shape)

  def test_unsharp_image_gaussian(self):
    image = generate_test_image()
    processed = unsharp_image_gaussian(image, 3)
    self.assertEqual(processed.shape, image.shape)

  def perspective_correct_image(self):
    image = generate_test_image()
    processed = perspective_correct_image(
      image, (10, 10), (90, 10), (90, 90), (10, 90))
    self.assertEqual(processed.shape[2], image.shape[2])

  def test_trim_image(self):
    image = generate_test_image()
    processed = trim_image(image, 10, 10, 10, 10)
    self.assertEqual(processed.shape[2], image.shape[2])

  def test_scale_image(self):
    image = generate_test_image()
    processed = scale_image(image, 200, 200)
    self.assertEqual(processed.shape[2], image.shape[2])
    processed = scale_image(image, 800, 800)
    self.assertEqual(processed.shape[2], image.shape[2])

  def test_apply_vignetting_image(self):
    image = generate_test_image()
    processed = apply_vignetting_image(image, 0.5, 0.2, 0.3)
    self.assertEqual(processed.shape[2], image.shape[2])
    processed = apply_vignetting_image(image, -0.5, 0.7, 0.8)
    self.assertEqual(processed.shape[2], image.shape[2])

  def test_write_caption(self):
    image = generate_test_image()
    processed = write_caption(image, "hello|2|f0f|T")
    self.assertEqual(processed.shape, image.shape)

  def test_run_command_help(self):
    saved_argv = sys.argv.copy()
    sys.argv = ["itb_stack.py", "--help"]
    f = io.StringIO()
    with contextlib.redirect_stdout(f), self.assertRaises(SystemExit) as cm:
      main()
    output = f.getvalue()
    self.assertIn("usage", output.lower())
    self.assertEqual(cm.exception.code, 0)
    sys.argv = saved_argv

  @patch.object(sys, "argv", [])
  def test_run_command_simple(self):
    sys.argv[:] = ["itb_stack.py", "--help"]
    f = io.StringIO()
    with contextlib.redirect_stdout(f), self.assertRaises(SystemExit) as cm:
      main()
    output = f.getvalue()
    self.assertIn("usage", output.lower())
    self.assertEqual(cm.exception.code, 0)

  @patch.object(sys, "argv", [])
  def test_run_command_simple(self):
    file1_path = os.path.join(self.temp_path, "output1.tif")
    file2_path = os.path.join(self.temp_path, "output2.jpg")
    sys.argv[:] = ["itb_stack.py", "[colorbar]", "--output", file1_path]
    main()
    self.assertTrue(os.path.exists(file1_path))
    sys.argv[:] = ["itb_stack.py", file1_path, "--output", file2_path]
    main()
    self.assertTrue(os.path.exists(file2_path))

  @patch.object(sys, "argv", [])
  def test_run_command_merge_average(self):
    output_path = os.path.join(self.temp_path, "output.tif")
    sys.argv[:] = ["itb_stack.py", "[colorbar]", "[colorbar]", "--output", output_path,
                   "--merge", "average"]
    main()
    self.assertTrue(os.path.exists(output_path))

  @patch.object(sys, "argv", [])
  def test_run_command_merge_stf(self):
    output_path = os.path.join(self.temp_path, "output.tif")
    sys.argv[:] = ["itb_stack.py", "[colorbar]", "[colorbar]", "--output", output_path,
                   "--average-exposure", "--align", "orb", "--merge", "weighted"]
    main()
    self.assertTrue(os.path.exists(output_path))

  @patch.object(sys, "argv", [])
  def test_run_command_merge_hdr(self):
    output_path = os.path.join(self.temp_path, "output.tif")
    sys.argv[:] = ["itb_stack.py", "[colorbar]", "[colorbar]", "--output", output_path,
                   "--align", "sift:nfeatures=10000:denoise=3:shift_limit=0.05",
                   "--merge", "debevec", "--tonemap", "reinhard"]
    main()
    self.assertTrue(os.path.exists(output_path))

  @patch.object(sys, "argv", [])
  def test_run_command_merge_denoise(self):
    output_path = os.path.join(self.temp_path, "output.tif")
    sys.argv[:] = ["itb_stack.py", "[colorbar]", "[colorbar]", "--output", output_path,
                   "--align", "s", "--merge", "denoise:clip_limit=0.5:blur_radius=4"]
    main()
    self.assertTrue(os.path.exists(output_path))

  @patch.object(sys, "argv", [])
  def test_run_command_merge_focus(self):
    output_path = os.path.join(self.temp_path, "output.tif")
    sys.argv[:] = ["itb_stack.py", "[colorbar]", "[colorbar]", "--output", output_path,
                   "--align", "ecc:use_affine:denoise=3",
                   "--merge", "focus"]
    main()
    self.assertTrue(os.path.exists(output_path))

  @patch.object(sys, "argv", [])
  def test_run_command_merge_grid(self):
    output_path = os.path.join(self.temp_path, "output.tif")
    sys.argv[:] = ["itb_stack.py", "[colorbar]", "[colorbar]", "--output", output_path,
                   "--merge", "grid:columns=2:margin=2:background=#282"]
    main()
    self.assertTrue(os.path.exists(output_path))

  @patch.object(sys, "argv", [])
  def test_run_command_edit_brightness(self):
    output_path = os.path.join(self.temp_path, "output.tif")
    sys.argv[:] = ["itb_stack.py", "[colorbar]", "[colorbar]", "--output", output_path,
                   "--level", "auto", "--linear", "0.8", "--gamma", "1.1", "--slog", "0.5",
                   "--sigmoid", "0.5:mid=0.3"]
    main()
    self.assertTrue(os.path.exists(output_path))

  @patch.object(sys, "argv", [])
  def test_run_command_edit_saturation(self):
    output_path = os.path.join(self.temp_path, "output.tif")
    sys.argv[:] = ["itb_stack.py", "[colorbar]", "[colorbar]", "--output", output_path,
                   "--saturation", "0.5", "--vibrance", "1"]
    main()
    self.assertTrue(os.path.exists(output_path))

  @patch.object(sys, "argv", [])
  def test_run_command_edit_blur_sharp(self):
    output_path = os.path.join(self.temp_path, "output.tif")
    sys.argv[:] = ["itb_stack.py", "[colorbar]", "[colorbar]", "--output", output_path,
                   "--denoise", "3", "--blur", "3", "--portrait", "auto",
                   "--texture", "3", "--unsharp", "3"]
    main()
    self.assertTrue(os.path.exists(output_path))

  @patch.object(sys, "argv", [])
  def test_run_command_edit_scaling(self):
    output_path = os.path.join(self.temp_path, "output.tif")
    sys.argv[:] = ["itb_stack.py", "[colorbar]", "[colorbar]", "--output", output_path,
                   "--trim", "10,10,10,10", "--pers", "10,10,10,90,0,100,100,0",
                   "--scale", "500", "--vignetting", "0.2"]
    main()
    self.assertTrue(os.path.exists(output_path))

  @patch.object(sys, "argv", [])
  def test_run_command_convert_profile(self):
    file1_path = os.path.join(self.temp_path, "output1.tif")
    file2_path = os.path.join(self.temp_path, "output2.jpg")
    sys.argv[:] = ["itb_stack.py", "[colorbar]", "--output", file1_path,
                   "--gamut", "displayp3"]
    main()
    self.assertTrue(os.path.exists(file1_path))
    sys.argv[:] = ["itb_stack.py", file1_path, "--output", file2_path,
                   "--gamut", "prophoto"]
    main()
    self.assertTrue(os.path.exists(file2_path))


if __name__ == "__main__":
  if "-v" in sys.argv:
    set_logging_level(logging.DEBUG)
  unittest.main()


# END OF FILE
