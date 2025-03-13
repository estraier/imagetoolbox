#! /usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile

import numpy as np
import cv2


from itb_stack import (
  set_logging_level,
  load_image, save_image, save_video,
  compute_brightness,
  lighten_image, darken_image, sigmoidal_contrast_image, inverse_sigmoidal_contrast_image,
  adjust_exposure,
  align_images_orb, align_images_sift, align_images_ecc,
  merge_images_average, merge_images_median, merge_images_minimum, merge_images_maximum,
  merge_images_weighted_average,
  merge_images_debevec, merge_images_robertson, merge_images_mertens,
  merge_images_focus_stacking, merge_images_stitch,
  tone_map_image_linear, tone_map_image_reinhard, tone_map_image_drago, tone_map_image_mantiuk,
  fill_black_margin_image,
  bilateral_denoise_image, gaussian_blur_image, gaussian_unsharp_image,
  trim_image, scale_image, write_caption,
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


def show_image(image):
  import matplotlib.pyplot as plt
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  plt.imshow(image_rgb)
  plt.axis("off")
  plt.show()


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
    image, bits = load_image("test_input.jpg")
    self.assertEqual(image.shape, (1000, 1000, 3))
    self.assertIn(bits, [8, 16, 32])

  @patch("cv2.imwrite", return_value=True)
  def test_save_image_mock(self, mock_imwrite):
    image = generate_test_image()
    save_image("test_output.jpg", image, 8)
    mock_imwrite.assert_called_once()

  def test_save_and_load_image_jpeg(self):
    image = generate_test_image()
    path = os.path.join(self.temp_path, "test_output.jpg")
    save_image(path, image, 8)
    loaded_image, bits = load_image(path)
    self.assertEqual(image.shape, loaded_image.shape)
    self.assertEqual(bits, 8)

  def test_save_and_load_image_tiff(self):
    image = generate_test_image()
    path = os.path.join(self.temp_path, "test_output.tif")
    save_image(path, image, 16)
    loaded_image, bits = load_image(path)
    self.assertEqual(image.shape, loaded_image.shape)
    self.assertEqual(bits, 16)

  @patch("cv2.VideoWriter")
  def test_save_video_mock(self, mock_video_writer):
    images = [generate_test_image() for _ in range(3)]
    path = os.path.join(self.temp_path, "test_output.mp4")
    mock_writer_instance = mock_video_writer.return_value
    save_video(path, images, 1)
    mock_video_writer.assert_called_once_with(
      path, cv2.VideoWriter_fourcc(*"mp4v"), 1, (images[0].shape[1], images[0].shape[0]))
    self.assertEqual(mock_writer_instance.write.call_count, len(images))
    mock_writer_instance.release.assert_called_once()

  def test_save_video(self):
    images = [generate_test_image() for _ in range(3)]
    path = os.path.join(self.temp_path, "test_output.mp4")
    save_video(path, images, 1)

  def test_compute_brightness(self):
    image = generate_test_image()
    brightness = compute_brightness(image)
    self.assertTrue(0 <= brightness <= 1)

  def test_lighten_image(self):
    image = generate_test_image() * 2
    processed = lighten_image(image, 1)
    self.assertEqual(processed.shape, image.shape)

  def test_darken_image(self):
    image = generate_test_image() * 2
    processed = darken_image(image, 1)
    self.assertEqual(processed.shape, image.shape)

  def test_sigmoidal_contrast_image(self):
    image = generate_test_image() * 2
    processed = sigmoidal_contrast_image(image, 1, 0.5)
    self.assertEqual(processed.shape, image.shape)

  def test_inverse_sigmoidal_contrast_image(self):
    image = generate_test_image() * 2
    processed = inverse_sigmoidal_contrast_image(image, 1, 0.5)
    self.assertEqual(processed.shape, image.shape)

  def test_adjust_exposure(self):
    image = generate_test_image()
    adjusted_image = adjust_exposure(image, 0.65)
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

  def test_fill_black_margin_image(self):
    image = generate_test_image()
    processed = fill_black_margin_image(image)
    self.assertEqual(processed.shape, image.shape)

  def test_bilateral_denoise_image(self):
    image = generate_test_image()
    processed = bilateral_denoise_image(image, 3)
    self.assertEqual(processed.shape, image.shape)

  def test_gaussian_blur_image(self):
    image = generate_test_image()
    processed = gaussian_blur_image(image, 3)
    self.assertEqual(processed.shape, image.shape)

  def test_gaussian_unsharp_image(self):
    image = generate_test_image()
    processed = gaussian_unsharp_image(image, 3)
    self.assertEqual(processed.shape, image.shape)

  def test_trim_image(self):
    image = generate_test_image()
    processed = trim_image(image, 10, 10, 10, 10)
    self.assertEqual(processed.shape[2], image.shape[2])

  def test_scale_image(self):
    image = generate_test_image()
    processed = scale_image(image, 200, 200)
    self.assertEqual(processed.shape[2], image.shape[2])

  def test_write_caption(self):
    image = generate_test_image()
    processed = write_caption(image, "hello|2|f0f|T")
    self.assertEqual(processed.shape, image.shape)


if __name__ == "__main__":
  if "-v" in sys.argv:
    set_logging_level(logging.DEBUG)
  unittest.main()
