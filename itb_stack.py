#! /usr/bin/python3
# -*- coding: utf-8 -*-
##################################################################################################
# Copyright (c) 2025 Mikio Hirabayashi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
##################################################################################################


import argparse
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import exifread
import numpy as np


PROG_NAME = "itb_stack.py"
PROG_VERSION = "0.0.2"
CMD_EXIFTOOL = "exiftool"
CMD_HUGIN_ALIGN = "align_image_stack"
EXTS_IMAGE = [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"]
EXTS_IMAGE_HEIF = [".heic", ".heif"]
EXTS_IMAGE_EXIF = [".jpg", ".jpeg", ".tiff", ".tif", ".webp", ".heic", ".heif"]
EXTS_VIDEO = [".mp4", ".mov"]
EXTS_NPZ = [".npz"]


logging.basicConfig(format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(PROG_NAME)
logger.setLevel(logging.INFO)
cmd_env = os.environ
cmd_env["PATH"] = cmd_env["PATH"] + ":/opt/homebrew/bin"
cmd_env["PATH"] = cmd_env["PATH"] + ":/usr/local/bin"
cmd_env["PATH"] = cmd_env["PATH"] + ":/Applications/Hugin/tools_mac"
cv2.setLogLevel(0)


def has_command(name):
  """Checks existence of a command."""
  return bool(shutil.which(name))


def generate_colorbar(width=640, height=480):
  """Generates an image of ARIB-like color bar."""
  img = np.zeros((height, width, 3), dtype=np.uint8)
  h1 = int(height * 0.60)
  h2 = int(height * 0.10)
  h3 = int(height * 0.10)
  h4 = height - (h1 + h2 + h3)
  top_colors = [
    (102, 102, 102),
    (192, 192, 192),
    (0, 255, 255),
    (255, 255, 0),
    (0, 255, 0),
    (255, 0, 255),
    (0, 0, 255),
    (255, 0, 0),
    (102, 102, 102),
  ]
  bar_width = width // len(top_colors)
  for i, color in enumerate(top_colors):
    img[0:h1, i*bar_width:(i+1)*bar_width] = color
  cyan_x = 0
  cyan_w = bar_width
  brown_w = bar_width
  brown_x = cyan_x + cyan_w
  blue_x = width - bar_width
  white_x = brown_x + brown_w
  white_w = width - cyan_w - brown_w - bar_width
  img[h1:h1+h2, cyan_x:cyan_x+cyan_w] = (255, 255, 0)
  img[h1:h1+h2, brown_x:brown_x+brown_w] = (42, 42, 165)
  img[h1:h1+h2, white_x:white_x+white_w] = (192, 192, 192)
  img[h1:h1+h2, blue_x:blue_x+bar_width] = (255, 0, 0)
  yellow_x = 0
  yellow_w = bar_width
  red_x = width - bar_width
  ramp_x = yellow_x + yellow_w
  ramp_w = width - yellow_w - bar_width
  img[h1+h2:h1+h2+h3, yellow_x:yellow_x+yellow_w] = (0, 255, 255)
  img[h1+h2:h1+h2+h3, red_x:red_x+bar_width] = (0, 0, 255)
  for i in range(ramp_w):
    val = int(i / ramp_w * 255)
    img[h1+h2:h1+h2+h3, ramp_x + i] = (val, val, val)
  gray_values = [255, 192, 128, 64, 38, 15, 0]
  block_w = width // len(gray_values)
  for i, val in enumerate(gray_values):
    img[h1+h2+h3:, i*block_w:(i+1)*block_w] = (val, val, val)
  return img.astype(np.float32) / 255


def show_image(image, title="show_image"):
  """Shows an image in the window."""
  image = image.astype(np.float32)
  if image.dtype == np.uint8:
    image = image / 255
  if len(image.shape) == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = np.clip(image, 0, 1)
  cv2.imshow(title, image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def normalize_input_image(image):
  """Normalizes the input image as linear RGB data in BGR space."""
  if image.dtype == np.uint32:
    image = image.astype(np.float32) / float((1<<32) - 1)
    bits = 32
  elif image.dtype == np.uint16:
    image = image.astype(np.float32) / float((1<<16) - 1)
    bits = 16
  else:
    image = image.astype(np.float32) / float((1<<8) - 1)
    bits = 8
  if len(image.shape) == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  elif image.shape[2] == 4:
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
  image = np.clip(image, 0, 1)
  image = srgb_to_linear(image)
  return image, bits


def load_image(file_path):
  """Loads an image and returns its linear RGB data as a NumPy array."""
  logger.debug(f"loading image: {file_path}")
  image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
  if image is None:
    raise ValueError(f"Failed to load image: {file_path}")
  image, bits = normalize_input_image(image)
  h, w = image.shape[:2]
  logger.debug(f"h={h}, w={w}, area={h*w}, bits={bits}")
  return image, bits


def save_image(file_path, image, bits):
  """Saves an image after converting it from linear RGB to sRGB."""
  assert image.dtype == np.float32
  logger.debug(f"saving image: {file_path}")
  image = linear_to_srgb(image)
  ext = os.path.splitext(file_path)[1].lower()
  if ext in [".jpg", ".jpeg", ".webp"]:
    image = (np.clip(image, 0, 1) * ((1<<8) - 1)).astype(np.uint8)
  elif ext in [".png", ".tiff", ".tif"]:
    if bits == 32:
      image = (np.clip(image, 0, 1) * ((1<<32) - 1)).astype(np.uint32)
    elif bits == 16:
      image = (np.clip(image, 0, 1) * ((1<<16) - 1)).astype(np.uint16)
    else:
      image = (np.clip(image, 0, 1) * ((1<<8) - 1)).astype(np.uint8)
  else:
    raise ValueError(f"Unsupported file format: {ext}")
  success = cv2.imwrite(file_path, image)
  if not success:
    raise ValueError(f"Failed to save image: {file_path}")


def load_images_heif(file_path):
  """Loads images in HEIF/HEIC and returns their linear RGB data as a tuple of a NumPy array."""
  from pillow_heif import open_heif
  logger.debug(f"loading HEIF/HIEC image: {file_path}")
  heif_file = open_heif(file_path)
  images = []
  for heif_image in heif_file:
    if heif_image.mode in ["RGB", "RGBA", "L", "LA"]:
      image = np.array(heif_image).astype(np.uint8)
    elif heif_image.mode == "I;16":
      image = np.array(heif_image).astype(np.uint16)
    else:
      raise ValueError(f"Unknown image mode: {heif_image.mode}")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image, bits = normalize_input_image(image)
    images.append((image, bits))
  return images


def load_video(file_path, mem_allowance, input_fps):
  """Loads images in video and returns their linear RGB data as a tuple of a NumPy array."""
  logger.debug(f"loading video: {file_path}")
  cap = cv2.VideoCapture(file_path)
  if not cap.isOpened():
    raise ValueError(f"Failed to open video: {file_path}")
  fps = cap.get(cv2.CAP_PROP_FPS)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  duration = total_frames / fps
  logger.debug(f"fps={fps}, frames={total_frames}, duration={duration:.2f}")
  frames = []
  total_mem_usage = 0
  if input_fps > 0:
    interval = 1 / input_fps
    current_time = 0.0
    while current_time < duration:
      cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
      ret, frame = cap.read()
      if not ret:
        break
      frame, bits = normalize_input_image(frame)
      frame_mem_size = estimate_image_memory_size(frame)
      if total_mem_usage + frame_mem_size > mem_allowance:
        logger.warning(f"Memory limit reached while processing")
        break
      frames.append((frame, bits))
      total_mem_usage += frame_mem_size
      current_time += interval
  else:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame, bits = normalize_input_image(frame)
      frame_mem_size = estimate_image_memory_size(frame)
      if total_mem_usage + frame_mem_size > mem_allowance:
        logger.warning(f"Memory limit reached while processing")
        break
      frames.append((frame, bits))
      total_mem_usage += frame_mem_size
  cap.release()
  return frames


def save_video(file_path, images, output_fps):
  """Saves a video of images."""
  assert all(image.dtype == np.float32 for image in images)
  logger.debug(f"saving video: {file_path}")
  h, w = images[0].shape[:2]
  ext = os.path.splitext(file_path)[-1].lower()
  if ext not in EXTS_VIDEO:
    raise ValueError(f"Unsupported file format: {ext}")
  codec = cv2.VideoWriter_fourcc(*"mp4v")
  out = cv2.VideoWriter(file_path, codec, output_fps, (w, h))
  for image in images:
    if image.shape[:2] != (h, w):
      image = crop_to_match(image, (h, w))
    srgb_image = linear_to_srgb(image)
    uint8_image = (srgb_image * 255).astype(np.uint8)
    out.write(uint8_image)
  out.release()


def load_npz(file_path, mem_allowance):
  """Loads images in NPZ and returns their linear RGB data as a tuple of a NumPy array."""
  logger.debug(f"loading NPZ: {file_path}")
  npz_data = np.load(file_path)
  frames = []
  total_mem_usage = 0
  for key in npz_data:
    frame = npz_data[key]
    if frame.dtype == np.uint32:
      frame = frame.astype(np.float32) / float((1<<32) - 1)
      bits = 32
    elif frame.dtype == np.uint16:
      frame = frame.astype(np.float32) / float((1<<16) - 1)
      bits = 16
    elif frame.dtype == np.uint8:
      frame = frame.astype(np.float32) / float((1<<8) - 1)
      bits = 8
    elif frame.dtype == np.float32:
      bits = 16
    frame_mem_size = estimate_image_memory_size(frame)
    if total_mem_usage + frame_mem_size > mem_allowance:
      logger.warning(f"Memory limit reached while processing")
      break
    frames.append((frame, bits))
    total_mem_usage += frame_mem_size
  return frames


def save_npz(file_path, images):
  """Saves an NPZ archive file of images."""
  assert all(image.dtype == np.float32 for image in images)
  logger.debug(f"saving NPZ: {file_path}")
  np.savez_compressed(file_path, *images)


def estimate_image_memory_size(image):
  """Estimates memory size of an image."""
  assert image.dtype == np.float32
  height, width = image.shape[:2]
  channels = 3
  depth = 4
  return width * height * channels * depth


def parse_boolean(text):
  """Parse a boolean expression and get its boolean value."""
  value = text.strip().lower()
  if value in ["true", "t", "1", "yes", "y"]:
    return True
  if value in ["false", "f", "0", "no", "n"]:
    return False
  raise ValueError(f"invalid boolean expression '{text}'")


def parse_numeric(text):
  """Parse a numeric expression and get its float value."""
  text = text.lower().strip()
  match = re.fullmatch("(-?\d+\.?\d*) */ *(-?\d+\.?\d*)", text)
  if match:
    return float(match.group(1)) / float(match.group(2))
  match = re.fullmatch("(-?\d+\.?\d*)", text)
  if match:
    return float(text)
  return float("nan")


def get_metadata(path):
  """Gets Exif data from a image file."""
  meta = {}
  if has_command(CMD_EXIFTOOL):
    ext = os.path.splitext(path)[1].lower()
    if ext in EXTS_IMAGE_EXIF:
      cmd = ["exiftool", "-s", "-t", "-n", path]
      logger.debug(f"running: {' '.join(cmd)}")
      content = subprocess.check_output(
        cmd, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      lines = content.decode("utf-8", "ignore").split("\n")
      for line in lines:
        fields = line.strip().split("\t", 1)
        if len(fields) < 2: continue
        name, value = fields[:2]
        if name == "ExposureTime":
          meta["_tv_"] = parse_numeric(str(value))
        if name == "FNumber":
          meta["_fv_"] = parse_numeric(str(value))
        if name == "ISO":
          meta["_sv_"] = parse_numeric(str(value))
  try:
    ext = os.path.splitext(path)[1].lower()
    if ext in EXTS_IMAGE_EXIF:
      with open(path, "rb") as f:
        tags = exifread.process_file(f)
        for name, value in tags.items():
          if name == "EXIF ExposureTime":
            meta["_tv_"] = parse_numeric(str(value))
          if name == "EXIF FNumber":
            meta["_fv_"] = parse_numeric(str(value))
          if name == "EXIF ISOSpeedRatings":
            meta["_sv_"] = parse_numeric(str(value))
  except:
    pass
  return meta


def get_light_value(meta):
  """Gets the luminance value from metadata."""
  tv = meta.get("_tv_")
  fv = meta.get("_fv_")
  sv = meta.get("_sv_") or 100.0
  if tv and fv:
    ev = math.log2(1 / tv / fv / fv)
    lv = ev + math.log2(sv / 100)
    return lv
  return None


def copy_metadata(source_path, target_path):
  """Copies EXIF data and ICC profile from source image to target image."""
  cmd = ["exiftool", "-TagsFromFile", source_path, "-icc_profile",
         "-thumbnailimage=", "-f", "-m", "-overwrite_original", target_path]
  logger.debug(f"running: {' '.join(cmd)}")
  try:
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  except:
    pass


def srgb_to_linear(image):
  """Converts sRGB to linear RGB using OpenCV (optimized for float32)."""
  assert image.dtype == np.float32
  image = np.where(image <= 0.04045,
                   image / 12.92, cv2.pow((image + 0.055) / 1.055, 2.4))
  return image.astype(np.float32)


def linear_to_srgb(image):
  """Converts linear RGB to sRGB using OpenCV (optimized for float32)."""
  assert image.dtype == np.float32
  image = np.where(image <= 0.0031308,
                   image * 12.92, 1.055 * cv2.pow(image, 1/2.4) - 0.055)
  return image.astype(np.float32)


def compute_brightness(image):
  """Computes the average brightness of an image in grayscale."""
  assert image.dtype == np.float32
  return np.mean(cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY))


def apply_gamma_image(image, gamma):
  """Adjusts image brightness by a gamma transformation."""
  assert image.dtype == np.float32
  if gamma < 1e-6:
    return image
  image = np.power(image, 1 / gamma)
  return image.astype(np.float32)


def apply_scaled_log_image(image, factor):
  """Adjust image brightness by a scaled log transformation."""
  assert image.dtype == np.float32
  if factor > 1e-6:
    image = np.log1p(image * factor) / np.log1p(factor)
  elif factor < -1e-6:
    factor = -factor
    image = (np.expm1(image * np.log1p(factor))) / factor
  return image.astype(np.float32)


def naive_sigmoid(x, gain, mid):
  """Computes naive sigmod conversion."""
  image = 1.0 / (1.0 + np.exp((mid - x) * gain))
  return image.astype(np.float32)


def naive_inverse_sigmoid(x, gain, mid):
  """Computes inverse naive sigmoid conversion."""
  min_val = naive_sigmoid(0.0, gain, mid)
  max_val = naive_sigmoid(1.0, gain, mid)
  a = (max_val - min_val) * x + min_val
  return -np.log(1.0 / a - 1.0) / gain


def sigmoidal_contrast_image(image, gain, mid):
  """Applies sigmoidal contrast adjustment with a scaled sigmoid function."""
  assert image.dtype == np.float32
  min_val = naive_sigmoid(0.0, gain, mid)
  max_val = naive_sigmoid(1.0, gain, mid)
  diff = max_val - min_val
  return np.clip((naive_sigmoid(image, gain, mid) - min_val) / diff, 0, 1)


def inverse_sigmoidal_contrast_image(image, gain, mid):
  """Applies inverse sigmoidal contrast adjustment."""
  assert image.dtype == np.float32
  min_val = naive_inverse_sigmoid(0.0, gain, mid)
  max_val = naive_inverse_sigmoid(1.0, gain, mid)
  diff = max_val - min_val
  return np.clip((naive_inverse_sigmoid(image, gain, mid) - min_val) / diff, 0, 1)


def apply_sigmoid_image(image, gain, mid=0.5):
  """Adjust image brightness by a sigmoid transformation."""
  assert image.dtype == np.float32
  if gain > 1e-6:
    min_val = naive_sigmoid(0.0, gain, mid)
    max_val = naive_sigmoid(1.0, gain, mid)
    diff = max_val - min_val
    image = (naive_sigmoid(image, gain, mid) - min_val) / diff
  elif gain < -1e-6:
    gain = -gain
    min_val = naive_inverse_sigmoid(0.0, gain, mid)
    max_val = naive_inverse_sigmoid(1.0, gain, mid)
    diff = max_val - min_val
    image = (naive_inverse_sigmoid(image, gain, mid) - min_val) / diff
  return np.clip(image, 0, 1).astype(np.float32)


def compute_auto_white_balance_factors(image, edge_weight=0.5, luminance_weight=0.5):
  """Computes the mean values for RGB channels for auto white balance."""
  assert image.dtype == np.float32
  image = cv2.GaussianBlur(image, (5, 5), 0)
  max_area = 1000000
  current_area = image.shape[0] * image.shape[1]
  if current_area > max_area:
    scale_factor = np.sqrt(max_area / current_area)
    image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor,
                       interpolation=cv2.INTER_AREA)
  std_range = 2
  if edge_weight > 0:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.clip(gray, 0, 1)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mask = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_mean = np.mean(edge_mask)
    edge_std = np.std(edge_mask) + 1e-6
    edge_mask = np.clip(edge_mask, edge_mean - std_range * edge_std,
                        edge_mean + std_range * edge_std)
    edge_mask = ((edge_mask - (edge_mean - std_range * edge_std)) /
                 (2 * std_range * edge_std + 1e-6))
    edge_mask = np.clip(edge_mask, 0, 1)
  else:
    edge_mask = np.zeros_like(image[:, :, 0], dtype=np.float32)
  if luminance_weight > 0:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    l_channel = np.clip(l_channel, 0, 100)
    l_mean = np.mean(l_channel)
    l_std = np.std(l_channel) + 1e-6
    luminance_mask = np.clip(l_channel, l_mean - std_range * l_std, l_mean + std_range * l_std)
    luminance_mask = ((luminance_mask - (l_mean - std_range * l_std)) /
              (2 * std_range * l_std + 1e-6))
    luminance_mask = np.clip(luminance_mask, 0, 1)
  else:
    luminance_mask = np.zeros_like(image[:, :, 0], dtype=np.float32)
  total_weight = edge_weight + luminance_weight
  if total_weight > 0:
    edge_weight /= total_weight
    luminance_weight /= total_weight
    combined_mask = edge_weight * edge_mask + luminance_weight * luminance_mask
  else:
    combined_mask = np.ones_like(image[:, :, 0], dtype=np.float32)
  mean_b = np.sum(image[:, :, 0] * combined_mask) / (np.sum(combined_mask) + 1e-6)
  mean_g = np.sum(image[:, :, 1] * combined_mask) / (np.sum(combined_mask) + 1e-6)
  mean_r = np.sum(image[:, :, 2] * combined_mask) / (np.sum(combined_mask) + 1e-6)
  return mean_r, mean_g, mean_b


def get_chromaticity_coordinates(kelvin):
  """Returns the chromaticity coordinates for a given color temperature."""
  if kelvin < 4000:
    x = (-0.2661239 * (10**9) / kelvin**3 -
         0.2343580 * (10**6) / kelvin**2 + 0.8776956 * (10**3) / kelvin + 0.179910)
  else:
    x = (-3.0258469 * (10**9) / kelvin**3 + 2.1070379 * (10**6) / kelvin**2 +
         0.2226347 * (10**3) / kelvin + 0.240390)
  if kelvin < 2222:
    y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
  elif kelvin < 4000:
    y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
  else:
    y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
  return x, y


def estimate_kelvin_to_rgb(kelvin):
  """Estimates RGB values from a color temperature."""
  kelvin = np.clip(kelvin, 1000, 40000)
  x, y = get_chromaticity_coordinates(kelvin)
  cie_y = 1.0
  cie_x = (cie_y / y) * x
  cie_z = (cie_y / y) * (1 - x - y)
  xyz_to_rgb = np.array([
    [ 3.2406, -1.5372, -0.4986],
    [-0.9689,  1.8758,  0.0415],
    [ 0.0557, -0.2040,  1.0570]
  ])
  rgb_linear = np.dot(xyz_to_rgb, np.array([cie_x, cie_y, cie_z]))
  rgb_linear /= np.max(rgb_linear)
  return float(rgb_linear[0]), float(rgb_linear[1]), float(rgb_linear[2])


def convert_kelvin_to_rgb(kelvin):
  """Converts a color temperature to RGB values."""
  cie_standard_data = [
    (2500, (1.000, 0.374, 0.068)),
    (3000, (1.000, 0.479, 0.154)),
    (3500, (1.000, 0.571, 0.259)),
    (4000, (1.000, 0.653, 0.377)),
    (4500, (1.000, 0.727, 0.502)),
    (5000, (1.000, 0.793, 0.629)),
    (5500, (1.000, 0.850, 0.755)),
    (5800, (1.000, 0.881, 0.828)),
    (6000, (1.000, 0.900, 0.876)),
    (6100, (1.000, 0.909, 0.900)),
    (6200, (1.000, 0.918, 0.923)),
    (6300, (1.000, 0.927, 0.947)),
    (6400, (1.000, 0.935, 0.970)),
    (6500, (1.000, 0.944, 0.993)),
    (6600, (0.985, 0.938, 1.000)),
    (6700, (0.964, 0.925, 1.000)),
    (6800, (0.944, 0.913, 1.000)),
    (7000, (0.906, 0.890, 1.000)),
    (7500, (0.828, 0.842, 1.000)),
    (8000, (0.766, 0.802, 1.000)),
    (8500, (0.715, 0.768, 1.000)),
    (9300, (0.652, 0.725, 1.000)),
    (9500, (0.639, 0.716, 1.000)),
  ]
  if kelvin < cie_standard_data[0][0]:
    lower_sample = upper_sample = cie_standard_data[0]
  elif kelvin > cie_standard_data[-1][0]:
    lower_sample = upper_sample = cie_standard_data[-1]
  else:
    lower_sample = None
    upper_sample = None
    for sample in cie_standard_data:
      if kelvin >= sample[0] and (not lower_sample or sample[0] > lower_sample[0]):
        lower_sample = sample
      if kelvin <= sample[0] and (not upper_sample or sample[0] < upper_sample[0]):
        upper_sample = sample
  lower_rgb = np.array(estimate_kelvin_to_rgb(lower_sample[0]))
  upper_rgb = np.array(estimate_kelvin_to_rgb(upper_sample[0]))
  lower_deviation = np.array(lower_sample[1]) / np.clip(lower_rgb, 1e-8, None)
  upper_deviation = np.array(upper_sample[1]) / np.clip(upper_rgb, 1e-8, None)
  if lower_sample[0] == upper_sample[0]:
    weight = 0.5
  else:
    weight = (kelvin - lower_sample[0]) / (upper_sample[0] - lower_sample[0])
  blended_deviation = (1 - weight) * lower_deviation + weight * upper_deviation
  estimated_rgb = np.array(estimate_kelvin_to_rgb(kelvin))
  corrected_rgb = estimated_rgb * blended_deviation
  return tuple(corrected_rgb / np.max(corrected_rgb))


def convert_rgb_to_kelvin(r, g, b):
  """Converts RGB values to a color temperature."""
  weights = np.array([2.0, 1.0, 2.0])
  rgb_input = np.array([r, g, b])
  rgb_input *= weights
  rgb_input /= np.linalg.norm(rgb_input)
  best_kelvin = 2000
  step = 500
  lowest = 2000
  highest = 20000
  for _ in range(3):
    best_similarity = -1
    for kelvin in np.arange(lowest, highest + step, step):
      r_est, g_est, b_est = convert_kelvin_to_rgb(kelvin)
      rgb_est = np.array([r_est, g_est, b_est])
      rgb_est *= weights
      rgb_est /= np.linalg.norm(rgb_est)
      similarity = np.dot(rgb_input, rgb_est)
      if similarity > best_similarity:
        best_similarity = similarity
        best_kelvin = kelvin
    lowest = best_kelvin - step
    highest = best_kelvin + step
    step /= 10
  return best_kelvin


def adjust_white_balance_image(image, expr="auto"):
  """Adjusts the white balance of an image."""
  assert image.dtype == np.float32
  WB_PRESETS = {
    "daylight": (1.1, 1.0, 0.9),
    "cloudy": (1.2, 1.0, 0.8),
    "shade": (1.3, 1.0, 0.7),
    "tungsten": (0.7, 1.0, 1.3),
    "fluorescent": (0.9, 1.0, 1.1),
    "flash": (1.1, 1.0, 0.9),
    "starlight": (1.4, 1.0, 0.6),
  }
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray = np.clip(gray, 0, 1)
  expr = expr.strip().lower()
  params = parse_name_opts_expression(expr)
  name = params["name"]
  if name in ["auto", "auto-scene", "auto-temp"]:
    kwargs = {}
    copy_param_to_kwargs(params, kwargs, "edge_weight", float)
    copy_param_to_kwargs(params, kwargs, "luminance_weight", float)
    mean_r, mean_g, mean_b = compute_auto_white_balance_factors(image, **kwargs)
    if name == "auto-temp":
      kelvin = convert_rgb_to_kelvin(mean_r, mean_g, mean_b)
      logger.debug(f"tempelature={kelvin:.0f}K, from={mean_r:.3f},{mean_g:.3f},{mean_b:.3f}")
      mean_r, mean_g, mean_b = convert_kelvin_to_rgb(kelvin)
    mean_gray = (mean_b + mean_g + mean_r) / 3
    scale_r = mean_gray / max(mean_r, 1e-6)
    scale_g = mean_gray / max(mean_g, 1e-6)
    scale_b = mean_gray / max(mean_b, 1e-6)
    if expr == "auto-scene":
      scale_rgb = np.array([scale_r, scale_g, scale_b])
      image_vector = np.array([scale_r, scale_g, scale_b])
      image_norm = np.linalg.norm(image_vector)
      best_preset = None
      best_similarity = -1
      for preset, reference_rgb in WB_PRESETS.items():
        preset_vector = np.array(reference_rgb)
        preset_norm = np.linalg.norm(preset_vector)
        similarity = np.dot(image_vector, preset_vector) / (image_norm * preset_norm + 1e-6)
        if similarity > best_similarity:
          best_similarity = similarity
          best_preset = preset
      logger.debug(f"scene={best_preset}, similarity={best_similarity:.3f},"
                   f" from={scale_r:.3f},{scale_g:.3f},{scale_b:.3f}")
      scale_r, scale_g, scale_b = WB_PRESETS[best_preset]
  elif expr in WB_PRESETS:
    scale_r, scale_g, scale_b = WB_PRESETS[expr]
  elif re.match(r"^\d+(\.\d+)?$", expr):
    kelvin = float(expr)
    if kelvin < 1000 or kelvin > 30000:
      raise ValueError(f"Invalid kelvin range: {expr}")
    mean_r, mean_g, mean_b = convert_kelvin_to_rgb(kelvin)
    mean_gray = (mean_b + mean_g + mean_r) / 3
    scale_r = mean_gray / max(mean_r, 1e-6)
    scale_g = mean_gray / max(mean_g, 1e-6)
    scale_b = mean_gray / max(mean_b, 1e-6)
  else:
    match = re.search(r"([\d\.]+)([^\d\.]+)([\d\.]+)([^\d\.]+)([\d\.]+)", expr)
    if match:
      mean_r = float(match.group(1))
      mean_g = float(match.group(3))
      mean_b = float(match.group(5))
    else:
      raise ValueError(f"Invalid white balance expresion: {expr}")
    if mean_r < 0 or mean_g < 0 or mean_b < 0:
      raise ValueError(f"Negative white balance expresion: {expr}")
    mean_gray = (mean_b + mean_g + mean_r) / 3
    scale_r = mean_gray / max(mean_r, 1e-6)
    scale_g = mean_gray / max(mean_g, 1e-6)
    scale_b = mean_gray / max(mean_b, 1e-6)
  logger.debug(f"R={scale_r:.3f}, G={scale_g:.3f}, B={scale_b:.3f}")
  white_threshold = 0.98
  weights = 1 - np.clip((gray - (1 - white_threshold)) / white_threshold, 0, 1)
  image[:, :, 0] = np.clip(image[:, :, 0] * (weights * scale_b + (1 - weights)), 0, 1)
  image[:, :, 1] = np.clip(image[:, :, 1] * (weights * scale_g + (1 - weights)), 0, 1)
  image[:, :, 2] = np.clip(image[:, :, 2] * (weights * scale_r + (1 - weights)), 0, 1)
  return image


def adjust_exposure_image(image, target_brightness, max_tries=10, max_dist=0.01):
  """Adjusts the exposure of an image to a target brightness."""
  assert image.dtype == np.float32
  brightness = compute_brightness(image) + 1e-6
  dist = abs(np.log(target_brightness / brightness))
  logger.debug(f"tries=0, gain=0.000, dist={dist:.3f},"
               f" brightness={brightness:.3f}")
  if dist < max_dist:
    return image
  increase = target_brightness > brightness
  num_tries = 0
  upper = 8.0
  lower = 1 / upper
  while True:
    num_tries += 1
    gain = (upper * lower) ** 0.5
    if not increase:
      gain *= -1
    tmp_image = apply_scaled_log_image(image, gain)
    brightness = compute_brightness(tmp_image) + 1e-6
    dist = abs(np.log(target_brightness / brightness))
    logger.debug(f"tries={num_tries}, gain={gain:.3f}, dist={dist:.3f},"
                 f" brightness={brightness:.3f}")
    if dist < max_dist or num_tries >= max_tries:
      return tmp_image
    if increase:
      if target_brightness > brightness:
        if num_tries < max_tries / 2:
          upper *= 2
        lower = gain
      else:
        upper = gain
    else:
      if target_brightness < brightness:
        if num_tries < max_tries / 2:
          upper *= 2
        lower = -gain
      else:
        upper = -gain
  return image


def is_homography_valid(m, width, height):
  """Checks if homography is valid."""
  scale_allowance = 0.05
  shift_allowance = 0.05
  rotation_allowance = 0.05
  sx, shear, dx = m[0]
  shear_y, sy, dy = m[1]
  scale_x = np.sqrt(sx**2 + shear**2)
  scale_y = np.sqrt(sy**2 + shear_y**2)
  rotation = np.arctan2(m[1, 0], m[0, 0]) / np.pi
  min_scale = 1 - scale_allowance
  max_scale = 1 / min_scale
  if not (-shift_allowance <= dx / width <= shift_allowance and
          -shift_allowance <= dy / height <= shift_allowance):
    return False
  if not (min_scale <= scale_x <= max_scale and
          min_scale <= scale_y <= max_scale):
    return False
  if abs(rotation) > rotation_allowance:
    return False
  return True


def log_homography_matrix(m):
  """Prints a log message of a homography matrix."""
  sx, shear, dx = m[0]
  shear_y, sy, dy = m[1]
  scale_x = np.sqrt(sx**2 + shear**2)
  scale_y = np.sqrt(sy**2 + shear_y**2)
  angle = np.arctan2(m[1, 0], m[0, 0]) * 180 / np.pi
  logger.debug(f"warping: tran=({dx:.2f}, {dy:.2f}), "
               f"scale=({scale_x:.2f}, {scale_y:.2f}), angle={angle:.2f}°")


def apply_clahe_gray_image(image, clip_limit, gamma=2.2):
  """Applies CLAHE on a gray image."""
  assert image.dtype == np.float32
  image = np.power(image, 1 / gamma) * 255.0
  byte_image = image.astype(np.uint8)
  undo_bytes = byte_image.astype(np.float32)
  float_ratio = np.where(byte_image > 0, undo_bytes / (image + 1e-6), image)
  tile_grid_size = (8, 8)
  clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
  new_image = clahe.apply(byte_image).astype(np.float32)
  new_image = np.power(new_image / 255.0, gamma)
  corrected = new_image / np.maximum(float_ratio, 1e-6)
  image = np.where((new_image == 0) | (float_ratio < 0.5), new_image, corrected)
  return np.clip(image, 0, 1).astype(np.float32)


def make_image_for_alignment(image, clahe_clip_limit=0, denoise=0):
  """Makes a byte-gray enhanced gray image for alignment."""
  assert image.dtype == np.float32
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray_image = np.clip(gray_image, 0, 1)
  if denoise > 0:
    ksize = math.ceil(2 * denoise) + 1
    sigma_color = min(0.05 * math.sqrt(ksize), 0.35)
    sigma_space = 10 * math.sqrt(ksize)
    gray_image = cv2.bilateralFilter(gray_image, denoise, sigma_color, sigma_space)
  if clahe_clip_limit > 0:
    gray_image = apply_clahe_gray_image(gray_image, clahe_clip_limit)
  byte_image = (np.clip(gray_image, 0, 1) * 255).astype(np.uint8)
  return np.clip(byte_image, 0, 255)


def align_images_orb(images, aligned_indices, nfeatures=5000, denoise=2, shift_limit=0.1):
  """Aligns images using ORB."""
  assert all(image.dtype == np.float32 for image in images)
  if len(images) < 2:
    return images
  ref_image = images[0]
  h, w = ref_image.shape[:2]
  sides_mean = (h + w) / 2
  check_results = []
  clahe_clip_limits = [1.0, 2.0, 4.0]
  for clahe_clip_limit in clahe_clip_limits:
    ref_gray = make_image_for_alignment(ref_image, clahe_clip_limit, denoise)
    orb = cv2.ORB_create(nfeatures=100000)
    ref_kp, ref_des = orb.detectAndCompute(ref_gray, None)
    if ref_des is None or len(ref_kp) < 10: continue
    score = abs(math.log(nfeatures * 2) - math.log(len(ref_kp)))
    logger.debug(f"checking reference image:"
                 f" clahe={clahe_clip_limit}, denoise={denoise}, kp={len(ref_kp)}")
    check_results.append((score, clahe_clip_limit, ref_gray))
  if not check_results:
    logger.debug("reference image has no descriptors")
    return images
  check_results = sorted(check_results)
  score, clahe_clip_limit, ref_gray = check_results[0]
  orb = cv2.ORB_create(nfeatures=nfeatures)
  ref_kp, ref_des = orb.detectAndCompute(ref_gray, None)
  check_results = None
  ref_grays = None
  logger.debug(f"best config: clahe={clahe_clip_limit}, denoise={denoise}, kp={len(ref_kp)}")
  aligned_indices.add(0)
  aligned_images = [ref_image]
  bounding_boxes = []
  for image in images[1:]:
    image_gray = make_image_for_alignment(image, clahe_clip_limit, denoise)
    kp, des = orb.detectAndCompute(image_gray, None)
    if des is None:
      logger.debug(f"image has no descriptors")
      aligned_images.append(image)
      continue
    logger.debug(f"detected {len(kp)} key points in the target")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(ref_des, des)
    cand_matches = sorted(matches, key=lambda x: x.distance)
    good_matches = []
    for m in cand_matches:
      dist = np.linalg.norm(np.array(ref_kp[m.queryIdx].pt) - np.array(kp[m.trainIdx].pt))
      dist /= sides_mean
      if dist > shift_limit: continue
      good_matches.append(m)
    if len(good_matches) > 10:
      logger.debug(f"matches found: {len(good_matches)} of {len(matches)}")
      src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
      dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
      m, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
      if m is not None:
        log_homography_matrix(m)
        aligned_image = cv2.warpPerspective(image, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if is_homography_valid(m, w, h):
          aligned_indices.add(len(aligned_images))
        aligned_images.append(aligned_image)
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, m)
        x_min = np.min(transformed_corners[:, :, 0]).item()
        y_min = np.min(transformed_corners[:, :, 1]).item()
        x_max = np.max(transformed_corners[:, :, 0]).item()
        y_max = np.max(transformed_corners[:, :, 1]).item()
        bounding_boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))
      else:
        logger.debug(f"no homography")
        aligned_images.append(image)
    else:
      logger.debug(f"not enough good matches in {len(matches)} matches")
      aligned_images.append(image)
  if bounding_boxes:
    x_min = max(b[0] for b in bounding_boxes)
    y_min = max(b[1] for b in bounding_boxes)
    x_max = min(b[2] for b in bounding_boxes)
    y_max = min(b[3] for b in bounding_boxes)
    x_min = np.clip(x_min, 0, w)
    y_min = np.clip(y_min, 0, h)
    x_max = np.clip(x_max, x_min + 1, w)
    y_max = np.clip(y_max, y_min + 1, h)
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    aspect_ratio = w / h
    cropped_aspect_ratio = crop_w / crop_h
    if cropped_aspect_ratio > aspect_ratio * 1.01:
      new_crop_w = int(crop_h * aspect_ratio)
      x_min = x_min + (crop_w - new_crop_w) // 2
      x_max = x_min + new_crop_w
    elif cropped_aspect_ratio < aspect_ratio * 0.99:
      new_crop_h = int(crop_w / aspect_ratio)
      y_min = y_min + (crop_h - new_crop_h) // 2
      y_max = y_min + new_crop_h
    logger.debug(f"cropping: x_min={x_min}, y_min={y_min},"
                 f" x_max={x_max}, y_max={y_max}")
    cropped_images = [img[y_min:y_max, x_min:x_max] for img in aligned_images]
  else:
    logger.debug(f"no need to crop")
    cropped_images = aligned_images
  return cropped_images


def align_images_sift(images, aligned_indices, nfeatures=30000, denoise=2, shift_limit=0.1):
  """Aligns images using SIFT."""
  assert all(image.dtype == np.float32 for image in images)
  if len(images) < 2:
    return images
  ref_image = images[0]
  h, w = ref_image.shape[:2]
  sides_mean = (h + w) / 2
  check_results = []
  clahe_clip_limits = [2.0, 4.0, 8.0]
  for clahe_clip_limit in clahe_clip_limits:
    ref_gray = make_image_for_alignment(ref_image, clahe_clip_limit, denoise)
    sift = cv2.SIFT_create(nfeatures=0)
    ref_kp, ref_des = sift.detectAndCompute(ref_gray, None)
    if ref_des is None or len(ref_kp) < 10: continue
    score = abs(math.log(nfeatures * 2) - math.log(len(ref_kp)))
    logger.debug(f"checking reference image:"
                 f" clahe={clahe_clip_limit}, denoise={denoise}, kp={len(ref_kp)}")
    check_results.append((score, clahe_clip_limit, ref_gray))
  if not check_results:
    logger.debug("reference image has no descriptors")
    return images
  check_results = sorted(check_results)
  score, clahe_clip_limit, ref_gray = check_results[0]
  sift = cv2.SIFT_create(nfeatures=nfeatures)
  ref_kp, ref_des = sift.detectAndCompute(ref_gray, None)
  check_results = None
  ref_grays = None
  logger.debug(f"best config: clahe={clahe_clip_limit}, denoise={denoise}, kp={len(ref_kp)}")
  aligned_indices.add(0)
  aligned_images = [ref_image]
  bounding_boxes = []
  for image in images[1:]:
    image_gray = make_image_for_alignment(image, clahe_clip_limit, denoise)
    kp, des = sift.detectAndCompute(image_gray, None)
    if des is None:
      logger.debug("image has no descriptors")
      aligned_images.append(image)
      continue
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(ref_des, des, k=2)
    cand_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    good_matches = []
    for m in cand_matches:
      dist = np.linalg.norm(np.array(ref_kp[m.queryIdx].pt) - np.array(kp[m.trainIdx].pt))
      dist /= sides_mean
      if dist > shift_limit: continue
      good_matches.append(m)
    if len(good_matches) > 10:
      logger.debug(f"matches found: {len(good_matches)} of {len(matches)}")
      src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
      dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
      m, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
      if m is not None:
        log_homography_matrix(m)
        aligned_image = cv2.warpPerspective(image, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if is_homography_valid(m, w, h):
          aligned_indices.add(len(aligned_images))
        aligned_images.append(aligned_image)
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, m)
        x_min = np.min(transformed_corners[:, :, 0]).item()
        y_min = np.min(transformed_corners[:, :, 1]).item()
        x_max = np.max(transformed_corners[:, :, 0]).item()
        y_max = np.max(transformed_corners[:, :, 1]).item()
        bounding_boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))
      else:
        logger.debug("no homography found")
        aligned_images.append(image)
    else:
      logger.debug(f"not enough good matches in {len(matches)} matches")
      aligned_images.append(image)
  if bounding_boxes:
    x_min = max(b[0] for b in bounding_boxes)
    y_min = max(b[1] for b in bounding_boxes)
    x_max = min(b[2] for b in bounding_boxes)
    y_max = min(b[3] for b in bounding_boxes)
    x_min = np.clip(x_min, 0, w)
    y_min = np.clip(y_min, 0, h)
    x_max = np.clip(x_max, x_min + 1, w)
    y_max = np.clip(y_max, y_min + 1, h)
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    aspect_ratio = w / h
    cropped_aspect_ratio = crop_w / crop_h
    if cropped_aspect_ratio > aspect_ratio * 1.01:
      new_crop_w = int(crop_h * aspect_ratio)
      x_min = x_min + (crop_w - new_crop_w) // 2
      x_max = x_min + new_crop_w
    elif cropped_aspect_ratio < aspect_ratio * 0.99:
      new_crop_h = int(crop_w / aspect_ratio)
      y_min = y_min + (crop_h - new_crop_h) // 2
      y_max = y_min + new_crop_h
    logger.debug(f"cropping: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
    cropped_images = [img[y_min:y_max, x_min:x_max] for img in aligned_images]
  else:
    logger.debug("no need to crop")
    cropped_images = aligned_images
  return cropped_images


def align_images_ecc(images, aligned_indices, use_affine=True, denoise=3):
  """Aligns images using ECC."""
  assert all(image.dtype == np.float32 for image in images)
  if len(images) < 2:
    return images
  ref_image = images[0]
  h, w = ref_image.shape[:2]
  clahe_clip_limit = 4
  ref_gray = make_image_for_alignment(ref_image, clahe_clip_limit, denoise)
  logger.debug(f"config: clahe={clahe_clip_limit}, use_affine={use_affine}, denoise={denoise}")
  aligned_indices.add(0)
  aligned_images = [ref_image]
  bounding_boxes = []
  for image in images[1:]:
    image_gray = make_image_for_alignment(image, clahe_clip_limit, denoise)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
    try:
      motion_mode = cv2.MOTION_AFFINE if use_affine else cv2.MOTION_TRANSLATION
      cc, warp_matrix = cv2.findTransformECC(
        ref_gray, image_gray, warp_matrix, motion_mode,
        criteria, inputMask=None, gaussFiltSize=5)
      log_homography_matrix(warp_matrix)
      aligned_image = cv2.warpAffine(
        image, warp_matrix, (w, h), flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE)
      if is_homography_valid(warp_matrix, w, h):
        aligned_indices.add(len(aligned_images))
      aligned_images.append(aligned_image)
      corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
      transformed_corners = cv2.transform(corners, warp_matrix)
      x_min = np.min(transformed_corners[:, :, 0]).item()
      y_min = np.min(transformed_corners[:, :, 1]).item()
      x_max = np.max(transformed_corners[:, :, 0]).item()
      y_max = np.max(transformed_corners[:, :, 1]).item()
      bounding_boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))
    except cv2.error:
      logger.debug("alignment failed")
      aligned_images.append(image)
  if bounding_boxes:
    x_min = max(min(b[0] for b in bounding_boxes), 0)
    y_min = max(min(b[1] for b in bounding_boxes), 0)
    x_max = min(max(b[2] for b in bounding_boxes), w)
    y_max = min(max(b[3] for b in bounding_boxes), h)
    x_min = np.clip(x_min, 0, w)
    y_min = np.clip(y_min, 0, h)
    x_max = np.clip(x_max, x_min + 1, w)
    y_max = np.clip(y_max, y_min + 1, h)
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    aspect_ratio = w / h
    cropped_aspect_ratio = crop_w / crop_h
    if cropped_aspect_ratio > aspect_ratio * 1.01:
      new_crop_w = int(crop_h * aspect_ratio)
      x_min += (crop_w - new_crop_w) // 2
      x_max = x_min + new_crop_w
    elif cropped_aspect_ratio < aspect_ratio * 0.99:
      new_crop_h = int(crop_w / aspect_ratio)
      y_min += (crop_h - new_crop_h) // 2
      y_max = y_min + new_crop_h
    logger.debug(f"cropping: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
    cropped_images = [img[y_min:y_max, x_min:x_max] for img in aligned_images]
  else:
    logger.debug("no need to crop")
    cropped_images = aligned_images
  return cropped_images


def align_images_hugin(images, input_paths, bits_list):
  """Aligns images using Hugin."""
  assert all(image.dtype == np.float32 for image in images)
  if len(images) < 2:
    return images
  temp_dir = tempfile.gettempdir()
  pid = os.getpid()
  tmp_paths = []
  prefix = f"itb_stack-{pid:08d}"
  full_prefix = os.path.join(temp_dir, prefix)
  cmd = [CMD_HUGIN_ALIGN, "-C", "-c", "64", "-a", f"{full_prefix}-output-"]
  align_input_paths = []
  for image, input_path, bits in zip(images, input_paths, bits_list):
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in EXTS_IMAGE:
      raise ValueError(f"Unsupported file format: {ext}")
    align_input_path = f"{full_prefix}-input-{len(align_input_paths)+1:04d}{ext}"
    align_input_paths.append(align_input_path)
    save_image(align_input_path, image, bits)
    cmd.append(align_input_path)
  logger.debug(f"running: {' '.join(cmd)}")
  subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  for align_input_path in align_input_paths:
    os.remove(align_input_path)
  aligned_images = []
  for name in os.listdir(temp_dir):
    if not name.startswith(prefix): continue
    align_output_path = os.path.join(temp_dir, name)
    image, bits = load_image(align_output_path)
    aligned_images.append(image)
    os.remove(align_output_path)
  return aligned_images


def adjust_size_images(images):
  """Adjusts the size of images."""
  assert all(image.dtype == np.float32 for image in images)
  max_h = max(image.shape[0] for image in images)
  max_w = max(image.shape[1] for image in images)
  resized_images = []
  for image in images:
    h, w = image.shape[:2]
    if h != max_h or w != max_w:
      image = cv2.resize(image, (max_w, max_h), interpolation=cv2.INTER_LANCZOS4)
    resized_images.append(image)
  return resized_images


def fix_overflown_image(image):
  """Replaces NaN and -inf with 0, and inf with 1."""
  assert image.dtype == np.float32
  return np.clip(np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0), 0, None)


def merge_images_average(images):
  """Merges images by average composition."""
  assert all(image.dtype == np.float32 for image in images)
  return np.mean(images, axis=0)


def merge_images_median(images):
  """Merges images by median composition."""
  assert all(image.dtype == np.float32 for image in images)
  return np.median(images, axis=0)


def merge_images_geometric_mean(images):
  """Merges images by geoemtric_mean composition."""
  assert all(image.dtype == np.float32 for image in images)
  return np.exp(np.mean(np.log(np.clip(images, 1e-6, 1)), axis=0))


def merge_images_minimum(images):
  """Merges images by minimum value composition."""
  assert all(image.dtype == np.float32 for image in images)
  return np.min(images, axis=0)


def merge_images_maximum(images):
  """Merges images by maximum value composition."""
  assert all(image.dtype == np.float32 for image in images)
  return np.max(images, axis=0)


def merge_images_denoise(images, clip_limit=0.4, blur_radius=3):
  """Merge images by blurred geometric mean-based median."""
  assert all(image.dtype == np.float32 for image in images)
  images = np.stack(images, axis=3).astype(np.float32)
  h, w, c, n = images.shape
  ksize = math.ceil(2 * blur_radius) + 1
  sigma_color = min(0.1 * math.sqrt(ksize), 0.5)
  sigma_space = 14 * math.sqrt(ksize)
  filtered_images = np.zeros_like(images, dtype=np.float32)
  for i in range(n):
    filtered_images[:, :, :, i] = cv2.bilateralFilter(
      images[:, :, :, i].astype(np.float32), ksize, sigma_color, sigma_space)
  smooth_image = np.exp(np.mean(np.log(np.clip(filtered_images, 1e-6, 1)), axis=3))
  blurred = cv2.GaussianBlur(smooth_image, (ksize, ksize), 0)
  smooth_image = cv2.addWeighted(smooth_image, 1.8, blurred, -0.8, 0)
  smooth_image = np.clip(smooth_image, 0, 1)
  gray_image = cv2.cvtColor(smooth_image, cv2.COLOR_BGR2GRAY)
  gray_image = np.clip(gray_image, 0, 1)
  sobel_x = np.abs(cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3))
  sobel_y = np.abs(cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3))
  sobel_e = np.sqrt(sobel_x**2 + sobel_y**2)
  edge_score = z_score_normalization(sobel_e)
  log_c = np.log1p(smooth_image)
  log_x = np.log1p(images)
  dist = np.abs(log_x - log_c[..., np.newaxis])
  dist_score = z_score_normalization(dist)
  total_score = edge_score[..., np.newaxis, np.newaxis] - dist_score
  threshold_value = np.percentile(total_score, clip_limit * 100)
  valid_pixels = total_score >= threshold_value
  def conditional_median(stack, valid_mask):
    filtered_stack = np.where(valid_mask, stack, np.nan)
    nan_mask = np.all(np.isnan(filtered_stack), axis=3, keepdims=True)
    filtered_stack = np.where(nan_mask, np.expand_dims(smooth_image, axis=3), filtered_stack)
    median_values = np.nanmedian(filtered_stack, axis=3)
    return median_values
  result = conditional_median(images, valid_pixels)
  return result


def calculate_stf_weights(f_numbers):
  """Calculates weights for each F-numbers for STF."""
  blur_radii = 1 / np.array(f_numbers)
  weights = blur_radii + np.min(blur_radii) / 4
  weights = weights / np.sum(weights)
  return weights


def merge_images_weighted_average(images, meta_list):
  """Merges images by weighted average composition."""
  assert all(image.dtype == np.float32 for image in images)
  f_numbers = [meta.get("_fv_") for meta in meta_list]
  if None in f_numbers:
    logger.debug(f"missing F-numbers")
    return np.mean(images, axis=0)
  logger.debug(f"F-numbers: {f_numbers}")
  weights = calculate_stf_weights(f_numbers)
  logger.debug(f"weights: {weights}")
  return np.average(images, axis=0, weights=weights)


def merge_images_debevec(images, meta_list):
  """Merges images by Debevec's method."""
  assert all(image.dtype == np.float32 for image in images)
  light_values = [get_light_value(meta) for meta in meta_list]
  if None in light_values:
    brightness_values = np.array([compute_brightness(image) for image in images])
    exposures = brightness_values / max(np.min(brightness_values), 0.0001)
  else:
    luminances = np.array([1 / (2 ** lv) for lv in light_values]).astype(np.float32)
    exposures = luminances / max(np.min(luminances), 0.0001)
  byte_images = [(np.clip(image, 0, 1) * 255).astype(np.uint8)
                 for image in images]
  merger = cv2.createMergeDebevec()
  hdr = merger.process(byte_images, times=exposures)
  return hdr


def merge_images_robertson(images, meta_list):
  """Merges images by Robertson's method."""
  assert all(image.dtype == np.float32 for image in images)
  light_values = [get_light_value(meta) for meta in meta_list]
  if None in light_values:
    brightness_values = np.array([compute_brightness(image) for image in images])
    exposures = brightness_values / max(np.min(brightness_values), 0.0001)
  else:
    luminances = np.array([1 / (2 ** lv) for lv in light_values]).astype(np.float32)
    exposures = luminances / max(np.min(luminances), 0.0001)
  byte_images = [(np.clip(image, 0, 1) * 255).astype(np.uint8)
                 for image in images]
  merger = cv2.createMergeRobertson()
  hdr = merger.process(byte_images, times=exposures)
  return hdr


def merge_images_mertens(images):
  """Merges images by Mertens's method."""
  assert all(image.dtype == np.float32 for image in images)
  byte_images = [(np.clip(image, 0, 1) * 255).astype(np.uint8)
                 for image in images]
  merger = cv2.createMergeMertens()
  hdr = merger.process(byte_images)
  hdr = normalize_negative_image(hdr)
  return hdr


def normalize_negative_image(image, clip_percentile=2.0):
  """Normalizes negaive pixels."""
  assert image.dtype == np.float32
  image = cv2.GaussianBlur(image, (5, 5), 0)
  min_val = np.percentile(image, clip_percentile)
  if min_val > -0.01:
    return np.clip(image, 0, None)
  image = np.clip(image, min_val, None)
  image -= min_val
  return np.clip(image, 0, None)


def z_score_normalization(image):
  """Applies Z-score normalization to stabilize feature scaling."""
  assert image.dtype == np.float32
  mean = np.mean(image)
  std = np.std(image)
  return (image - mean) / (std + 1e-6)


def percentile_normalization(image, low=1, high=99):
  """Applies percentile normalization to adjust domain into [0,1]."""
  assert image.dtype == np.float32
  low_val = np.percentile(image, low)
  high_val = np.percentile(image, high)
  image = (image - low_val) / max(high_val - low_val, 1e-6)
  return np.clip(image, 0, 1)


def estimate_white_noise_level(image, num_tiles=400, percentile=10):
  """Estimates white noise level for Laplacian."""
  assert image.dtype == np.float32
  lap = np.abs(cv2.Laplacian(image, cv2.CV_32F, ksize=3))
  h, w = lap.shape
  area = h * w
  tile_unit = max(round(math.sqrt(area) / math.sqrt(num_tiles)), 1)
  tile_size_max = int(tile_unit * 1.5)
  tile_means = []
  x = 0
  while x < w:
    tile_w = tile_unit
    if x + tile_size_max >= w:
      tile_w = w - x
    y = 0
    while y < h:
      tile_h = tile_unit
      if y + tile_size_max >= h:
        tile_h = h - y
      tile = lap[y:y+tile_h, x:x+tile_w]
      if tile.size > 0:
        tile_means.append(np.mean(tile))
      y += tile_h
    x += tile_w
  if not tile_means:
    return 0.0
  tile_means = np.array(tile_means)
  k = max(1, int(len(tile_means) * percentile / 100))
  lowest_k = np.partition(tile_means, k)[:k]
  return float(np.mean(lowest_k))


def estimate_foreground_isolation(image, num_tiles=100):
  """Estimate how easily the foreground can be isolated from the background."""
  assert image.dtype == np.float32
  h, w = image.shape
  area = h * w
  tile_unit = max(round(math.sqrt(area) / math.sqrt(num_tiles)), 1)
  tile_size_max = int(tile_unit * 1.5)
  tile_means = []
  x = 0
  while x < w:
    tile_w = tile_unit
    if x + tile_size_max >= w:
      tile_w = w - x
    y = 0
    while y < h:
      tile_h = tile_unit
      if y + tile_size_max >= h:
        tile_h = h - y
      tile = image[y:y+tile_h, x:x+tile_w]
      if tile.size > 0:
        tile_means.append(np.mean(tile))
      y += tile_h
    x += tile_w
  if not tile_means:
    return 0.0
  values = np.array(tile_means)
  values = values[values >= 0]
  values = np.sort(values + 1e-10)
  n = len(values)
  index = np.arange(1, n + 1)
  gini = ((2 * np.sum(index * values)) / (n * np.sum(values))) - (n + 1) / n
  return float(np.clip(gini, 0.0, 1.0))


def compute_sharpness_naive(
    image, base_area=1000000, blur_radius=2,
    rescale=True, clahe_clip_limit=0.3, high_low_balance=0.5, suppress_noise=0.5):
  """Computes sharpness using normalized Laplacian and Sobel filters."""
  assert image.dtype == np.float32
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  if clahe_clip_limit > 0:
    gray = apply_clahe_gray_image(gray, clahe_clip_limit)
  h, w = gray.shape[:2]
  area = h * w
  is_scaled = False
  if area > base_area * 2:
    scale = (base_area / area) ** 0.5
    gray = cv2.resize(gray, (math.ceil(w * scale), math.ceil(h * scale)),
                      interpolation=cv2.INTER_AREA)
    is_scaled = True
  if blur_radius > 1:
    ksize = math.ceil(2 * blur_radius) + 1
    gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)
  laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
  if suppress_noise > 0:
    noise_floor = min(estimate_white_noise_level(gray), np.mean(laplacian) * 0.5)
    laplacian = np.clip(laplacian - suppress_noise * noise_floor, 0, None)
  laplacian = z_score_normalization(laplacian)
  sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
  sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
  sobel_e = np.sqrt(sobel_x**2 + sobel_y**2)
  sobel = z_score_normalization(sobel_e)
  sharpness = high_low_balance * laplacian + (1 - high_low_balance) * sobel
  if is_scaled and rescale:
    sharpness = cv2.resize(sharpness, (w, h), interpolation=cv2.INTER_LANCZOS4)
  sharpness = z_score_normalization(sharpness)
  return np.clip(sharpness, -10, 10)


def compute_sharpness_adaptive(
    image, base_area=1000000, rescale=True,
    clahe_clip_limit=0.3, suppress_noise=0.9):
  """Computes the best sharpness map with adaptive parameters."""
  assert image.dtype == np.float32
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  if clahe_clip_limit > 0:
    gray = apply_clahe_gray_image(gray, clahe_clip_limit)
  h, w = gray.shape[:2]
  area = h * w
  is_scaled = False
  if area > base_area * 2:
    scale = (base_area / area) ** 0.5
    gray = cv2.resize(gray, (math.ceil(w * scale), math.ceil(h * scale)),
                      interpolation=cv2.INTER_AREA)
    is_scaled = True
  best_score = -1
  best_map_raw = None
  for blur_radius in [0, 1, 2]:
    if blur_radius > 0:
      ksize = math.ceil(2 * blur_radius) + 1
      blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    else:
      blurred = gray
    laplacian = np.abs(cv2.Laplacian(blurred, cv2.CV_32F, ksize=3))
    if suppress_noise > 0:
      noise_floor = min(estimate_white_noise_level(blurred), np.mean(laplacian) * 0.5)
      laplacian = np.clip(laplacian - suppress_noise * noise_floor, 0, None)
    laplacian = z_score_normalization(laplacian)
    sobel_x = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))
    sobel_y = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3))
    sobel_e = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = z_score_normalization(sobel_e)
    for high_low_balance in [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
      sharp = high_low_balance * laplacian + (1 - high_low_balance) * sobel
      isolation = estimate_foreground_isolation(sharp)
      if isolation > best_score:
        best_score = isolation
        best_map_raw = sharp
  best_map = z_score_normalization(best_map_raw)
  if is_scaled and rescale:
    best_map = cv2.resize(best_map, (w, h), interpolation=cv2.INTER_LANCZOS4)
  return np.clip(best_map, -10, 10)


def extract_mean_tiles(image, num_tiles=100, attractor=(0.5, 0.5), attractor_weight=0.1):
  """Extracts tiles with stats from an image."""
  assert image.dtype == np.float32
  h, w = image.shape[:2]
  area = h * w
  tile_unit = max(round(math.sqrt(area) / math.sqrt(num_tiles)), 1)
  tile_size_max = int(tile_unit * 1.5)
  cx = attractor[0] * w
  cy = attractor[1] * h
  tiles = []
  x = 0
  while x < w:
    tile_w = tile_unit
    if x + tile_size_max >= w:
      tile_w = w - x
    tile_column = []
    y = 0
    while y < h:
      tile_h = tile_unit
      if y + tile_size_max >= h:
        tile_h = h - y
      tile = image[y:y + tile_h, x:x + tile_w]
      score = float(np.mean(tile))
      tile_cx = x + tile_w / 2
      tile_cy = y + tile_h / 2
      dx = (tile_cx - cx) / (w / 2)
      dy = (tile_cy - cy) / (h / 2)
      dist2_norm = dx * dx + dy * dy
      weight = math.exp(-attractor_weight * dist2_norm)
      tile_column.append((score * weight, x, y, tile_w, tile_h))
      y += tile_h
    tiles.append(tile_column)
    x += tile_w
  return tiles


def find_best_rect(tiles, max_window_size=5, area_penalty=0.5):
  """Finds the best rectangle of the largest average values."""
  tile_h_count = len(tiles[0])
  tile_w_count = len(tiles)
  best_score = -np.inf
  best_rect = (0, 0, 0, 0)
  for win_h in range(1, max_window_size + 1):
    for win_w in range(1, max_window_size + 1):
      for i in range(tile_w_count - win_w + 1):
        for j in range(tile_h_count - win_h + 1):
          total_score = 0.0
          x0, y0 = tiles[i][j][1], tiles[i][j][2]
          x1, y1 = x0, y0
          for dx in range(win_w):
            for dy in range(win_h):
              score, x, y, w_tile, h_tile = tiles[i + dx][j + dy]
              total_score += score
              x1 = max(x1, x + w_tile)
              y1 = max(y1, y + h_tile)
          num_tiles_in_window = win_w * win_h
          avg_score = total_score / num_tiles_in_window
          weighted_score = avg_score * (num_tiles_in_window ** (1 - area_penalty))
          if weighted_score > best_score:
            best_score = weighted_score
            best_rect = (x0, y0, x1 - x0, y1 - y0)
  return best_rect


def find_good_focus_tiles(tiles, sep_area_balance=0.5, area_penalty=0.75):
  """Finds a good subset of focus tiles combining variance reduction and size-based score."""
  flat_tiles = [tile for col in tiles for tile in col]
  flat_tiles = sorted(flat_tiles, key=lambda t: t[0], reverse=True)
  all_scores = [t[0] for t in flat_tiles]
  total_var = np.var(all_scores)
  selected_scores = []
  non_selected_scores = all_scores.copy()
  best_score = -np.inf
  best_index = 0
  running_total = 0.0
  for i in range(1, len(flat_tiles)):
    score_i = non_selected_scores.pop(0)
    selected_scores.append(score_i)
    running_total += score_i
    p = len(selected_scores) / len(all_scores)
    var_sel = np.var(selected_scores)
    var_non = np.var(non_selected_scores)
    split_variance = p * var_sel + (1 - p) * var_non
    variance_gain = 1.0 - split_variance / (total_var + 1e-6)
    variance_score = variance_gain ** 0.5
    size_score = (running_total / i) * (i ** (1 - area_penalty))
    score = sep_area_balance * variance_score + (1 - sep_area_balance) * size_score
    if score > best_score:
      best_score = score
      best_index = i
  return flat_tiles[:best_index]


def compute_centroid(image, rect):
  """Computes the centroid of a region."""
  x, y, w, h = rect
  roi = image[y:y+h, x:x+w]
  if roi.ndim == 3:
    roi = roi.mean(axis=2)
  rh, rw = roi.shape
  ry, rx = np.meshgrid(np.arange(y, y+rh), np.arange(x, x+rw), indexing='ij')
  total_weight = np.sum(roi)
  if total_weight < 1e-6:
    return x + w / 2, y + h / 2
  cx = np.sum(rx * roi) / total_weight
  cy = np.sum(ry * roi) / total_weight
  return float(cx), float(cy)


def draw_rectangle(image, x, y, width, height, thickness=0, color=(0.5, 0.5, 0.5)):
  """Draws a rectangle."""
  assert image.dtype == np.float32
  h, w = image.shape[:2]
  if not thickness:
    thickness = max(2, int(((h * w) ** 0.5) / 256))
  pt1 = (int(x), int(y))
  pt2 = (int(x + width), int(y + height))
  color = color[2], color[1], color[0]
  image = cv2.rectangle(image, pt1, pt2, color, thickness)
  return image


def larger_rect(rect, area_ratio, w, h):
   """Compute a rectangle within the boundary."""
   x, y, rw, rh = rect
   cx = x + rw / 2
   cy = y + rh / 2
   scale = math.sqrt(area_ratio)
   new_w = rw * scale
   new_h = rh * scale
   new_x = max(0, int(round(cx - new_w / 2)))
   new_y = max(0, int(round(cy - new_h / 2)))
   new_x2 = min(w, int(round(cx + new_w / 2)))
   new_y2 = min(h, int(round(cy + new_h / 2)))
   return new_x, new_y, new_x2 - new_x, new_y2 - new_y


def compute_focus_grabcut(image, attractor=(0.5, 0.5), attractor_weight=0.1,
                          rect_closed=1.0, rect_open=1.0, tiles_closed=1.0, tiles_open=1.0,
                          use_rms=True, sharpness_percentile=95):
  """Computes focus mask using multiple GrabCut strategies and blends them."""
  assert image.dtype == np.float32
  h, w = image.shape[:2]
  sharpness_map = compute_sharpness_adaptive(image)
  sharpness_map = percentile_normalization(sharpness_map, 2, 98)
  small_h, small_w = sharpness_map.shape[:2]
  if attractor_weight < 0:
    unit_size = (small_w * small_h) ** 0.5
    rect_w = int(unit_size // 4)
    rect_h = int(unit_size // 4)
    rect = (round(attractor[0] * small_w) - rect_w // 2,
            round(attractor[1] * small_h) - rect_h // 2,
            rect_w, rect_h)
    good_tiles = []
  else:
    tiles = extract_mean_tiles(
      sharpness_map, attractor=attractor, attractor_weight=attractor_weight)
    rect = find_best_rect(tiles)
    good_tiles = find_good_focus_tiles(tiles)
  rect_large = larger_rect(rect, 2, small_w, small_h)
  centroid = compute_centroid(sharpness_map, rect)
  centroid_rect = center_rect(rect_large, centroid, 0.5)
  small_area = small_h * small_w
  base_area = 100000
  if base_area < small_area:
    base_scale = (base_area / small_area) ** 0.5
    small_w = round(small_w * base_scale)
    small_h = round(small_h * base_scale)
    small_image = cv2.resize(image, (small_w, small_h),
                             interpolation=cv2.INTER_AREA)
    centroid_rect = [round(x * base_scale) for x in centroid_rect]
    rect_large = [round(x * base_scale) for x in rect_large]
    good_tiles = [(s, round(x * base_scale), round(y * base_scale),
                   round(w * base_scale), round(h * base_scale))
                  for s, x, y, w, h in good_tiles]
    sharpness_map = cv2.resize(sharpness_map, (small_w, small_h),
                               interpolation=cv2.INTER_LINEAR)
  else:
    small_image = image
  sharp_threshold = np.percentile(sharpness_map, sharpness_percentile)
  byte_image = (np.clip(small_image * 255, 0, 255)).astype(np.uint8)
  all_masks = []
  total_weight = 0.0
  def run_grabcut_with_rect(rect, definite_background=False):
    x, y, rw, rh = rect
    if definite_background:
      mask = np.full((small_h, small_w), cv2.GC_BGD, np.uint8)
      mask[y:y+rh, x:x+rw] = cv2.GC_PR_FGD
      mask[(mask == cv2.GC_BGD) & (sharpness_map > sharp_threshold)] = cv2.GC_PR_BGD
    else:
      mask = np.full((small_h, small_w), cv2.GC_PR_BGD, np.uint8)
      mask[y:y+rh, x:x+rw] = cv2.GC_PR_FGD
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(byte_image, mask, None,
                bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    mask_binary = np.where((mask == cv2.GC_FGD) |
                           (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    return mask_binary
  def run_grabcut_with_tiles(tile_list, definite_background=False):
    if definite_background:
      mask = np.full((small_h, small_w), cv2.GC_BGD, np.uint8)
      for s, x, y, tw, th in tile_list:
        mask[y:y+th, x:x+tw] = cv2.GC_PR_FGD
      mask[(mask == cv2.GC_BGD) & (sharpness_map > sharp_threshold)] = cv2.GC_PR_BGD
    else:
      mask = np.full((small_h, small_w), cv2.GC_PR_BGD, np.uint8)
      for s, x, y, tw, th in tile_list:
        mask[y:y+th, x:x+tw] = cv2.GC_PR_FGD
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(byte_image, mask, None,
                bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    mask_binary = np.where((mask == cv2.GC_FGD) |
                           (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    return mask_binary
  def apply_area_penalty(weight, mask, threshold=0.10):
    fg_ratio = np.mean(mask)
    if fg_ratio <= threshold:
      return weight
    penalty = 1.0 - (fg_ratio - threshold) / (1.0 - threshold)
    return weight * penalty
  if rect_closed > 0 and rect:
    mask = run_grabcut_with_rect(centroid_rect, definite_background=True)
    weight = apply_area_penalty(rect_closed, mask)
    all_masks.append((weight, mask))
    total_weight += weight
  if rect_open > 0 and rect:
    mask = run_grabcut_with_rect(rect_large, definite_background=False)
    weight = apply_area_penalty(rect_open, mask)
    all_masks.append((weight, mask))
    total_weight += weight
  if tiles_closed > 0 and good_tiles:
    mask = run_grabcut_with_tiles(good_tiles, definite_background=True)
    weight = apply_area_penalty(tiles_closed, mask)
    all_masks.append((weight, mask))
    total_weight += weight
  if tiles_open > 0 and good_tiles:
    mask = run_grabcut_with_tiles(good_tiles, definite_background=False)
    weight = apply_area_penalty(tiles_open, mask)
    all_masks.append((weight, mask))
    total_weight += weight
  if total_weight == 0:
    return np.zeros((h, w), np.float32)
  combined = np.zeros((small_h, small_w), np.float32)
  if use_rms:
    numerator = np.zeros((small_h, small_w), np.float32)
    for weight, mask in all_masks:
      numerator += weight * (mask ** 2)
    combined = np.sqrt(numerator / (total_weight + 1e-8))
  else:
    for weight, mask in all_masks:
      combined += weight * mask
    combined /= total_weight
  mask_soft = cv2.GaussianBlur(combined, (11, 11), 0)
  mask_fullsize = cv2.resize(mask_soft, (w, h), interpolation=cv2.INTER_LINEAR)
  return np.clip(mask_fullsize, 0, 1)


def make_gaussian_pyramid(image, levels):
  """Generate Gaussian pyramid."""
  assert image.dtype == np.float32
  pyramid = [image]
  for _ in range(levels):
    image = cv2.pyrDown(image)
    pyramid.append(image)
  return pyramid


def make_laplacian_pyramid(image, levels):
  """Generate Laplacian pyramid."""
  assert image.dtype == np.float32
  gaussian_pyr = make_gaussian_pyramid(image, levels)
  laplacian_pyr = []
  for i in range(levels):
    size = (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0])
    expanded = cv2.pyrUp(gaussian_pyr[i+1], dstsize=size)
    laplacian_pyr.append(cv2.subtract(gaussian_pyr[i], expanded))
  laplacian_pyr.append(gaussian_pyr[-1])
  return laplacian_pyr


def merge_images_laplacian_pyramids_focus(images, weights, pyramid_levels):
  """Merges images by Laplacian Pyramids for focus stacking."""
  assert all(image.dtype == np.float32 for image in images)
  weight_pyramids = [make_gaussian_pyramid(weights[i], pyramid_levels)
                     for i in range(len(images))]
  laplacian_pyramids = [make_laplacian_pyramid(images[i], pyramid_levels)
                        for i in range(len(images))]
  blended_pyramids = []
  for level in range(pyramid_levels+1):
    blended = np.zeros_like(laplacian_pyramids[0][level])
    for i in range(len(images)):
      weight_resized = np.expand_dims(weight_pyramids[i][level], axis=-1)
      blended += laplacian_pyramids[i][level] * weight_resized
    blended_pyramids.append(blended)
  stacked_image = blended_pyramids[-1]
  for i in range(len(blended_pyramids) - 2, -1, -1):
    size = (blended_pyramids[i].shape[1], blended_pyramids[i].shape[0])
    stacked_image = cv2.pyrUp(stacked_image, dstsize=size)
    stacked_image = cv2.add(stacked_image, blended_pyramids[i])
  return stacked_image


def merge_images_focus_stacking(images, smoothness=0.5, pyramid_levels=8):
  """Merges images by focus stacking."""
  assert all(image.dtype == np.float32 for image in images)
  h, w = images[0].shape[:2]
  pyramid_levels = min(pyramid_levels, int(math.log2(min(h, w))) - 3)
  sharpness_maps = np.array([compute_sharpness_naive(img) for img in images])
  images_array = np.stack(images, axis=0)
  if smoothness <= 0:
    best_focus_index = np.argmax(sharpness_maps, axis=0)
    stacked_image = images_array[best_focus_index, np.arange(h)[:, None], np.arange(w)]
    return np.clip(stacked_image, 0, 1)
  weights = sharpness_maps.copy()
  weights -= np.max(weights, axis=0, keepdims=True)
  tau = max(np.std(weights) * smoothness, 1e-4)
  weights = np.exp(weights / tau)
  weights = weights / (np.sum(weights, axis=0, keepdims=True) + 1e-8)
  if pyramid_levels <= 1 or min(h, w) < 256:
    stacked_image = np.sum(weights[..., np.newaxis] * images_array, axis=0)
    return np.clip(stacked_image, 0, 1)
  factor = 2 ** pyramid_levels
  new_h = ((h + factor - 1) // factor) * factor
  new_w = ((w + factor - 1) // factor) * factor
  expanded_images = np.array([
    cv2.copyMakeBorder(img, 0, new_h - h, 0, new_w - w, cv2.BORDER_REPLICATE)
    for img in images
  ])
  expanded_weights = np.array([
    cv2.copyMakeBorder(wt, 0, new_h - h, 0, new_w - w, cv2.BORDER_REPLICATE)
    for wt in weights
  ])
  expanded_stacked = merge_images_laplacian_pyramids_focus(
    expanded_images, expanded_weights, pyramid_levels)
  trimmed = expanded_stacked[:h, :w]
  return np.clip(trimmed, 0, 1)


def merge_images_grid(images, columns=1, margin=0, background=(0.5, 0.5, 0.5)):
  """Merges images in a grid."""
  assert all(image.dtype == np.float32 for image in images)
  num_images = len(images)
  rows = (num_images + columns - 1) // columns
  widths = [0] * columns
  heights = [0] * rows
  for i, img in enumerate(images):
    row, col = divmod(i, columns)
    h, w = img.shape[:2]
    widths[col] = max(widths[col], w)
    heights[row] = max(heights[row], h)
  grid_width = sum(widths) + (columns + 1) * margin
  grid_height = sum(heights) + (rows + 1) * margin
  background = background[2], background[1], background[0]
  output = np.full((grid_height, grid_width, 3), background, dtype=np.float32)
  y_offset = margin
  for row in range(rows):
    x_offset = margin
    for col in range(columns):
      idx = row * columns + col
      if idx >= num_images:
        continue
      h, w = images[idx].shape[:2]
      output[y_offset:y_offset+h, x_offset:x_offset+w] = images[idx]
      x_offset += widths[col] + margin
    y_offset += heights[row] + margin
  return output


def merge_images_stitch(images):
  """Stitches images as a panoramic photo and removes black margins."""
  assert all(image.dtype == np.float32 for image in images)
  byte_images = [(np.clip(image, 0, 1) * 255).astype(np.uint8) for image in images]
  stitcher = cv2.Stitcher_create()
  status, stitched_image = stitcher.stitch(byte_images)
  if status != cv2.Stitcher_OK:
    raise ValueError(f"Stitching failed with status {status}")
  return np.clip(stitched_image.astype(np.float32) / 255, 0, 1)


def tone_map_image_linear(image):
  """Applies tone mapping by linear normalization."""
  assert image.dtype == np.float32
  return cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)


def tone_map_image_reinhard(image):
  """Applies tone mapping by Reinhard's method."""
  assert image.dtype == np.float32
  tonemap = cv2.createTonemapReinhard(gamma=1.0, intensity=0, light_adapt=0.5,
                                      color_adapt=0.5)
  ldr = np.clip(tonemap.process(image), 0, 1)
  return ldr


def tone_map_image_drago(image):
  """Applies tone mapping by Drago's method."""
  assert image.dtype == np.float32
  tonemap = cv2.createTonemapDrago(gamma=1.0, saturation=1.0, bias=0.9)
  ldr = np.clip(tonemap.process(image), 0, 1)
  return ldr


def tone_map_image_mantiuk(image):
  """Applies tone mapping by Mantiuk's method."""
  assert image.dtype == np.float32
  tonemap = cv2.createTonemapMantiuk(gamma=1.0, scale=0.9, saturation=1.0)
  ldr = np.clip(tonemap.process(image), 0, 1)
  return ldr


def fill_black_margin_image(image):
  """Fills black margin on the sides with neighbor colors."""
  assert image.dtype == np.float32
  padding = 10
  padded = cv2.copyMakeBorder(image, padding, padding, padding, padding,
                              borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
  byte_image = (padded * 255).astype(np.uint8)
  undo_bytes = byte_image.astype(np.float32)
  float_ratio = np.where(byte_image > 0, undo_bytes / (byte_image + 1e-6), padded)
  gray = cv2.cvtColor(byte_image, cv2.COLOR_BGR2GRAY)
  gray = np.clip(gray, 0, 1)
  h, w = gray.shape
  mask = np.zeros((h + 2, w + 2), np.uint8)
  for x in range(w):
    if gray[0, x] == 0:
      cv2.floodFill(gray, mask, (x, 0), 255, flags=4)
    if gray[h - 1, x] == 0:
      cv2.floodFill(gray, mask, (x, h - 1), 255, flags=4)
  for y in range(h):
    if gray[y, 0] == 0:
      cv2.floodFill(gray, mask, (0, y), 255, flags=4)
    if gray[y, w - 1] == 0:
      cv2.floodFill(gray, mask, (w - 1, y), 255, flags=4)
  black_margin_mask = (mask[1:-1, 1:-1] == 1).astype(np.uint8)
  inpainted = cv2.inpaint(byte_image, black_margin_mask, 5, cv2.INPAINT_TELEA)
  inpainted = cv2.GaussianBlur(inpainted, (5, 5), 0)
  inpainted = np.clip(inpainted.astype(np.float32) / 255.0, 0, 1)
  restored = np.where(black_margin_mask[:, :, None] == 1, inpainted, padded)
  corrected = restored / np.maximum(float_ratio, 1e-6)
  corrected = np.where((restored == 0) | (float_ratio < 0.5), restored, corrected)
  restored = np.clip(corrected, 0, 1)
  return restored[padding:-padding, padding:-padding]


def apply_global_histeq_image(image, gamma=2.2, white_level=255, restore_color=True):
  """Applies global histogram equalization contrast enhancement."""
  assert image.dtype == np.float32
  lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
  l, a, b = cv2.split(lab)
  l = np.clip(l, 0, 100)
  a = np.clip(a, -128, 127)
  b = np.clip(b, -128, 127)
  l = np.power(l / 100, 1 / gamma) * white_level
  byte_l = (l).astype(np.uint8)
  undo_bytes = byte_l.astype(np.float32)
  float_ratio = np.where(byte_l > 0, undo_bytes / (l + 1e-6), l)
  new_l = cv2.equalizeHist(byte_l).astype(np.float32)
  new_l = np.power(new_l / white_level, gamma) * 100
  corrected = new_l / np.maximum(float_ratio, 1e-6)
  corrected = np.where((new_l == 0) | (float_ratio < 0.5), new_l, corrected)
  new_l = np.clip(corrected, 0, 100)
  lab = cv2.merge((new_l, a, b))
  enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
  if restore_color:
    old_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, old_s, old_v = cv2.split(old_hsv)
    new_hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
    new_h, _, new_v = cv2.split(new_hsv)
    s_estimate_ratio = (1 - new_v) / (1 - old_v + 1e-10)
    s_estimate_ratio = np.clip(s_estimate_ratio, 1 / 8, 8)
    merged_s = np.clip(old_s * (s_estimate_ratio ** 0.5), 0, 1)
    final_hsv = cv2.merge((new_h, merged_s, new_v))
    enhanced_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
  return np.clip(enhanced_image, 0, 1)


def apply_clahe_image(image, clip_limit, gamma=2.2, white_level=245, restore_color=True):
  """Applies CLAHE contrast enhancement."""
  assert image.dtype == np.float32
  lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
  l, a, b = cv2.split(lab)
  l = np.clip(l, 0, 100)
  a = np.clip(a, -128, 127)
  b = np.clip(b, -128, 127)
  l = np.power(l / 100, 1 / gamma) * white_level
  byte_l = (l).astype(np.uint8)
  undo_bytes = byte_l.astype(np.float32)
  float_ratio = np.where(byte_l > 0, undo_bytes / (l + 1e-6), l)
  tile_grid_size = (8, 8)
  clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
  new_l = clahe.apply(byte_l).astype(np.float32)
  new_l = np.power(new_l / white_level, gamma) * 100
  corrected = new_l / np.maximum(float_ratio, 1e-6)
  corrected = np.where((new_l == 0) | (float_ratio < 0.5), new_l, corrected)
  new_l = np.clip(corrected, 0, 100)
  lab = cv2.merge((new_l, a, b))
  enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
  if restore_color:
    old_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, old_s, old_v = cv2.split(old_hsv)
    new_hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
    new_h, _, new_v = cv2.split(new_hsv)
    s_estimate_ratio = (1 - new_v) / (1 - old_v + 1e-10)
    s_estimate_ratio = np.clip(s_estimate_ratio, 1 / 8, 8)
    merged_s = np.clip(old_s * (s_estimate_ratio ** 0.5), 0, 1)
    final_hsv = cv2.merge((new_h, merged_s, new_v))
    enhanced_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
  return np.clip(enhanced_image, 0, 1)


def saturate_colors_image(image, factor):
  """Saturates colors of the image by applying a scaled log transformation."""
  assert image.dtype == np.float32
  if factor > 1e-6:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = (np.log1p(s * factor) / np.log1p(factor)).astype(np.float32)
    hsv = cv2.merge((h, s, v))
    mod_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
  elif factor < -1e-6:
    factor = -factor
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = ((np.expm1(s * np.log1p(factor))) / factor).astype(np.float32)
    hsv = cv2.merge((h, s, v))
    mod_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
  return mod_image.astype(np.float32)


def apply_artistic_filter_image(image, name):
  """Applies an artistic filter."""
  assert image.dtype == np.float32
  name = name.strip().lower()
  if name == "pencil":
    gamma = 2.2
    gamma_image = np.power(image, 1 / gamma)
    byte_image = (gamma_image * 255).astype(np.uint8)
    undo_bytes = byte_image.astype(np.float32)
    float_ratio = np.where(byte_image > 0, undo_bytes / (byte_image + 1e-6), gamma_image)
    _, converted = cv2.pencilSketch(byte_image)
    restored = converted.astype(np.float32) / 255
    corrected = restored / np.maximum(float_ratio, 1e-6)
    corrected = np.where((restored == 0) | (float_ratio < 0.5), restored, corrected)
    restored = np.clip(corrected, 0, 1)
    image = np.clip(np.power(restored, gamma), 0, 1)
  elif name == "stylized":
    gamma = 2.2
    gamma_image = np.power(image, 1 / gamma)
    byte_image = (gamma_image * 255).astype(np.uint8)
    undo_bytes = byte_image.astype(np.float32)
    float_ratio = np.where(byte_image > 0, undo_bytes / (byte_image + 1e-6), gamma_image)
    converted = cv2.stylization(byte_image)
    restored = converted.astype(np.float32) / 255
    corrected = restored / np.maximum(float_ratio, 1e-6)
    corrected = np.where((restored == 0) | (float_ratio < 0.5), restored, corrected)
    restored = np.clip(corrected, 0, 1)
    image = np.clip(np.power(restored, gamma), 0, 1)
  elif name == "oil":
    gamma = 2.2
    gamma_image = np.power(image, 1 / gamma)
    byte_image = (gamma_image * 255).astype(np.uint8)
    undo_bytes = byte_image.astype(np.float32)
    float_ratio = np.where(byte_image > 0, undo_bytes / (byte_image + 1e-6), gamma_image)
    converted = cv2.xphoto.oilPainting(byte_image, size=7, dynRatio=1)
    restored = converted.astype(np.float32) / 255
    corrected = restored / np.maximum(float_ratio, 1e-6)
    corrected = np.where((restored == 0) | (float_ratio < 0.5), restored, corrected)
    restored = np.clip(corrected, 0, 1)
    image = np.clip(np.power(restored, gamma), 0, 1)
  elif name == "cartoon":
    gamma = 1.2
    gamma_image = np.power(image, 1 / gamma)
    byte_image = (gamma_image * 255).astype(np.uint8)
    undo_bytes = byte_image.astype(np.float32)
    float_ratio = np.where(byte_image > 0, undo_bytes / (byte_image + 1e-6), gamma_image)
    gray = cv2.cvtColor(byte_image, cv2.COLOR_BGR2GRAY)
    gray = np.clip(gray, 0, 1)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)
    color = cv2.bilateralFilter(byte_image, 9, 250, 250)
    converted = cv2.bitwise_and(color, color, mask=edges).astype(np.float32)
    restored = converted.astype(np.float32) / 255
    corrected = restored / np.maximum(float_ratio, 1e-6)
    corrected = np.where((restored == 0) | (float_ratio < 0.5), restored, corrected)
    restored = np.clip(corrected, 0, 1)
    image = np.clip(np.power(restored, gamma), 0, 1)
  else:
    raise ValueError(f"Unknown artistic filter: {name}")
  return image


def convert_grayscale_image(image, expr):
  """Converts the image into grayscale."""
  assert image.dtype == np.float32
  params = parse_name_opts_expression(expr)
  name = params["name"]
  confs = [(("bt601", "601", "gray"), (0.299, 0.587, 0.114)),
           (("bt709", "709"), (0.2126, 0.7152, 0.0722)),
           (("bt2020", "2020"), (0.2627, 0.6780, 0.0593)),
           (("red", "r"), (1.0, 0.41, 0.08)),
           (("orange", "o"), (1.0, 0.83, 0.166)),
           (("yellow", "y"), (0.6, 1.0, 0.2)),
           (("green", "g"), (0.3, 1.0, 0.2)),
           (("blue", "b"), (0.2, 0.5, 1.0)),
           (("mean", "m"), (1.0, 1.0, 1.0))]
  color_map = None
  for conf in confs:
    if name in conf[0]:
      color_map = conf[1]
      break
  if color_map:
    sum_ratio = sum(color_map)
    weights = np.array([x / sum_ratio for x in color_map])
    gray_image = np.dot(image[..., :3], weights).astype(np.float32)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return np.clip(gray_image, 0, 1)
  elif name in ["lab", "luminance"]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    l = np.clip(l, 0, 100)
    gray_image = l / 100
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return np.clip(gray_image, 0, 1)
  elif name in ["hsv", "value"]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    gray_image = np.clip(v, 0, 1)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return np.clip(gray_image, 0, 1)
  elif name in ["hsl", "lightness"]:
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    _, l, _ = cv2.split(hls)
    gray_image = np.clip(l, 0, 1)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return np.clip(gray_image, 0, 1)
  elif name in ["laplacian"]:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.clip(gray_image, 0, 1)
    gray_image = np.abs(cv2.Laplacian(gray_image, cv2.CV_32F))
    gray_image = normalize_edge_image(gray_image)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return np.clip(gray_image, 0, 1)
  elif name in ["sobel"]:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.clip(gray_image, 0, 1)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    gray_image = cv2.magnitude(sobel_x, sobel_y)
    gray_image = normalize_edge_image(gray_image)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return np.clip(gray_image, 0, 1)
  elif name in ["stddev"]:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.clip(gray_image, 0, 1)
    square_image = gray_image ** 2
    level = max(1, compute_levels_blur_image_portrait(gray_image) - 2)
    mean = blur_image_pyramid(gray_image, level)
    mean_sq = blur_image_pyramid(square_image, level)
    stddev = np.sqrt(np.clip(mean_sq - mean**2, 1e-6, None))
    gray_image = normalize_edge_image(stddev)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return np.clip(gray_image, 0, 1)
  elif name in ["sharpness"]:
    if "adaptive" in params:
      gray_image = compute_sharpness_adaptive(image)
    else:
      gray_image = compute_sharpness_naive(image)
    gray_image = normalize_edge_image(gray_image)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return np.clip(gray_image, 0, 1)
  elif name in ["focus"]:
    gray_image = compute_sharpness_adaptive(image)
    gray_image = normalize_edge_image(gray_image)
    h, w = gray_image.shape[:2]
    attractor = parse_coordinate(params.get("attractor") or "0.5,0.5")
    attractor_weight = float(params.get("attractor_weight") or "0")
    if float(attractor_weight) < 0:
      tiles = []
      unit_size = (w * h) ** 0.5
      rect_w = int(unit_size // 4)
      rect_h = int(unit_size // 4)
      rect = (round(attractor[0] * w) - rect_w // 2, round(attractor[1] * h) - rect_h // 2,
              rect_w, rect_h)
    else:
      kwargs = {}
      copy_param_to_kwargs(params, kwargs, "attractor", parse_coordinate)
      copy_param_to_kwargs(params, kwargs, "attractor_weight", float)
      tiles = extract_mean_tiles(gray_image, **kwargs)
      rect = find_best_rect(tiles)

    rect_large = larger_rect(rect, 2, w, h)
    centroid = compute_centroid(gray_image, rect)
    centroid_rect = center_rect(rect_large, centroid, 0.5)
    color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    color_image = draw_rectangle(color_image, *rect, 0, (1, 0, 0))
    color_image = draw_rectangle(color_image, *rect_large, 0, (0, 0, 1))
    color_image = draw_rectangle(color_image, *centroid_rect, 0, (0, 1, 0))
    good_tiles = find_good_focus_tiles(tiles)
    for tile in good_tiles:
      tile_rect = tile[1:]
      color_image = draw_rectangle(color_image, *tile_rect, 2, (0, 0.5, 0.5))
    return np.clip(color_image, 0, 1)
  elif name in ["lcs"]:
    color_image = convert_image_lcs(image)
    if "tricolor" in params:
      color_image = convert_image_lcs_tricolor(color_image)
    return np.clip(color_image, 0, 1)
  elif name in ["grabcut"]:
    gray_image = compute_sharpness_adaptive(image)
    gray_image = normalize_edge_image(gray_image)
    color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    confs = [
      ("rect_closed", (0.5, 0, 0)),
      ("rect_open",   (0.5, 0, 0)),
      ("tiles_closed",(0, 0, 0.5)),
      ("tiles_open",  (0, 0, 0.5)),
    ]
    overlay = np.zeros_like(color_image, dtype=np.float32)
    for name_mode, color in confs:
      kwargs = {"rect_closed": 0.0, "rect_open": 0.0, "tiles_closed": 0.0, "tiles_open": 0.0}
      kwargs[name_mode] = 1.0
      copy_param_to_kwargs(params, kwargs, "attractor", parse_coordinate)
      copy_param_to_kwargs(params, kwargs, "attractor_weight", float)
      mask = compute_focus_grabcut(image, **kwargs)
      color_bgr = (color[2], color[1], color[0])
      for c in range(3):
        overlay[..., c] += mask * color_bgr[c]
    color_image = 0.5 * color_image + 0.5 * overlay
    return np.clip(color_image, 0, 1)
  raise ValueError(f"Unknown grayscale name: {name}")


def convert_image_lcs(image):
  """Convert BGR image into LCS pseudo-color space."""
  assert image.dtype == np.float32
  sharp = compute_sharpness_adaptive(image)
  sharp = percentile_normalization(sharp, 2, 98)
  lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
  l, a, b = cv2.split(lab)
  l = np.clip(l, 0, 100)
  a = np.clip(a, -128, 127)
  b = np.clip(b, -128, 127)
  l_norm = percentile_normalization(l, 2, 98)
  sharp_norm = percentile_normalization(sharp, 2, 98)
  chroma = np.sqrt(a ** 2 + b ** 2)
  chroma_norm = percentile_normalization(chroma, 2, 98)
  lcs_image = cv2.merge([l_norm, chroma_norm, sharp_norm])
  return np.clip(lcs_image, 0, 1)


def convert_image_lcs_tricolor(image):
  assert image.dtype == np.float32
  """Convert LCS pseudo-color space into tricolor BGR."""
  l, c, s = cv2.split(image)
  h_red = np.full_like(l, 0)
  l_red = l
  s_red = s
  hls_red = cv2.merge([h_red, l_red, s_red])
  rgb_red = cv2.cvtColor(hls_red, cv2.COLOR_HLS2BGR)
  h_blue = np.full_like(l, 200)
  l_blue = l
  s_blue = c
  hls_blue = cv2.merge([h_blue, l_blue, s_blue])
  rgb_blue = cv2.cvtColor(hls_blue, cv2.COLOR_HLS2BGR)
  rgb_combined = (rgb_red + rgb_blue) / 2
  hls_combined = cv2.cvtColor(rgb_combined, cv2.COLOR_BGR2HLS)
  new_s = np.clip(((s_red ** 2 + s_blue ** 2) / 2) ** 0.5, 0, 1)
  new_h, new_l, _ = cv2.split(hls_combined)
  hls_combined = cv2.merge([new_h, new_l, new_s])
  rgb_restored = cv2.cvtColor(hls_combined, cv2.COLOR_HLS2BGR)
  return np.clip(rgb_restored, 0, 1)


def center_rect(rect, center, area_ratio):
  """Compute a rectangle around the center."""
  x, y, w, h = rect
  if center is None:
    cx, cy = x + w / 2, y + h / 2
  else:
    cx, cy = center
  scale = math.sqrt(area_ratio)
  rel_cx = (cx - x) / w
  rel_cy = (cy - y) / h
  new_w = w * scale
  new_h = h * scale
  new_x = cx - rel_cx * new_w
  new_y = cy - rel_cy * new_h
  return round(new_x), round(new_y), round(new_w), round(new_h)


def normalize_edge_image(image):
  """Normalized edge image to be [0,1]."""
  assert image.dtype == np.float32
  image = z_score_normalization(image)
  image = percentile_normalization(image, 2, 98)
  return np.clip(image, 0, 1)


def bilateral_denoise_image(image, radius):
  """Applies bilateral denoise."""
  assert image.dtype == np.float32
  ksize = math.ceil(2 * radius) + 1
  sigma_color = min(0.05 * math.sqrt(ksize), 0.35)
  sigma_space = 10 * math.sqrt(ksize)
  return cv2.bilateralFilter(image, ksize, sigma_color, sigma_space)


def blur_image_gaussian(image, radius):
  """Applies Gaussian blur."""
  assert image.dtype == np.float32
  ksize = math.ceil(2 * radius) + 1
  return cv2.GaussianBlur(image, (ksize, ksize), 0)




def circular_blur_kernel(radius):
  """Creates a normalized circular (disc-shaped) kernel."""
  size = radius * 2 + 1
  y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
  mask = x**2 + y**2 <= radius**2
  kernel = np.zeros((size, size), dtype=np.float32)
  kernel[mask] = 1
  kernel /= np.sum(kernel)
  return kernel

def ring_blur_kernel(radius_outer, radius_inner=0):
  size = radius_outer * 2 + 1
  center = radius_outer
  Y, X = np.ogrid[:size, :size]
  dist = np.sqrt((X - center) ** 2 + (Y - center) ** 2)
  kernel = np.where((dist >= radius_inner) & (dist <= radius_outer), 1.0, 0.0)
  kernel /= np.sum(kernel)
  return kernel.astype(np.float32)


def circular_blur(image, radius=5):
  """
  Applies a circular disc-shaped blur, similar to optical bokeh.
  radius: radius of the circular kernel, controls strength.
  """
  kernel = circular_blur_kernel(radius)
  #kernel = ring_blur_kernel(radius)
  blurred = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)
  return blurred


def pyramid_down(image):
  tx, ty = 0.5, 0.5
  m = np.float32([[1, 0, tx], [0, 1, ty]])
  image = cv2.warpAffine(image, m, (image.shape[1], image.shape[0]),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
  resized = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
  resized = cv2.GaussianBlur(resized, (5, 5), sigmaX=0.45, sigmaY=0.45)
  return resized


def pyramid_up(image):
  image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
  tx, ty = -0.5, -0.5
  m = np.float32([[1, 0, tx], [0, 1, ty]])
  image = cv2.warpAffine(image, m, (image.shape[1], image.shape[0]),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
  image = cv2.GaussianBlur(image, (3, 3), sigmaX=0.3, sigmaY=0.3)
  return image




def make_gaussian_pyramid_resize(image, levels):
  assert image.dtype == np.float32
  pyramid = [image]
  for _ in range(levels):
    image = pyramid_down(image)
    pyramid.append(image)
  return pyramid


  #pyramid = make_gaussian_pyramid_resize(expanded, levels)
  #diffused = resize_up(diffused)



def blur_image_pyramid(image, levels, decay=0.0, contrast=1.0):
  """Applies pyramid blur."""
  assert image.dtype == np.float32
  h, w = image.shape[:2]
  levels = min(levels, int(math.log2(min(h, w))) - 1)
  factor = 2 ** levels
  new_h = ((h + factor - 1) // factor) * factor
  new_w = ((w + factor - 1) // factor) * factor
  expanded = cv2.copyMakeBorder(image, 0, new_h - h, 0, new_w - w, cv2.BORDER_REPLICATE)
  bokehs = [2 ** i for i in range(0, levels)]
  alpha_weights = [b / bokehs[-1] for b in reversed(bokehs)]
  pyramid = make_gaussian_pyramid(expanded, levels)
  diffused = pyramid[-1]
  for i in range(levels - 1, -1, -1):
    size = (pyramid[i].shape[1], pyramid[i].shape[0])
    diffused = cv2.pyrUp(diffused, dstsize=size)
    gamma_decay = decay ** 2
    alpha = alpha_weights[i] * gamma_decay + (1 - gamma_decay)
    std_ref = pyramid[i].std()
    std_diff = diffused.std()
    gain = std_ref / (std_diff + 1e-6) * contrast
    gain = np.clip(gain, 1.0, 2.0)
    diffused = alpha * diffused * gain + (1 - alpha) * pyramid[i]
  trimmed = diffused[:h, :w]
  return np.clip(trimmed, 0, 1)


def compute_levels_blur_image_portrait(image):
  """Computes a decent level for blur_image_portrait."""
  h, w = image.shape[:2]
  area_root = math.sqrt(h * w)
  return max(int(math.log2(area_root)) - 6, 2)


def blur_image_naive_ecpb(image, levels, decay=0.0, contrast=1.0, edge_threshold=0.8):
  """Applies portrait blur by naive Edge-Contained Pyramid Blur."""
  assert image.dtype == np.float32
  h, w = image.shape[:2]
  levels = min(levels, int(math.log2(min(h, w))) - 1)
  factor = 2 ** levels
  new_h = ((h + factor - 1) // factor) * factor
  new_w = ((w + factor - 1) // factor) * factor
  expanded = cv2.copyMakeBorder(image, 0, new_h - h, 0, new_w - w, cv2.BORDER_REPLICATE)
  sharp_full = compute_sharpness_naive(expanded)
  sharp_full = percentile_normalization(sharp_full, 2, 98)
  edge_full = (sharp_full > edge_threshold).astype(np.float32)
  gauss_pyr = [expanded]
  for _ in range(levels):
    gauss_pyr.append(cv2.pyrDown(gauss_pyr[-1]))
  lap_pyr = []
  for i in range(levels):
    up = cv2.pyrUp(gauss_pyr[i + 1], dstsize=(gauss_pyr[i].shape[1], gauss_pyr[i].shape[0]))
    lap = gauss_pyr[i] - up
    lap_pyr.append(lap)
  lap_pyr.append(gauss_pyr[-1])
  restored = lap_pyr[-1]
  restored_list = [restored]
  for i in range(levels - 1, -1, -1):
    size = (lap_pyr[i].shape[1], lap_pyr[i].shape[0])
    restored = cv2.pyrUp(restored, dstsize=size) + lap_pyr[i]
    restored_list.insert(0, restored)
  def hard_pool_half(mask):
    h, w = mask.shape
    h2, w2 = h // 2, w // 2
    mask = mask[:h2*2, :w2*2]
    reshaped = mask.reshape(h2, 2, w2, 2)
    pooled = np.sum(reshaped, axis=(1, 3))
    return (pooled >= 2).astype(np.float32)
  edge_pyr = [edge_full]
  for lvl in range(levels):
    sharp_lvl = compute_sharpness_naive(gauss_pyr[lvl + 1])
    sharp_lvl = percentile_normalization(sharp_lvl, 2, 98)
    edge_lvl = (sharp_lvl > edge_threshold).astype(np.float32)
    wall_lvl = hard_pool_half(edge_pyr[-1])
    edge_comb = np.maximum(wall_lvl, edge_lvl)
    edge_pyr.append(edge_comb)
  def edge_aware_pyrdown(img, edge_mask, fallback):
    mask_inv = 1.0 - edge_mask
    mask_inv_3d = mask_inv[:, :, None]
    numerator = cv2.pyrDown(img * mask_inv_3d)
    denominator = cv2.pyrDown(mask_inv)
    denom_safe = np.clip(denominator, 1e-6, None)
    averaged = numerator / denom_safe[:, :, None]
    blend_weight = np.clip(1.0 - denominator[:, :, None], 0, 1)
    return (1 - blend_weight) * averaged + blend_weight * fallback
  edged_gauss_pyr = [expanded]
  for lvl in range(levels):
    fallback = restored_list[lvl+1]
    down = edge_aware_pyrdown(edged_gauss_pyr[-1], edge_pyr[lvl], fallback)
    edged_gauss_pyr.append(down)
  bokehs = [2 ** i for i in range(0, levels)]
  alpha_weights = [b / bokehs[-1] for b in reversed(bokehs)]
  def edge_aware_pyrup(img_low, edge_mask, size, fallback):
    mask_inv = 1.0 - edge_mask
    mask_inv_3d = mask_inv[:, :, None]
    numerator = cv2.pyrUp(img_low * mask_inv_3d, dstsize=size)
    denominator = cv2.pyrUp(mask_inv, dstsize=size)
    denom_safe = np.clip(denominator, 1e-6, None)
    averaged = numerator / denom_safe[:, :, None]
    blend_weight = np.clip(1.0 - denominator[:, :, None], 0, 1)
    return (1 - blend_weight) * averaged + blend_weight * fallback
  diffused = edged_gauss_pyr[-1]
  for lvl in range(levels - 1, -1, -1):
    size = (gauss_pyr[lvl].shape[1], gauss_pyr[lvl].shape[0])
    fallback = restored_list[lvl]
    diffused = edge_aware_pyrup(diffused, edge_pyr[lvl + 1], size, fallback)
    gamma_decay = decay ** 2
    alpha = alpha_weights[lvl] * gamma_decay + (1 - gamma_decay)
    std_ref = edged_gauss_pyr[lvl].std()
    std_diff = diffused.std()
    gain = std_ref / (std_diff + 1e-6) * contrast
    gain = np.clip(gain, 1.0, 2.0)
    diffused = alpha * diffused * gain + (1 - alpha) * edged_gauss_pyr[lvl]
  result = diffused[:h, :w]
  return np.clip(result, 0, 1)


def blur_image_stacked_ecpb(image, max_level, decay=0.0, contrast=1.0, edge_threshold=0.8,
                            bokeh_balance=0.75):
  """Applies portrait blur by naive stacked Edge-Contained Pyramid Blur."""
  assert image.dtype == np.float32
  h, w = image.shape[:2]
  max_level = min(max_level, int(math.log2(min(h, w))) - 1)
  factor = 2 ** max_level
  new_h = ((h + factor - 1) // factor) * factor
  new_w = ((w + factor - 1) // factor) * factor
  expanded = cv2.copyMakeBorder(image, 0, new_h - h, 0, new_w - w, cv2.BORDER_REPLICATE)
  min_thresh = edge_threshold * 0.75
  max_thresh = edge_threshold + (1 - edge_threshold) * 0.5
  edge_thresholds = np.linspace(min_thresh, max_thresh, max_level + 1).tolist()
  gauss_pyr_full = [expanded]
  for _ in range(max_level):
    gauss_pyr_full.append(cv2.pyrDown(gauss_pyr_full[-1]))
  lap_pyr_full = []
  sizes_full = []
  for i in range(max_level):
    up = cv2.pyrUp(gauss_pyr_full[i + 1],
                   dstsize=(gauss_pyr_full[i].shape[1], gauss_pyr_full[i].shape[0]))
    lap = gauss_pyr_full[i] - up
    lap_pyr_full.append(lap)
    sizes_full.append((gauss_pyr_full[i].shape[1], gauss_pyr_full[i].shape[0]))
  sizes_full.append((gauss_pyr_full[-1].shape[1], gauss_pyr_full[-1].shape[0]))
  sharp_full = compute_sharpness_naive(expanded)
  sharp_full = percentile_normalization(sharp_full, 2, 98)
  edge_full = (sharp_full > edge_thresholds[0]).astype(np.float32)
  sharpness_pyr_full = [
    percentile_normalization(compute_sharpness_naive(gauss_pyr_full[i + 1]), 2, 98)
    for i in range(max_level)
  ]
  results = []
  for levels in range(1, max_level + 1):
    gauss_pyr = gauss_pyr_full[:levels + 1]
    lap_pyr = lap_pyr_full[:levels] + [gauss_pyr_full[levels]]
    sizes = sizes_full[:levels + 1]
    sharp_pyr = sharpness_pyr_full[:levels]
    restored = lap_pyr[-1]
    restored_list = [restored]
    for i in range(levels - 1, -1, -1):
      size = sizes[i]
      restored = cv2.pyrUp(restored, dstsize=size) + lap_pyr[i]
      restored_list.insert(0, restored)
    def hard_pool_half(mask):
      h2, w2 = mask.shape[0] // 2, mask.shape[1] // 2
      pooled = mask[:h2*2, :w2*2].reshape(h2, 2, w2, 2).sum(axis=(1, 3))
      return (pooled >= 2).astype(np.float32)
    edge_pyr = [edge_full]
    for lvl in range(levels):
      edge_lvl = (sharp_pyr[lvl] > edge_thresholds[levels]).astype(np.float32)
      wall_lvl = hard_pool_half(edge_pyr[-1])
      edge_comb = np.maximum(wall_lvl, edge_lvl)
      edge_pyr.append(edge_comb)
    def edge_aware_pyrdown(img, edge_mask, fallback):
      mask_inv = 1.0 - edge_mask
      mask_inv_3d = mask_inv[:, :, None]
      numerator = cv2.pyrDown(img * mask_inv_3d)
      denominator = cv2.pyrDown(mask_inv)
      denom_safe = np.clip(denominator, 1e-6, None)
      averaged = numerator / denom_safe[:, :, None]
      blend_weight = np.clip(1.0 - denominator[:, :, None], 0, 1)
      return (1 - blend_weight) * averaged + blend_weight * fallback
    edged_gauss_pyr = [expanded]
    for lvl in range(levels):
      fallback = restored_list[lvl+1]
      down = edge_aware_pyrdown(edged_gauss_pyr[-1], edge_pyr[lvl], fallback)
      edged_gauss_pyr.append(down)
    def edge_aware_pyrup(img_low, edge_mask, size, fallback):
      mask_inv = 1.0 - edge_mask
      mask_inv_3d = mask_inv[:, :, None]
      numerator = cv2.pyrUp(img_low * mask_inv_3d, dstsize=size)
      denominator = cv2.pyrUp(mask_inv, dstsize=size)
      denom_safe = np.clip(denominator, 1e-6, None)
      averaged = numerator / denom_safe[:, :, None]
      blend_weight = np.clip(1.0 - denominator[:, :, None], 0, 1)
      return (1 - blend_weight) * averaged + blend_weight * fallback
    bokehs = [2 ** i for i in range(0, levels)]
    alpha_weights = [b / bokehs[-1] for b in reversed(bokehs)]
    diffused = edged_gauss_pyr[-1]
    for lvl in range(levels - 1, -1, -1):
      size = sizes[lvl]
      fallback = restored_list[lvl]
      diffused = edge_aware_pyrup(diffused, edge_pyr[lvl + 1], size, fallback)
      gamma_decay = decay ** 2
      alpha = alpha_weights[lvl] * gamma_decay + (1 - gamma_decay)
      std_ref = edged_gauss_pyr[lvl].std()
      std_diff = diffused.std()
      gain = std_ref / (std_diff + 1e-6) * contrast
      gain = np.clip(gain, 1.0, 2.0)
      diffused = alpha * diffused * gain + (1 - alpha) * edged_gauss_pyr[lvl]
    results.append(diffused[:h, :w])
  def compute_geometric_weights():
    scores = []
    for level in range(1, max_level + 1):
      bokeh_benefit = math.log2(level + 1)
      artifact_cost = math.sqrt(level)
      score = bokeh_balance * bokeh_benefit - (1 - bokeh_balance) * artifact_cost
      scores.append(score)
    scores = np.array(scores)
    weights = np.exp(scores - np.max(scores))
    return weights / np.sum(weights)
  weights = compute_geometric_weights()
  log_sum = np.zeros_like(results[0])
  for r, w in zip(results, weights):
    log_sum += w * np.log(np.clip(r, 1e-6, 1.0))
  geo_mean = np.exp(log_sum)
  return np.clip(geo_mean, 0, 1)


def blur_image_portrait(image, max_level, decay=0.0, contrast=1.0, edge_threshold=0.8,
                        bokeh_balance=0.75, repeat=1,
                        grabcut=1.0, attractor=(0.5, 0.5), attractor_weight=0.1,
                        finish_edge=1.0):
  """Applies portrait blur according to the configuration."""
  blurred = image
  for i in range(repeat):
    if max_level > 0:
      blurred = blur_image_stacked_ecpb(
        blurred, max_level, decay=decay, contrast=contrast,
        edge_threshold=edge_threshold, bokeh_balance=bokeh_balance)
    elif max_level >= -100:
      max_level = max_level * -1
      blurred = blur_image_naive_ecpb(
        blurred, max_level, decay=decay, contrast=contrast,
        edge_threshold=edge_threshold)
    else:
      max_level = max_level * -1 - 100
      blurred = blur_image_pyramid(blurred, max_level, decay=decay, contrast=contrast)
  restored = blurred
  if grabcut > 0:
    mask = compute_focus_grabcut(image, attractor=attractor, attractor_weight=attractor_weight)
    mask *= grabcut
    restored = image * mask[..., None] + restored * (1 - mask[..., None])
  if finish_edge > 0:
    mask = compute_sharpness_naive(image, blur_radius=0, base_area=sys.maxsize)
    mask = percentile_normalization(mask, 2, 98)
    mask = sigmoidal_contrast_image(mask, gain=10, mid=0.9) * finish_edge
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    restored = mask * image + (1 - mask) * restored
  return np.clip(restored, 0, 1)


def unsharp_image_gaussian(image, radius):
  """Applies unsharp mask by Gaussian blur."""
  assert image.dtype == np.float32
  ksize = math.ceil(2 * radius) + 1
  if radius == 1:
    sigma = 0.8
    amount = 1.5
    threshold=0.01
  else:
    sigma = radius / 2
    amount = 1.2
    threshold=0.03
  blur_image = cv2.GaussianBlur(image, (ksize, ksize), sigma)
  diff_image = image - blur_image
  mask = np.abs(diff_image) > threshold
  diff_image *= amount * mask.astype(np.float32)
  sharp_image = image + diff_image
  return np.clip(sharp_image, 0, 1)


def trim_image(image, top, right, bottom, left):
  """Trims a image by percentages from sides."""
  assert image.dtype == np.float32
  h, w = image.shape[:2]
  top_px = int(h * top)
  right_px = int(w * right)
  bottom_px = int(h * bottom)
  left_px = int(w * left)
  new_top = min(top_px, h - 1)
  new_bottom = max(h - bottom_px, 1)
  new_left = min(left_px, w - 1)
  new_right = max(w - right_px, 1)
  cropped_image = image[new_top:new_bottom, new_left:new_right]
  return cropped_image


def perspective_correct_image(image, tl, tr, br, bl):
  """Apply perspective correction on the image."""
  assert image.dtype == np.float32
  h, w = image.shape[:2]
  def ratio_to_pixels(rx, ry):
    return int(rx * w), int(ry * h)
  tl = ratio_to_pixels(*tl)
  tr = ratio_to_pixels(*tr)
  br = ratio_to_pixels(*br)
  bl = ratio_to_pixels(*bl)
  src_points = np.array([tl, tr, br, bl], dtype=np.float32)
  dst_points = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
  matrix = cv2.getPerspectiveTransform(src_points, dst_points)
  warped_image = cv2.warpPerspective(image, matrix, (w, h))
  return warped_image


def get_scaled_image_size(image, long_size):
  """Gets the new width and height if an image is scaled."""
  assert image.dtype == np.float32
  h, w = image.shape[:2]
  if h > w:
    scale = long_size / h
  else:
    scale = long_size / w
  new_width = max(round(w * scale), 1)
  new_height = max(round(h * scale), 1)
  return new_width, new_height


def scale_image(image, width, height):
  """Scales the image into the new geometry."""
  assert image.dtype == np.float32
  h, w = image.shape[:2]
  if width * height > h * w:
    interpolation = cv2.INTER_LANCZOS4
  else:
    interpolation = cv2.INTER_AREA
  return cv2.resize(image, (width, height), interpolation=interpolation)


def apply_vignetting_image(image, strength, cx=0.5, cy=0.5):
  """Apply radial vignetting to the image."""
  assert image.dtype == np.float32
  h, w = image.shape[:2]
  y, x = np.ogrid[:h, :w]
  center_x = cx * w
  center_y = cy * h
  dx = (x - center_x) / (w / 2)
  dy = (y - center_y) / (h / 2)
  distance = np.sqrt(dx**2 + dy**2)
  mask = 1 - np.abs(strength) * (distance**2)
  mask = np.clip(mask, 0, 1).astype(np.float32)
  if strength >= 0:
    image = image * mask[..., np.newaxis]
  else:
    correction = 1 / np.clip(mask, 1e-6, 1)
    image = image * correction[..., np.newaxis]
  return np.clip(image, 0, 1)


def parse_color_expr(expr):
  """Parses a color expression and returns a R, G, B tuple."""
  expr = expr.strip().lower()
  named_colors = {
    "black": "#000000", "white": "#ffffff",
    "red": "#ff0000", "green": "#008000", "blue": "#0000ff",
    "yellow": "#ffff00", "cyan": "#00ffff", "magenta": "#ff00ff",
    "gray": "#808080", "silver": "#c0c0c0", "maroon": "#800000",
    "olive": "#808000", "lime": "#00ff00", "teal": "#008080",
    "navy": "#000080", "fuchsia": "#ff00ff", "aqua": "#00ffff",
    "purple": "#800080", "orange": "#ffa500"
  }
  if expr in named_colors:
    expr = named_colors[expr]
  match = re.fullmatch(r"#?([0-9a-fA-F]{3,6})", expr)
  if match:
    expr = match.group(1)
    if len(expr) == 3:
      r, g, b = (int(expr[0] * 2, 16), int(expr[1] * 2, 16),
                 int(expr[2] * 2, 16))
      return r / 255, g / 255, b / 255
    elif len(expr) == 6:
      r, g, b = (int(expr[0:2], 16), int(expr[2:4], 16), int(expr[4:6], 16))
      return r / 255, g / 255, b / 255
  raise ValueError(f"invalid color expression '{expr}'")


def write_caption(image, capexpr):
  """Write a text on an image."""
  h, w = image.shape[:2]
  fields = capexpr.split("|")
  text = fields[0]
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_ratio = 1.0
  if len(fields) > 1:
    match = re.fullmatch(r" *(\d+\.\d*) *", fields[1])
    if match:
      font_ratio = float(match.group(1))
  font_scale = 0.002 * np.sqrt(h * w) * font_ratio
  thickness = max(2, int(font_scale * 1.5))
  r, g, b = (0.5, 0.5, 0.5)
  if len(fields) > 2:
    r, g, b = parse_color_expr(fields[2])
  font_color = (b, g, r)
  line_type = cv2.LINE_AA
  tw, th = cv2.getTextSize(text, font, font_scale, thickness)[0]
  tx = (w - tw) // 2
  ty = (h + th) // 2
  if len(fields) > 3:
    match = re.fullmatch(r" *([a-xA-X]+) *", fields[3])
    if match:
      expr = match.group(1).lower()
      if "t" in expr:
        ty = th + h // 40
      elif "b" in expr:
        ty = h - h // 40
      if "l" in expr:
        tx = w // 50
      if "r" in expr:
        tx = w - tw - w // 40
  logger.debug(f"text={text}, scale={font_scale:.2f},"
               f" r={r:.2f}, g={g:.2f}, b={b:.2f},"
               f" x={tx}, y={ty}, w={tw}, h={th}")
  cv2.putText(image, text, (tx, ty), font, font_scale, font_color,
              thickness, line_type)
  return image


def parse_name_opts_expression(expr):
  """Parses name:option expression and returns key-value map."""
  expr = expr.strip()
  if re.match(r"^[a-zA-Z]+:", expr):
    fields = re.split(":", expr)
  elif re.match(r"^[a-zA-Z]+,", expr):
    fields = re.split(",", expr)
  elif re.match(r"^[a-zA-Z]+;", expr):
    fields = re.split(";", expr)
  else:
    fields = re.split(r"[ ,\|:;]+", expr)
  params = {"name": fields[0]}
  for field in fields[1:]:
    columns = field.split("=", 1)
    name = columns[0].strip()
    if len(columns) > 1:
      params[name] = columns[1].strip()
    else:
      params[name] = "true"
  return params


def parse_num_opts_expression(expr, defval=0):
  """Parses num:option expression and returns key-value map."""
  expr = expr.strip()
  if re.match(r"^[a-zA-Z]+:", expr):
    fields = re.split(":", expr)
  elif re.match(r"^[a-zA-Z]+,", expr):
    fields = re.split(",", expr)
  elif re.match(r"^[a-zA-Z]+;", expr):
    fields = re.split(";", expr)
  else:
    fields = re.split(r"[ ,\|:;]+", expr)
  try:
    num = float(fields[0])
  except:
    num = defval
  params = {"num": num}
  for field in fields[1:]:
    columns = field.split("=", 1)
    name = columns[0].strip()
    if len(columns) > 1:
      params[name] = columns[1].strip()
    else:
      params[name] = "true"
  return params


def copy_param_to_kwargs(params, kwargs, name, convert_type=None):
  """Copies an option parameter into the kwargs."""
  if name in params:
    value = params[name]
    if callable(convert_type):
      value = convert_type(value)
    elif type(convert_type) == bool:
      value = convert_type
    kwargs[name] = value


def parse_coordinate(expr):
  """Parses coordinate expression and returns XY ratios."""
  expr = expr.strip()
  if not expr:
    return None
  values = list(map(float, re.split(r'[ ,\|:;]+', expr.strip())))
  values = [float(v) for v in values]
  if len(values) == 2:
    x, y = values[0], values[1]
  else:
    raise ValueError("coordinate expression must contain 2 values")
  return x, y


def parse_trim_expression(expr):
  """Parses trimming expression and returns TLBR ratios."""
  expr = expr.strip()
  if not expr:
    return None
  values = list(map(float, re.split(r'[ ,\|:;]+', expr.strip())))
  values = [min(max(float(v), 0), 100) / 100 for v in values]
  if len(values) == 1:
    t = r = b = l = values[0]
  elif len(values) == 2:
    t, b = values[0], values[0]
    l, r = values[1], values[1]
  elif len(values) == 3:
    t, l, b = values
    r = l
  elif len(values) == 4:
    t, r, b, l = values
  else:
    raise ValueError("trim expression must contain 1 to 4 values")
  return t, r, b, l


def parse_pers_expression(expr):
  """Parses perspective correction expression and returns TL,TR,BR,TL ratios."""
  expr = expr.strip()
  if not expr:
    return None
  values = list(map(float, re.split(r'[ ,\|:;]+', expr.strip())))
  values = [min(max(float(v), 0), 100) / 100 for v in values]
  if len(values) == 4:
    coords = [
      (values[0], values[0]),
      (1 - values[1], values[1]),
      (1 - values[2], 1 - values[2]),
      (values[3], 1 - values[3])
    ]
  elif len(values) == 8:
    coords = [
      (values[0], values[1]),
      (values[2], values[3]),
      (values[4], values[5]),
      (values[6], values[7])
    ]
  else:
    raise ValueError("perspective expression must contain 4 or 8 values")
  center_x = sum(x for x, _ in coords) / 4
  center_y = sum(y for _, y in coords) / 4
  def angle_from_center(point):
    x, y = point
    return np.arctan2(y - center_y, x - center_x)
  sorted_points = sorted(coords, key=angle_from_center)
  tops = sorted(sorted_points[:2], key=lambda p: p[0])
  bottoms = sorted(sorted_points[2:], key=lambda p: p[0])
  return tops[0], tops[1], bottoms[1], bottoms[0]


def parse_scale_expression(expr):
  """Parses scaling expression and returns WH pixels."""
  expr = expr.strip()
  if not expr:
    return None
  values = list(map(float, re.split(r'[ ,\|:;]+', expr.strip())))
  values = [int(v) for v in values]
  if len(values) == 1:
    w, h = values[0], None
  elif len(values) == 2:
    w, h = values[0], values[1]
  else:
    raise ValueError("scale expression must contain 1 to 2 values")
  return w, h


def set_logging_level(level):
  """Sets the logging level."""
  logger.setLevel(level)


def make_ap_args():
  """Makes arguments of the argument parser."""
  description = "Stack and combine images."
  epilog = f"{PROG_NAME} version {PROG_VERSION}. Powered by OpenCV2."
  ap = argparse.ArgumentParser(
    prog=PROG_NAME, description=description, epilog=epilog,
    formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
  ap.add_argument("inputs", nargs='+', help="input image paths")
  ap.add_argument("--output", "-o", default="output.jpg", metavar="path",
                  help="output image path (dafault=output.jpg)")
  ap.add_argument("--white-balance", "-wb", default="", metavar="expr",
                  help="choose a white balance:"
                  " none (default), auto, auto-scene, daylight, cloudy, shade, tungsten,"
                  " fluorescent, flash, or a kelvin in K or three weights of RGB like 11,13,12")
  ap.add_argument("--average-exposure", "-ax", action='store_true',
                  help="average input exposure")
  ap.add_argument("--align", "-a", default="", metavar="name",
                  help="choose an alignment method for input images:"
                  " none (default), orb, sift, ecc, hugin")
  ap.add_argument("--ignore-unaligned", "-iu", action='store_true',
                  help="ignore unaligned images")
  ap.add_argument("--merge", "-m", default="average", metavar="name",
                  help="choose a processing method for merging:"
                  " average (default), median, geomean, maximum, minimum, denoise, weighted,"
                  " debevec, robertson, mertens, focus, grid, stitch")
  ap.add_argument("--tonemap", "-t", default="linear", metavar="name",
                  help="choose a tone mapping method for debevec:"
                  " linear (default), reinhard, drago, mantiuk")
  ap.add_argument("--no-restore", "-nr", action='store_true',
                  help="do not apply auto restoration of brightness")
  ap.add_argument("--fill-margin", "-fm", action='store_true',
                  help="fill black marin with the color of nearest pixels")
  ap.add_argument("--gamma", type=float, default=1.0, metavar="num",
                  help="gamma brightness adjustment."
                  " less than 1.0 to darken, less than 1.0 to lighten")
  ap.add_argument("--slog", type=float, default=0, metavar="num",
                  help="scaled log brightness adjustment."
                  " positive to lighten, negative to darken")
  ap.add_argument("--sigmoid", default="0", metavar="num",
                  help="sigmoidal contrast adjustment."
                  " positive to strengthen, negative to weaken")
  ap.add_argument("--histeq", default="0", metavar="num",
                  help="apply histogram equalization by the clip limit. negative means global")
  ap.add_argument("--saturate", type=float, default=0, metavar="num",
                  help="saturate colors. positive for vivid, negative for muted")
  ap.add_argument("--art", default="", metavar="text",
                  help="apply an artistic filter: pencil, stylized, oil, cartoon")
  ap.add_argument("--gray", default="", metavar="text",
                  help="convert to grayscale: bt601, bt709, bt2020,"
                  " red, orange, yellow, green, blue, mean, lab, hsv, laplacian, sobel")
  ap.add_argument("--denoise", type=int, default=0, metavar="num",
                  help="apply bilateral denoise by the pixel radius.")
  ap.add_argument("--blur", default="0", metavar="num",
                  help="apply Gaussian blur by the pixel radius. negative uses pyramid blur")
  ap.add_argument("--portrait", default="0", metavar="num",
                  help="apply portrait blur by the pyramid level")
  ap.add_argument("--unsharp", type=int, default=0, metavar="num",
                  help="apply Gaussian unsharp mask by the pixel radius.")
  ap.add_argument("--trim", default="", metavar="numlist",
                  help="trim sides: TOP,LEFT,BOTTOM,RIGHT in percentage eg. 5,10,3,7")
  ap.add_argument("--pers", default="", metavar="numlist",
                  help="perspective change: X1,Y1,X2,Y2,X3,Y3,X4,Y4 in percentage"
                  " eg. 10,10,10,90,0,100,100,0")
  ap.add_argument("--scale", default="", metavar="numlist",
                  help="scale change: WIDTH,HEIGHT in pixels eg. 1920,1080")
  ap.add_argument("--vignetting", default="0", metavar="num",
                  help="apply vignetting by the light reduction ratio at the corners")
  ap.add_argument("--caption", default="", metavar="text",
                  help="put a caption text: TEXT|SIZE|COLOR|POS eg. Hello|5|ddeeff|tl")
  ap.add_argument("--input-video-fps", type=float, default=1, metavar="num",
                  help="input video files with the FPS")
  ap.add_argument("--output-video-fps", type=float, default=1, metavar="num",
                  help="output a video file with the FPS")
  ap.add_argument("--max-memory-usage", type=float, default=8, metavar="num",
                  help="maximum memory usage in GiB")
  ap.add_argument("--debug", action='store_true', help="print debug messages")
  return ap.parse_args()


def main():
  """Executes all operations."""
  args = make_ap_args()
  start_time = time.time()
  if args.debug:
    set_logging_level(logging.DEBUG)
  logger.debug(f"{PROG_NAME}={PROG_VERSION},"
               f" OpenCV={cv2.__version__}, NumPy={np.__version__}")
  logger.info(f"Process started: input={args.inputs}, output={args.output}")
  for path in args.inputs:
    if not os.path.exists(path):
      ValueError(f"{path} doesn't exist")
  logger.info(f"Loading the input files")
  images_data = load_input_images(args)
  images, bits_list = zip(*images_data)
  meta_list = [get_metadata(input_path) for input_path in args.inputs]
  if logger.isEnabledFor(logging.DEBUG):
    log_image_stats(images[0][0], "first")
  brightness_values = np.array([compute_brightness(image) for image in images])
  mean_brightness = np.mean(brightness_values)
  logger.debug(f"mean_brightness={mean_brightness:.3f}")
  if args.white_balance and args.white_balance != "none":
    logger.info(f"Adjusting white balance")
    images = [adjust_white_balance_image(image, args.white_balance) for image in images]
  if args.average_exposure:
    logger.info(f"Adjusting input exposure to the mean")
    images = [adjust_exposure_image(image, mean_brightness) for image in images]
  aligned_indices = set()
  align_params = parse_name_opts_expression(args.align)
  align_name = align_params["name"]
  if align_name in ["none", ""]:
    pass
  elif align_name in ["orb", "o"]:
    logger.info(f"Aligning images by ORB")
    kwargs = {}
    copy_param_to_kwargs(align_params, kwargs, "nfeatures", int)
    copy_param_to_kwargs(align_params, kwargs, "shift_limit", float)
    copy_param_to_kwargs(align_params, kwargs, "denoise", int)
    images = align_images_orb(images, aligned_indices, **kwargs)
  elif align_name in ["sift", "s"]:
    logger.info(f"Aligning images by SIFT")
    kwargs = {}
    copy_param_to_kwargs(align_params, kwargs, "nfeatures", int)
    copy_param_to_kwargs(align_params, kwargs, "shift_limit", float)
    copy_param_to_kwargs(align_params, kwargs, "denoise", int)
    images = align_images_sift(images, aligned_indices, **kwargs)
  elif align_name in ["ecc", "e"]:
    logger.info(f"Aligning images by ECC")
    kwargs = {}
    copy_param_to_kwargs(align_params, kwargs, "use_affine", parse_boolean)
    copy_param_to_kwargs(align_params, kwargs, "denoise", int)
    images = align_images_ecc(images, aligned_indices, **kwargs)
  elif align_name in ["hugin", "h"]:
    logger.info(f"Aligning images by Hugin")
    images = align_images_hugin(images, args.inputs, bits_list)
  else:
    raise ValueError(f"Unknown align method: f{align_name}")
  images = adjust_size_images(images)
  if len(aligned_indices) > 0:
    logger.debug(f"aligned indices: {sorted(list(aligned_indices))}")
    if args.ignore_unaligned:
      old_num = len(images)
      images, bits_list, meta_list = zip(
        *[(image, bits, meta)
          for i, (image, bits, meta) in enumerate(zip(images, bits_list, meta_list))
          if i in aligned_indices])
      if len(images) != old_num:
        logger.debug(f"{old_num - len(images)} of {old_num} images are removed")
  ext = os.path.splitext(args.output)[1].lower()
  if ext in EXTS_NPZ:
    postprocess_npz(args, images)
  elif ext in EXTS_VIDEO:
    postprocess_video(args, images)
  elif ext in EXTS_IMAGE:
    postprocess_images(args, images, bits_list, meta_list, mean_brightness)
  else:
    raise ValueError(f"Unsupported file format: {ext}")
  elapsed_time = time.time() - start_time
  logger.info(f"Process done: time={elapsed_time:.2f}s")


def load_input_images(args):
  """Loads input images."""
  limit_mem_size = args.max_memory_usage * (1<<30)
  total_mem_size = 0
  images_data = []
  for input_path in args.inputs:
    ext = os.path.splitext(input_path)[1].lower()
    mem_allowance = limit_mem_size - total_mem_size
    match = re.fullmatch(r"\[([a-z]+?)(:.*)?\]", input_path)
    if match:
      name = match.group(1).lower()
      if name == "colorbar":
        image = generate_colorbar()
        bits = 8
      else:
        raise ValueError(f"Unsupported image generation: {name}")
      total_mem_size += estimate_image_memory_size(image)
      if total_mem_size > limit_mem_size:
        raise SystemError(f"Exceeded memory limit: {total_mem_size} vs {limit_mem_size}")
      images_data.append((image, bits))
    elif ext in EXTS_NPZ:
      npz_image_data = load_npz(input_path, mem_allowance)
      for image, bits in npz_image_data:
        total_mem_size += estimate_image_memory_size(image)
        if total_mem_size > limit_mem_size:
          raise SystemError(f"Exceeded memory limit: {total_mem_size} vs {limit_mem_size}")
        images_data.append((image, bits))
    elif ext in EXTS_VIDEO:
      video_image_data = load_video(input_path, mem_allowance, args.input_video_fps)
      for image, bits in video_image_data:
        total_mem_size += estimate_image_memory_size(image)
        if total_mem_size > limit_mem_size:
          raise SystemError(f"Exceeded memory limit: {total_mem_size} vs {limit_mem_size}")
        images_data.append((image, bits))
    elif ext in EXTS_IMAGE_HEIF:
      for image, bits in load_images_heif(input_path):
        total_mem_size += estimate_image_memory_size(image)
        if total_mem_size > limit_mem_size:
          raise SystemError(f"Exceeded memory limit: {total_mem_size} vs {limit_mem_size}")
        images_data.append((image, bits))
    elif ext in EXTS_IMAGE:
      image, bits = load_image(input_path)
      total_mem_size += estimate_image_memory_size(image)
      if total_mem_size > limit_mem_size:
        raise SystemError(f"Exceeded memory limit: {total_mem_size} vs {limit_mem_size}")
      images_data.append((image, bits))
    else:
      raise ValueError(f"Unsupported file format: {ext}")
  return images_data


def postprocess_npz(args, images):
  """Postprocess images as a NumPy compressed."""
  assert all(image.dtype == np.float32 for image in images)
  images = [edit_image(image, args) for image in images]
  logger.info(f"Saving the output file as a NumPy compressed")
  save_npz(args.output, images)


def postprocess_video(args, images):
  """Postprocess images as a video."""
  assert all(image.dtype == np.float32 for image in images)
  images = [edit_image(image, args) for image in images]
  logger.info(f"Saving the output file as a video")
  save_video(args.output, images, args.output_video_fps)
  if has_command(CMD_EXIFTOOL):
    logger.info(f"Copying metadata")
    copy_metadata(args.inputs[0], args.output)


def crop_to_match(image, target_size):
  """Crops the center of the image to match the target size."""
  assert image.dtype == np.float32
  h, w = image.shape[:2]
  th, tw = target_size
  y_offset = max((h - th) // 2, 0)
  x_offset = max((w - tw) // 2, 0)
  return image[y_offset:y_offset+th, x_offset:x_offset+tw]


def log_image_stats(image, prefix):
  """prints logs of an image."""
  assert image.dtype == np.float32
  has_nan = np.isnan(image).any()
  if has_nan:
    image = fix_overflown_image(image)
  min = np.min(image)
  max = np.max(image)
  mean = np.mean(image)
  stddev = np.std(image)
  skew = np.mean((image - np.mean(image))**3) / (np.std(image)**3)
  kurt = np.mean((image - np.mean(image))**4) / np.std(image)**4 - 3
  logger.debug(f"{prefix} stats: min={min:.3f}, max={max:.3f}, mean={mean:.3f},"
               f" stddev={stddev:.3f}, skew={skew:.3f}, kurt={kurt:.3f}, nan={has_nan}")


def edit_image(image, args):
  """Edits an image."""
  assert image.dtype == np.float32
  if args.fill_margin:
    logger.info(f"Filling the margin")
    image = fill_black_margin_image(image)
  if args.gamma != 1.0 and args.gamma > 0:
    logger.info(f"Adjust brightness by a gamma")
    image = apply_gamma_image(image, args.gamma)
  if args.slog != 0:
    logger.info(f"Adjust brightness by a scaled log")
    image = apply_scaled_log_image(image, args.slog)
  sigmoid_params = parse_num_opts_expression(args.sigmoid)
  sigmoid_num = sigmoid_params["num"]
  if sigmoid_num != 0:
    logger.info(f"Adjust brightness by a sigmoid")
    kwargs = {}
    copy_param_to_kwargs(sigmoid_params, kwargs, "mid", float)
    image = apply_sigmoid_image(image, sigmoid_num, **kwargs)
  histeq_params = parse_num_opts_expression(args.histeq)
  histeq_num = histeq_params["num"]
  if histeq_num > 0:
    logger.info(f"Applying CLAHE enhancement")
    kwargs = {}
    copy_param_to_kwargs(histeq_params, kwargs, "gamma", float)
    copy_param_to_kwargs(histeq_params, kwargs, "restore_color", parse_boolean)
    image = apply_clahe_image(image, histeq_num, **kwargs)
  elif histeq_num < 0:
    logger.info(f"Applying global HE enhancement")
    kwargs = {}
    copy_param_to_kwargs(histeq_params, kwargs, "gamma", float)
    copy_param_to_kwargs(histeq_params, kwargs, "restore_color", parse_boolean)
    image = apply_global_histeq_image(image, **kwargs)
  if args.saturate != 0:
    logger.info(f"Saturating colors")
    image = saturate_colors_image(image, args.saturate)
  if args.art and args.gray != "none":
    logger.info(f"Applying an artistic filter")
    image = apply_artistic_filter_image(image, args.art)
  if args.gray and args.gray != "none":
    logger.info(f"Converting to grayscale")
    image = convert_grayscale_image(image, args.gray)
  if args.denoise > 0:
    logger.info(f"Applying birateral denoise")
    image = bilateral_denoise_image(image, args.denoise)
  blur_params = parse_num_opts_expression(args.blur)
  blur_num = blur_params["num"]
  if blur_num > 0:
    logger.info(f"Applying Gaussian blur")
    image = blur_image_gaussian(image, blur_num)
  if blur_num < 0:
    logger.info(f"Applying Pyramid blur")
    kwargs = {}
    copy_param_to_kwargs(blur_params, kwargs, "decay", float)
    copy_param_to_kwargs(blur_params, kwargs, "contrast", float)
    image = blur_image_pyramid(image, int(blur_num) * -1, **kwargs)
  portrait_params = parse_name_opts_expression(args.portrait)
  portrait_name = portrait_params["name"]
  if portrait_name == "auto":
    portrait_name = str(compute_levels_blur_image_portrait(image))
  portrait_levels = int(portrait_name)
  if portrait_levels != 0:
    kwargs = {}
    copy_param_to_kwargs(portrait_params, kwargs, "decay", float)
    copy_param_to_kwargs(portrait_params, kwargs, "contrast", float)
    copy_param_to_kwargs(portrait_params, kwargs, "edge_threshold", float)
    copy_param_to_kwargs(portrait_params, kwargs, "bokeh_balance", float)
    copy_param_to_kwargs(portrait_params, kwargs, "repeat", int)
    copy_param_to_kwargs(portrait_params, kwargs, "grabcut", float)
    copy_param_to_kwargs(portrait_params, kwargs, "attractor", parse_coordinate)
    copy_param_to_kwargs(portrait_params, kwargs, "attractor_weight", float)
    copy_param_to_kwargs(portrait_params, kwargs, "finish_edge", float)
    logger.info(f"Applying portrait blur by {portrait_levels} levels")
    image = blur_image_portrait(image, portrait_levels, **kwargs)
  if args.unsharp > 0:
    logger.info(f"Applying Gaussian unsharp mask")
    image = unsharp_image_gaussian(image, args.unsharp)
  trim_params = parse_trim_expression(args.trim)
  if trim_params:
    logger.info(f"Trimming the image")
    image = trim_image(image, *trim_params)
  pers_params = parse_pers_expression(args.pers)
  if pers_params:
    logger.info(f"Doing perspective correction of the image")
    image = perspective_correct_image(image, *pers_params)
  scale_params = parse_scale_expression(args.scale)
  if scale_params:
    logger.info(f"Scaling the image")
    if scale_params[1] is None:
      scale_params = get_scaled_image_size(image, scale_params[0])
    image = scale_image(image, *scale_params)
  vignetting_params = parse_num_opts_expression(args.vignetting)
  vignetting_num = vignetting_params["num"]
  if vignetting_num != 0:
    kwargs = {}
    copy_param_to_kwargs(vignetting_params, kwargs, "cx", float)
    copy_param_to_kwargs(vignetting_params, kwargs, "cy", float)
    image = apply_vignetting_image(image, vignetting_num, **kwargs)
  if len(args.caption) > 0:
    logger.info(f"Writing the caption")
    image = write_caption(image, args.caption)
  return image


def postprocess_images(args, images, bits_list, meta_list, mean_brightness):
  """Postprocesses images as a merged image."""
  assert all(image.dtype == np.float32 for image in images)
  merge_params = parse_name_opts_expression(args.merge)
  merge_name = merge_params["name"]
  be_adjusted = False
  is_hdr = False
  if merge_name in ["average", "a", "mean"]:
    logger.info(f"Merging images by average composition")
    merged_image = merge_images_average(images)
    be_adjusted = len(images) > 1
  elif merge_name in ["median", "mdn"]:
    logger.info(f"Merging images by median composition")
    merged_image = merge_images_median(images)
    be_adjusted = len(images) > 1
  elif merge_name in ["geomean", "gm"]:
    logger.info(f"Merging images by geometric mean composition")
    merged_image = merge_images_geometric_mean(images)
    be_adjusted = len(images) > 1
  elif merge_name in ["minimum", "min"]:
    logger.info(f"Merging images by minimum composition")
    merged_image = merge_images_minimum(images)
    be_adjusted = len(images) > 1
  elif merge_name in ["maximum", "max"]:
    logger.info(f"Merging images by maximum composition")
    merged_image = merge_images_maximum(images)
    be_adjusted = len(images) > 1
  elif merge_name in ["denoise", "dn", "bgmbm"]:
    logger.info(f"Merging images by denoise composition")
    kwargs = {}
    copy_param_to_kwargs(merge_params, kwargs, "clip_limit", float)
    copy_param_to_kwargs(merge_params, kwargs, "blur_radius", float)
    merged_image = merge_images_denoise(images, **kwargs)
    be_adjusted = True
  elif merge_name in ["weighted", "w"]:
    logger.info(f"Merging images by weighted average composition")
    merged_image = merge_images_weighted_average(images, meta_list)
    be_adjusted = len(images) > 1
  elif merge_name in ["debevec", "d"]:
    logger.info(f"Merging images by Debevec's method as an HDRI")
    merged_image = merge_images_debevec(images, meta_list)
    be_adjusted = True
    is_hdr = True
  elif merge_name in ["robertson", "r"]:
    logger.info(f"Merging images by Robertson's method")
    merged_image = merge_images_robertson(images, meta_list)
    be_adjusted = True
    is_hdr = True
  elif merge_name in ["mertens", "m"]:
    logger.info(f"Merging images by Mertens's method")
    merged_image = merge_images_mertens(images)
    be_adjusted = True
    is_hdr = True
  elif merge_name in ["focus", "f"]:
    logger.info(f"Merging images by focus stacking")
    kwargs = {}
    copy_param_to_kwargs(merge_params, kwargs, "smoothness", float)
    copy_param_to_kwargs(merge_params, kwargs, "pyramid_levels", float)
    merged_image = merge_images_focus_stacking(images, **kwargs)
  elif merge_name in ["grid", "g"]:
    logger.info(f"Merging images in a grid")
    kwargs = {}
    copy_param_to_kwargs(merge_params, kwargs, "columns", int)
    copy_param_to_kwargs(merge_params, kwargs, "margin", int)
    copy_param_to_kwargs(merge_params, kwargs, "background", parse_color_expr)
    merged_image = merge_images_grid(images, **kwargs)
  elif merge_name in ["stitch", "s"]:
    logger.info(f"Stitching images as a panoramic photo")
    merged_image = merge_images_stitch(images)
  else:
    raise ValueError(f"Unknown merge method: {merge_name}")
  if logger.isEnabledFor(logging.DEBUG):
    log_image_stats(merged_image, "merged")
  merged_image = fix_overflown_image(merged_image)
  if is_hdr:
    if args.tonemap in ["linear", "l"]:
      logger.info(f"Tone mapping images by linear reduction")
      merged_image = tone_map_image_linear(merged_image)
    elif args.tonemap in ["reinhard", "r"]:
      logger.info(f"Tone mapping images by Reinhard's method")
      merged_image = tone_map_image_reinhard(merged_image)
    elif args.tonemap in ["drago", "d"]:
      logger.info(f"Tone mapping images by Drago's method")
      merged_image = tone_map_image_drago(merged_image)
    elif args.tonemap in ["mantiuk", "m"]:
      logger.info(f"Tone mapping images by Mantiuk's method")
      merged_image = tone_map_image_mantiuk(merged_image)
    else:
      raise ValueError(f"Unknown tone method: {args.tonemap}")
    if logger.isEnabledFor(logging.DEBUG):
      log_image_stats(merged_image, "tonemapped")
    merged_image = fix_overflown_image(merged_image)
  if be_adjusted and not args.no_restore:
    logger.info(f"Applying auto restoration of brightness")
    merged_image = adjust_exposure_image(merged_image, mean_brightness)
  merged_image = edit_image(merged_image, args)
  logger.info(f"Saving the output file as an image")
  ext = os.path.splitext(args.output)[1].lower()
  save_image(args.output, merged_image, bits_list[0])
  if has_command(CMD_EXIFTOOL):
    logger.info(f"Copying metadata")
    copy_metadata(args.inputs[0], args.output)


if __name__ == "__main__":
  main()
