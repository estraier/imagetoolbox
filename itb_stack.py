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
  logger.debug(f"saving NPZ: {file_path}")
  np.savez_compressed(file_path, *images)


def estimate_image_memory_size(image):
  """Estimates memory size of an image."""
  height, width = image.shape[:2]
  channels = 3
  depth = 4
  return width * height * channels * depth


def parse_boolean(text):
  """Parse a boolean expression and get its boolean value."""
  return text.strip().lower() in ["true", "t", "1", "yes", "y"]


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
  image = np.where(image <= 0.04045,
                   image / 12.92, cv2.pow((image + 0.055) / 1.055, 2.4))
  return image.astype(np.float32)


def linear_to_srgb(image):
  """Converts linear RGB to sRGB using OpenCV (optimized for float32)."""
  image = np.where(image <= 0.0031308,
                   image * 12.92, 1.055 * cv2.pow(image, 1/2.4) - 0.055)
  return image.astype(np.float32)


def compute_brightness(image):
  """Computes the average brightness of an image in grayscale."""
  return np.mean(cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY))


def apply_gamma_image(image, gamma):
  """Adjusts image brightness by a gamma transformation."""
  if gamma < 1e-6:
    return image
  image = np.power(image, 1 / gamma)
  return image.astype(np.float32)


def apply_scaled_log_image(image, factor):
  """Adjust image brightness by a scaled log transformation."""
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
  min_val = naive_sigmoid(0.0, gain, mid)
  max_val = naive_sigmoid(1.0, gain, mid)
  diff = max_val - min_val
  return np.clip((naive_sigmoid(image, gain, mid) - min_val) / diff, 0, 1)


def inverse_sigmoidal_contrast_image(image, gain, mid):
  """Applies inverse sigmoidal contrast adjustment."""
  min_val = naive_inverse_sigmoid(0.0, gain, mid)
  max_val = naive_inverse_sigmoid(1.0, gain, mid)
  diff = max_val - min_val
  return np.clip((naive_inverse_sigmoid(image, gain, mid) - min_val) / diff, 0, 1)


def apply_sigmoid_image(image, gain, mid):
  """Adjust image brightness by a sigmoid transformation."""
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
  expr = expr.strip().lower()
  params = parse_name_opts_expression(expr)
  name = params["name"]
  if name in ["auto", "auto-scene", "auto-temp"]:
    kwargs = {}
    if "edge_weight" in params:
      kwargs["edge_weight"] = int(params["edge_weight"])
    if "luminance_weight" in params:
      kwargs["luminance_weight"] = float(params["luminance_weight"])
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
  """Print a log message of a homography matrix."""
  sx, shear, dx = m[0]
  shear_y, sy, dy = m[1]
  scale_x = np.sqrt(sx**2 + shear**2)
  scale_y = np.sqrt(sy**2 + shear_y**2)
  angle = np.arctan2(m[1, 0], m[0, 0]) * 180 / np.pi
  logger.debug(f"warping: tran=({dx:.2f}, {dy:.2f}), "
               f"scale=({scale_x:.2f}, {scale_y:.2f}), angle={angle:.2f}°")


def make_image_for_alignment(image, clahe_clip_limit=0, denoise=0):
  """Make a byte-gray enhanced image for alignment."""
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  if denoise > 0:
    ksize = math.ceil(2 * denoise) + 1
    sigma_color = min(0.05 * math.sqrt(ksize), 0.35)
    sigma_space = 10 * math.sqrt(ksize)
    gray_image = cv2.bilateralFilter(gray_image, denoise, sigma_color, sigma_space)
  byte_image = (np.clip(gray_image, 0, 1) * 255).astype(np.uint8)
  tile_grid_size = (8, 8)
  if clahe_clip_limit > 0:
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=tile_grid_size)
    byte_image = clahe.apply(byte_image)
  return np.clip(byte_image, 0, 255)


def align_images_orb(images, aligned_indices, nfeatures=5000, denoise=2, shift_limit=0.1):
  """Aligns images using ORB."""
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


def fix_overflown_image(image):
  """Replaces NaN and -inf with 0, and inf with 1."""
  return np.clip(np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0), 0, None)


def merge_images_average(images):
  """Merges images by average composition."""
  return np.mean(images, axis=0)


def merge_images_median(images):
  """Merges images by median composition."""
  return np.median(images, axis=0)


def merge_images_geometric_mean(images):
  """Merges images by geoemtric_mean composition."""
  return np.exp(np.mean(np.log(np.clip(images, 1e-6, 1)), axis=0))


def merge_images_minimum(images):
  """Merges images by minimum value composition."""
  return np.min(images, axis=0)


def merge_images_maximum(images):
  """Merges images by maximum value composition."""
  return np.max(images, axis=0)


def merge_images_denoise(images, clip_limit=0.4, blur_radius=3):
  """Merge images by blurred geometric mean-based median."""
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
  byte_images = [(np.clip(image, 0, 1) * 255).astype(np.uint8)
                 for image in images]
  merger = cv2.createMergeMertens()
  hdr = merger.process(byte_images)
  hdr = normalize_negative_image(hdr)
  return hdr


def normalize_negative_image(image, clip_percentile=2.0):
  """Normalizes negaive pixels."""
  image = cv2.GaussianBlur(image, (5, 5), 0)
  min_val = np.percentile(image, clip_percentile)
  if min_val > -0.01:
    return np.clip(image, 0, None)
  image = np.clip(image, min_val, None)
  image -= min_val
  return np.clip(image, 0, None)


def z_score_normalization(image):
  """Applies Z-score normalization to stabilize feature scaling."""
  mean = np.mean(image)
  std = np.std(image)
  return (image - mean) / (std + 1e-6)


def compute_sharpness(image):
  """Computes sharpness using normalized Laplacian and Sobel filters."""
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  laplacian = np.abs(cv2.Laplacian(blurred, cv2.CV_32F, ksize=3))
  laplacian = z_score_normalization(laplacian)
  sobel_x = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))
  sobel_y = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3))
  sobel_e = np.sqrt(sobel_x**2 + sobel_y**2)
  sobel = z_score_normalization(sobel_e)
  sharpness = 0.5 * laplacian + 0.5 * sobel
  sharpness = np.clip(sharpness, -10, 10)
  return sharpness


def make_gaussian_pyramid(image, levels):
  """Generate Gaussian pyramid."""
  pyramid = [image]
  for _ in range(levels):
    image = cv2.pyrDown(image)
    pyramid.append(image)
  return pyramid


def make_laplacian_pyramid(image, levels):
  """Generate Laplacian pyramid."""
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
  h, w, c = images[0].shape
  pyramid_levels = min(pyramid_levels, int(math.log2(min(h, w))) - 3)
  sharpness_maps = np.array([compute_sharpness(img) for img in images])
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
  byte_images = [(np.clip(image, 0, 1) * 255).astype(np.uint8) for image in images]
  stitcher = cv2.Stitcher_create()
  status, stitched_image = stitcher.stitch(byte_images)
  if status != cv2.Stitcher_OK:
    raise ValueError(f"Stitching failed with status {status}")
  return np.clip(stitched_image.astype(np.float32) / 255, 0, 1)


def tone_map_image_linear(image):
  """Applies tone mapping by linear normalization."""
  return cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)


def tone_map_image_reinhard(image):
  """Applies tone mapping by Reinhard's method."""
  tonemap = cv2.createTonemapReinhard(gamma=1.0, intensity=0, light_adapt=0.5,
                                      color_adapt=0.5)
  ldr = np.clip(tonemap.process(image), 0, 1)
  return ldr


def tone_map_image_drago(image):
  """Applies tone mapping by Drago's method."""
  tonemap = cv2.createTonemapDrago(gamma=1.0, saturation=1.0, bias=0.9)
  ldr = np.clip(tonemap.process(image), 0, 1)
  return ldr


def tone_map_image_mantiuk(image):
  """Applies tone mapping by Mantiuk's method."""
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


def apply_global_histeq_image(image, gamma=2.2, restore_color=True):
  """Applies global histogram equalization contrast enhancement."""
  lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
  l, a, b = cv2.split(lab)
  l = np.power(l / 100, 1 / gamma) * 255
  byte_l = (l).astype(np.uint8)
  undo_bytes = byte_l.astype(np.float32)
  float_ratio = np.where(byte_l > 0, undo_bytes / (l + 1e-6), l)
  new_l = cv2.equalizeHist(byte_l).astype(np.float32)
  new_l = np.power(new_l / 255, gamma) * 100
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


def apply_clahe_image(image, clip_limit, gamma=2.2, restore_color=True):
  """Applies CLAHE contrast enhancement."""
  lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
  l, a, b = cv2.split(lab)
  l = np.power(l / 100, 1 / gamma) * 100
  byte_l = (l).astype(np.uint8)
  undo_bytes = byte_l.astype(np.float32)
  float_ratio = np.where(byte_l > 0, undo_bytes / (l + 1e-6), l)
  tile_grid_size = (8, 8)
  clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
  new_l = clahe.apply(byte_l).astype(np.float32)
  new_l = np.power(new_l / 100, gamma) * 100
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


def convert_grayscale_image(image, name):
  """Converts the image into grayscale."""
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
    gray_image = l / 100
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return np.clip(gray_image, 0, 1)
  elif name in ["hsv", "value"]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    gray_image = v
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return np.clip(gray_image, 0, 1)
  elif name in ["hsl", "lightness"]:
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    _, l, _ = cv2.split(hls)
    gray_image = l
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return np.clip(gray_image, 0, 1)
  elif name in ["laplacian"]:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.Laplacian(gray_image, cv2.CV_32F)
    gray_image = normalize_edge_image(gray_image)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return np.clip(gray_image, 0, 1)
  elif name in ["sobel"]:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    gray_image = cv2.magnitude(sobel_x, sobel_y)
    gray_image = normalize_edge_image(gray_image)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return np.clip(gray_image, 0, 1)
  elif name in ["sharpness"]:
    gray_image = compute_sharpness(image)
    gray_image = normalize_edge_image(gray_image)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return np.clip(gray_image, 0, 1)
  raise ValueError(f"Unknown grayscale name: {name}")


def normalize_edge_image(image):
  """Normalized edge image to be [0,1]."""
  image = np.abs(image)
  mean = np.mean(image)
  std = np.std(image) + 1e-20
  image = (image - mean) / std
  low_perc = 2
  high_perc = 98
  low_val = np.percentile(image, low_perc)
  high_val = np.percentile(image, high_perc)
  image = (image - low_val) / (high_val - low_val)
  return np.clip(image, 0, 1).astype(np.float32)


def bilateral_denoise_image(image, radius):
  """Applies bilateral denoise."""
  ksize = math.ceil(2 * radius) + 1
  sigma_color = min(0.05 * math.sqrt(ksize), 0.35)
  sigma_space = 10 * math.sqrt(ksize)
  return cv2.bilateralFilter(image, ksize, sigma_color, sigma_space)


def blur_image_gaussian(image, radius):
  """Applies Gaussian blur."""
  ksize = math.ceil(2 * radius) + 1
  return cv2.GaussianBlur(image, (ksize, ksize), 0)


def blur_image_pyramid(image, levels, decay=0.0, contrast=1.0):
  """Applies pyramid blur."""
  h, w = image.shape[:2]
  levels = min(levels, int(math.log2(min(h, w))) - 1)
  factor = 2 ** levels
  new_h = ((h + factor - 1) // factor) * factor
  new_w = ((w + factor - 1) // factor) * factor
  expanded = cv2.copyMakeBorder(image, 0, new_h - h, 0, new_w - w, cv2.BORDER_REPLICATE)
  pyramid = make_gaussian_pyramid(expanded, levels)
  diffused = pyramid[-1]
  bokehs = [2 ** i for i in range(levels - 1, -1, -1)]
  sum_bokehs = sum(bokehs)
  raw_alphas = [b / sum_bokehs for b in bokehs]
  for i in range(levels - 1, -1, -1):
    size = (pyramid[i].shape[1], pyramid[i].shape[0])
    diffused = cv2.pyrUp(diffused, dstsize=size)
    alpha = raw_alphas[i] * decay + (1 - decay)
    rev_alpha = raw_alphas[levels - i - 1]
    std_ref = pyramid[i].std()
    std_diff = diffused.std()
    gain = std_ref / (std_diff + 1e-5) * contrast
    gain = np.clip(gain, 1.0, 2.0)
    diffused = alpha * diffused * gain + (1 - alpha) * pyramid[i]
  trimmed = diffused[:h, :w]
  return np.clip(trimmed, 0, 1)


def unsharp_image_gaussian(image, radius):
  """Applies unsharp mask by Gaussian blur."""
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
  h, w = image.shape[:2]
  if width * height > h * w:
    interpolation = cv2.INTER_LANCZOS4
  else:
    interpolation = cv2.INTER_AREA
  return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


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
      return r, g, b
    elif len(expr) == 6:
      r, g, b = (int(expr[0:2], 16), int(expr[2:4], 16), int(expr[4:6], 16))
      return r, g, b
  return None


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
  r, g, b = (255, 255, 255)
  if len(fields) > 2:
    color = parse_color_expr(fields[2])
    if color:
      r, g, b = color
  r, g, b = float(r) / 255, float(g) / 255, float(b) / 255
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
  fields = re.split(r'[ ,\|:;]+', expr.strip())
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
  fields = re.split(r'[ ,\|:;]+', expr.strip())
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
  ap.add_argument("--sigmoid", type=float, default=0, metavar="num",
                  help="sigmoidal contrast adjustment."
                  " positive to strengthen, negative to weaken")
  ap.add_argument("--histeq", type=str, default="0", metavar="num",
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
  ap.add_argument("--blur", type=str, default="0", metavar="num",
                  help="apply Gaussian blur by the pixel radius. negative uses pyramid blur")
  ap.add_argument("--unsharp", type=int, default=0, metavar="num",
                  help="apply Gaussian unsharp mask by the pixel radius.")
  ap.add_argument("--trim", default="", metavar="numlist",
                  help="trim sides: TOP,LEFT,BOTTOM,RIGHT in percentage eg. 5,10,3,7")
  ap.add_argument("--pers", default="", metavar="numlist",
                  help="perspective change: X1,Y1,X2,Y2,X3,Y3,X4,Y4 in percentage"
                  " eg. 10,10,10,90,0,100,100,0")
  ap.add_argument("--scale", default="", metavar="numlist",
                  help="scale change: WIDTH,HEIGHT in pixels eg. 1920,1080")
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
    if "nfeatures" in align_params:
      kwargs["nfeatures"] = int(align_params["nfeatures"])
    if "shift_limit" in align_params:
      kwargs["shift_limit"] = float(align_params["shift_limit"])
    if "denoise" in align_params:
      kwargs["denoise"] = int(align_params["denoise"])
    images = align_images_orb(images, aligned_indices, **kwargs)
  elif align_name in ["sift", "s"]:
    logger.info(f"Aligning images by SIFT")
    kwargs = {}
    if "nfeatures" in align_params:
      kwargs["nfeatures"] = int(align_params["nfeatures"])
    if "shift_limit" in align_params:
      kwargs["shift_limit"] = float(align_params["shift_limit"])
    if "denoise" in align_params:
      kwargs["denoise"] = int(align_params["denoise"])
    images = align_images_sift(images, aligned_indices, **kwargs)
  elif align_name in ["ecc", "e"]:
    logger.info(f"Aligning images by ECC")
    kwargs = {}
    if "use_affine" in align_params:
      kwargs["use_affine"] = parse_boolean(align_params["use_affine"])
    if "denoise" in align_params:
      kwargs["denoise"] = int(align_params["denoise"])
    images = align_images_ecc(images, aligned_indices, **kwargs)
  elif align_name in ["hugin", "h"]:
    logger.info(f"Aligning images by Hugin")
    images = align_images_hugin(images, args.inputs, bits_list)
  else:
    raise ValueError(f"Unknown align method")
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
    if ext in EXTS_NPZ:
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
  images = [edit_image(image, args) for image in images]
  logger.info(f"Saving the output file as a NumPy compressed")
  save_npz(args.output, images)


def postprocess_video(args, images):
  """Postprocess images as a video."""
  images = [edit_image(image, args) for image in images]
  logger.info(f"Saving the output file as a video")
  save_video(args.output, images, args.output_video_fps)
  if has_command(CMD_EXIFTOOL):
    logger.info(f"Copying metadata")
    copy_metadata(args.inputs[0], args.output)


def crop_to_match(image, target_size):
  """Crops the center of the image to match the target size."""
  h, w = image.shape[:2]
  th, tw = target_size
  y_offset = max((h - th) // 2, 0)
  x_offset = max((w - tw) // 2, 0)
  return image[y_offset:y_offset+th, x_offset:x_offset+tw]


def log_image_stats(image, prefix):
  has_nan = np.isnan(image).any()
  if has_nan:
    image = fix_overflown_image(image)
  min = np.min(image)
  max = np.max(image)
  mean = np.mean(image)
  stddev = np.std(image)
  logger.debug(f"{prefix} stats: min={min:.3f}, max={max:.3f},"
               f" mean={mean:.3f}, stddev={stddev:.3f}, nan={has_nan}")


def edit_image(image, args):
  """Edits an image."""
  if args.fill_margin:
    logger.info(f"Filling the margin")
    image = fill_black_margin_image(image)
  if args.gamma != 1.0 and args.gamma > 0:
    logger.info(f"Adjust brightness by a gamma")
    image = apply_gamma_image(image, args.gamma)
  if args.slog != 0:
    logger.info(f"Adjust brightness by a scaled log")
    image = apply_scaled_log_image(image, args.slog)
  if args.sigmoid != 0:
    logger.info(f"Adjust brightness by a sigmoid")
    image = apply_sigmoid_image(image, args.sigmoid, 0.5)
  histeq_params = parse_num_opts_expression(args.histeq)
  histeq_num = histeq_params["num"]
  if histeq_num > 0:
    logger.info(f"Applying CLAHE enhancement")
    kwargs = {}
    if "gamma" in histeq_params:
      kwargs["gamma"] = float(histeq_params["gamma"])
    if "no-restore_color" in histeq_params:
      kwargs["restore_color"] = False
    image = apply_clahe_image(image, histeq_num, **kwargs)
  elif histeq_num < 0:
    logger.info(f"Applying global HE enhancement")
    kwargs = {}
    if "gamma" in histeq_params:
      kwargs["gamma"] = float(histeq_params["gamma"])
    if "no-restore_color" in histeq_params:
      kwargs["restore_color"] = False
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
    if "decay" in blur_params:
      kwargs["decay"] = float(blur_params["decay"])
    if "contrast" in blur_params:
      kwargs["contrast"] = int(blur_params["contrast"])
    image = blur_image_pyramid(image, int(blur_num) * -1, **kwargs)
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
  if len(args.caption) > 0:
    logger.info(f"Writing the caption")
    image = write_caption(image, args.caption)
  return image


def postprocess_images(args, images, bits_list, meta_list, mean_brightness):
  """Postprocesses images as a merged image."""
  merge_params = parse_name_opts_expression(args.merge)
  merge_name = merge_params["name"]
  is_hdr = False
  if merge_name in ["average", "a", "mean"]:
    logger.info(f"Merging images by average composition")
    merged_image = merge_images_average(images)
  elif merge_name in ["median", "mdn"]:
    logger.info(f"Merging images by median composition")
    merged_image = merge_images_median(images)
  elif merge_name in ["geomean", "gm"]:
    logger.info(f"Merging images by geometric mean composition")
    merged_image = merge_images_geometric_mean(images)
  elif merge_name in ["minimum", "min"]:
    logger.info(f"Merging images by minimum composition")
    merged_image = merge_images_minimum(images)
  elif merge_name in ["maximum", "max"]:
    logger.info(f"Merging images by maximum composition")
    merged_image = merge_images_maximum(images)
  elif merge_name in ["denoise", "dn", "bgmbm"]:
    logger.info(f"Merging images by denoise composition")
    kwargs = {}
    if "clip_limit" in merge_params:
      kwargs["clip_limit"] = float(merge_params["clip_limit"])
    if "blur_radius" in merge_params:
      kwargs["blur_radius"] = int(merge_params["blur_radius"])
    merged_image = merge_images_denoise(images, **kwargs)
  elif merge_name in ["weighted", "w"]:
    logger.info(f"Merging images by weighted average composition")
    merged_image = merge_images_weighted_average(images, meta_list)
  elif merge_name in ["debevec", "d"]:
    logger.info(f"Merging images by Debevec's method as an HDRI")
    merged_image = merge_images_debevec(images, meta_list)
    is_hdr = True
  elif merge_name in ["robertson", "r"]:
    logger.info(f"Merging images by Robertson's method")
    merged_image = merge_images_robertson(images, meta_list)
    is_hdr = True
  elif merge_name in ["mertens", "m"]:
    logger.info(f"Merging images by Mertens's method")
    merged_image = merge_images_mertens(images)
    is_hdr = True
  elif merge_name in ["focus", "f"]:
    logger.info(f"Merging images by focus stacking")
    kwargs = {}
    if "smoothness" in merge_params:
      kwargs["smoothness"] = float(merge_params["smoothness"])
    if "pyramid_levels" in merge_params:
      kwargs["pyramid_levels"] = int(merge_params["pyramid_levels"])
    merged_image = merge_images_focus_stacking(images, **kwargs)
  elif merge_name in ["grid", "g"]:
    logger.info(f"Merging images in a grid")
    kwargs = {}
    if "columns" in merge_params:
      kwargs["columns"] = int(merge_params["columns"])
    if "margin" in merge_params:
      kwargs["margin"] = int(merge_params["margin"])
    if "background" in merge_params:
      color = parse_color_expr(merge_params["background"])
      if color:
        kwargs["background"] = (color[0] / 255, color[1] / 255, color[2] / 255)
    merged_image = merge_images_grid(images, **kwargs)
  elif merge_name in ["stitch", "s"]:
    logger.info(f"Stitching images as a panoramic photo")
    merged_image = merge_images_stitch(images)
  else:
    raise ValueError(f"Unknown merge method")
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
      raise ValueError(f"Unknown tone method")
    if logger.isEnabledFor(logging.DEBUG):
      log_image_stats(merged_image, "tonemapped")
    merged_image = fix_overflown_image(merged_image)
  if not args.no_restore:
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
