#! /usr/bin/python3
# -*- coding: utf-8 -*-
##################################################################################################
# Copyright (c) 2026 Mikio Hirabayashi
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
import stat
import subprocess
import sys
import tempfile

import cv2


PROG_NAME = "shrink_large_files.py"
PROG_VERSION = "0.0.3"
DEFAULT_TARGET_SIZE = 800 * 1024
SPECULATIVE_AREA_RATIO_THRESHOLD = 0.70
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ITB_STACK_PATH = os.path.join(SCRIPT_DIR, "itb_stack.py")


logging.basicConfig(format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(PROG_NAME)
logger.setLevel(logging.INFO)


def parse_size(text):
  """Parses a file size expression and returns its size in bytes."""
  match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kKmM]?)(?:[iI]?[bB])?\s*", text)
  if not match:
    raise argparse.ArgumentTypeError(f"invalid size expression: {text}")
  value = float(match.group(1))
  unit = match.group(2).lower()
  multiplier = 1
  if unit == "k":
    multiplier = 1024
  elif unit == "m":
    multiplier = 1024 * 1024
  size = round(value * multiplier)
  if size <= 0:
    raise argparse.ArgumentTypeError("target size must be greater than zero")
  return size


def format_size(size):
  """Formats a file size in binary units."""
  if size >= 1024 * 1024:
    return f"{size / (1024 * 1024):.1f}MiB"
  if size >= 1024:
    return f"{size / 1024:.1f}KiB"
  return f"{size}B"


def make_command_environment():
  """Makes an environment whose PATH includes the script directory."""
  env = os.environ.copy()
  current_path = env.get("PATH", "")
  if current_path:
    env["PATH"] = current_path + os.pathsep + SCRIPT_DIR
  else:
    env["PATH"] = SCRIPT_DIR
  return env


def get_output_path(input_path):
  """Gets the WebP output path corresponding to an input path."""
  stem, ext = os.path.splitext(input_path)
  if ext.lower() == ".webp":
    return input_path
  return stem + ".webp"


def get_scaled_size(input_path, source_size, target_size):
  """Gets dimensions whose area is reduced by the target-to-source size ratio."""
  image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
  if image is None:
    raise ValueError(f"failed to read image: {input_path}")
  height, width = image.shape[:2]
  area_ratio = target_size / source_size
  scale = math.sqrt(area_ratio)
  new_width = max(math.floor(width * scale), 1)
  new_height = max(math.floor(height * scale), 1)
  if new_width == width and new_height == height:
    if width >= height and width > 1:
      new_width -= 1
    elif height > 1:
      new_height -= 1
  return width, height, new_width, new_height


def should_try_speculative_no_resize(source_size, target_size):
  """Checks whether the estimated resized area would retain more than 70%."""
  area_ratio = target_size / source_size
  return area_ratio > SPECULATIVE_AREA_RATIO_THRESHOLD


def make_temporary_path(output_path):
  """Makes a temporary WebP path in the output directory."""
  output_dir = os.path.dirname(os.path.abspath(output_path))
  output_name = os.path.basename(output_path)
  fd, temporary_path = tempfile.mkstemp(
    prefix=f".{output_name}.", suffix=".tmp.webp", dir=output_dir)
  os.close(fd)
  return temporary_path


def run_itb_stack(input_path, temporary_path, env, width=None, height=None):
  """Runs itb_stack.py to save an image as WebP, optionally scaling it."""
  command = [
    sys.executable,
    ITB_STACK_PATH,
    "--output", temporary_path,
  ]
  if width is not None or height is not None:
    if width is None or height is None:
      raise ValueError("width and height must be both specified or both omitted")
    command += ["--scale", f"{width},{height}"]
  command += ["--", input_path]
  subprocess.run(command, check=True, env=env)


def replace_file(input_path, output_path, temporary_path, original_mode):
  """Atomically installs the new file and removes the old name when necessary."""
  os.chmod(temporary_path, original_mode)
  if output_path == input_path:
    os.replace(temporary_path, input_path)
    return
  os.replace(temporary_path, output_path)
  os.unlink(input_path)


def attempt_conversion(input_path, temporary_path, env, width=None, height=None):
  """Creates a temporary converted file and returns its size."""
  run_itb_stack(input_path, temporary_path, env, width=width, height=height)
  if not os.path.isfile(temporary_path):
    raise ValueError("itb_stack.py did not create the output file")
  output_size = os.path.getsize(temporary_path)
  if output_size <= 0:
    raise ValueError("itb_stack.py created an empty output file")
  return output_size


def process_file(input_path, target_size, env):
  """Shrinks one file when it exceeds the target size."""
  if not os.path.isfile(input_path):
    raise ValueError(f"not a regular file: {input_path}")
  source_stat = os.stat(input_path)
  source_size = source_stat.st_size
  if source_size <= target_size:
    logger.info(f"SKIP {input_path}: {format_size(source_size)}")
    return False

  output_path = get_output_path(input_path)
  original_mode = stat.S_IMODE(source_stat.st_mode)

  if should_try_speculative_no_resize(source_size, target_size):
    temporary_path = make_temporary_path(output_path)
    try:
      logger.info(
        f"TRY   {input_path}: speculative no-resize recompression "
        f"for {format_size(source_size)} -> <= {format_size(target_size)}")
      output_size = attempt_conversion(input_path, temporary_path, env)
      if output_size <= target_size:
        replace_file(input_path, output_path, temporary_path, original_mode)
        temporary_path = None
        logger.info(f"DONE  {output_path}: {format_size(output_size)} (no resize)")
        return True
      logger.info(
        f"MISS  {input_path}: speculative no-resize result {format_size(output_size)} "
        f"still exceeds {format_size(target_size)}")
    finally:
      if temporary_path and os.path.exists(temporary_path):
        os.unlink(temporary_path)

  width, height, new_width, new_height = get_scaled_size(
    input_path, source_size, target_size)
  temporary_path = make_temporary_path(output_path)
  try:
    logger.info(
      f"SHRINK {input_path}: {format_size(source_size)}, "
      f"{width}x{height} -> {new_width}x{new_height}")
    output_size = attempt_conversion(
      input_path, temporary_path, env, width=new_width, height=new_height)
    replace_file(input_path, output_path, temporary_path, original_mode)
    temporary_path = None
    logger.info(f"DONE  {output_path}: {format_size(output_size)}")
    if output_size > target_size:
      logger.warning(
        f"WARNING {output_path}: output still exceeds target "
        f"{format_size(target_size)}")
    return True
  finally:
    if temporary_path and os.path.exists(temporary_path):
      os.unlink(temporary_path)


def make_ap_args():
  """Makes arguments of the argument parser."""
  description = "Shrink oversized images received as newline-separated paths on standard input."
  version_msg = f"{PROG_NAME} version {PROG_VERSION}."
  ap = argparse.ArgumentParser(
    prog=PROG_NAME, description=description, epilog=version_msg,
    formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
  ap.add_argument("--version", action="version", version=version_msg)
  ap.add_argument(
    "--target-size", type=parse_size, default=DEFAULT_TARGET_SIZE, metavar="size",
    help="target file size; k means KiB and m means MiB (default=800k)")
  return ap.parse_args()


def main():
  """Runs the command."""
  args = make_ap_args()
  if not os.path.isfile(ITB_STACK_PATH):
    logger.error(f"ERROR missing command: {ITB_STACK_PATH}")
    return 1
  env = make_command_environment()
  num_errors = 0
  for line in sys.stdin:
    input_path = line.rstrip("\r\n")
    if not input_path:
      continue
    try:
      process_file(input_path, args.target_size, env)
    except Exception as error:
      logger.error(f"ERROR {input_path}: {error}")
      num_errors += 1
  return 1 if num_errors else 0


if __name__ == "__main__":
  sys.exit(main())
