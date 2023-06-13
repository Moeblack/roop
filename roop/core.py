#!/usr/bin/env python3

import os
import sys
# single thread doubles performance of gpu-mode - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List, Any
import platform
import signal
import shutil
import argparse
import psutil
import torch
import onnxruntime
import tensorflow
import multiprocessing
from opennsfw2 import predict_video_frames, predict_image
import cv2
import importlib

import roop.globals
import roop.ui as ui
from roop.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp
from roop.analyser import get_one_face
from roop.processors import get_frame_processor_modules, process_image, process_video

if 'ROCMExecutionProvider' in roop.globals.execution_providers:
    del torch

warnings.simplefilter(action='ignore', category=FutureWarning)


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--face', help='use a face image', dest='source_path')
    parser.add_argument('-t', '--target', help='replace image or video with face', dest='target_path')
    parser.add_argument('-o', '--output', help='save output to this file', dest='output_path')
    parser.add_argument('--frame-processor', help='list of frame processors to run', dest='frame_processor', default=['face_swapper'], choices=['face_swapper', 'face_enhancer'], nargs='+')
    parser.add_argument('--keep-fps', help='maintain original fps', dest='keep_fps', action='store_true', default=False)
    parser.add_argument('--keep-audio', help='maintain original audio', dest='keep_audio', action='store_true', default=True)
    parser.add_argument('--keep-frames', help='keep frames directory', dest='keep_frames', action='store_true', default=False)
    parser.add_argument('--many-faces', help='swap every face in the frame', dest='many_faces', action='store_true', default=False)
    parser.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libx264', choices=['libx264', 'libx265'])
    parser.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int, default=18)
    parser.add_argument('--max-memory', help='maximum amount of RAM in GB to be used', dest='max_memory', type=int, default=suggest_max_memory())
    parser.add_argument('--cpu-cores', help='number of CPU cores to use', dest='cpu_cores', type=int, default=suggest_cpu_cores())
    parser.add_argument('--execution-provider', help='execution provider', dest='execution_provider', default=['CPUExecutionProvider'], choices=onnxruntime.get_available_providers(), nargs='+')
    parser.add_argument('--execution-threads', help='number of threads to be use for the GPU', dest='execution_threads', type=int, default=suggest_execution_threads())

    args = parser.parse_known_args()[0]

    roop.globals.source_path = args.source_path
    roop.globals.target_path = args.target_path
    roop.globals.output_path = args.output_path
    roop.globals.frame_processors = args.frame_processor
    roop.globals.headless = args.source_path or args.target_path or args.output_path
    roop.globals.keep_fps = args.keep_fps
    roop.globals.keep_audio = args.keep_audio
    roop.globals.keep_frames = args.keep_frames
    roop.globals.many_faces = args.many_faces
    roop.globals.video_encoder = args.video_encoder
    roop.globals.video_quality = args.video_quality
    roop.globals.max_memory = args.max_memory
    roop.globals.cpu_cores = args.cpu_cores
    roop.globals.execution_providers = args.execution_provider
    roop.globals.execution_threads = args.execution_threads

    if 'CUDAExecutionProvider' in roop.globals.execution_providers and 'face_enhancer' in roop.globals.frame_processors:
        roop.globals.frame_processors.remove('face_enhancer')


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_cpu_cores() -> int:
    if platform.system().lower() == 'darwin':
        return 2
    return int(max(psutil.cpu_count() / 2, 1))


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in roop.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in roop.globals.execution_providers:
        return 2
    return 8


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def pre_check() -> None:
    if sys.version_info < (3, 9):
        quit('Python version is not supported - please upgrade to 3.9 or higher.')
    if not shutil.which('ffmpeg'):
        quit('ffmpeg is not installed.')


#del
def update_status(message: str) -> None:
    value = 'Status: ' + message
    print(value)
    if not roop.globals.headless:
        ui.update_status(value)


def start() -> None:
    if not roop.globals.source_path or not os.path.isfile(roop.globals.source_path):
        update_status('Select an image that contains a face.')
        return
    elif not roop.globals.target_path or not os.path.isfile(roop.globals.target_path):
        update_status('Select an image or video target!')
        return
    test_face = get_one_face(cv2.imread(roop.globals.source_path))
    if not test_face:
        update_status('No face detected in source image. Please try with another one!')
        return
    # process image to image
    if has_image_extension(roop.globals.target_path):
        if predict_image(roop.globals.target_path) > 0.85:
            destroy()
        process_image()
        """ 
        if 'face-swapper' in roop.globals.frame_processors:
            update_status('Swapping in progress...')
            processors.face_swapper.process_image(roop.globals.source_path, roop.globals.target_path, roop.globals.output_path)
        if 'CUDAExecutionProvider' in roop.globals.execution_providers and 'face-enhancer' in roop.globals.frame_processors:
            update_status('Enhancing in progress...')
            processors.face_enhancer.process_image(roop.globals.source_path, roop.globals.target_path, roop.globals.output_path)
        """
        if is_image(roop.globals.target_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return
    # process image to videos
    seconds, probabilities = predict_video_frames(video_path=roop.globals.target_path, frame_interval=100)
    if any(probability > 0.85 for probability in probabilities):
        destroy()
    update_status('Creating temp resources...')
    create_temp(roop.globals.target_path)
    update_status('Extracting frames...')
    extract_frames(roop.globals.target_path)
    temp_frame_paths = get_temp_frame_paths(roop.globals.target_path)
    process_video(temp_frame_paths)

    # if 'face-swapper' in roop.globals.frame_processors:
    #     update_status('Swapping in progress...')
    #     conditional_process_video(roop.globals.source_path, temp_frame_paths, processors.face_swapper.process_video)
    # release_resources()
    # # limit to one execution thread
    # roop.globals.execution_threads = 1
    # if 'CUDAExecutionProvider' in roop.globals.execution_providers and 'face-enhancer' in roop.globals.frame_processors:
    #     update_status('Enhancing in progress...')
    #     conditional_process_video(roop.globals.source_path, temp_frame_paths, processors.face_enhancer.process_video)
    # release_resources()
    if roop.globals.keep_fps:
        update_status('Detecting fps...')
        fps = detect_fps(roop.globals.target_path)
        update_status(f'Creating video with {fps} fps...')
        create_video(roop.globals.target_path, fps)
    else:
        update_status('Creating video with 30.0 fps...')
        create_video(roop.globals.target_path)
    if roop.globals.keep_audio:
        if roop.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(roop.globals.target_path, roop.globals.output_path)
    else:
        move_temp(roop.globals.target_path, roop.globals.output_path)
    clean_temp(roop.globals.target_path)
    if is_video(roop.globals.target_path):
        update_status('Processing to video succeed!')
    else:
        update_status('Processing to video failed!')


def destroy() -> None:
    if roop.globals.target_path:
        clean_temp(roop.globals.target_path)
    quit()


def run() -> None:
    parse_args()
    pre_check()

    limit_resources()
    if roop.globals.headless:
        start()
    else:
        window = ui.init(start, destroy)
        window.mainloop()
