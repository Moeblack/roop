import sys
from typing import List, Any
import importlib
import multiprocessing
import torch

import roop.globals
import roop.ui as ui

if 'ROCMExecutionProvider' in roop.globals.execution_providers:
    del torch

FRAME_PROCESSOR_MODULES = []

def get_frame_processor_modules(processors: List[str]) -> List[Any]:
    global FRAME_PROCESSOR_MODULES
    module_names = [f'processors.{module}' for module in processors]
    try:
        for module in module_names:
            loaded_module = importlib.import_module(module)
            FRAME_PROCESSOR_MODULES.append(loaded_module)
    except ImportError:
        print('Failed when importing module: ImportError')
        sys.exit()
    return FRAME_PROCESSOR_MODULES


def release_resources() -> None:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        torch.cuda.empty_cache()


def conditional_process_video(source_path: str, temp_frame_paths: List[str], process_video) -> None:
    pool_amount = len(temp_frame_paths) // roop.globals.cpu_cores
    if pool_amount > 2 and roop.globals.cpu_cores > 1 and roop.globals.execution_providers == ['CPUExecutionProvider']:
        POOL = multiprocessing.Pool(roop.globals.cpu_cores, maxtasksperchild=1)
        pools = []
        for i in range(0, len(temp_frame_paths), pool_amount):
            pool = POOL.apply_async(process_video, args=(source_path, temp_frame_paths[i:i + pool_amount], 'multi-processing'))
            pools.append(pool)
        for pool in pools:
            pool.get()
        POOL.close()
        POOL.join()
    else:
         process_video(roop.globals.source_path, temp_frame_paths, 'multi-threading')


def update_status(message: str) -> None:
    value = 'Status: ' + message
    print(value)
    if not roop.globals.headless:
        ui.update_status(value)


def process_image():
    for module in FRAME_PROCESSOR_MODULES:
        update_status(f'{module.__name__} in progress...')
        module.process_image(roop.globals.source_path, roop.globals.target_path, roop.globals.output_path)


def process_video(frame_paths):
    for module in FRAME_PROCESSOR_MODULES:
        update_status(f'{module.__name__} in progress...')
        conditional_process_video(roop.globals.source_path, frame_paths, module.process_video)
        release_resources()


def module_pre_check()