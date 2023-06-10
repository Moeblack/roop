import os
import cv2
import numpy as np
import torch
import threading
from tqdm import tqdm
import roop.globals

from torchvision.transforms.functional import normalize

from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
from codeformer.basicsr.utils.download_util import load_file_from_url
from codeformer.basicsr.utils.registry import ARCH_REGISTRY
from codeformer.basicsr.utils import img2tensor, tensor2img

pretrain_model_url = {
    "codeformer": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    "detection": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "parsing": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
    "realesrgan": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
}

# download weights
if not os.path.exists("CodeFormer/weights/CodeFormer/codeformer.pth"):
    load_file_from_url(
        url=pretrain_model_url["codeformer"], model_dir="CodeFormer/weights/CodeFormer", progress=True, file_name=None
    )
if not os.path.exists("CodeFormer/weights/facelib/detection_Resnet50_Final.pth"):
    load_file_from_url(
        url=pretrain_model_url["detection"], model_dir="CodeFormer/weights/facelib", progress=True, file_name=None
    )
if not os.path.exists("CodeFormer/weights/facelib/parsing_parsenet.pth"):
    load_file_from_url(
        url=pretrain_model_url["parsing"], model_dir="CodeFormer/weights/facelib", progress=True, file_name=None
    )
if not os.path.exists("CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth"):
    load_file_from_url(
        url=pretrain_model_url["realesrgan"], model_dir="CodeFormer/weights/realesrgan", progress=True, file_name=None
    )

#FACE_HELPER = None
CODE_FORMER = None
THREAD_LOCK = threading.Lock()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = "CodeFormer/weights/CodeFormer/codeformer.pth"
checkpoint = torch.load(ckpt_path)["params_ema"]

def get_facepaste_enhancer():
    global CODE_FORMER
    with THREAD_LOCK:
        if CODE_FORMER is None:
            CODE_FORMER = ARCH_REGISTRY.get("CodeFormer")(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=["32", "64", "128", "256"],
            ).to(device)
            CODE_FORMER.load_state_dict(checkpoint)
            CODE_FORMER.eval()
        return CODE_FORMER

def get_facepaste_back(FACE_HELPER):
    if FACE_HELPER is None:
        FACE_HELPER = FaceRestoreHelper(
        upscale_factor = int(2),
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        use_parse=True,
        device=device,
    )
    return FACE_HELPER


def enhance_face_in_frame(cropped_faces):
    try:
        for idx, cropped_face in enumerate(cropped_faces):
            face_t = data_preprocess(cropped_face)
            face_enhanced = restore_face(face_t)
            return face_enhanced
    except RuntimeError as error:
        print(f"Failed inference for CodeFormer-code: {error}")


def process_faces(source_face: any, frame: any) -> any:
    try:
        face_helper = get_facepaste_back(None)
        face_helper.read_image(frame)
        # get face landmarks for each face
        face_helper.get_face_landmarks_5(
            only_center_face=False, resize=640, eye_dist_threshold=5
        )
        # align and warp each face
        face_helper.align_warp_face()
        cropped_faces = face_helper.cropped_faces
        face_enhanced = enhance_face_in_frame(cropped_faces)
        face_helper.add_restored_face(face_enhanced)
        face_helper.get_inverse_affine()
        result = face_helper.paste_faces_to_input_image()
        face_helper.clean_all()
        return result
    except RuntimeError as error:
        print(f"Failed inference for CodeFormer-code-paste: {error}")




def data_preprocess(frame):
    frame_t = img2tensor(frame / 255.0, bgr2rgb=True, float32=True)
    normalize(frame_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    return frame_t.unsqueeze(0).to(device)


def generate_output(frame_t, codeformer_fidelity = 0.6):
    with torch.no_grad():
        output = get_facepaste_enhancer()(frame_t, w=codeformer_fidelity, adain=True)[0]
    return output


def postprocess_output(output):
    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
    return restored_face.astype("uint8")


def restore_face(face_t):
    try:
        output = generate_output(face_t)
        restored_face = postprocess_output(output)
        del output
    except RuntimeError as error:
        print(f"Failed inference for CodeFormer-tensor: {error}")
        restored_face = postprocess_output(face_t)
    return restored_face


# def paste_face_back(face_enhanced):
#     try:
#         with THREAD_LOCK:
#             get_facepaste_back().add_restored_face(face_enhanced)
#             get_facepaste_back().get_inverse_affine()
#             enhanced_img = get_facepaste_back().paste_faces_to_input_image()
#         return enhanced_img
#     except:
#         print('error')

        
# def process_faces(source_face: any, frame: any) -> any:
#     try:
#         result = process_faces(frame)
#         get_facepaste_back().clean_all()
#         return result
#     except RuntimeError as error:
#         print(f"Failed inference for CodeFormer-code-paste: {error}")


def process_frames(source_path: str, frame_paths: list[str], progress=None) -> None:
    source_face = None #get_one_face(cv2.imread(source_path))
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        try:
            result = process_faces(source_face, frame)
            cv2.imwrite(frame_path, result)
        except Exception as exception:
            print(exception)
            pass
        if progress:
            progress.update(1)


def multi_process_frame(source_img, frame_paths, progress) -> None:
    threads = []
    frames_per_thread = len(frame_paths) // roop.globals.gpu_threads
    remaining_frames = len(frame_paths) % roop.globals.gpu_threads
    start_index = 0
    # create threads by frames
    for _ in range(roop.globals.gpu_threads):
        end_index = start_index + frames_per_thread
        if remaining_frames > 0:
            end_index += 1
            remaining_frames -= 1
        thread_frame_paths = frame_paths[start_index:end_index]
        thread = threading.Thread(target=process_frames, args=(source_img, thread_frame_paths, progress))
        threads.append(thread)
        thread.start()
        start_index = end_index
    # join threads
    for thread in threads:
        thread.join()


# def process_image(source_path: str, target_path: str, output_file) -> None:
#     frame = cv2.imread(target_path)
#     target_frame = get_one_face(frame)
#     source_face = get_one_face(cv2.imread(source_path))
#     result = get_face_swapper().get(frame, target_frame, source_face, paste_back=True)
#     cv2.imwrite(output_file, result)


def process_video(source_path: str, frame_paths: list[str], mode: str) -> None:
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    total = len(frame_paths)
    with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        if mode == 'cpu':
            progress.set_postfix({'mode': mode, 'cores': roop.globals.cpu_cores, 'memory': roop.globals.max_memory})
            process_frames(source_path, frame_paths, progress)
        elif mode == 'gpu':
            progress.set_postfix({'mode': mode, 'threads': roop.globals.gpu_threads, 'memory': roop.globals.max_memory})
            multi_process_frame(source_path, frame_paths, progress)




