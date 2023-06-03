import shutil
from tqdm import tqdm
from codeformer.app import inference_app

def upscale_video(frame_paths):
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'

    with tqdm(total=len(frame_paths), desc="Processing", unit="frame", dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        for frame_path in frame_paths:
            try:
                result = inference_app(
                    image=frame_path,
                    background_enhance=True,
                    face_upsample=True,
                    upscale=2,
                    codeformer_fidelity=0.7,
                )
                shutil.move(result, frame_path)
            except Exception:
                progress.set_postfix(status='E', refresh=True)
                pass
            progress.update(1)

def upscale_img(output_file):
     result = inference_app(
        image=output_file,
        background_enhance=True,
        face_upsample=True,
        upscale=2,
        codeformer_fidelity=0.7,
     )
     shutil.move(result, output_file)
