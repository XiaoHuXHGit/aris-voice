import sys
import os
import argparse
import logging
import time

from voice_remover.models.vocal_remover.separator import Separator

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


def inference(
        # 必填参数
        audio_file: str,

        # 选填参数
        debug: bool = False,
        output_format: str = 'wav',
        output_dir: str = 'results',
        use_cpu: bool = False,
        # 模型设置
        model_filename: str = '5_HP-Karaoke-UVR.pth',  # 5_HP人声和声分离模型
        model_file_dir: str = 'voice_remover/pretrain/VR_Models',  # 模型文件
        extra_output_dir: str | None = None,
        invert_spect: bool = True,
        normalization: float = 0.9,
        single_stem: bool | None = None,
        save_another_stem: bool = False,
        # 模型推理设置
        vr_batch_size: int = 4,
        vr_window_size: int = 320,
        vr_aggression: int = 5,
        vr_enable_tta: bool = False,
        vr_high_end_process: bool = False,
        vr_enable_post_process: bool = True,
        vr_post_process_threshold: float = 0.2,
):
    logger = logging.getLogger(__name__)
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter(fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(module)s - %(message)s",
                                      datefmt="%H:%M:%S")
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)

    log_level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(log_level)

    start_time = time.time()
    # logger.info(f"Separator beginning with input file or folder: {audio_file}")
    print(f"音乐分离开始 初始文件夹: {audio_file}")

    separator = Separator(
        log_formatter=log_formatter,
        log_level=log_level,
        model_file_dir=model_file_dir,
        output_dir=output_dir,
        extra_output_dir=extra_output_dir,
        output_format=output_format,
        normalization_threshold=normalization,
        output_single_stem=single_stem,
        invert_using_spec=invert_spect,
        use_cpu=use_cpu,
        save_another_stem=save_another_stem,
        vr_params={
            "batch_size": vr_batch_size,
            "window_size": vr_window_size,
            "aggression": vr_aggression,
            "enable_tta": vr_enable_tta,
            "enable_post_process": vr_enable_post_process,
            "post_process_threshold": vr_post_process_threshold,
            "high_end_process": vr_high_end_process,
        },
    )
    separator.load_model(model_filename=model_filename)
    output_files = separator.separate(audio_file)
    # logger.info(f"Separator finished in {time.time() - start_time:.2f} seconds.")
    # logger.info(f"Results are saved to: {output_files}")
    print(f"分离使用时间： {time.time() - start_time:.2f} 秒.")
    return output_files
