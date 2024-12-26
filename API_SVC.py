import os
import io
import json
import torch
import urllib
import uvicorn
import logging
import soundfile
import numpy as np

from scipy.io import wavfile
from pydub import AudioSegment
from fastapi.responses import StreamingResponse, FileResponse
from fastapi import FastAPI, Request, Query, HTTPException, File, UploadFile, Form

# svc 人工智能包导入
from inference import infer_tool
from inference.infer_tool import Svc
from spkmix import spk_mix_map
# 背景音乐声音分离人工智能接口导入
from voice_remover.uvr_inference import inference


class SVCInferenceConfig:
    def __init__(self):
        pass

    # ================ 全局默认参数(没事别碰！) ================
    model_name = 'G_0.pth'  # 模型名
    config_name = 'config.json'  # 模型embeddings设置参数
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 推演设备设置
    cluster_model_name = r'feature_and_index.pkl'  # 聚类/特征检索模型路径
    enhance = False  # NSF-HIFIGAN增强器
    diffusion_model_name = 'model_10000.pt'  # 浅扩散模型
    diffusion_config_name = 'diffusion.yaml'  # 浅扩散模型配置
    shallow_diffusion = False  # 是否启用diffusion模型
    only_diffusion = False  # 只用diffusion模型推演
    use_spk_mix = False  # 使用角色融合
    feature_retrieval = True  # 使用特征检索
    wav_format = 'wav'
    svc_export_path = 'results/svc_export'
    file_name = 'result.wav'

    # ================ 模型默认参数 ================
    model_path = r"logs/44k"
    config_path = r"configs"

    # ================ 模型推理参数 ================
    # 必填参数
    clean_name = ''  # 需要传入的是wav字节流文件
    spk_list = ''  # 说话人名称
    # 可选参数
    tran = 0  # 音高调整，支持正负（半音）
    cluster_infer_ratio = 0.2  # 聚类方案或特征检索占比，范围0-1，若没有训练聚类模型或特征检索则默认0即可
    k_step = 100  # 扩散步数，越大越接近扩散模型的结果
    clip = 0  # 音频强制切片，默认0为自动切片，单位为秒/s
    f0_predictor = 'rmvpe'  # 选择F0预测器
    auto_predict_f0 = False  # 语音转换自动预测音高，转换歌声时不要打开这个会严重跑调
    enhancer_adaptive_key = 0  # 使增强器适应更高的音域(单位为半音数)|默认为0
    linear_gradient = 0  # 两段音频切片的交叉淡入长度，如果强制切片后出现人声不连贯可调整该数值
    loudness_envelope_adjustment = 1  # 输入源响度包络替换输出响度包络融合比例，越靠近1越使用输出响度包络
    # 下面的别动了
    slice_db = -40  # 默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50
    noice_scale = 0.4  # 噪音级别，会影响咬字和音质
    pad_seconds = 0.5  # 推理音频pad秒数
    linear_gradient_retain = 0.75  # 自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭
    f0_filter_threshold = 0.05  # F0过滤阈值，只有使用crepe时有效
    second_encoding = False  # 二次编码，浅扩散前会对原始音频进行二次编码

    # ================ 声音合成参数 ================
    loud_add = 4
    send_format = 'mp3'
    audio_export_path = 'results/audio_export'


class VRInferenceDefaultConfig:
    def __init__(self):
        pass

    '''
    说明：使用vr分离会返回三个音频
    _Instrumental -> _Instrumental
    _Vocals -> (del)
    _Vocals_No Reverb -> _Vocals
    '''

    # 必填参数
    audio_file_path = r"raw"
    audio_file_name = r'Immortals-fall out boy.flac'

    # 选填参数
    debug = False
    output_format = 'wav'
    output_dir = 'results/audio_separate'
    use_cpu = False

    # 模型设置
    vr_model_filename = '5_HP-Karaoke-UVR.pth'  # 5_HP人声和声分离模型
    de_reverb_model_filename = 'UVR-DeEcho-DeReverb.pth'  # DeEcho-DeReverb混响分离模型
    model_file_dir = 'voice_remover/pretrain/VR_Models'  # 模型文件
    extra_output_dir = None
    invert_spect = True
    normalization = 0.9
    single_stem = None
    save_another_stem = True
    # 模型推理设置
    vr_batch_size = 4
    vr_window_size = 320
    vr_aggression = 5
    vr_enable_tta = False
    vr_high_end_process = False
    vr_enable_post_process = True
    vr_post_process_threshold = 0.2


# API设置
app = FastAPI()
host = '172.18.62.224'
port = 9960


# AI模型
# svc_model = None


# 获取角色
def character_get(character: str):
    with open('model_save.json', 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        # print(json_data)
    try:
        return json_data['svc_model'][character]
    except KeyError:
        return None


# 背景音乐、人声分离，混响去除
def music_voice_remove(file_name, path=VRInferenceDefaultConfig.audio_file_path):
    if path is None:
        path = VRInferenceDefaultConfig.audio_file_path
    clean_name, extension = os.path.splitext(file_name)
    output_dir = os.path.join(VRInferenceDefaultConfig.output_dir, clean_name)
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 分离人声
    inference(os.path.join(path, file_name),
              output_dir=output_dir,
              output_format=VRInferenceDefaultConfig.output_format,
              model_filename=VRInferenceDefaultConfig.vr_model_filename)
    vocals_with_reverb = os.path.join(VRInferenceDefaultConfig.output_dir, clean_name,
                                      f'{clean_name}_Vocals.{VRInferenceDefaultConfig.output_format}')
    # 去混响
    inference(audio_file=vocals_with_reverb,
              output_dir=output_dir,
              output_format=VRInferenceDefaultConfig.output_format,
              model_filename=VRInferenceDefaultConfig.de_reverb_model_filename)
    os.remove(vocals_with_reverb)
    os.remove(os.path.join(VRInferenceDefaultConfig.output_dir, clean_name,
                           f'{clean_name}_Vocals_Reverb.{VRInferenceDefaultConfig.output_format}'))
    os.rename(os.path.join(VRInferenceDefaultConfig.output_dir, clean_name,
                           f'{clean_name}_Vocals_No Reverb.{VRInferenceDefaultConfig.output_format}'),
              os.path.join(VRInferenceDefaultConfig.output_dir, clean_name,
                           f'{clean_name}_Vocals_DeReverb.{VRInferenceDefaultConfig.output_format}'))


# svc模型搭载
def svc_model_init(
        model_name=SVCInferenceConfig.model_name,
        config_name=SVCInferenceConfig.config_name,
        device=SVCInferenceConfig.device,
        cluster_model=SVCInferenceConfig.cluster_model_name,
        enhance=SVCInferenceConfig.enhance,
        diffusion_model_name=SVCInferenceConfig.diffusion_model_name,
        diffusion_config_name=SVCInferenceConfig.diffusion_config_name,
        shallow_diffusion=SVCInferenceConfig.shallow_diffusion,
        only_diffusion=SVCInferenceConfig.only_diffusion,
        use_spk_mix=SVCInferenceConfig.use_spk_mix,
        feature_retrieval=SVCInferenceConfig.feature_retrieval
):
    return Svc(
        net_g_path=os.path.join(SVCInferenceConfig.model_path, model_name),
        config_path=os.path.join(SVCInferenceConfig.config_path, config_name),
        device=device,
        cluster_model_path=cluster_model,  # cluster_model,
        nsf_hifigan_enhance=enhance,
        diffusion_model_path=os.path.join(SVCInferenceConfig.model_path, 'diffusion', diffusion_model_name),
        diffusion_config_path=os.path.join(SVCInferenceConfig.config_path, diffusion_config_name),
        shallow_diffusion=shallow_diffusion,
        only_diffusion=only_diffusion,
        spk_mix_enable=use_spk_mix,
        feature_retrieval=feature_retrieval,  # feature_retrieval,
    )


# svc模型推演
def svc_inference(
        # 必填参数
        # clean_name,
        target_audio_path,
        spk,
        # 可选参数
        svc_model_argparse={
            'model_name': SVCInferenceConfig.model_name,
            'config_name': SVCInferenceConfig.config_name,
            'device': SVCInferenceConfig.device,
            'cluster_model': SVCInferenceConfig.cluster_model_name,
            'enhance': SVCInferenceConfig.enhance,
            'diffusion_model_name': SVCInferenceConfig.diffusion_model_name,
            'diffusion_config_name': SVCInferenceConfig.diffusion_config_name,
            'shallow_diffusion': SVCInferenceConfig.shallow_diffusion,
            'only_diffusion': SVCInferenceConfig.only_diffusion,
            'use_spk_mix': SVCInferenceConfig.use_spk_mix,
            'feature_retrieval': SVCInferenceConfig.feature_retrieval
        },
        tran=SVCInferenceConfig.tran,
        clip=SVCInferenceConfig.clip,
        cluster_infer_ratio=SVCInferenceConfig.cluster_infer_ratio,
        # shallow_diffusion=SVCInferenceConfig.shallow_diffusion,
        k_step=SVCInferenceConfig.k_step,
        # only_diffusion=SVCInferenceConfig.only_diffusion,
        auto_predict_f0=SVCInferenceConfig.auto_predict_f0,
        f0_predictor=SVCInferenceConfig.f0_predictor,
        enhancer_adaptive_key=SVCInferenceConfig.enhancer_adaptive_key,
        linear_gradient=SVCInferenceConfig.linear_gradient,
        # 不要动的参数
        slice_db=SVCInferenceConfig.slice_db,
        noice_scale=SVCInferenceConfig.noice_scale,
        pad_seconds=SVCInferenceConfig.pad_seconds,
        linear_gradient_retain=SVCInferenceConfig.linear_gradient,
        f0_filter_threshold=SVCInferenceConfig.f0_filter_threshold,
        use_spk_mix=SVCInferenceConfig.use_spk_mix,
        second_encoding=SVCInferenceConfig.second_encoding,
        loudness_envelope_adjustment=SVCInferenceConfig.loudness_envelope_adjustment,
        wav_format=SVCInferenceConfig.wav_format,
        file_name=SVCInferenceConfig.file_name,
):
    svc_model = svc_model_init(**svc_model_argparse)
    # 你小子，居然要列表
    # clean_name = clean_name.split(",")
    # spk_list = spk_list.split(",")

    infer_tool.mkdir(["raw", "results"])

    # if len(spk_mix_map) <= 1:
    #     use_spk_mix = False
    # if use_spk_mix:
    #     spk_list = [spk_mix_map]

    # infer_tool.fill_a_to_b(tran, clean_name)
    # for clean_name, tran in zip(clean_name, tran):
    #     raw_audio_path = f"raw/{clean_name}"
    #     if "." not in raw_audio_path:
    #         raw_audio_path += ".wav"
    #     infer_tool.format_wav(raw_audio_path)
    #     for spk in spk_list:
    kwarg = {
        "raw_audio_path": target_audio_path,
        "spk": spk,
        "tran": tran,
        "slice_db": slice_db,
        "cluster_infer_ratio": cluster_infer_ratio,  # cluster_infer_ratio,
        "auto_predict_f0": auto_predict_f0,
        "noice_scale": noice_scale,
        "pad_seconds": pad_seconds,
        "clip_seconds": clip,
        "lg_num": linear_gradient,
        "lgr_num": linear_gradient_retain,
        "f0_predictor": f0_predictor,
        "enhancer_adaptive_key": enhancer_adaptive_key,
        "cr_threshold": f0_filter_threshold,
        "k_step": k_step,
        "use_spk_mix": use_spk_mix,
        "second_encoding": second_encoding,
        "loudness_envelope_adjustment": loudness_envelope_adjustment
    }
    audio = svc_model.slice_inference(**kwarg)
    # key = "auto" if auto_predict_f0 else f"{tran}key"
    # cluster_name = "" if cluster_infer_ratio == 0 else f"_{cluster_infer_ratio}"
    # is_diffusion = "sovits"
    # if shallow_diffusion:
    #     is_diffusion = "sovdiff"
    # if only_diffusion:
    #     is_diffusion = "diff"
    # if use_spk_mix:
    #     spk = "spk_mix"
    if not os.path.exists(SVCInferenceConfig.svc_export_path):
        os.makedirs(SVCInferenceConfig.svc_export_path)
    res_path = f'{SVCInferenceConfig.svc_export_path}'
    # file_name = f'{clean_name}_{key}_{spk}{cluster_name}_{is_diffusion}_{f0_predictor}.{wav_format}'
    soundfile.write(os.path.join(res_path, file_name), audio, svc_model.target_sample, format=wav_format)
    # 弃用多处理
    svc_model.clear_empty()
    # # 返回音频流文件
    # audio_stream = io.BytesIO()
    # sample_rate = svc_model.target_sample
    # soundfile.write(audio_stream, audio, sample_rate, format='wav')
    # audio_stream.seek(0)
    # return audio_stream


"""
add_echo：添加回声

delay: 控制回声出现的时间间隔（以秒为单位）。
decay: 控制回声的音量衰减（从 0 到 1 之间，1 表示没有衰减，0 完全消失）。
feedback: 控制回声的递归程度（决定回声的重复层级，越高回声越明显）

分别对应Adobe AU的参数：
delay-延迟时间(delay)
decay-回声电平(volume)
feedback-反馈(feedback)
"""


def add_echo(data, sample_rate, delay, decay, feedback=0):
    num_samples = int(sample_rate * delay)  # 计算延迟样本数
    # 计算输出数据长度
    echo_length = len(data) + num_samples * (feedback + 1)
    echo_data = np.zeros(echo_length)

    # 添加原始声音
    echo_data[:len(data)] += data

    # 添加回声
    current_echo = decay * data
    for i in range(1, feedback + 1):  # 从1开始到feedback次数
        start_index = i * num_samples
        # 确保加到 echo_data 时不超出范围
        if start_index < echo_length:
            echo_data[start_index:start_index + len(current_echo)] += current_echo

        # 更新回声衰减后的音频数据
        current_echo = decay * current_echo

    return echo_data.astype(data.dtype)


def auto_mix(
        AI_audio: AudioSegment,
        # 轨道响度参数
        ai_loud_add,
        # 混响回声参数
        delay,
        decay,
        feedback
) -> AudioSegment:
    # 增加AI生成声音的响度
    AI_audio = AI_audio.apply_gain(ai_loud_add)
    return AI_audio
    # 添加回声
    # AI_audio, sample_rate = np.array(AI_audio.get_array_of_samples()), AI_audio.frame_rate
    # print(f'已得到数据，采样率为：{sample_rate}')
    # AI_audio = add_echo(AI_audio, sample_rate, delay, decay, feedback).tobytes()
    # return AudioSegment(
    #     AI_audio,
    #     frame_rate=sample_rate,
    #     sample_width=2,
    #     channels=1
    # )


# 多轨道音频合成
def multitrack_combination(background_music_path: str,
                           harmony_voice_path: str,
                           AI_voice_list: list,
                           is_maxLength=True,
                           ai_loud_add=0,
                           harmony=False) -> AudioSegment:
    # 加载音频轨道
    background_music = AudioSegment.from_file(background_music_path)
    background_music_len = len(background_music)
    # 等把msst背景音乐分离整出来再说
    if harmony:
        harmony_voice = AudioSegment.from_file(harmony_voice_path)
        harmony_voice_len = len(harmony_voice)
    else:
        harmony_voice = AudioSegment.from_file(background_music_path).silent(duration=background_music_len)
        harmony_voice_len = background_music_len
    AI_voice_len = [len(voice) for voice in AI_voice_list]
    AI_voice = [AudioSegment.from_file(voice_path) for voice_path in AI_voice_list]

    if is_maxLength:
        # 找到最长的音轨长度
        max_length = max(background_music_len, harmony_voice_len, *AI_voice_len)
        # 填充音频轨道
        if len(background_music) < max_length:
            silence = AudioSegment.silent(duration=max_length - background_music_len)
            background_music = background_music + silence
        if len(harmony_voice) < max_length:
            silence = AudioSegment.silent(duration=max_length - harmony_voice_len)
            harmony_voice = harmony_voice + silence
        # 合成音频轨道
        combined_audio = background_music.overlay(harmony_voice)
        # 用同样的方法处理AI生成声音的序列
        for AI_audio, AI_audio_len in zip(AI_voice, AI_voice_len):
            if len(AI_audio) < max_length:
                silence = AudioSegment.silent(duration=max_length - AI_audio_len)
                AI_audio = AI_audio + silence
            # 合成音频轨道
            combined_audio = combined_audio.overlay(auto_mix(
                AI_audio, ai_loud_add, delay=0.17, decay=0.12, feedback=2
            ))
    else:
        # 找到最短的音轨长度
        min_length = min(background_music_len, harmony_voice_len, *AI_voice_len)
        # 填充音频轨道，使每个音轨的长度都与最短音轨一致
        if len(background_music) > min_length:
            background_music = background_music[:min_length]
        if len(harmony_voice) > min_length:
            harmony_voice = harmony_voice[:min_length]
        # 合成音频轨道
        combined_audio = background_music.overlay(harmony_voice)
        # 用同样的方法处理AI生成的声音序列
        for AI_audio, AI_audio_len in zip(AI_voice, AI_voice_len):
            if AI_audio_len > min_length:
                AI_audio = AI_audio[:min_length]
            # 合成音频轨道
            combined_audio = combined_audio.overlay(auto_mix(
                AI_audio, ai_loud_add, delay=0.17, decay=0.12, feedback=2
            ))
        # 返回合成的音频轨道
    return combined_audio


# 一键处理函数
def audio_process(
        # 必填参数
        file_name,
        spk_list: list | str,
        # 可选参数
        tran: list | str = SVCInferenceConfig.tran,
        clip=SVCInferenceConfig.clip,
        cluster_infer_ratio=SVCInferenceConfig.cluster_infer_ratio,
        shallow_diffusion=SVCInferenceConfig.shallow_diffusion,
        k_step=SVCInferenceConfig.k_step,
        only_diffusion=SVCInferenceConfig.only_diffusion,
        auto_predict_f0=SVCInferenceConfig.auto_predict_f0,
        f0_predictor=SVCInferenceConfig.f0_predictor,
        enhancer_adaptive_key=SVCInferenceConfig.enhancer_adaptive_key,
        linear_gradient=SVCInferenceConfig.linear_gradient,
        loud_add=SVCInferenceConfig.loud_add,
        send_format=SVCInferenceConfig.send_format,
        feature_retrieval=SVCInferenceConfig.feature_retrieval,
        # 不要动的参数
        slice_db=SVCInferenceConfig.slice_db,
        noice_scale=SVCInferenceConfig.noice_scale,
        pad_seconds=SVCInferenceConfig.pad_seconds,
        linear_gradient_retain=SVCInferenceConfig.linear_gradient_retain,
        f0_filter_threshold=SVCInferenceConfig.f0_filter_threshold,
        use_spk_mix=SVCInferenceConfig.use_spk_mix,
        second_encoding=SVCInferenceConfig.second_encoding,
        loudness_envelope_adjustment=SVCInferenceConfig.loudness_envelope_adjustment,
        wav_format=SVCInferenceConfig.wav_format,
        is_gradio_argparse=False
):
    character_svc_model_config = character_get(spk_list)
    if character_svc_model_config is None:
        print("找不到角色")
        return JSONResponse(
            status_code=400,
            content={"message": f"character error: character named \"{spk_list}\" not found"})
    clean_name, extension = os.path.splitext(os.path.basename(file_name) if is_gradio_argparse else file_name)
    complete_voice_path = {
        'instrumental': os.path.join(VRInferenceDefaultConfig.output_dir, clean_name,
                                     f'{clean_name}_Instrumental.{VRInferenceDefaultConfig.output_format}'),
        'prompt_audio': os.path.join(VRInferenceDefaultConfig.output_dir, clean_name,
                                     f'{clean_name}_Vocals_DeReverb.{VRInferenceDefaultConfig.output_format}'),
    }
    if not os.path.exists(complete_voice_path['instrumental']) and not os.path.exists(
            complete_voice_path['prompt_audio']):
        print('启动第一阶段VOICE REMOVE推演')
        music_voice_remove(clean_name + extension, path=os.path.dirname(file_name) if is_gradio_argparse else None)
        print('人工智能VOICE REMOVER推演完成')

    # SVC推演，文件命名
    if isinstance(tran, str):
        tran = eval(tran)
    key = "auto" if auto_predict_f0 else f"{tran}key"
    is_diffusion = "sovits"
    if shallow_diffusion:
        is_diffusion = "sovdiff"
    if only_diffusion:
        is_diffusion = "diff"
    cluster_name = "" if cluster_infer_ratio == 0 else f"_{cluster_infer_ratio}"

    spk_name = character_svc_model_config["speaker_name"]
    svc_process_file_name = f'{clean_name}_{key}_{spk_name}{cluster_name}_{is_diffusion}_{f0_predictor}.{wav_format}'
    svc_process_path = os.path.join(SVCInferenceConfig.svc_export_path, svc_process_file_name)
    if not os.path.exists(svc_process_path):
        # svc主推演模型
        cluster_model = os.path.join(SVCInferenceConfig.model_path,
                                     character_svc_model_config['cluster_model']['feature_and_index'])
        # 如果没有特征模型那就直接取消特征模型推演
        if not (cluster_infer_ratio and (character_svc_model_config['cluster_model']['feature_and_index'] or
                                         character_svc_model_config['cluster_model']['k_means'])):
            print('\033[31m' + f"没有聚类/特征检索模型" + '\033[0m')
            cluster_model = ""
            cluster_infer_ratio = 0
            feature_retrieval = False
        else:
            print('\033[31m' + f"检测到聚类/特征检索模型" + '\033[0m')
            print(
                f'cluster_model = {cluster_model}\ncluster_infer_ratio = {cluster_infer_ratio}\nfeature_retrieval = {feature_retrieval}')

        svc_inference(
            # clean_name,
            complete_voice_path['prompt_audio'],
            # 这里记得把默认参数全改了
            character_svc_model_config['speaker_name'],
            {
                'model_name': character_svc_model_config['model_name'],
                'config_name': character_svc_model_config['config_name'],
                'device': SVCInferenceConfig.device,
                'cluster_model': cluster_model,
                'enhance': SVCInferenceConfig.enhance,
                'diffusion_model_name': character_svc_model_config['diffusion_model_name'],
                'diffusion_config_name': character_svc_model_config['diffusion_config_name'],
                'shallow_diffusion': SVCInferenceConfig.shallow_diffusion,
                'only_diffusion': SVCInferenceConfig.only_diffusion,
                'use_spk_mix': SVCInferenceConfig.use_spk_mix,
                'feature_retrieval': feature_retrieval
            },
            # 可选参数
            tran,
            clip,
            cluster_infer_ratio,
            # shallow_diffusion,
            k_step,
            # only_diffusion,
            auto_predict_f0,
            f0_predictor,
            enhancer_adaptive_key,
            linear_gradient,
            # 不要动的参数
            slice_db,
            noice_scale,
            pad_seconds,
            linear_gradient_retain,
            f0_filter_threshold,
            use_spk_mix,
            second_encoding,
            loudness_envelope_adjustment,
            wav_format,
            svc_process_file_name,
        )
    loudness = f'{loud_add}db'
    export_file_name = f'{clean_name}_{key}_{spk_name}{cluster_name}_{is_diffusion}_{f0_predictor}_{loudness}.{wav_format}'
    export_path = f"{SVCInferenceConfig.audio_export_path}/{export_file_name}"
    # send_format = InferenceConfig.send_format
    send_file_name, send_path = export_file_name, export_path
    if not os.path.exists(SVCInferenceConfig.audio_export_path):
        os.makedirs(SVCInferenceConfig.audio_export_path)
    if send_format != 'wav':
        send_file_name = send_file_name.replace(f'.{wav_format}', f'.{send_format}')
        send_path = send_path.replace(f'.{wav_format}', f'.{send_format}')
    if not os.path.exists(send_path):
        print(f'启动第三阶段 多轨道合成')
        print(f'合成音频 轨道数：{len([svc_process_path]) + 2}')
        # 多轨道合成
        combined = multitrack_combination(
            background_music_path=complete_voice_path['instrumental'],
            harmony_voice_path='',
            AI_voice_list=[svc_process_path],
            is_maxLength=True,
            ai_loud_add=loud_add,
        )

        # 导出合成后的音频
        # combined.export(export_path, format="wav")  # 导出wav无损格式
        combined.export(send_path, format=send_format)  # 导出设定的格式
        print(f'多轨道音频合成完毕')
    return send_path, send_file_name, send_format


def file_response(send_path, send_file_name, send_format):
    media_type = 'mpeg' if send_format == 'mp3' else send_format
    response = FileResponse(send_path, media_type="audio/" + media_type)
    # print(f"inline; filename={urllib.parse.quote(send_file_name)}")
    response.headers["Content-Disposition"] = f"inline; filename={urllib.parse.quote(send_file_name)};"
    response.headers["Accept-Ranges"] = "bytes"
    return response


# @app.post('/change_model')
# def svc_model_change(
# ):
#     global svc_model
#     svc_model.clear_empty()
#     svc_model = svc_model_init()


@app.get("/")
async def api_get(
        # 必填参数
        file_name,
        spk_list,
        # 可选参数
        tran=SVCInferenceConfig.tran,
        clip=SVCInferenceConfig.clip,
        cluster_infer_ratio=SVCInferenceConfig.cluster_infer_ratio,
        shallow_diffusion=SVCInferenceConfig.shallow_diffusion,
        k_step=SVCInferenceConfig.k_step,
        only_diffusion=SVCInferenceConfig.only_diffusion,
        auto_predict_f0=SVCInferenceConfig.auto_predict_f0,
        f0_predictor=SVCInferenceConfig.f0_predictor,
        enhancer_adaptive_key=SVCInferenceConfig.enhancer_adaptive_key,
        linear_gradient=SVCInferenceConfig.linear_gradient,
        loud_add=SVCInferenceConfig.loud_add,
        send_format=SVCInferenceConfig.send_format,
        # 不要动的参数
        slice_db=SVCInferenceConfig.slice_db,
        noice_scale=SVCInferenceConfig.noice_scale,
        pad_seconds=SVCInferenceConfig.pad_seconds,
        linear_gradient_retain=SVCInferenceConfig.linear_gradient_retain,
        f0_filter_threshold=SVCInferenceConfig.f0_filter_threshold,
        use_spk_mix=SVCInferenceConfig.use_spk_mix,
        second_encoding=SVCInferenceConfig.second_encoding,
        loudness_envelope_adjustment=SVCInferenceConfig.loudness_envelope_adjustment,
        wav_format=SVCInferenceConfig.wav_format,
        return_pcm=True,
):
    return file_response(*audio_process(
        # 必填参数
        file_name,
        spk_list,
        # 可选参数
        tran=tran,
        clip=clip,
        cluster_infer_ratio=cluster_infer_ratio,
        shallow_diffusion=shallow_diffusion,
        k_step=k_step,
        only_diffusion=only_diffusion,
        auto_predict_f0=auto_predict_f0,
        f0_predictor=f0_predictor,
        enhancer_adaptive_key=enhancer_adaptive_key,
        linear_gradient=linear_gradient,
        loud_add=loud_add,
        send_format=send_format,
        # 不要动的参数
        slice_db=slice_db,
        noice_scale=noice_scale,
        pad_seconds=pad_seconds,
        linear_gradient_retain=linear_gradient_retain,
        f0_filter_threshold=f0_filter_threshold,
        use_spk_mix=use_spk_mix,
        second_encoding=second_encoding,
        loudness_envelope_adjustment=loudness_envelope_adjustment,
        wav_format=wav_format,
    ))


@app.post("/")
async def api_response(
        file: UploadFile = File(...),
        # 必填参数
        file_name: str = Form(...),
        spk_list: str = Form(...),
        # 可选参数，添加默认值
        tran: str = Form(SVCInferenceConfig.tran),
        clip: float = Form(SVCInferenceConfig.clip),
        cluster_infer_ratio: float = Form(SVCInferenceConfig.cluster_infer_ratio),
        shallow_diffusion: bool = Form(SVCInferenceConfig.shallow_diffusion),
        k_step: int = Form(SVCInferenceConfig.k_step),
        only_diffusion: bool = Form(SVCInferenceConfig.only_diffusion),
        auto_predict_f0: bool = Form(SVCInferenceConfig.auto_predict_f0),
        f0_predictor: str = Form(SVCInferenceConfig.f0_predictor),
        enhancer_adaptive_key: str = Form(SVCInferenceConfig.enhancer_adaptive_key),
        linear_gradient: float = Form(SVCInferenceConfig.linear_gradient),
        loud_add: float = Form(SVCInferenceConfig.loud_add),
        send_format: str = Form(SVCInferenceConfig.send_format),
        # 不要动的参数
        slice_db: float = Form(SVCInferenceConfig.slice_db),
        noice_scale: float = Form(SVCInferenceConfig.noice_scale),
        pad_seconds: float = Form(SVCInferenceConfig.pad_seconds),
        linear_gradient_retain: float = Form(SVCInferenceConfig.linear_gradient_retain),
        f0_filter_threshold: float = Form(SVCInferenceConfig.f0_filter_threshold),
        use_spk_mix: bool = Form(SVCInferenceConfig.use_spk_mix),
        second_encoding: bool = Form(SVCInferenceConfig.second_encoding),
        loudness_envelope_adjustment: float = Form(SVCInferenceConfig.loudness_envelope_adjustment),
        wav_format: str = Form(SVCInferenceConfig.wav_format),
):
    # 接收文件流并写入
    file_name = file_name.replace(" ", "_")

    with open(fr'raw\{file_name}', "wb+") as return_file:
        content = await file.read()  # 读取文件内容
        return_file.write(content)
    print(fr'写入了，写入到raw\{file_name}里面了，看看')
    return file_response(*audio_process(
        # 必填参数
        file_name,
        spk_list,
        # 可选参数
        tran=tran,
        clip=clip,
        cluster_infer_ratio=cluster_infer_ratio,
        shallow_diffusion=shallow_diffusion,
        k_step=k_step,
        only_diffusion=only_diffusion,
        auto_predict_f0=auto_predict_f0,
        f0_predictor=f0_predictor,
        enhancer_adaptive_key=enhancer_adaptive_key,
        linear_gradient=linear_gradient,
        loud_add=loud_add,
        send_format=send_format,
        # 不要动的参数
        slice_db=slice_db,
        noice_scale=noice_scale,
        pad_seconds=pad_seconds,
        linear_gradient_retain=linear_gradient_retain,
        f0_filter_threshold=f0_filter_threshold,
        use_spk_mix=use_spk_mix,
        second_encoding=second_encoding,
        loudness_envelope_adjustment=loudness_envelope_adjustment,
        wav_format=wav_format,
    ))


if __name__ == '__main__':
    # 测试网址在宿舍：http://172.18.62.224:9960/?clean_name=Counting_Stars&spk_list=Lee
    # 测试网址在学校：http://172.17.115.171:9960/?clean_name=Counting_Stars&spk_list=Lee
    # 测试网址在宿舍：http://172.18.62.224:9960/?file_name=Dream_it_Possible.flac&spk_list=azusa
    uvicorn.run(app, host=host, port=port, workers=1)
    # audio_process('Brave_Shine-Aimer.flac', 'mika', loud_add=4, tran=0)
