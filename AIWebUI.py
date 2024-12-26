import os
import json
import shutil
import tempfile
import numpy as np
import gradio as gr

from API_SVC import audio_process
# from API_TTS import get_tts_wav, dict_language

# 全局参数
MAX_FILE_SIZE_MB = 12  # 设置文件大小限制为12MB
audio_export_path = r'results\audio_export'


def storage_limiter(audio_file):
    if audio_file is not None:
        file_size = os.path.getsize(audio_file.name) / (1024 * 1024)  # 转换为MB
        if file_size > MAX_FILE_SIZE_MB:
            return False
    return True


def clear_gradio_temp():
    # 获取用户临时目录
    temp_dir = tempfile.gettempdir()
    gradio_dir = os.path.join(temp_dir, 'gradio')

    # 检查 Gradio 文件夹是否存在
    if os.path.exists(gradio_dir):
        # 清理 Gradio 文件夹内的所有内容
        for item in os.listdir(gradio_dir):
            item_path = os.path.join(gradio_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)  # 删除文件
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # 删除文件夹及其内容
        # print(f"{gradio_dir} 中的所有内容已被清理。")


def get_characters(model):
    with open('model_save.json', 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        # print(json_data)
    # 目前可选：svc_model, tts_model
    get_model_argparse = json_data[model]
    characters = get_model_argparse.keys()
    characters_name_cn = [get_model_argparse[character_name]['name'] for character_name in characters]
    try:
        return dict(zip(characters_name_cn, characters))
    except KeyError:
        return None


def sing_voice_conversion(file_name, speaker, tran, loud_add, enhancer_adaptive_key):
    speaker = get_characters('svc_model')[speaker]
    # 返回生成的音频文件路径
    file_path = audio_process(file_name=file_name, spk_list=speaker, tran=tran, loud_add=loud_add,
                              enhancer_adaptive_key=enhancer_adaptive_key, is_gradio_argparse=True)[0]
    return file_path  # 示例输出文件名


def refresh_speaker_list(model):
    choices = list(get_characters(model=model).keys())
    return gr.Dropdown(label="选择说话人", choices=choices, value=choices[0] if choices else "",
                       allow_custom_value=True, interactive=True)


def refresh_preview_list():
    return gr.FileExplorer(root_dir=audio_export_path, file_count='single')


def refresh_emotion_list(character):
    with open('model_save.json', 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    if not isinstance(character, str):
        character = character.value
    character = json_data['tts_model'][get_characters('tts_model')[character]]
    get_model_argparse = character['emotions']
    choices = list(get_model_argparse.keys())
    return gr.Dropdown(label="选择情绪", choices=choices, value=choices[0] if choices else "",
                       allow_custom_value=True, interactive=True)


# def tts_character_inference(tts_speaker, emotion, tts_text, tts_text_language, how_to_cut):
#     with open('model_save.json', 'r', encoding='utf-8') as json_file:
#         json_data = json.load(json_file)
#     if not isinstance(tts_speaker, str):
#         tts_speaker = tts_speaker.value
#         emotion = emotion.value
#     tts_speaker = get_characters('tts_model')[tts_speaker]
#     character = json_data['tts_model'][tts_speaker]
#     emotion = character['emotions'][emotion]
#     SoVIT_weight = character['SoVIT_weight']
#     GPT_weight = character['GPT_weight']
#     refer_wav_path = emotion['path']
#     prompt_text = emotion['prompt_text']
#     prompt_language = emotion['prompt_language']
#
#     inp_refs = []
#
#     audio = []  # 用于存储所有的numpy数组
#     sampling_rate = None  # 用于存储采样率
#
#     # for rate, data in get_tts_wav(sovits_path=SoVIT_weight,
#     #                               gpt_path=GPT_weight,
#     #                               ref_wav_path=refer_wav_path,
#     #                               prompt_text=prompt_text,
#     #                               prompt_language=prompt_language,
#     #                               text=tts_text,
#     #                               text_language=tts_text_language,
#     #                               inp_refs=inp_refs):
#     #     # 将采样率赋值（假设所有返回的采样率是相同的）
#     #     if sampling_rate is None:
#     #         sampling_rate = rate
#     #
#     #     # 将numpy数组添加到audio列表中
#     #     audio.append(data)
#     #     import sounddevice as sd
#     #     sd.play(data, samplerate=rate)
#     #
#     #     # 合并音频数据
#     # if audio:
#     #     audio = np.concatenate(audio)  # 合并所有numpy数组为一个
#
#     return get_tts_wav(sovits_path=SoVIT_weight,
#                        gpt_path=GPT_weight,
#                        ref_wav_path=refer_wav_path,
#                        prompt_text=prompt_text,
#                        prompt_language=prompt_language,
#                        text=tts_text,
#                        text_language=tts_text_language,
#                        inp_refs=inp_refs)

    # yield get_tts_wav(sovits_path=SoVIT_weight, gpt_path=GPT_weight, ref_wav_path=refer_wav_path,
    #                   prompt_text=prompt_text, prompt_language=prompt_language, text=tts_text,
    #                   text_language=tts_text_language, inp_refs=inp_refs)


# Gradio界面
def ai_voice_generate():
    clear_gradio_temp()
    # ==================== UI ====================
    with gr.Blocks(title='AI SVC') as UI:
        # 标题
        gr.Markdown("# AI音乐一键合成 by小狐")

        with gr.Tabs():
            with gr.TabItem('AI歌声转换'):
                gr.Markdown("带宽不够o(TヘTo)<br>请不要上传大于12MB的文件<br>也不要DDOS我，感谢o(TヘTo)")
                # 文本输入区域
                prompt_audio = gr.Audio(label="参考音频(直接放音乐就行)", type="filepath")
                # 下拉菜单
                with gr.Row():
                    svc_speaker = refresh_speaker_list(model='svc_model')
                    tran = gr.Number(label="变调(填整数 可以是负数)", value=0)
                with gr.Row():
                    loud_add = gr.Number(label="AI响度增加", value=4)
                    enhancer_adaptive_key = gr.Number(label="NSF-HIFIGAN增强", value=0)
                refresh_speakers = gr.Button("刷新模型")
                # 操作按钮
                generate_button = gr.Button("生成", variant="primary")
                # 音频输出区域
                audio_output = gr.Audio(label="生成的音频")

            # with gr.TabItem('AI文本转语音'):
            #     gr.Markdown("施工中，估计全是bug")
            #     with gr.Row():
            #         tts_speaker = refresh_speaker_list(model='tts_model')
            #         # 这里是写情绪的，先把所有文件都放到辅助参考音频里面，然后选择的是主推演音频
            #         emotion = refresh_emotion_list(tts_speaker)
            #     with gr.Row():
            #         tts_text_language = gr.Dropdown(
            #             label="选择语言", choices=list(dict_language.keys()), value=list(dict_language.keys())[0],
            #             allow_custom_value=True, interactive=True)
            #         tts_text = gr.Textbox(label="输入想要生成的文本")
            #     with gr.Row():
            #         how_to_cut = gr.Dropdown(
            #             label="切分方式",
            #             choices=["不切", "四句切分", "50字切分", "以中文句号切分",
            #                      "以英文句号切分", "按标点符号切", ],
            #             value="四句切分",
            #             interactive=True, scale=1
            #         )
            #         tts_result_preview = gr.Audio(label="生成的音频", format='wav')
            #     tts_generate_button = gr.Button("合成", variant="primary")

            with gr.TabItem('大家的作品'):
                preview_refresh_button = gr.Button('刷新一下')
                audio_select = refresh_preview_list()
                preview_button = gr.Button("既然你做了 那我就听听", variant="primary")
                audio_preview_component = gr.Audio(label="在这里播放哟~")

        # ==================== 连接按钮和事件 ====================
        # AI歌声转换页面事件
        generate_button.click(sing_voice_conversion,
                              inputs=[prompt_audio, svc_speaker, tran, loud_add, enhancer_adaptive_key],
                              outputs=audio_output)
        refresh_speakers.click(refresh_speaker_list, outputs=svc_speaker)
        # AI文本转语音页面事件
        # tts_generate_button.click(
        #     tts_character_inference,
        #     inputs=[tts_speaker, emotion, tts_text, tts_text_language, how_to_cut],
        #     outputs=[tts_result_preview])
        # tts_speaker.change(
        #     fn=refresh_emotion_list,
        #     inputs=tts_speaker,
        #     outputs=emotion
        # )
        # 大家的作品页面事件
        preview_button.click(lambda path: path, inputs=audio_select, outputs=audio_preview_component)
        preview_refresh_button.click(refresh_preview_list, outputs=audio_select)

    UI.launch(server_name="0.0.0.0", server_port=7860, share=False, favicon_path='assets/icon/icon.png')
    UI.close()
    # http://localhost:9880?text=你好小狐，我是你的小梓！&text_language=zh
    # -dr "G:\Artificial-Intelligence\AI-voice-synthesis\GPT-SoVITS\GPT-SoVITS\wav\azusa\Azusa_Peaceful.wav" -dt "この力をどういう風に使うか、先生に決めてほしい。" -dl "all_ja"


if __name__ == '__main__':
    # 测试网址在学校：http://172.17.115.171:9960
    # 测试网址在宿舍：http://172.18.62.224:9960

    ai_voice_generate()
    clear_gradio_temp()
    # print(list(get_characters().values()))

#
# 　　 ┏┓　　  ┏┓
# 　　┏┛┻━━━━━┛┻┓
# 　　┃　　　　   ┃
# 　　┃　　 ━　  ┃
# 　　┃　┳┛　┗┳　┃
# 　　┃　　　　　 ┃
# 　　┃　　 ┻　　 ┃
# 　　┃　　　　　 ┃
# 　　┗━┓　　　┏━┛Codes are far away from bugs with the animal protecting
# 　　　　┃　　　┃    神兽保佑,代码无bug
# 　　　　┃　　　┃
# 　　　　┃　　　┗━━━┓
# 　　　　┃　　　　　 ┣┓
# 　　　　┃　　　　 ┏┛
# 　　　　┗┓┓┏━┳┓┏┛
# 　　　　　┃┫┫　┃┫┫
# 　　　　　┗┻┛　┗┻┛
#

