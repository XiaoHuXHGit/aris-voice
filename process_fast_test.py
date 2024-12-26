import os
import time
import subprocess

"""
need args:
-m model path
-c config file path
-cl clip length, default 0, auto clip
-n wavPath  # file name + file format, all reference audio are in directory 'raw'
-t pitch range, default 0, normal pitch range
-s speaker name: liv  # test only(not Unicode)
-f0p f0 predictor, default rmvpe
-wf output file format, default 'wav'

command head:
interface-main.py

command options:
-m {modelPath}
-c {configPath}
-cl 0
-n {wavPath}
-t 0
-s liv
-f0p rmvpe
-wf wav
"""


class TestInferenceOptions:
    # model and wav data detail
    # 心如止水_Ice_Paper_(Vocals_only) Мокрые_губы(Vocals_Only) За_спиной_АДЛИН_Harmony
    wav_name = 'Call_of_Silence'
    wav_path = f'{wav_name}.wav'
    main_model_name = 'Nanami_'
    height = 0
    speaker_name = 'Nanami'
    inference_device = 'cuda:0'

    # model start steps and final steps
    model_start_step = 66400
    model_final_step = 67600 + 1
    step = model_start_step
    eval_interval = 400  # steps
    model_path = rf'logs\44k\{main_model_name}G_'
    config_path = r'configs\config.json'

    # diffusion start steps and final steps and configs
    diffusion_apply = False
    diffusion_start_step = 50000
    diffusion_final_step = 1000 + 1
    diffusion_step = diffusion_start_step
    diffusion_eval_interval = 1000  # steps
    diffusion_path = r'logs\44k\diffusion\model_'
    diffusion_config_path = r'configs\diffusion.yaml'

    # feature_and_index and kmeans model path and options
    feature_and_kmeans_apply = True  # revise to True, the parameters below will take effect
    feature_and_kmeans = 'feature'  # this option has 2 argparse 'feature' and 'kmeans'
    feature_model_path = rf'logs\44k\{main_model_name}feature_and_index.pkl'
    kmeans_model_path = rf'logs\44k\{main_model_name}kmeans_10000.pt'
    cluster_infer_ratio = 0.5

    # output options
    TimeSleep = 180  # second
    file_format = 'wav'
    result_path = r'results'

    # simple configs
    python_path = r'G:\Artificial-Intelligence\AI-voice-synthesis\so-vits-svc\so-vits-svc-integration\workenv\python.exe'
    infinite_loop = True  # if this config argparse is 'false', please check model_start_step and model_final_step
    time_out = 0  # this option can constrain the max limit of infinite_loop, 0 is lift the restrictions

    # NSF-HIFIGAN
    enhance_apply = False

    # ignore this argparse


class LOGGER:
    log: str = ''
    log_output: bool = True
    colors = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'purple': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'reset': '\033[0m'
    }

    def __init__(self, log: str = '') -> None:
        self.log += log

    def log_add(self, extend_log: str, color: str = 'reset',
                enter: bool = False, record_time: bool = False) -> None:
        if record_time:
            self.time_record()
        extend_log += '\n' if enter else ''
        self.log += f'{self.colors[color]}{extend_log}{self.colors["reset"]}'

    def log_auto_record(self, message: str, color: str = 'reset',
                        enter: bool = False, record_time: bool = False) -> None:
        self.log_add(message, color, enter, record_time)
        self.output()
        self.log_clear()

    def log_clear(self) -> None:
        self.log = ''

    def time_record(self, color: str = 'blue') -> None:
        from datetime import datetime
        now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        self.log += f'{self.colors[color]}{now}{self.colors["reset"]} '

    def output(self) -> None:
        if self.log_output:
            print(self.log)


logger = LOGGER()


# this function can test main model
"""
:param
mode 1: test main model; 2: test diffusion model;
"""


def model_eval_inference():
    # main model inference
    command = [
        TestInferenceOptions.python_path, 'inference_main.py',
        '-m', TestInferenceOptions.model_path + str(TestInferenceOptions.step) + '.pth',
        '-c', TestInferenceOptions.config_path,
        '-cl', '0',
        '-n', TestInferenceOptions.wav_path,
        '-t', str(TestInferenceOptions.height),
        '-s', TestInferenceOptions.speaker_name,
        '-f0p', 'rmvpe',
        '-wf', 'wav',
        '-d', TestInferenceOptions.inference_device
    ]
    model_detection_log = f'已检测到目录更新\n主模型：{TestInferenceOptions.main_model_name}G_{TestInferenceOptions.step}.pth'
    model_inference_log = f'AI推演完成(模型信息)\n主模型训练步数：{TestInferenceOptions.step}steps'
    rename_additional_information = f'_G{TestInferenceOptions.step}'
    # check if mode equal 1 will test diffusion model and ignore main model
    if TestInferenceOptions.diffusion_apply:
        command.extend([
            '-shd',
            '-dm', TestInferenceOptions.diffusion_path + str(TestInferenceOptions.diffusion_step) + '.pt',
            '-dc', TestInferenceOptions.diffusion_config_path,
        ])
        model_detection_log += f'\n浅扩散模型：model_{TestInferenceOptions.diffusion_step}.pt'
        model_inference_log += f'\n浅扩散模型训练步数：{TestInferenceOptions.diffusion_step}steps'
        rename_additional_information += f'_diff{TestInferenceOptions.diffusion_step}'
    # recognize if cluster model is been applied
    if TestInferenceOptions.feature_and_kmeans_apply:
        if TestInferenceOptions.feature_and_kmeans[0] == 'f':
            model_detection_log += f'\n特征检索模型：' + TestInferenceOptions.feature_model_path.split("\\")[-1]
            model_inference_log += '\n使用特征检索模型'
            command.extend(['-fr', '-cm', TestInferenceOptions.feature_model_path, '-cr', str(TestInferenceOptions.cluster_infer_ratio)])
            rename_additional_information += '_feature'
        else:
            model_detection_log += f'\n聚类模型：' + TestInferenceOptions.kmeans_model_path.split("\\")[-1]
            model_inference_log += '\n使用聚类模型'
            command.extend(['-cm', TestInferenceOptions.kmeans_model_path, '-cr', str(TestInferenceOptions.cluster_infer_ratio)])
            rename_additional_information += '_kmeans'
    # NSF-HIFIGAN enhancer, but it is not compatible with the diffusion model below
    if TestInferenceOptions.enhance_apply:
        command.append('-eh')
        rename_additional_information += '_enhance'
    rename_additional_information += '.' + TestInferenceOptions.file_format
    # prepare the sub-progress to detect models
    logger.log_auto_record(model_detection_log, 'green', record_time=True)
    logger.log_auto_record(f'正在准备子进程', 'cyan', record_time=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()  # process blocking, until subprocess complete
    if stderr != '' and 'sample time step:' not in stderr:
        logger.log_auto_record('AI测试推演发生错误：\n' + stderr, 'red', record_time=True)
        exit()
    logger.log_auto_record(model_inference_log, 'purple', record_time=True)
    # result rename
    get_result_name = ''
    for step in os.listdir(r'results'):
        if step.startswith(TestInferenceOptions.wav_path):
            get_result_name = step
            break
    result_path = TestInferenceOptions.result_path + '\\' + get_result_name
    result_rename = result_path.replace('.wav', '') + rename_additional_information
    os.rename(result_path, result_rename)


# when the file not found, program will into cold down mode
def fast_test_cold_down(log: str):
    logger.log_auto_record(log, 'purple', record_time=True)
    # countdown by reading class variables 'TestInferenceOptions.TimeSleep'
    try:  # catch the keyboard interrupt
        time.sleep(TestInferenceOptions.TimeSleep)
    except KeyboardInterrupt:
        logger.log_auto_record(f'已退出实时模型测试程序', 'cyan')
        exit()  # interrupt by hardware input


def program_execution(mode: str = 'main_model'):
    modes = ('main_model', 'diffusion_model')
    try:  # find the index in modes and assignment 'integer' type to the variable 'mode'
        mode = modes.index(mode)
    except ValueError:
        logger.log_auto_record(f'未找到模式\"{mode}\"', 'red')
    print(f'mode = {mode} [{modes[mode]}]')
    if TestInferenceOptions.infinite_loop:
        while TestInferenceOptions.infinite_loop:
            if os.path.exists(TestInferenceOptions.model_path + str(TestInferenceOptions.step) + '.pth'):
                model_eval_inference()
                if mode == 0:
                    TestInferenceOptions.step += TestInferenceOptions.eval_interval
                elif mode == 1:
                    TestInferenceOptions.diffusion_step += TestInferenceOptions.diffusion_eval_interval
            else:
                # countdown by reading class variables 'TestInferenceOptions.TimeSleep'
                fast_test_cold_down(f'未检测G_{TestInferenceOptions.step}.pth文件进入训练等待')
    else:
        for step in range(TestInferenceOptions.model_start_step,
                          TestInferenceOptions.model_final_step,
                          TestInferenceOptions.eval_interval):
            TestInferenceOptions.step = step
            model_eval_inference()


if __name__ == '__main__':
    # program_execution(mode='main_model')
    model_eval_inference()
