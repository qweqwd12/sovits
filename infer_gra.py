import os
import subprocess
import shutil
import soundfile
import shlex
import locale
import gradio as gr
from ruamel.yaml import YAML

class WebUI:
    def __init__(self):
        self.train_config_path = 'configs/train.yaml'
        self.info = Info()
        self.names2 = []
        self.voice_names = []
        self.base_config_path = 'configs/base.yaml'
        if not os.path.exists(self.train_config_path):
            shutil.copyfile(self.base_config_path, self.train_config_path)
            print(i18n("初始化成功"))
        else:
            print(i18n("就绪"))
        self.main_ui()

    def main_ui(self):
        with gr.Blocks(theme=gr.themes.Base(primary_hue=gr.themes.colors.orange)) as ui:
            gr.Markdown('# so-vits-语音转个性化语音界面')

            with gr.Tab(i18n("推理")):
                with gr.Accordion(i18n('推理说明'), open=False):
                    gr.Markdown(self.info.inference)

                gr.Markdown(i18n('### 推理参数设置'))

                with gr.Row():
                    with gr.Column():
                        self.keychange = gr.Slider(-24, 24, value=0, step=1, label=i18n('变调'))
                        self.file_list = gr.Markdown(value="", label=i18n("文件列表"))
                        
                        with gr.Row():
                            self.resume_model2 = gr.Dropdown(
                                choices=sorted(self.names2), 
                                label='Select the model you want to export',
                                info=i18n('选择要导出的模型'), 
                                interactive=True
                            )
                            
                            with gr.Column():
                                self.bt_refersh2 = gr.Button(value=i18n('刷新模型和音色'))
                                self.bt_out_model = gr.Button(value=i18n('导出模型'), variant="primary")

                        with gr.Row():
                            self.resume_voice = gr.Dropdown(
                                choices=sorted(self.voice_names), 
                                label='Select the sound file',
                                info=i18n('选择音色文件'), 
                                interactive=True
                            )

                        with gr.Row():
                            self.input_wav = gr.Audio(type='filepath', label=i18n('选择待转换音频'), source='upload')

                        with gr.Row():
                            self.bt_infer = gr.Button(value=i18n('开始转换'), variant="primary")

                        with gr.Row():
                            self.output_wav = gr.Audio(label=i18n('输出音频'), interactive=False)

            # 绑定事件
            self.bt_out_model.click(fn=self.out_model, inputs=[self.resume_model2])
            self.bt_infer.click(
                fn=self.inference, 
                inputs=[self.input_wav, self.resume_voice, self.keychange], 
                outputs=[self.output_wav]
            )
            self.bt_refersh2.click(
                fn=self.refresh_model_and_voice, 
                outputs=[self.resume_model2, self.resume_voice]
            )

        ui.launch(inbrowser=True, server_port=2333, share=False)

    def out_model(self, resume_model2):
        print(i18n('开始导出模型'))
        try:
            subprocess.Popen(
                f'python -u svc_export.py -c {self.train_config_path} -p "chkpt/sovits5.0/{resume_model2}"',
                stdout=subprocess.PIPE
            )
            print(i18n('导出模型成功'))
        except Exception as e:
            print(i18n("出现错误："), e)

    def refresh_model2(self):
        model_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chkpt/sovits5.0")
        self.names2 = []
        try:
            for name in os.listdir(model_root):
                if name.endswith(".pt"):
                    self.names2.append(name)
            return {"choices": sorted(self.names2), "__type__": "update"}
        except FileNotFoundError:
            return {"label": i18n("缺少模型文件"), "__type__": "update"}

    def refresh_voice(self):
        voice_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_svc/singer")
        self.voice_names = []
        try:
            for name in os.listdir(voice_root):
                if name.endswith(".npy"):
                    self.voice_names.append(name)
            return {"choices": sorted(self.voice_names), "__type__": "update"}
        except FileNotFoundError:
            return {"label": i18n("缺少文件"), "__type__": "update"}

    def refresh_model_and_voice(self):
        model_update = self.refresh_model2()
        voice_update = self.refresh_voice()
        return model_update, voice_update

    def inference(self, input, resume_voice, keychange):
        if os.path.exists("test.wav"):
            os.remove("test.wav")
            print(i18n("已清理残留文件"))
        else:
            print(i18n("无需清理残留文件"))
            
        print(i18n('开始推理'))
        shutil.copy(input, ".")
        input_name = os.path.basename(input)
        os.rename(input_name, "test.wav")
        
        if not input_name.endswith(".wav"):
            data, samplerate = soundfile.read(input_name)
            input_name = input_name.rsplit(".", 1)[0] + ".wav"
            soundfile.write(input_name, data, samplerate)
            
        cmd = [
            "python", "-u", "svc_inference.py",
            "--config", shlex.quote(self.train_config_path),
            "--model", "sovits5.0.pth",
            "--spk", f"data_svc/singer/{resume_voice}",
            "--wave", "test.wav",
            "--shift", shlex.quote(str(keychange))
        ]
        
        train_process = subprocess.run(cmd, shell=False, capture_output=True, text=True)
        print(train_process.stdout)
        print(train_process.stderr)
        print(i18n("推理成功"))
        return "svc_out.wav"

class Info:
    def __init__(self) -> None:
        self.inference = '### 推理功能说明：上传音频文件，选择模型和音色，调整变调参数后点击开始转换'

class I18nAuto:
    def __init__(self, language='zh_CN'):
        self.language_list = ['zh_CN', 'en_US']
        self.language_all = {
            'zh_CN': {
                '初始化成功': '初始化成功',
                '就绪': '就绪',
                '推理': '推理',
                '推理说明': '推理说明',
                '### 推理参数设置': '### 推理参数设置',
                '变调': '变调',
                '文件列表': '文件列表',
                '选择要导出的模型': '选择要导出的模型',
                '刷新模型和音色': '刷新模型和音色',
                '导出模型': '导出模型',
                '选择音色文件': '选择音色文件',
                '选择待转换音频': '选择待转换音频',
                '开始转换': '开始转换',
                '输出音频': '输出音频',
                '开始导出模型': '开始导出模型',
                '导出模型成功': '导出模型成功',
                '出现错误：': '出现错误：',
                '缺少模型文件': '缺少模型文件',
                '缺少文件': '缺少文件',
                '已清理残留文件': '已清理残留文件',
                '无需清理残留文件': '无需清理残留文件',
                '开始推理': '开始推理',
                '推理成功': '推理成功'
            },
            'en_US': {
                '初始化成功': 'Initialization successful',
                '就绪': 'Ready',
                "推理": "Inference",
                "推理说明": "Inference instructions",
                "### 推理参数设置": "### Inference parameter settings",
                "变调": "Pitch shift",
                "文件列表": "File list",
                "选择要导出的模型": "Select the model to export",
                "刷新模型和音色": "Refresh model and timbre",
                "导出模型": "Export model",
                "选择音色文件": "Select timbre file",
                "选择待转换音频": "Select audio to be converted",
                "开始转换": "Start conversion",
                "输出音频": "Output audio",
                "开始导出模型": "Start exporting model",
                "导出模型成功": "Model exported successfully",
                "出现错误：": "An error occurred:",
                "缺少模型文件": "Missing model file",
                '缺少文件': 'Missing file',
                "已清理残留文件": "Residual files cleaned up",
                "无需清理残留文件": "No need to clean up residual files",
                "开始推理": "Start inference",
                "推理成功": "Inference successful"
            }
        }
        
        self.language_map = {}
        self.language = language or locale.getdefaultlocale()[0]
        if self.language not in self.language_list:
            self.language = 'zh_CN'
        self.read_language(self.language_all['zh_CN'])

    def read_language(self, lang_dict: dict):
        self.language_map.update(lang_dict)

    def __call__(self, key):
        return self.language_map[key]

if __name__ == "__main__":
    i18n = I18nAuto()
    webui = WebUI()