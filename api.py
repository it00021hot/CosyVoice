# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from fastapi import FastAPI, File, Form, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import io
import soundfile as sf
import tempfile
import uvicorn
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
os.system('cd pretrained_models/CosyVoice-ttsfrd/ && pip install ttsfrd_dependency-0.1-py3-none-any.whl && pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl && apt install -y unzip && unzip resource.zip -d .')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = ['3s极速复刻', '自然语言控制']
instruct_dict = {'3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮',
                 '自然语言控制': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入instruct文本\n3. 点击生成音频按钮'}
stream_mode_list = [('否', False), ('是', True)]
max_val = 0.8

# 创建FastAPI应用
app = FastAPI(title="CosyVoice API", description="API for CosyVoice ASR and TTS")

# Pydantic模型用于TTS接口
class TTSRequest(BaseModel):
    tts_text: str
    mode: str = "3s极速复刻"
    prompt_text: Optional[str] = ""
    instruct_text: Optional[str] = ""
    seed: int = 0
    stream: bool = False

def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]

def prompt_wav_recognition(prompt_wav):
    res = asr_model.generate(input=prompt_wav,
                             language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
                             use_itn=True,
    )
    text = res[0]["text"].split('|>')[-1]
    return text

def save_audio_to_file(audio_data, sample_rate):
    """保存音频数据到临时文件"""
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_file.name, audio_data, sample_rate, format='WAV')
    return temp_file.name

def generate_audio_stream(tts_text, mode_checkbox_group, prompt_text, prompt_wav, instruct_text,
                   seed, stream):
    """生成音频数据的生成器，用于流式响应"""
    sft_dropdown, speed = '', 1.0
    
    # 验证模式和参数
    if mode_checkbox_group in ['自然语言控制']:
        if instruct_text == '':
            raise HTTPException(status_code=400, detail="自然语言控制模式需要输入instruct文本")
        if prompt_wav is None:
            raise HTTPException(status_code=400, detail="自然语言控制模式需要输入prompt音频")
            
    # 验证prompt音频
    if mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
        if prompt_wav is None:
            raise HTTPException(status_code=400, detail="prompt音频为空，请提供prompt音频")
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            raise HTTPException(status_code=400, detail=f"prompt音频采样率{torchaudio.info(prompt_wav).sample_rate}低于{prompt_sr}")
        
    # 验证prompt文本
    if mode_checkbox_group in ['3s极速复刻']:
        if prompt_text == '':
            raise HTTPException(status_code=400, detail="prompt文本为空，请提供prompt文本")
        info = torchaudio.info(prompt_wav)
        if info.num_frames / info.sample_rate > 10:
            raise HTTPException(status_code=400, detail="请限制输入音频在10s内，避免推理效果过低")

    # 根据不同模式生成音频
    if mode_checkbox_group == '预训练音色':
        logging.info('get sft inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
            yield i['tts_speech'].numpy().flatten()
    elif mode_checkbox_group == '3s极速复刻':
        logging.info('get zero_shot inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
            yield i['tts_speech'].numpy().flatten()
    elif mode_checkbox_group == '跨语种复刻':
        logging.info('get cross_lingual inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
            yield i['tts_speech'].numpy().flatten()
    else:
        logging.info('get instruct inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k, stream=stream, speed=speed):
            yield i['tts_speech'].numpy().flatten()

@app.post("/v1/asr")
async def recognize_speech(
    file: UploadFile = File(...),
    keys: str = Form("audio"),
    lang: str = Form("auto")
):
    """
    语音识别API
    - file: wav/mp3音频文件 (16KHz)
    - keys: 音频名称（逗号分隔）
    - lang: 语音内容的语言 (默认: auto)
    """
    # 保存上传的音频文件
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    try:
        temp_file.write(await file.read())
        temp_file.close()
        
        # 使用ASR模型识别语音
        result = asr_model.generate(
            input=temp_file.name,
            language=lang,  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
        )
        
        # 格式化结果
        text = result[0]["text"].split('|>')[-1]
        
        # 拆分keys
        key_list = keys.split(",")
        
        return {
            "texts": {key_list[0]: text}
        }
    finally:
        # 清理临时文件
        os.unlink(temp_file.name)

@app.post("/v1/invoke")
async def generate_speech(
    request: Optional[TTSRequest] = None,
    tts_text: Optional[str] = Form(None),
    mode: Optional[str] = Form("3s极速复刻"),
    prompt_text: Optional[str] = Form(""),
    instruct_text: Optional[str] = Form(""),
    seed: Optional[int] = Form(0),
    stream: Optional[bool] = Form(False),
    prompt_file: Optional[UploadFile] = File(None)
):
    """
    语音合成API
    - 接受JSON请求体或表单数据
    - 支持上传prompt音频文件
    - 返回合成的音频文件
    """
    # 合并JSON和表单参数
    if request:
        tts_text = request.tts_text
        mode = request.mode
        prompt_text = request.prompt_text
        instruct_text = request.instruct_text
        seed = request.seed
        stream = request.stream
    
    if not tts_text:
        raise HTTPException(status_code=400, detail="合成文本不能为空")
    
    # 保存上传的prompt音频文件
    prompt_wav = None
    if prompt_file:
        temp_prompt = tempfile.NamedTemporaryFile(delete=False, suffix=Path(prompt_file.filename).suffix)
        temp_prompt.write(await prompt_file.read())
        temp_prompt.close()
        prompt_wav = temp_prompt.name

    try:
        # 生成音频
        if stream:
            # 返回流式响应
            return StreamingResponse(
                generate_audio_stream(tts_text, mode, prompt_text, prompt_wav, instruct_text, seed, stream),
                media_type="audio/wav"
            )
        else:
            # 生成完整音频
            audio_generator = generate_audio_stream(tts_text, mode, prompt_text, prompt_wav, instruct_text, seed, stream)
            audio_chunks = []
            for chunk in audio_generator:
                audio_chunks.append(chunk)
            
            # 合并所有音频片段
            audio_data = np.concatenate(audio_chunks) if audio_chunks else np.zeros(0)
            
            # 保存为临时文件并返回
            output_file = save_audio_to_file(audio_data, target_sr)
            return FileResponse(output_file, media_type="audio/wav", filename="generated.wav")
    finally:
        # 清理临时文件
        if prompt_wav:
            try:
                os.unlink(prompt_wav)
            except:
                pass

def run_gradio():
    """运行原有的Gradio界面"""
    with gr.Blocks() as demo:
        gr.Markdown("### 代码库 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    预训练模型 [CosyVoice2-0.5B](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B) \
                    [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### 请输入需要合成的文本，选择推理模式，并按照提示步骤进行操作")

        tts_text = gr.Textbox(label="输入合成文本", lines=1, value="CosyVoice迎来全面升级，提供更准、更稳、更快、 更好的语音生成能力。CosyVoice is undergoing a comprehensive upgrade, providing more accurate, stable, faster, and better voice generation capabilities.")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='选择推理模式', value=inference_mode_list[0])
            instruction_text = gr.Text(label="操作步骤", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            stream = gr.Radio(choices=stream_mode_list, label='是否流式推理', value=stream_mode_list[0][1])
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="随机推理种子")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制prompt音频文件')
        prompt_text = gr.Textbox(label="prompt文本", lines=1, placeholder="请输入prompt文本，支持自动识别，您可以自行修正识别结果...", value='')
        instruct_text = gr.Textbox(label="输入instruct文本", lines=1, placeholder="请输入instruct文本.例如:用四川话说这句话。", value='')

        generate_button = gr.Button("生成音频")

        audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
        prompt_wav_upload.change(fn=prompt_wav_recognition, inputs=[prompt_wav_upload], outputs=[prompt_text])
        prompt_wav_record.change(fn=prompt_wav_recognition, inputs=[prompt_wav_record], outputs=[prompt_text])
    demo.queue(max_size=2, default_concurrency_limit=4).launch(server_port=50000)

def generate_audio(tts_text, mode_checkbox_group, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream):
    """原有的音频生成函数，保留用于Gradio界面"""
    sft_dropdown, speed = '', 1.0
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['自然语言控制']:
        if instruct_text == '':
            gr.Warning('您正在使用自然语言控制模式, 请输入instruct文本')
            yield (target_sr, default_data)
        if prompt_wav is None:
            gr.Info('您正在使用自然语言控制模式, 请输入prompt音频')
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['跨语种复刻']:
        if cosyvoice.frontend.instruct is True:
            gr.Warning('您正在使用跨语种复刻模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M模型'.format(args.model_dir))
            yield (target_sr, default_data)
        if instruct_text != '':
            gr.Info('您正在使用跨语种复刻模式, instruct文本会被忽略')
        if prompt_wav is None:
            gr.Warning('您正在使用跨语种复刻模式, 请提供prompt音频')
            yield (target_sr, default_data)
        gr.Info('您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言')
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
        if prompt_wav is None:
            gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
            yield (target_sr, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('prompt音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            yield (target_sr, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['预训练音色']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！')
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['3s极速复刻']:
        if prompt_text == '':
            gr.Warning('prompt文本为空，您是否忘记输入prompt文本？')
            yield (target_sr, default_data)
        if instruct_text != '':
            gr.Info('您正在使用3s极速复刻模式，预训练音色/instruct文本会被忽略！')
        info = torchaudio.info(prompt_wav)
        if info.num_frames / info.sample_rate > 10:
            gr.Warning('请限制输入音频在10s内，避免推理效果过低')
            yield (target_sr, default_data)

    if mode_checkbox_group == '预训练音色':
        logging.info('get sft inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
            yield (target_sr, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '3s极速复刻':
        logging.info('get zero_shot inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (target_sr, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '跨语种复刻':
        logging.info('get cross_lingual inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (target_sr, i['tts_speech'].numpy().flatten())
    else:
        logging.info('get instruct inference request')
        logging.info('get instruct inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (target_sr, i['tts_speech'].numpy().flatten())


if __name__ == '__main__':
    load_jit = True if os.environ.get('jit') == '1' else False
    load_onnx = True if os.environ.get('onnx') == '1' else False
    load_trt = True if os.environ.get('trt') == '1' else False
    logging.info('cosyvoice args load_jit {} load_onnx {} load_trt {}'.format(load_jit, load_onnx, load_trt))
    cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=load_jit, load_onnx=load_onnx, load_trt=load_trt)
    sft_spk = cosyvoice.list_avaliable_spks()
    prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
    for stream in [True, False]:
        for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=stream)):
            continue
    prompt_sr, target_sr = 16000, 24000
    default_data = np.zeros(target_sr)

    model_dir = "iic/SenseVoiceSmall"
    asr_model = AutoModel(
        model=model_dir,
        disable_update=True,
        log_level='DEBUG',
        device="cuda:0")
    
    # 选择运行模式：API或UI
    import argparse
    parser = argparse.ArgumentParser(description='启动CosyVoice服务')
    parser.add_argument('--mode', type=str, default='api', choices=['api', 'ui'], help='运行模式：api或ui')
    parser.add_argument('--port', type=int, default=8000, help='服务器端口')
    args = parser.parse_args()
    
    if args.mode == 'ui':
        # 运行Gradio界面
        main()
    else:
        # 运行FastAPI服务
        uvicorn.run(app, host="0.0.0.0", port=args.port)