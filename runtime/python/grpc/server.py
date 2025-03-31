# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
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
from concurrent import futures
import argparse
import cosyvoice_pb2
import cosyvoice_pb2_grpc
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import grpc
import torch
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from funasr import AutoModel

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


class CosyVoiceServiceImpl(cosyvoice_pb2_grpc.CosyVoiceServicer):
    def __init__(self, args):
        try:
            self.cosyvoice = CosyVoice(args.model_dir)
        except Exception:
            try:
                self.cosyvoice = CosyVoice2(args.model_dir)
            except Exception:
                raise TypeError('no valid model_type!')
        
        # 初始化ASR模型
        try:
            self.asr_model = AutoModel(
                model="../../../pretrained_models/SenseVoiceSmall",
                disable_update=True,
                log_level='DEBUG',
                device="cuda:0" if torch.cuda.is_available() else "cpu"
            )
        except Exception as e:
            logging.error(f"Failed to initialize ASR model: {str(e)}")
            self.asr_model = None
            
        logging.info('grpc service initialized')

    def _process_request(self, request):
        """处理请求的内部方法，供Inference和InferenceNonStream共用"""
        if request.HasField('sft_request'):
            logging.info('get sft inference request')
            model_output = self.cosyvoice.inference_sft(request.sft_request.tts_text, request.sft_request.spk_id)
        elif request.HasField('zero_shot_request'):
            logging.info('get zero_shot inference request')
            prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(request.zero_shot_request.prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
            prompt_speech_16k = prompt_speech_16k.float() / (2**15)
            model_output = self.cosyvoice.inference_zero_shot(request.zero_shot_request.tts_text,
                                                              request.zero_shot_request.prompt_text,
                                                              prompt_speech_16k)
        elif request.HasField('cross_lingual_request'):
            logging.info('get cross_lingual inference request')
            prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(request.cross_lingual_request.prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
            prompt_speech_16k = prompt_speech_16k.float() / (2**15)
            model_output = self.cosyvoice.inference_cross_lingual(request.cross_lingual_request.tts_text, prompt_speech_16k)
        else:
            logging.info('get instruct inference request')
            prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(request.instruct_request.prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
            prompt_speech_16k = prompt_speech_16k.float() / (2**15)
            
            # 如果prompt_text为空，进行自动语音识别
            prompt_text = request.instruct_request.prompt_text
            if not prompt_text and self.asr_model is not None:
                try:
                    logging.info('prompt_text为空，开始自动语音识别')
                    res = self.asr_model.generate(
                        input=request.instruct_request.prompt_audio,
                        use_itn=True
                    )
                    prompt_text = res[0]["text"].split('|>')[-1]
                    logging.info(f'语音识别结果: {prompt_text}')
                except Exception as e:
                    logging.error(f"语音识别失败: {str(e)}")
                    raise Exception(f"语音识别失败: {str(e)}")
            
            model_output = self.cosyvoice.inference_instruct2(request.instruct_request.tts_text,
                                                              prompt_text,
                                                              prompt_speech_16k,
                                                              stream=False, speed=1)
        return model_output

    def Inference(self, request, context):
        try:
            model_output = self._process_request(request)
            logging.info('send inference response')
            for i in model_output:
                response = cosyvoice_pb2.Response()
                response.tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
                yield response
        except Exception as e:
            logging.error(f"Inference error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Inference failed: {str(e)}")
            return

    def InferenceNonStream(self, request, context):
        try:
            # 获取模型输出
            model_output = self._process_request(request)
            
            # 合并所有音频片段
            logging.info('合并音频片段为单一响应')
            all_audio_segments = []
            for i in model_output:
                audio_segment = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16)
                all_audio_segments.append(audio_segment)
            
            # 连接所有音频片段
            if all_audio_segments:
                combined_audio = np.concatenate(all_audio_segments, axis=1)
                audio_bytes = combined_audio.tobytes()
                
                # 创建非流式响应
                return cosyvoice_pb2.NonStreamResponse(
                    tts_audio=audio_bytes,
                    success=True,
                    error_message=""
                )
            else:
                return cosyvoice_pb2.NonStreamResponse(
                    tts_audio=b'',
                    success=False,
                    error_message="No audio generated"
                )
                
        except Exception as e:
            logging.error(f"NonStream inference error: {str(e)}")
            return cosyvoice_pb2.NonStreamResponse(
                tts_audio=b'',
                success=False,
                error_message=str(e)
            )

    def SpeechRecognition(self, request, context):
        if self.asr_model is None:
            return cosyvoice_pb2.ASRResponse(
                text="",
                success=False,
                error_message="ASR model not initialized"
            )
            
        try:
            # 直接使用音频bytes进行识别
            res = self.asr_model.generate(
                input=request.audio_data,
                use_itn=True
            )
            
            # 提取识别结果
            text = res[0]["text"].split('|>')[-1]
            
            return cosyvoice_pb2.ASRResponse(
                text=text,
                success=True,
                error_message=""
            )
            
        except Exception as e:
            logging.error(f"Speech recognition error: {str(e)}")
            return cosyvoice_pb2.ASRResponse(
                text="",
                success=False,
                error_message=str(e)
            )


def main():
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_conc), maximum_concurrent_rpcs=args.max_conc)
    cosyvoice_pb2_grpc.add_CosyVoiceServicer_to_server(CosyVoiceServiceImpl(args), grpcServer)
    grpcServer.add_insecure_port('0.0.0.0:{}'.format(args.port))
    grpcServer.start()
    logging.info("server listening on 0.0.0.0:{}".format(args.port))
    grpcServer.wait_for_termination()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=9090)
    parser.add_argument('--max_conc',
                        type=int,
                        default=4)
    parser.add_argument('--model_dir',
                        type=str,
                        default='../../../pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    main()
