import os

import gradio as gr
from paddlespeech.cli.tts import TTSExecutor
import glob

choice = {
    "speedyspeech_csmsc": {
        "am": "fastspeech2_mix",
        "am_config": None,
        "am_ckpt": None,
        "am_stat": None,
        "phones_dict": None,
        "tones_dict": None,
        "speaker_dict": None,
        "lang": "zh",
    },
    "fastspeech2_csmsc": {
        "am": "fastspeech2_csmsc",
        "am_config": None,
        "am_ckpt": None,
        "am_stat": None,
        "phones_dict": None,
        "tones_dict": None,
        "speaker_dict": None,
        "lang": "zh",
    },
    "fastspeech2_ljspeech": {
        "am": "fastspeech2_ljspeech",
        "am_config": None,
        "am_ckpt": None,
        "am_stat": None,
        "phones_dict": None,
        "tones_dict": None,
        "speaker_dict": None,
        "lang": "en",
    },
    "fastspeech2_aishell3": {
        "am": "fastspeech2_aishell3",
        "am_config": None,
        "am_ckpt": None,
        "am_stat": None,
        "phones_dict": None,
        "tones_dict": None,
        "speaker_dict": None,
        "lang": "zh",
    },
    "fastspeech2_vctk": {
        "am": "fastspeech2_vctk",
        "am_config": None,
        "am_ckpt": None,
        "am_stat": None,
        "phones_dict": None,
        "tones_dict": None,
        "speaker_dict": None,
        "lang": "en",
    },
    "fastspeech2_cnndecoder_csmsc": {
        "am": "fastspeech2_cnndecoder_csmsc",
        "am_config": None,
        "am_ckpt": None,
        "am_stat": None,
        "phones_dict": None,
        "tones_dict": None,
        "speaker_dict": None,
        "lang": "zh",
    },
    "fastspeech2_mix": {
        "am": "fastspeech2_mix",
        "am_config": None,
        "am_ckpt": None,
        "am_stat": None,
        "phones_dict": None,
        "tones_dict": None,
        "speaker_dict": None,
        "lang": "mix",
    },
    "tacotron2_csmsc": {
        "am": "tacotron2_csmsc",
        "am_config": None,
        "am_ckpt": None,
        "am_stat": None,
        "phones_dict": None,
        "tones_dict": None,
        "speaker_dict": None,
        "lang": "zh",
    },
    "tacotron2_ljspeech": {
        "am": "tacotron2_ljspeech",
        "am_config": None,
        "am_ckpt": None,
        "am_stat": None,
        "phones_dict": None,
        "tones_dict": None,
        "speaker_dict": None,
        "lang": "en",
    },
    "fastspeech2_male": {
        "am": "fastspeech2_male",
        "am_config": None,
        "am_ckpt": None,
        "am_stat": None,
        "phones_dict": None,
        "tones_dict": None,
        "speaker_dict": None,
        "lang": "zh",
    },
    "fastspeech2_canton": {
        "am": "fastspeech2_canton",
        "am_config": None,
        "am_ckpt": None,
        "am_stat": None,
        "phones_dict": None,
        "tones_dict": None,
        "speaker_dict": None,
        "lang": "canton",
    },
    "夜兰-原神-微调": {
        "am": "fastspeech2_mix",
        "am_config": "./PaddleSpeech/examples/other/tts_finetune/tts3/models/fastspeech2_mix_ckpt_1.2.0/default.yaml",
        "am_ckpt": "./model_store/夜兰/snapshot_iter_101448.pdz",
        "am_stat": "./PaddleSpeech/examples/other/tts_finetune/tts3/models/fastspeech2_mix_ckpt_1.2.0/speech_stats.npy",
        "phones_dict": "./model_store/夜兰/phone_id_map.txt",
        "tones_dict": None,
        "speaker_dict": "./model_store/夜兰/speaker_id_map.txt",
        # "voc": "hifigan_aishell3",
        # "voc_config": "./PaddleSpeech/examples/other/tts_finetune/tts3/models/hifigan_aishell3_ckpt_0.2.0/default.yaml",
        # "voc_ckpt": "./PaddleSpeech/examples/other/tts_finetune/tts3/models/hifigan_aishell3_ckpt_0.2.0/snapshot_iter_2500000.pdz",
        # "voc_stat": "./PaddleSpeech/examples/other/tts_finetune/tts3/models/hifigan_aishell3_ckpt_0.2.0/feats_stats.npy",
        "lang": "mix",
    },
}
choices = list(choice.keys())

vocs = [
    "pwgan_csmsc",
    "pwgan_ljspeech",
    "pwgan_aishell3",
    "pwgan_vctk",
    "mb_melgan_csmsc",
    "style_melgan_csmsc",
    "hifigan_csmsc	",
    "hifigan_ljspeech",
    "hifigan_aishell3",
    "hifigan_vctk",
    "wavernn_csmsc",
    "pwgan_male",
    "hifigan_male",
]

example = {
    "女-基础-1": {
        "am": "speedyspeech_csmsc",
        "am_config": None,
        "am_ckpt": None,
        "am_stat": None,
        "phones_dict": None,
        "tones_dict": None,
        "speaker_dict": None,
        "lang": "zh",
        "voc": "hifigan_aishell3",
        "voc_config": None,
        "voc_ckpt": None,
        "voc_stat": None,
        "spk_id": 0,
    },
    "女-基础-2": {
        "am": "fastspeech2_canton",
        "am_config": None,
        "am_ckpt": None,
        "am_stat": None,
        "phones_dict": None,
        "tones_dict": None,
        "speaker_dict": None,
        "lang": "canton",
        "voc": "hifigan_aishell3",
        "voc_config": None,
        "voc_ckpt": None,
        "voc_stat": None,
        "spk_id": 0,
    },
    "男-基础": {
        "am": "fastspeech2_male",
        "am_config": None,
        "am_ckpt": None,
        "am_stat": None,
        "phones_dict": None,
        "tones_dict": None,
        "speaker_dict": None,
        "lang": "zh",
        "voc": "hifigan_aishell3",
        "voc_config": None,
        "voc_ckpt": None,
        "voc_stat": None,
        "spk_id": 0,
    },
    # "夜兰-原神-微调": {
    #     "am": "fastspeech2_mix",
    #     "am_config": "./PaddleSpeech/examples/other/tts_finetune/tts3/models/fastspeech2_mix_ckpt_1.2.0/default.yaml",
    #     "am_ckpt": "./model_store/夜兰/snapshot_iter_101448.pdz",
    #     "am_stat": "./PaddleSpeech/examples/other/tts_finetune/tts3/models/fastspeech2_mix_ckpt_1.2.0/speech_stats.npy",
    #     "phones_dict": "./model_store/夜兰/phone_id_map.txt",
    #     "tones_dict": None,
    #     "speaker_dict": "./model_store/夜兰/speaker_id_map.txt",
    #     "lang": "mix",
    #     "voc": "hifigan_aishell3",
    #     "voc_config": "./PaddleSpeech/examples/other/tts_finetune/tts3/models/hifigan_aishell3_ckpt_0.2.0/default.yaml",
    #     "voc_ckpt": "./PaddleSpeech/examples/other/tts_finetune/tts3/models/hifigan_aishell3_ckpt_0.2.0/snapshot_iter_2500000.pdz",
    #     "voc_stat": "./PaddleSpeech/examples/other/tts_finetune/tts3/models/hifigan_aishell3_ckpt_0.2.0/feats_stats.npy",
    #     "spk_id": 0,
    # },
}

home_dir = "/root/individual/hello2mao/playground-python/PaddleSpeech-tts-finetune/"


# 找到最新生成的模型
def find_max_ckpt(model_path):
    max_ckpt = 0
    for filename in os.listdir(model_path):
        if filename.endswith(".pdz"):
            files = filename[:-4]
            a1, a2, it = files.split("_")
            if int(it) > max_ckpt:
                max_ckpt = int(it)
    return max_ckpt


all_file_name_list = glob.glob(home_dir + "work/trained/*")
for file_name in all_file_name_list:
    try:
        name = os.path.basename(file_name)
        output_dir = os.path.join(home_dir, "work/trained", name, "exp")
        dump_dir = os.path.join(home_dir, "work/trained", name, "dump")
        model_path = os.path.join(output_dir, "checkpoints")
        ckpt = find_max_ckpt(model_path)
        example[name] = {
            "am": "fastspeech2_mix",
            "am_config": "./PaddleSpeech/examples/other/tts_finetune/tts3/models/fastspeech2_mix_ckpt_1.2.0/default.yaml",
            "am_ckpt": model_path + f"/snapshot_iter_{ckpt}.pdz",
            "am_stat": "./PaddleSpeech/examples/other/tts_finetune/tts3/models/fastspeech2_mix_ckpt_1.2.0/speech_stats.npy",
            "phones_dict": dump_dir + "/phone_id_map.txt",
            "tones_dict": None,
            "speaker_dict": dump_dir + "/speaker_id_map.txt",
            "lang": "mix",
            "voc": "hifigan_aishell3",
            "voc_config": "./PaddleSpeech/examples/other/tts_finetune/tts3/models/hifigan_aishell3_ckpt_0.2.0/default.yaml",
            "voc_ckpt": "./PaddleSpeech/examples/other/tts_finetune/tts3/models/hifigan_aishell3_ckpt_0.2.0/snapshot_iter_2500000.pdz",
            "voc_stat": "./PaddleSpeech/examples/other/tts_finetune/tts3/models/hifigan_aishell3_ckpt_0.2.0/feats_stats.npy",
            "spk_id": 0,
        }
    except Exception as e:
        print(f"err: {e}")
        continue

examples = list(example.keys())

tts_executor = None


def speech_generate(
    text: str, choice_name, progress=gr.Progress(track_tqdm=True)
) -> os.PathLike:
    assert isinstance(text, str) and len(text) > 0, "Input Chinese text..."
    global tts_executor
    if tts_executor is None:
        gr.Error("请先加载人物")
    if choice_name is None:
        gr.Error("请先加载人物")
    print(f"speech_generate, config: {example[choice_name]}")
    wav_file = tts_executor(
        am=example[choice_name]["am"],
        am_config=example[choice_name]["am_config"],
        am_ckpt=example[choice_name]["am_ckpt"],
        am_stat=example[choice_name]["am_stat"],
        phones_dict=example[choice_name]["phones_dict"],
        tones_dict=example[choice_name]["tones_dict"],
        speaker_dict=example[choice_name]["speaker_dict"],
        lang=example[choice_name]["lang"],
        voc=example[choice_name]["voc"],
        voc_config=example[choice_name]["voc_config"],
        voc_ckpt=example[choice_name]["voc_ckpt"],
        voc_stat=example[choice_name]["voc_stat"],
        spk_id=example[choice_name]["spk_id"],
        device="gpu",
        text=text,
        output="output.wav",
    )
    return wav_file


def load(text, choice_name, progress=gr.Progress(track_tqdm=True)):
    global tts_executor
    del tts_executor
    print(f"load, config: {example[choice_name]}")
    tts_executor = TTSExecutor()
    wav_file = tts_executor(
        am=example[choice_name]["am"],
        am_config=example[choice_name]["am_config"],
        am_ckpt=example[choice_name]["am_ckpt"],
        am_stat=example[choice_name]["am_stat"],
        phones_dict=example[choice_name]["phones_dict"],
        tones_dict=example[choice_name]["tones_dict"],
        speaker_dict=example[choice_name]["speaker_dict"],
        lang=example[choice_name]["lang"],
        voc=example[choice_name]["voc"],
        voc_config=example[choice_name]["voc_config"],
        voc_ckpt=example[choice_name]["voc_ckpt"],
        voc_stat=example[choice_name]["voc_stat"],
        spk_id=example[choice_name]["spk_id"],
        device="gpu",
        text=text,
        output="output.wav",
    )
    return wav_file, gr.update(), gr.update(interactive=True)


with gr.Blocks() as app:
    # choice_name = gr.Radio(choices=choices, value=choices[0], label="声学模型")
    # voc = gr.Radio(choices=vocs, value="hifigan_aishell3", label="声码器")
    # spk = gr.Number(value=0, label="说话人id", precision=0)
    voice = gr.Radio(choices=examples, label="加载人物")
    with gr.Row():
        with gr.Column(scale=4):
            input = gr.Textbox(
                placeholder="请输入文字...", value="那好吧，我们明天六点在人民广场见面，记得准时到哦", label="输入文本"
            )
        with gr.Column(scale=1):
            input_btn = gr.Button(value="文本转语音", variant="primary", interactive=False)
    gr.Examples(
        examples=[
            "北京冲量在线科技有限公司成立于2020年8月，是国内领先的隐私计算科技创新企业",
            "都什么时候了你还在乎利息，你这卡里还有多少钱。",
            "那好吧，我们明天六点在人民广场见面，记得准时到哦",
            "你别不信，我上网查了真的有这个人，不信我再查给你看",
            "我有分辨能力，我知道我自己能看什么书，我求你别管我，也别替我操心了行吗？",
            "是的！安迪，你怎么进来的呀？",
            "许之一，你能干什么，餐餐点不明白，租金租金交不起，你就是个废柴，你就是。",
        ],
        inputs=[input],
    )
    audio = gr.Audio(label="转换结果", autoplay=True)
    response = voice.change(
        fn=load,
        inputs=[input, voice],
        outputs=[audio, voice, input_btn],
    )
    input_btn.click(
        fn=speech_generate,
        inputs=[input, voice],
        outputs=[audio],
    )
app.queue(1)
app.launch(
    server_name="0.0.0.0",
    # server_port=8077,
    debug=False,
    inbrowser=False,
    show_api=False,
    share=True,
)
