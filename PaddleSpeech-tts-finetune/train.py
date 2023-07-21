import subprocess
import os
import glob
import logging
import shutil

home_dir = "/root/individual/hello2mao/playground-python/PaddleSpeech-tts-finetune/"


# 命令行执行函数，可以进入指定路径下执行
def run_cmd(cmd, cwd_path):
    p = subprocess.Popen(cmd, shell=True, cwd=cwd_path)
    res = p.wait()
    print(cmd)
    print("运行结果：", res)
    if res == 0:
        # 运行成功
        print("运行成功")
        return True
    else:
        # 运行失败
        print("运行失败")
        return False


def train(name):
    # 试验路径
    exp_dir = home_dir + "work/trained/" + name
    # 配置试验相关路径信息
    cwd_path = home_dir + "PaddleSpeech/examples/other/tts_finetune/tts3"
    # 可以参考 env.sh 文件，查看模型下载信息
    pretrained_model_dir = "models/fastspeech2_mix_ckpt_1.2.0"

    # # 同时上传了 wav+标注文本 以及本地生成的 textgrid 对齐文件
    # 输入数据集路径
    data_dir = home_dir + "work/dataset/" + name + "/wav"
    # 如果上传了 MFA 对齐结果，则使用已经对齐的文件
    mfa_dir = home_dir + "work/dataset/" + name + "/textgrid"

    # 输出文件路径
    wav_output_dir = os.path.join(exp_dir, "output")
    os.makedirs(wav_output_dir, exist_ok=True)

    dump_dir = os.path.join(exp_dir, "dump")
    output_dir = os.path.join(exp_dir, "exp")
    lang = "zh"

    new_dir = "work/dataset/" + name + "/textgrid/newdir"

    in_label = (
        home_dir + "PaddleSpeech/examples/other/tts_finetune/tts3/conf/finetune.yaml"
    )
    shutil.copy(in_label, exp_dir)

    # check oov
    cmd = f"""
        python3 local/check_oov.py \
            --input_dir={data_dir} \
            --pretrained_model_dir={pretrained_model_dir} \
            --newdir_name={new_dir} \
            --lang={lang}
    """

    # 执行该步骤
    run_cmd(cmd, cwd_path)

    cmd = f"""
    python3 local/generate_duration.py \
        --mfa_dir={mfa_dir}
    """

    # 执行该步骤
    run_cmd(cmd, cwd_path)

    cmd = f"""
    python3 local/extract_feature.py \
        --duration_file="./durations.txt" \
        --input_dir={data_dir} \
        --dump_dir={dump_dir}\
        --pretrained_model_dir={pretrained_model_dir}
    """

    run_cmd(cmd, cwd_path)

    cmd = f"""
    python3 local/prepare_env.py \
        --pretrained_model_dir={pretrained_model_dir} \
        --output_dir={output_dir}
    """
    run_cmd(cmd, cwd_path)

    epoch = 250
    config_path = os.path.join(exp_dir, "finetune.yaml")

    cmd = f"""
    python3 local/finetune.py \
        --pretrained_model_dir={pretrained_model_dir} \
        --dump_dir={dump_dir} \
        --output_dir={output_dir} \
        --ngpu=1 \
        --epoch={epoch} \
        --finetune_config={config_path}
    """
    run_cmd(cmd, cwd_path)


all_file_name_list = glob.glob(
    "/root/individual/hello2mao/playground-python/PaddleSpeech-tts-finetune/work/dataset/*"
)
for file_name in all_file_name_list:
    name = os.path.basename(file_name)
    logging.info(f"start train: {name}")
    try:
        train(name)
    except Exception as e:
        logging.error(f"train err: {e}")
        continue
    logging.info(f"train: {name} done")
