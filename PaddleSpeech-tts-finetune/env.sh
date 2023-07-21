# 配置 MFA 环境
cd /root/individual/hello2mao/playground-python/PaddleSpeech-tts-finetune/PaddleSpeech/examples/other/tts_finetune/tts3
mkdir -p tools/aligner
cd tools
# download MFA
wget https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_linux.tar.gz
cp montreal-forced-aligner_linux.tar.gz ./
# extract MFA
cd /root/individual/hello2mao/playground-python/PaddleSpeech-tts-finetune/PaddleSpeech/examples/other/tts_finetune/tts3
tar xvf montreal-forced-aligner_linux.tar.gz
# fix .so of MFA
cd montreal-forced-aligner/lib
ln -snf libpython3.6m.so.1.0 libpython3.6m.so
cd -
# download align models and dicts
cd aligner
wget https://paddlespeech.bj.bcebos.com/MFA/ernie_sat/aishell3_model.zip
wget https://paddlespeech.bj.bcebos.com/MFA/AISHELL-3/with_tone/simple.lexicon
wget https://paddlespeech.bj.bcebos.com/MFA/ernie_sat/vctk_model.zip
wget https://paddlespeech.bj.bcebos.com/MFA/LJSpeech-1.1/cmudict-0.7b
unzip aishell3_model.zip
cd ../../

# 下载预训练模型(此处为中英双语模型)
cd /root/individual/hello2mao/playground-python/PaddleSpeech-tts-finetune/PaddleSpeech/examples/other/tts_finetune/tts3
mkdir models
cd models
wget https://paddlespeech.bj.bcebos.com/t2s/chinse_english_mixed/models/fastspeech2_mix_ckpt_1.2.0.zip
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_aishell3_ckpt_0.2.0.zip
unzip fastspeech2_mix_ckpt_1.2.0.zip
unzip hifigan_aishell3_ckpt_0.2.0.zip
