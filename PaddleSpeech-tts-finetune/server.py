from paddlespeech.server.bin.paddlespeech_server import ServerExecutor

server_executor = ServerExecutor()
server_executor(
    config_file="./conf/tts_online_application_finetune.yaml",
    log_file="./log/paddlespeech.log",
)
