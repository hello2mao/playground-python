# chatchat

## 模型
| 模型 | 版本 | 支持进度 |
| :--: | :--: | :--: | 
| ChatGPT | ChatGPT-免费版 <br> ChatGPT-收费版| 程序员 | 
| ChatGLM | 30 | 设计师 | 
| Baichuan | 28 | 产品经理 |

## 插件支持
| 插件 | 功能 | 进度 |  
| :--: | :--: | :--: |  
| 张三 | 25 | 程序员 |  
| 李四 | 30 | 设计师 |  
| 王五 | 28 | 产品经理 |

## Run
```bash
# ubuntu 20.04
sudo apt install -y swig

conda create -n chatchat python=3.10
conda activate chatchat
pip install -r requirements.txt
python main.py
```

# Ref
https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese