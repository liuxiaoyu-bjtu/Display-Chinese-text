import gradio as gr
import torch
from transformers import BertTokenizer, BertModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)

def process_file(file):
    # 加载BERT模型和分词器
    model_name = '/Display-Chinese-text/bert-base-chinese1/model_path'
    
    # 尝试加载本地模型并捕获异常
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
    except OSError as e:
        # 如果加载模型失败，返回错误信息
        return f"加载模型时出错：{str(e)}", gr.update(visible=True), gr.update(visible=False), None, gr.update(visible=False)

    # 读取txt文档，并将每一行文本以字符串的形式存在列表中
    sentences = []
    if file.name.endswith('.txt'):
        try:
            with open(file.name, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f]  # 去掉每行的首尾空格或换行符
        except FileNotFoundError:
            return "文件未找到", gr.update(visible=True), None, gr.update(visible=False)
        except Exception as e:
            return f"读取文件时出错：{e}", gr.update(visible=True), None, gr.update(visible=False)
    else:
        return "不支持的数据文件格式。", gr.update(visible=True), None, gr.update(visible=False)
    
    # 将列表内容转换为带换行符的字符串
    sentences_text = "\n".join(sentences)

    # 获取每个句子的嵌入向量
    embeddings = []
    for sentence in sentences:
        # 将每个句子转换为BERT嵌入
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        # 使用[CLS]标记的隐藏状态作为句子的嵌入向量
        sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(sentence_embedding)

    # 将列表转换为NumPy数组
    embeddings = np.array(embeddings)
    # 使用t-SNE进行降维，设置perplexity为小于样本数量的值
    tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 可视化句子向量在二维空间中的分布
    plt.figure(figsize=(12, 8))
    for i, (x, y) in enumerate(embeddings_2d):
        plt.scatter(x, y, color='blue', alpha=0.5)
        plt.text(x, y, sentences[i], fontsize=14)
    plt.xlabel('第1个维度', fontsize=14)
    plt.ylabel('第2个维度', fontsize=14)
    plt.title('句子可视化', fontsize=14)
    plt.rcParams['xtick.labelsize'] = 12  # 设置横坐标刻度字体大小
    plt.rcParams['ytick.labelsize'] = 12  # 设置纵坐标刻度字体大小  
    plt.grid(True)
    plt.tight_layout()
    plt.axis('equal')
    image_path = 'output_display.png'
    plt.savefig(image_path)
    
    return sentences_text, gr.update(visible=False), image_path, gr.update(visible=True)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="上传txt文本文档文件（支持TXT格式", file_types=["txt"])
            submit_button = gr.Button("提交")  # 添加提交按钮
        with gr.Column():
            data_ = gr.Text(value="数据待上传", label="自然语言文本：")
            text_placeholder = gr.Markdown("数据待上传", visible=True)  # 用于显示提示信息
            output_image = gr.Image(visible=False)  # 图片显示区域
    # 文件上传后调用 process_file 函数
    submit_button.click(process_file, inputs=file_input, outputs=[data_, text_placeholder, output_image, output_image])

demo.launch(share=True)
