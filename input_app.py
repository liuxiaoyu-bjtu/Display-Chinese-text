import gradio as gr
import torch
from transformers import BertTokenizer, BertModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)

font_path = 'SimHei.ttf'  # 替换为实际的字体文件路径
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'SimHei'  # 将 'custom_font' 替换为您为该字体指定的名称
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示字符

def get_sentence_embedding(sentence):
    # 加载BERT模型和分词器
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # 使用[CLS]标记的隐藏状态作为句子的嵌入向量
    sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return sentence_embedding

def process_text(input_text):
    # 将输入文本按行分割
    sentences = input_text.splitlines()  # 按行分割文本
    # 获取每个句子的嵌入向量
    embeddings = [get_sentence_embedding(sentence) for sentence in sentences]

    # 将列表转换为NumPy数组
    embeddings = np.array(embeddings)
    # 使用t-SNE进行降维，设置perplexity为小于样本数量的值
    tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 可视化句子向量在二维空间中的分布
    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(embeddings_2d):
        plt.scatter(x, y, color='blue', alpha=0.5)
        plt.text(x, y, sentences[i], fontsize=12)
    plt.xlabel('降维后的第1个维度')
    plt.ylabel('降维后的第2个维度')
    plt.axis('equal')
    plt.title('句子向量二维空间可视化')
    plt.grid(True)
    plt.tight_layout()
    image_path = 'output_display.png'
    plt.savefig(image_path)
    return image_path, gr.update(visible=True)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_textbox = gr.Textbox(label="请输入文本，每行代表一个句子", lines=5)  # 输入框，支持多行
            submit_button = gr.Button("提交")  # 提交按钮
        with gr.Column():
            output_image = gr.Image(label="待提交数据")  # 图片显示区域

    # 绑定按钮点击事件，调用process_text处理文本
    submit_button.click(process_text, inputs=input_textbox, outputs=[output_image, output_image])

demo.launch(share=True)
