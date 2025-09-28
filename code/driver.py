import os
os.environ.setdefault("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
os.environ.setdefault("QWEN_API_KEY", "************")
os.environ.setdefault("BRAIN_MODEL", "qwen2.5-72b-instruct")
import gradio as gr 
from llm_client import LLMClient
from brain import Brain
from page_functions import *

os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["no_proxy"]  = "127.0.0.1,localhost,::1"
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(k, None)


brain_client = LLMClient(os.getenv("QWEN_API_BASE"), os.getenv("QWEN_API_KEY"), os.getenv("BRAIN_MODEL"))
brain = Brain(brain_client)
with gr.Blocks(title="AnnoAgent") as demo:
    webpage_title ="""
    <div style="text-align: center; margin: 20px 0;">
        <span style="font-size: 2.5em; font-weight: bold; color: #2c3e50;">
            AnnoAgent for single cell Annotation
        </span>
    </div>
    """
    gr.Markdown(webpage_title)
    arg_brain = gr.State({"brain":brain})
    arg_paras = gr.State({ "mode":"REFERENCE", "x":None ,"y":None, "x_test":None ,"y_test":None,"train_mask":None, "valid_mask":None})
    chat = gr.Chatbot(height=520, label="AnnoAgent", value=[("Start chat!","Please provide the reference dataset!")])
    user_msg = gr.Textbox(
        placeholder="Train;Predict;Yes;No",
        label="Chat with AnnoAgent",
    )
    user_msg.submit(text_submit, [chat, user_msg, arg_brain, arg_paras], [chat, user_msg, arg_brain, arg_paras])

    gr.Markdown("---")
    gr.Markdown("📤 Upload CSV\n")
    file_input = gr.File(label="SELECT FILE", file_types=[".csv"])
    up_btn = gr.Button("Upload and Handle")
    up_btn.click(file_upload, [chat ,file_input, arg_brain, arg_paras], [chat, arg_brain, arg_paras])
demo.launch(server_name="0.0.0.0", server_port=8000, share=False, show_error=True)