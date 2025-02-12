import gradio as gr
import torch
import os
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image
import numpy as np

# Load model and processor
model_path = "deepseek-ai/Janus-Pro-7B"
config = AutoConfig.from_pretrained(model_path)
language_config = config.language_config
language_config._attn_implementation = 'eager'
vl_gpt = AutoModelForCausalLM.from_pretrained(model_path,
                                             language_config=language_config,
                                             trust_remote_code=True)

if torch.cuda.is_available():
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda()
else:
    vl_gpt = vl_gpt.to(torch.float16)

vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.inference_mode()
def multimodal_understanding(image, question, seed, top_p, temperature):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    conversation = [
        {"role": "<|User|>", "content": f"<image_placeholder>\n{question}", "images": [image]},
        {"role": "<|Assistant|>", "content": ""},
    ]

    pil_images = [Image.fromarray(image)]
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(cuda_device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

@torch.inference_mode()
def generate_image(prompt, seed=None, guidance=5, t2i_temperature=1.0):
    torch.cuda.empty_cache()
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    # Placeholder implementation (adjust as per your model's needs)
    return f"Generated image for prompt: {prompt}"

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(value="# Multimodal Understanding")
    with gr.Row():
        image_input = gr.Image()
        with gr.Column():
            question_input = gr.Textbox(label="Question")
            und_seed_input = gr.Number(label="Seed", precision=0, value=42)
            top_p = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.05, label="top_p")
            temperature = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.05, label="temperature")
        
    understanding_button = gr.Button("Chat")
    understanding_output = gr.Textbox(label="Response")

    prompt_input = gr.Textbox(label="Prompt")
    seed_input = gr.Number(label="Seed (Optional)", precision=0, value=12345)

    generation_button = gr.Button("Generate Images")
    image_output = gr.Textbox(label="Generated Image Output")

    understanding_button.click(
        multimodal_understanding,
        inputs=[image_input, question_input, und_seed_input, top_p, temperature],
        outputs=understanding_output
    )

    generation_button.click(
        generate_image,
        inputs=[prompt_input, seed_input],
        outputs=image_output
    )

# Ensure the app binds to the correct port on Render
port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
