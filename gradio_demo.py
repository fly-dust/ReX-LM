import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import spaces

# Load model and tokenizer
model_name = "yuchenlin/Rex-v0.1-1.5B"

device = "cuda" # the device to load the model onto
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, rex_size=3)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto"
)
model.to(device)

@spaces.GPU(enable_queue=True)
def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens=512,
    temperature=0.5,
    top_p=1.0,
    repetition_penalty=1.1,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})
 
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens = max_tokens,
        temperature = temperature,
        top_p = top_p,
        repetition_penalty=repetition_penalty,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a helpful AI assistant and your name is RexLM.", label="System message"),
        gr.Slider(minimum=1, maximum=4096, value=1024, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.5, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
        gr.Slider(minimum=0.5, maximum=1.5, value=1.1, step=0.1, label="Repetation Penalty"),
    ],
)


if __name__ == "__main__":
    demo.launch(share=True)