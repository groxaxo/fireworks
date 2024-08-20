import os
import streamlit as st
import fireworks.client
from fireworks.client.image import ImageInference, Answer

# Streamlit app
st.subheader("Fireworks Playground")
with st.sidebar:
    fireworks_api_key = st.text_input("Fireworks API Key", type="password")
    option = st.selectbox("Select Model", [
        "Text: Meta Llama 3.1 405B Instruct",
        "Text: Meta Llama 3.1 70B Instruct",
        "Text: Meta Llama 3.1 8B Instruct",
        "Text: Llama 3 70B Instruct",
        "Text: Mixtral MoE 8x22B Instruct",
        "Text: Mixtral MoE 8x7B Instruct",
        "Text: Firefunction V2",
        "Text: FireLLaVA-13B",
        "Text: Chronos Hermes 13B v2",
        "Text: CodeGemma 2B",
        "Text: CodeGemma 7B",
        "Text: Code Llama 13B",
        "Text: Code Llama 13B Instruct",
        "Text: Code Llama 13B Python",
        "Text: Code Llama 34B",
        "Text: Code Llama 34B Instruct",
        "Text: Code Llama 34B Python",
        "Text: Code Llama 70B",
        "Text: Code Llama 70B Instruct",
        "Text: Code Llama 70B Python",
        "Text: Code Llama 7B",
        "Text: Code Llama 7B Instruct",
        "Text: Code Llama 7B Python",
        "Text: Code Qwen 1.5 7B",
        "Text: DeepSeek Coder 1.3B Base",
        "Text: DeepSeek Coder 33B Instruct",
        "Text: DeepSeek Coder 6.7B Base",
        "Text: DeepSeek Coder 7B Base v1.5",
        "Text: DeepSeek Coder 7B Instruct v1.5",
        "Text: DeepSeek Coder V2 Lite Base",
        "Text: DeepSeek Coder V2 Instruct",
        "Text: Dolphin 2 9 2 Qwen2 72b",
        "Text: Dolphin 2.6 Mixtral 8x7b",
        "Text: ELYZA-japanese-Llama-2-7b",
        "Text: FireFunction V1",
        "Text: Gemma 2 9B Instruct",
        "Text: Gemma 7B",
        "Text: Hermes 2 Pro Mistral 7b",
        "Text: Japanese StableLM Instruct Beta 70B",
        "Text: Japanese Stable LM Instruct Gamma 7B",
        "Text: Japanese Stable VLM",
        "Text: Llama Guard v2 8B",
        "Text: Llama Guard 7B",
        "Text: Llama 2 7B",
        "Text: Llama 3 70B Instruct (HF version)",
        "Text: Llama 3 8B Instruct",
        "Text: Llama 3 8B Instruct (HF version)",
        "Text: LLaVA V1.6 Yi 34B",
        "Text: Mistral 7B",
        "Text: Mistral 7B v0.2",
        "Text: Mistral Nemo Base 2407",
        "Text: Mistral Nemo Instruct 2407",
        "Text: Mixtral Moe 8x22B",
        "Text: Mixtral MoE 8x22B",
        "Text: Mixtral MoE 8x7B Instruct (HF version)",
        "Text: MythoMax L2 13b",
        "Text: Nous Capybara 7B V1.9",
        "Text: Nous Hermes 2 - Mixtral 8x7B - DPO",
        "Text: Nous Hermes 2 - Mixtral 8x7B - DPO (fp8)",
        "Text: Nous Hermes 2 - Yi 34B",
        "Text: Nous Hermes Llama2 13B",
        "Text: Nous Hermes Llama2 70B",
        "Text: Nous Hermes Llama2 7B",
        "Text: OpenChat 3.5 0106",
        "Text: OpenHermes 2 - Mistral 7B",
        "Text: OpenHermes 2.5 - Mistral 7B",
        "Text: Mistral 7B OpenOrca",
        "Text: Phi-2",
        "Text: Phi 3 Mini 128K Instruct",
        "Text: Phi 3 Vision 128K Instruct",
        "Text: Phind CodeLlama 34B Python v1",
        "Text: Phind CodeLlama 34B v1",
        "Text: Phind CodeLlama 34B v2",
        "Text: Pythia 12B",
        "Text: Qwen 14B Chat",
        "Text: Qwen1.5 72B Chat",
        "Text: Qwen 72B Chat",
        "Text: Snorkel Mistral PairRM DPO",
        "Text: Stable Code 3B",
        "Text: StableLM 2 Zephyr 1.6B",
        "Text: StableLM Zephyr 3B",
        "Text: StarCoder 15.5B",
        "Text: StarCoder2 15B",
        "Text: StarCoder2 3B",
        "Text: StarCoder2 7B",
        "Text: StarCoder 7B",
        "Text: Toppy M 7B",
        "Text: Yi 34B",
        "Text: Capybara 34B",
        "Text: Yi 34B Chat",
        "Text: Yi 6B",
        "Text: Yi-Large",
        "Text: Zephyr 7B Beta",
        "Image: Stable Diffusion XL",
        "Image: Stable Diffusion 3 Large",
        "Image: Stable Diffusion 3 Medium",
        "Image: Playground v2 1024",
        "Image: Playground v2.5 1024",
        "Image: Segmind Stable Diffusion 1B (SSD-1B)",
        "Image: Japanese Stable Diffusion XL",
        "Image: Stable Diffusion 3 Turbo",
    ])

os.environ["FIREWORKS_API_KEY"] = fireworks_api_key
prompt = st.text_input("Prompt", label_visibility="collapsed")

# If Generate button is clicked
if st.button("Generate"):
    if not fireworks_api_key.strip() or not prompt.strip():
        st.error("Please provide the missing fields.")
    else:
        try:
            with st.spinner("Please wait..."):
                fireworks.client.api_key = fireworks_api_key
                if option == "Text: Meta Llama 3.1 405B Instruct":
                    response = fireworks.client.ChatCompletion.create(
                        model="accounts/fireworks/models/llama-v3p1-405b-instruct",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=131072,
                    )
                    st.success(response.choices[0].message.content)
                elif option == "Text: Meta Llama 3.1 70B Instruct":
                    response = fireworks.client.ChatCompletion.create(
                        model="accounts/fireworks/models/llama-v3p1-70b-instruct",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=131072,
                    )
                    st.success(response.choices[0].message.content)
                # ... (rest of the models)
                elif option == "Text: Mistral Nemo Base 2407":
                    response = fireworks.client.ChatCompletion.create(
                        model="accounts/fireworks/models/mistral-nemo-base-2407",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=128000,
                    )
                    st.success(response.choices[0].message.content)
                elif option == "Text: Mistral Nemo Instruct 2407":
                    response = fireworks.client.ChatCompletion.create(
                        model="accounts/fireworks/models/mistral-nemo-instruct-2407",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=128000,
                    )
                    st.success(response.choices[0].message.content)
                # ... (rest of the models)
                elif option == "Image: Stable Diffusion XL":
                    client = ImageInference(model="stable-diffusion-xl-1024-v1-0")
                    answer: Answer = client.text_to_image(
                        prompt=prompt,
                        cfg_scale=7,
                        height=1024,
                        width=1024,
                        sampler=None,
                        steps=30,
                        seed=0,
                        safety_check=False,
                        output_image_format="PNG",
                    )
                    st.image(answer.image)
                # ... (rest of the image models)
        except Exception as e:
            st.exception(f"Exception: {e}")
