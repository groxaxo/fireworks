import os, streamlit as st
import fireworks.client
from fireworks.client.image import ImageInference, Answer

# Streamlit app
st.subheader("Fireworks Playground")
with st.sidebar:
  fireworks_api_key = st.text_input("Fireworks API Key", type="password")
  option = st.selectbox("Select Model", [
    "Text: Meta Llama 3 70B Instruct",
    "Text: Google Gemma 2 9B Instruct",
    "Text: Mixtral MoE 8x7B Instruct",
    "Text: 01 Yi Large",
    "Image: Stable Diffusion XL"]
    )

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
        if option == "Text: Meta Llama 3 70B Instruct":
          # Run llama-v3-70b-instruct model on Fireworks AI
          response = fireworks.client.ChatCompletion.create(
              model="accounts/fireworks/models/llama-v3-8b-instruct",
              messages=[{
                  "role": "user",
                  "content": prompt,
              }],
          )
          st.success(response.choices[0].message.content)
        elif option == "Text: Google Gemma 2 9B Instruct":
          # Run gemma2-9b-it model on Fireworks AI
          response = fireworks.client.ChatCompletion.create(
              model="accounts/fireworks/models/gemma2-9b-it",
              messages=[{
                  "role": "user",
                  "content": prompt,
              }],
          )
          st.success(response.choices[0].message.content)
        elif option == "Text: Mixtral MoE 8x7B Instruct":
          # Run mixtral-8x7b-instruct model on Fireworks AI
          response = fireworks.client.ChatCompletion.create(
              model="accounts/fireworks/models/mixtral-8x7b-instruct",
              messages=[{
                  "role": "user",
                  "content": prompt,
              }],
          )
          st.success(response.choices[0].message.content)
        elif option == "Text: 01 Yi Large":
          # Run yi-large model on Fireworks AI
          response = fireworks.client.ChatCompletion.create(
              model="accounts/fireworks/models/yi-large",
              messages=[{
                  "role": "user",
                  "content": prompt,
              }],
          )
          st.success(response.choices[0].message.content)
        elif option == "Image: Stable Diffusion XL":
          # Run stable-diffusion-xl-1024-v1-0 model on Fireworks AI
          client = ImageInference(model="stable-diffusion-xl-1024-v1-0")
          answer : Answer = client.text_to_image(
              prompt=prompt,
              cfg_scale=7,
              height=1024,
              width=1024,
              sampler=None,
              steps=30,
              seed=0,
              safety_check=False,
              output_image_format="PNG"
          )
          st.image(answer.image)
    except Exception as e:
      st.exception(f"Exception: {e}")
