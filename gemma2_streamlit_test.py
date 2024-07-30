#### Env: test_env
import streamlit as st
from huggingface_hub import InferenceClient
from huggingface_hub import login
import os
os.environ['CURL_CA_BUNDLE'] = '' # Solving SSL Certification Error // 해당 에러가 발생했다 안했다 하는 상황....

# -------------- Hugging Face Setting --------------
# Hugging Face Login
access_token_read = "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"
access_token_write = "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"
login(token = access_token_read)

# Gemma Client Parameter Setting
generate_kwargs = dict(
            temperature=0.49,
            max_new_tokens=1600,
            top_p=0.49,
            repetition_penalty=0.99,
            do_sample=True)

# 모델 지정
client = InferenceClient("google/gemma-1.1-7b-it") # google/gemma-1.1-2b-it

# -------------- Gemma Response --------------
# gemma 프롬프트 형식 설정 -> 나중에 History를 추가하고 싶으면 값들을 받아서 얹는 방식으로 실행 (7b-it로 실행해야 프롬프트를 알아먹음)
def prompt_format(query:str):
    prompt = f"<start_of_turn>user{query}<end_of_turn><start_of_turn>model"
    return str(prompt)

#def gemma_response(query:str, kwargs:dict):
#    stream_response = client.text_generation(prompt_format(query), **generate_kwargs, stream = True, details=True, return_full_text=True)
    #stream_response = client.text_generation('Write me a poem about Machine Learning.', **generate_kwargs, stream=True)

#    output = ""
#    for response in stream_response:
#        output += (response.token.text).replace('<eos>', '') # 이것보다 좋은 방법이 있을 것 같은데..
#
#    return output

# -------------- Streamlit setting --------------
st.title('Gemma 1.1 Chatbot')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [] # 여기다가 계속 append.

# Message History Show
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
# prompt 변수를 동시에 선언하고 할당하는 것을 의미
if prompt := st.chat_input("Message To Gemma"):
    # User Message History 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    # User 부분에 메시지 할당
    with st.chat_message('user'):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Response Stream 기능 설정
        #response = st.write_stream(gemma_response(str(prompt), generate_kwargs))
        response = st.write_stream(client.text_generation(prompt_format(prompt), **generate_kwargs, stream = True))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

