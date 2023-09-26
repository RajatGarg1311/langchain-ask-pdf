#from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import tiktoken
import time


def count_tokens(message, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(message, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(message, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    num_tokens += len(encoding.encode(message))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def main():
    #load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

    OPENAI_API_KEY = ""
    llm_model = ""
    st.sidebar.header("Settings")
    with st.sidebar:
        llm_model = st.selectbox("LLM Models", ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613"]) 
        OPENAI_API_KEY=st.text_input("AI API Key", type="password")
        #"[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    
    # Selected Open Model
    st.write("Selected LLM Model: ",llm_model)
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # count tokens from chunks
      num_tokens = sum([count_tokens(chunk, llm_model) for chunk in chunks])
      cost_per_token = 0.0099
      estimated_cost = (num_tokens/500) * cost_per_token

      st.write("Number of tokens: ", str(num_tokens))
      st.write("Estimated cost: ", str(round(estimated_cost, 2)), "$")

      while OPENAI_API_KEY == "":
        with st.spinner('Checking API Key'):
              st.toast('Please Add API Key', icon='🚨')
              time.sleep(5)
        if OPENAI_API_KEY != "":
          st.toast('Done!', icon='🎉')

      isAgreed = st.checkbox("I agree, to bear the cost")

      if (isAgreed == True and OPENAI_API_KEY != ""):
        try:
          # create embeddings         
          embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
          knowledge_base = FAISS.from_texts(chunks, embeddings)
          isDisabled = False
        except Exception as e:
          st.error(e, icon= "🔥")
          isDisabled = True
        
        # show user input
        user_question = st.text_input("Ask a question about your PDF:", disabled=isDisabled)
        if user_question:
          docs = knowledge_base.similarity_search(user_question)
          try:
            llm = OpenAI(temperature=0.0, openai_api_key=OPENAI_API_KEY)
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
              response = chain.run(input_documents=docs, question=user_question)
              print(cb)
              
            st.write(response)
          except Exception as e:
            st.error(e, icon= "🔥")

if __name__ == '__main__':
    main()
