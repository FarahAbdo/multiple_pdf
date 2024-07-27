import streamlit as st
import transformers
from torch import cuda
import torch
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList
import os
from dotenv import load_dotenv

load_dotenv()

# Set up the model and tokenizer
model_id = 'meta-llama/Llama-2-7b-chat-hf'
hf_auth = os.getenv('HF_AUTH')
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# Load the model without quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map='auto',
    use_auth_token=hf_auth
)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)
stop_list = ['\nHuman:', '\n```\n']
stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    task='text-generation',
    stopping_criteria=stopping_criteria,
    temperature=0.1,
    max_new_tokens=512,
    repetition_penalty=1.1
)

# Load documents
web_links = ["https://drive.google.com/drive/folders/1k6QvVG2yl7jvlpAKcwX2ECKgMmIFOjfG"]
loader = WebBaseLoader(web_links)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

# Initialize FAISS
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
vectorstore = FAISS.from_documents(all_splits, embeddings)

# Initialize Conversational Retrieval Chain
chain = ConversationalRetrievalChain.from_llm(model, vectorstore.as_retriever(), return_source_documents=True)

# Streamlit app
st.title("Document-based Question Answering")

# Text input for the user's question
query = st.text_input("Enter your question:")

# Button to submit the question
if st.button("Get Answer"):
    chat_history = []
    result = chain({"question": query, "chat_history": chat_history})
    st.write("Answer:", result['answer'])
    st.write("Source Documents:", result['source_documents'])

if __name__ == "__main__":
    st.run()

