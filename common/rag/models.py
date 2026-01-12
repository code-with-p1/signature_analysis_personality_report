import warnings
warnings.filterwarnings(action='ignore')
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()

def load_huggingface_model():
    chat_llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        temperature=0.65,
        max_new_tokens=1024,
        top_p=0.92,
        repetition_penalty=1.05
    )
    model = ChatHuggingFace(llm=chat_llm)
    return model