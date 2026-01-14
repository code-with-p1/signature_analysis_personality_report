import warnings
warnings.filterwarnings(action='ignore')
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()

def load_huggingface_model():
    """
    Initialize and return a Hugging Face chat model wrapper for RAG-based graphology analysis.

    Creates a HuggingFaceEndpoint instance connected to the Qwen2.5-7B-Instruct model
    and wraps it with ChatHuggingFace for conversational compatibility with LangChain.

    Configuration:
    - Model: Qwen/Qwen2.5-7B-Instruct (7B parameter instruction-tuned model)
    - Temperature: 0.65 (balanced between creativity and coherence)
    - Max new tokens: 1024
    - Top-p: 0.92 (nucleus sampling)
    - Repetition penalty: 1.05 (light discouragement of repetitions)

    Returns
    -------
    ChatHuggingFace
        Configured LangChain-compatible chat model ready to be used in chains

    Notes
    -----
    - Requires HUGGINGFACEHUB_API_TOKEN to be set in environment variables
      (loaded via dotenv)
    - Uses inference endpoint (cloud-based inference) — no local GPU/CPU loading
    - Model is reloaded every time this function is called
    - Current settings are optimized for structured, precise graphological analysis
      with controlled creativity
    - Consider adjusting temperature/max_new_tokens based on response length needs

    Raises
    ------
    ValueError
        If HUGGINGFACEHUB_API_TOKEN is missing or invalid
    """

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