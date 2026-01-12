from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from common.rag.embeddings import fetch_relevant_document
from common.rag.models import load_huggingface_model


def generate_personality_summary(trait):
    """
    This is your RAG / LLM call.
    Replace this with your actual implementation.
    """
    if trait is None:
        return ""

    system_message = """You are talented graphalogy expert who has PHD in it. You are best in reading and analysing the handwritting. You are a helpful assistant. Answer ONLY from the provided transcript context. If the context is insufficient, just say you don't know."""

    question = "Explain the detailed analysis on shared topic"

    context = fetch_relevant_document(topic=trait)

    model = load_huggingface_model()

    # Create RAG prompt template
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_message}"),
        ("human", """Context information:\n\n{context}\n\nQuestion:\n\n{question}\n\nTopic:{topic}\n\nAnswer:""")
    ])

    simple_rag_chain = (
        rag_prompt
        | model
        | StrOutputParser()
    )

    # Just invoke with dictionary of the three variables
    answer = simple_rag_chain.invoke({
        "system_message": system_message, # or just pass topic and build inside
        "context": context,
        "question": question,
        "topic": trait
    })

    return answer