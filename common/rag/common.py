from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from common.rag.embeddings import fetch_relevant_document
from common.rag.models import load_huggingface_model


def generate_personality_summary(trait):
    """
    Generate a graphological personality summary for a given trait/topic using RAG.

    This function performs a Retrieval-Augmented Generation (RAG) query to analyze
    handwriting characteristics and derive corresponding personality traits/psychological
    interpretations — but **only** from information explicitly present in retrieved documents.

    Important:
        The analysis is strictly limited to the content found in the vector store.
        No external/pre-trained graphological knowledge is used by the LLM.

    Parameters
    ----------
    trait : str or None
        The personality trait, psychological characteristic, behavioral pattern or
        topic for which handwriting analysis should be retrieved and interpreted.
        Examples: "introversion", "high ambition", "emotional instability", "leadership"

    Returns
    -------
    str
        Graphological analysis containing:
        - observed handwriting features (if any were found)
        - their professional graphological interpretation
        - overall personality impression
        OR one of the following safety messages:
        - "The provided context contains insufficient information for handwriting analysis"
        - empty string (when trait is None)

    Notes
    -----
    - The function is intentionally very strict about hallucination prevention.
    - Quality of the result depends heavily on the relevance and richness of documents
      stored in the vector database for the given trait.
    """

    if trait is None:
        return ""

    system_message = """
    You are a highly experienced professional graphologist with a PhD in Graphology and more than 20 years of practical experience in forensic and psychological handwriting analysis.

    Your only task is to analyze handwriting features and give interpretations STRICTLY based on the information provided in the retrieved context/transcript.

    Rules you must follow:
    • Never use knowledge or assumptions from your training data
    • Never invent or assume handwriting characteristics that are not explicitly described in the provided context
    • If the context contains insufficient information for a meaningful analysis → answer only: "The provided context contains insufficient information for handwriting analysis"
    • Use professional graphological terminology
    • Structure your answer clearly: first describe observed features, then psychological/personality interpretation (if enough data)

    Be objective, precise, and stay 100% within the provided context.
    """

    question = f"Analyze the handwriting features and personality traits of a person characterized as: {trait}, using ONLY the information present in the provided context."

    context = fetch_relevant_document(topic=trait)

    model = load_huggingface_model()

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_message}"),
        ("human", """Context information:\n\n{context}\n\nQuestion:\n\n{question}\n\nTopic:{topic}\n\nAnswer:""")
    ])

    simple_rag_chain = (
        rag_prompt
        | model
        | StrOutputParser()
    )

    answer = simple_rag_chain.invoke({
        "system_message": system_message,
        "context": context,
        "question": question,
        "topic": trait
    })

    return answer