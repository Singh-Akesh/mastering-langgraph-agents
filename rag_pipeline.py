from typing import List
from langchain.schema import Document
from agent_react import retriever, llm

def combine_docs(docs: List[Document], max_chars: int = 20000) -> str:
    """Make a single text block from docs (keeps source metadata)."""
    if not docs:
        return ""

    parts = []
    for i, d in enumerate(docs, 1):
        meta = dict(d.metadata or {})
        header = f"--- EMAIL {i} ---\n"
        if meta:
            header += "METADATA: " + str(meta) + "\n"

        content = d.page_content.strip()
        parts.append(header + content)

    text = "\n\n".join(parts)

    # truncate if too long
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n--- TRUNCATED ---"

    return text



SYSTEM_INSTRUCTIONS = (
    "You are an assistant that answers strictly using the provided email content. Ignore '_id'."
    "You MUST base your answer only on that content. "
    "Do not hallucinate or use other knowledge."
)

PROMPT_TEMPLATE = SYSTEM_INSTRUCTIONS + "\n\nEMAILS:\n{context}\n\nQUESTION: {question}\n\nANSWER:"

def rag_query(query: str):
    docs = retriever.invoke(query)
    print("Retrieved from qdrant: ", docs)
    if not docs:
        return {"answer": "I could not find the information in the emails.", "source_documents": []}

    context = combine_docs(docs)
    prompt_text = PROMPT_TEMPLATE.format(context=context, question=query)

    resp = llm.invoke(prompt_text)

    # resp might be a string or an object; normalize
    if isinstance(resp, dict):
        # sometimes returns {"content": "..."} or similar
        answer = resp.get("content") or resp.get("result") or str(resp)
    else:
        answer = str(resp)

    return {"answer": answer, "source_documents": docs}


if __name__ == "__main__":
    q = "How many uber receipts do I have?"
    out = rag_query(q)
    print("=== ANSWER ===")
    print(out["answer"])

