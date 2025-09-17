from langgraph.prebuilt import create_react_agent
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.tools.retriever import create_retriever_tool
from langchain.chains import RetrievalQA
from qdrant_util import client, COLLECTION_NAME

# 1️⃣ Embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# 2️⃣ Vectorstore
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
    content_payload_key="text"
)

# 3️⃣ Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5, "score_threshold": 0.75}
)

# 4️⃣ RAG Tool
rag_tool = create_retriever_tool(
    retriever,
    name="email_search",
    description=(
        "Always use this tool to search the user's emails before answering. "
        "You must ground your response in the retrieved email content. "
        "If the information is not found in the emails, say you cannot find it "
        "instead of making something up."
    ),
)

# 5️⃣ Ollama LLM (local)
llm = ChatOllama(
    model="llama3-groq-tool-use",
    temperature=0,
    base_url="http://localhost:11434"
)

# 6️⃣ ReAct agent

prompt = """
You have access to a tool called email_search that can search the user's emails for relevant information.
You **must** use the email_search tool to retrieve relevant emails before answering.
If the email_search tool returns no results, respond: 'I could not find the information in the emails.'
Do not hallucinate or use any other information.
"""

agent = create_react_agent(
    model=llm,
    tools=[rag_tool],
    prompt=prompt,
    # max_iterations=3,   # ensures agent can call tool multiple times # NOT SUPPORTED
    # verbose=True # set to True to see the reasoning and tool calls # NOT SUPPORTED
)

# RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,  # optional: returns which emails were used
    chain_type="stuff",  # or 'map_reduce' depending on your preference
)


# 7️⃣ Invoke
# query = "Summarize the email I received about Job referral"
# agentResult = agent.invoke(
#     {"messages": [{"role": "user", "content": query}]}
# )
#
# chainResult = qa_chain.invoke({"query": query})
#
# # 8️⃣ Print the result
# print(agentResult)
# print(chainResult)

