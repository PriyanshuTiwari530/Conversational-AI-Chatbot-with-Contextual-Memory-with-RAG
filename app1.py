import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
import os

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path)
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY", "")


st.set_page_config(page_title="RAG Chatbot with Memory", page_icon="")
st.title(" ")
st.caption("Here To Assist You, upload a document to ground answers in it. Conversation memory persists per session.")

# ---------- Session state setup ----------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user-session-1"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None  # built lazily on first message or on doc upload
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None


@st.cache_resource
def get_model():
    return init_chat_model(model="command-r7b-12-2024", model_provider="cohere")


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_retriever(file_path: str, file_name: str):
    if file_name.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def build_chatbot(retriever=None):
    """retriever=None -> plain chat, no document grounding.
    retriever=<FAISS retriever> -> RAG-grounded chat using the uploaded doc."""
    model = get_model()

    trimmer = trim_messages(
        max_tokens=1000,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
    )

    if retriever is not None:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Answer using this context: {context}"),
            MessagesPlaceholder("messages"),
        ])
    else:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder("messages"),
        ])

    def call_model(state: MessagesState):
        trimmed_messages = trimmer.invoke(state["messages"])

        if retriever is not None:
            last_user_message = state["messages"][-1].content
            retrieved_docs = retriever.invoke(last_user_message)
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)
            prompt = prompt_template.invoke({
                "messages": trimmed_messages,
                "context": context
            })
        else:
            prompt = prompt_template.invoke({"messages": trimmed_messages})

        response = model.invoke(prompt)
        return {"messages": response}

    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ---------- Sidebar: file upload (optional) ----------
with st.sidebar:
    st.header(" Upload a document (optional)")
    st.caption("Without a document, you can still chat normally below.")
    uploaded_file = st.file_uploader("Choose a .txt or .pdf file", type=["txt", "pdf"])

    if uploaded_file is not None:
        if st.button("Build knowledge base"):
            with st.spinner("Processing document..."):
                suffix = "." + uploaded_file.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                retriever = build_retriever(tmp_path, uploaded_file.name)
                st.session_state.retriever = retriever
                st.session_state.chatbot = build_chatbot(retriever)
                st.session_state.doc_name = uploaded_file.name
                st.session_state.messages = []
                os.unlink(tmp_path)
            st.success(f"Knowledge base built from {uploaded_file.name}! Answers will now use this document.")

    if st.session_state.doc_name:
        st.info(f" Currently grounded on: **{st.session_state.doc_name}**")
        if st.button("Remove document (go back to plain chat)"):
            st.session_state.retriever = None
            st.session_state.doc_name = None
            st.session_state.chatbot = build_chatbot(None)
            st.session_state.messages = []
            st.rerun()

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = st.session_state.thread_id + "-new"
        st.rerun()

# ---------- Chat UI ----------
# Build a plain (non-RAG) chatbot automatically if none exists yet,
# so the user can chat immediately without uploading anything.
if st.session_state.chatbot is None:
    st.session_state.chatbot = build_chatbot(st.session_state.retriever)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

placeholder_text = (
    "Ask something about your document..."
    if st.session_state.doc_name
    else "Ask me anything..."
)
user_input = st.chat_input(placeholder_text)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.spinner("Thinking..."):
        output = st.session_state.chatbot.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        answer = output["messages"][-1].content

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
