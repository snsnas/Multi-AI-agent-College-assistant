import streamlit as st
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="College Assistant",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD RESOURCES
# ============================================================

@st.cache_resource(show_spinner="Loading college knowledge base...")
def load_resources():

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # --------------------------------------------------------
    # PDF -> Chunks -> FAISS -> Retriever
    # --------------------------------------------------------

    def build_retriever(pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100)

        chunks = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        return retriever
    

    academic_retriever = build_retriever( "academics_handbook.pdf")

    fee_retriever = build_retriever("fee_structure.pdf")


    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.3)

    return (academic_retriever, fee_retriever, llm)


academic_retriever, fee_retriever, llm = load_resources()


# ============================================================
# LANGGRAPH STATE
# ============================================================

class State(TypedDict):
    programme: str
    messages: Annotated[list,add_messages]
    query_type: str
    retrieved_context: str


# ============================================================
# CLASSIFIER NODE
# ============================================================

def classifier_node(state: State) -> dict:
    last_message = (state["messages"][-1].content)


    prompt = f"""
You are a query classifier for a college AI assistant.

Classify the student's query into EXACTLY ONE category:

academic
fee
general


ACADEMIC includes questions about:

attendance
examinations
grading
credits
promotion rules
course structure
degree requirements
academic policies
summer training
internships mentioned in academic documents
semester rules


FEE includes questions about:

tuition
semester fees
annual fees
payment
refund
late charges
scholarships
security deposits
student activity fees
money-related college questions


GENERAL includes:

greetings
casual conversation
general questions
anything unrelated to academic or fee policies


Student query:

{last_message}


Return ONLY ONE WORD:

academic

OR

fee

OR

general
"""


    response = llm.invoke(prompt)


    category = ( response.content.strip().lower())


    if "academic" in category:
        category = "academic"

    elif "fee" in category:
        category = "fee"

    else:
        category = "general"

    return { "query_type": category}


# ============================================================
# ACADEMIC RAG NODE
# ============================================================

def academic_rag_node(state: State) -> dict:
    query = ( state["messages"][-1].content)

    docs = academic_retriever.invoke(query)

    context_parts = []

    for doc in docs:
        page = doc.metadata.get("page")

        if page is not None:
            source = (f"Academic Handbook - Page {page + 1}")

        else:
            source = ("Academic Handbook")

        context_parts.append(
                    f"""
        SOURCE:
        {source}
        
        CONTENT:
        {doc.page_content}
        """
                )

    context = "\n\n".join(context_parts)

    return {"retrieved_context": context}


# ============================================================
# FEE RAG NODE
# ============================================================

def fee_rag_node(state: State) -> dict:
    query = (state["messages"][-1].content)

    docs = fee_retriever.invoke( query)
    context_parts = []

    for doc in docs:
        page = doc.metadata.get("page")

        if page is not None:
            source = (f"Fee Structure - Page {page + 1}")

        else:
            source = ("Fee Structure")


        context_parts.append(
            f"""
SOURCE:
{source}

CONTENT:
{doc.page_content}
"""
        )


    context = "\n\n".join(
        context_parts
    )


    return {
        "retrieved_context": context
    }


# ============================================================
# GENERAL NODE
# ============================================================

def general_node(state: State) -> dict:

    return {
        "retrieved_context":
            "NO_RETRIEVAL_NEEDED"
    }


# ============================================================
# RESPONSE NODE
# ============================================================

def response_node(state: State) -> dict:

    query = (
        state["messages"][-1].content
    )


    programme = state.get(
        "programme",
        "Unknown"
    )


    query_type = state.get(
        "query_type",
        "general"
    )


    context = state.get(
        "retrieved_context",
        ""
    )


    # ========================================================
    # GENERAL QUERY
    # ========================================================

    if context == "NO_RETRIEVAL_NEEDED":

        prompt = f"""
You are a professional and friendly AI college assistant.

The student is enrolled in:

{programme}


Student message:

{query}


Respond naturally and concisely.

If the student greets you, greet them and briefly explain that
you can help with:

academic rules
attendance
examinations
fees
college policies

Do not invent college-specific information.
"""


    # ========================================================
    # RAG QUERY
    # ========================================================

    else:

        prompt = f"""
You are an AI college assistant.

You are currently helping a:

{programme} student


The query was classified as:

{query_type}


Use ONLY the official college document context below to answer
college-specific questions.


==================================================
OFFICIAL COLLEGE DOCUMENT CONTEXT
==================================================

{context}


==================================================
STUDENT QUESTION
==================================================

{query}


==================================================
INSTRUCTIONS
==================================================

1. Answer the question directly.

2. Prioritize information specifically related to:

   {programme}

3. If the document contains information for multiple programmes,
   clearly identify the information relevant to {programme}.

4. Do NOT invent:

   fees
   percentages
   dates
   policies
   attendance requirements
   academic requirements
   payment information

5. If the retrieved context does not contain enough information,
   say:

   "I couldn't find enough information about that in the
   available college documents."

6. Use bullet points when useful.

7. Keep the answer clear and reasonably concise.

8. When the retrieved context provides a page reference,
   mention the relevant source at the end of the answer.

Example:

Source: Academic Handbook - Page 12
"""

    response = llm.invoke(prompt)

    return {"messages": [( "ai",response.content.strip())]}

# ============================================================
# ROUTER
# ============================================================

def route_query(state: State):
    query_type = state.get("query_type","general")


    if query_type == "academic":
        return "academic_rag"


    elif query_type == "fee":
        return "fee_rag"

    return "general"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

@st.cache_resource(show_spinner=False)
def build_graph():
    graph = StateGraph(State)
    
    # Nodes
  
    graph.add_node("classifier",classifier_node)
    graph.add_node( "academic_rag",academic_rag_node)
    graph.add_node( "fee_rag",fee_rag_node)
    graph.add_node("general",general_node)
    graph.add_node("response",response_node)

    # --------------------------------------------------------
    # START -> CLASSIFIER
    # --------------------------------------------------------

    graph.add_edge(START, "classifier")

    # --------------------------------------------------------
    # CONDITIONAL ROUTING
    # --------------------------------------------------------

    graph.add_conditional_edges("classifier",route_query)

    # --------------------------------------------------------
    # RAG -> RESPONSE
    # --------------------------------------------------------

    graph.add_edge("academic_rag","response")
    graph.add_edge("fee_rag","response")
    graph.add_edge("general","response")

    # --------------------------------------------------------
    # RESPONSE -> END
    # --------------------------------------------------------

    graph.add_edge("response",END)

    return graph.compile()


# ============================================================
# CREATE LANGGRAPH APP
# ============================================================

app = build_graph()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = []


# ============================================================
# UI CSS
# ============================================================

st.markdown("""
<style>

/* ==========================================================
   GLOBAL
========================================================== */

html,
body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {

    background:
        #ffffff !important;

    color:
        #111111 !important;
}


.block-container {

    max-width:
        900px;

    padding-top:
        2rem;

    padding-bottom:
        8rem;
}


/* ==========================================================
   STREAMLIT HEADER
========================================================== */

[data-testid="stHeader"] {

    background:
        #ffffff !important;
}


/* ==========================================================
   MAIN HEADER
========================================================== */

.app-header {

    text-align:
        center;

    margin-top:
        20px;

    margin-bottom:
        38px;
}


.logo-box {

    width:
        54px;

    height:
        54px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    margin:
        0 auto 16px auto;

    background:
        #111111;

    color:
        #ffffff !important;

    border-radius:
        14px;

    font-size:
        15px;

    font-weight:
        800;

    letter-spacing:
        -0.4px;
}


.app-title {

    color:
        #111111 !important;

    font-size:
        40px;

    font-weight:
        750;

    letter-spacing:
        -1.5px;

    margin-bottom:
        7px;
}


.app-subtitle {

    color:
        #6b7280 !important;

    font-size:
        15px;
}


/* ==========================================================
   WELCOME CARD
========================================================== */

.welcome-card {

    background:
        #fafafa;

    border:
        1px solid #e5e7eb;

    border-radius:
        18px;

    padding:
        26px;

    margin-bottom:
        28px;
}


.welcome-title {

    color:
        #111111 !important;

    font-size:
        18px;

    font-weight:
        700;

    margin-bottom:
        8px;
}


.welcome-description {

    color:
        #6b7280 !important;

    font-size:
        14px;

    line-height:
        1.7;
}


/* ==========================================================
   SECTION LABEL
========================================================== */

.section-label {

    color:
        #9ca3af !important;

    font-size:
        10px;

    font-weight:
        700;

    letter-spacing:
        1px;

    text-transform:
        uppercase;

    margin-bottom:
        12px;
}


/* ==========================================================
   BUTTONS
========================================================== */

.stButton > button {

    background:
        #ffffff !important;

    color:
        #262626 !important;

    border:
        1px solid #e5e7eb !important;

    border-radius:
        12px !important;

    min-height:
        50px;

    font-weight:
        500;

    box-shadow:
        0 1px 3px
        rgba(0,0,0,0.04);

    transition:
        all 0.2s ease;
}


.stButton > button:hover {

    background:
        #f5f5f5 !important;

    color:
        #000000 !important;

    border-color:
        #a3a3a3 !important;

    transform:
        translateY(-1px);
}


/* ==========================================================
   CHAT MESSAGES
========================================================== */

[data-testid="stChatMessage"] {

    background:
        #ffffff !important;

    border:
        1px solid #e5e7eb;

    border-radius:
        20px;

    padding:
        14px;

    margin-bottom:
        12px;

    box-shadow:
        0 2px 7px
        rgba(0,0,0,0.025);
}


[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {

    color:
        #1f2937 !important;

    line-height:
        1.65;
}


/* ==========================================================
   MONOCHROME CHAT AVATARS
========================================================== */

[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {

    background:
        #ffffff !important;

    border:
        1.5px solid #111111 !important;

    border-radius:
        11px !important;

    box-shadow:
        none !important;
}


[data-testid="stChatMessageAvatarUser"] svg,
[data-testid="stChatMessageAvatarAssistant"] svg {

    color:
        #111111 !important;

    fill:
        #111111 !important;
}


[data-testid="stChatMessageAvatarUser"] *,
[data-testid="stChatMessageAvatarAssistant"] * {

    color:
        #111111 !important;
}


/* ==========================================================
   ROUTING BADGES
========================================================== */

.query-badge {

    display:
        inline-block;

    padding:
        4px 10px;

    background:
        #f5f5f5;

    color:
        #111111 !important;

    border:
        1px solid #d4d4d4;

    border-radius:
        20px;

    font-size:
        10px;

    font-weight:
        700;

    letter-spacing:
        0.6px;

    margin-bottom:
        8px;
}


.badge-academic,
.badge-fee,
.badge-general {

    background:
        #f5f5f5 !important;

    color:
        #111111 !important;

    border:
        1px solid #d4d4d4 !important;
}


/* ==========================================================
   SIDEBAR
========================================================== */

[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {

    background:
        #fafafa !important;
}


[data-testid="stSidebar"] {

    border-right:
        1px solid #e5e7eb;
}


[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {

    color:
        #374151 !important;
}


.sidebar-logo {

    color:
        #111111 !important;

    font-size:
        20px;

    font-weight:
        750;
}


.sidebar-subtitle {

    color:
        #9ca3af !important;

    font-size:
        12px;

    margin-top:
        3px;
}


/* ==========================================================
   SELECT BOX
========================================================== */

[data-baseweb="select"] > div {

    background:
        #ffffff !important;

    color:
        #111111 !important;

    border-color:
        #d1d5db !important;
}


[data-baseweb="select"] span {

    color:
        #111111 !important;
}


/* ==========================================================
   STATUS CARD
========================================================== */

.status-card {

    background:
        #ffffff;

    border:
        1px solid #e5e7eb;

    border-radius:
        12px;

    padding:
        14px;

    margin-top:
        15px;
}


.status-row {

    display:
        flex;

    align-items:
        center;
}


.status-dot {

    width:
        8px;

    height:
        8px;

    background:
        #111111;

    border-radius:
        50%;

    display:
        inline-block;

    margin-right:
        8px;
}


.status-text {

    color:
        #374151 !important;

    font-size:
        13px;

    font-weight:
        600;
}


.status-sub {

    color:
        #9ca3af !important;

    font-size:
        11px;

    margin-top:
        6px;
}


/* ==========================================================
   KNOWLEDGE SOURCES
========================================================== */

.source-card {

    background:
        #ffffff;

    border:
        1px solid #e5e7eb;

    border-radius:
        10px;

    padding:
        11px 12px;

    margin-bottom:
        8px;

    color:
        #4b5563 !important;

    font-size:
        12px;
}


/* ==========================================================
   BOTTOM AREA

   Keeps the entire bottom section white.
========================================================== */

[data-testid="stBottom"],
[data-testid="stBottom"] > div,
.stBottom,
.stBottom > div {

    background:
        #ffffff !important;
}


/* ==========================================================
   CHAT INPUT
========================================================== */

/* ==========================================================
   ROUNDED CHAT INPUT
========================================================== */

[data-testid="stChatInput"] {
    background: #ffffff !important;

    border: 1px solid #e5e7eb !important;

    border-radius: 28px !important;

    overflow: hidden !important;

    box-shadow:
        0 6px 24px rgba(0, 0, 0, 0.06) !important;
}


/* Make all inner containers follow the rounded shape */

[data-testid="stChatInput"] > div {
    background: #ffffff !important;
    border-radius: 28px !important;
}


/* Text area */

[data-testid="stChatInput"] textarea {
    background: #ffffff !important;

    color: #111111 !important;

    caret-color: #111111 !important;

    border-radius: 28px !important;
}


/* Placeholder */

[data-testid="stChatInput"] textarea::placeholder {
    color: #9ca3af !important;
    opacity: 1 !important;
}


/* Send button */

[data-testid="stChatInputSubmitButton"] {
    background: #111111 !important;

    color: #ffffff !important;

    border-radius: 50% !important;

    width: 48px !important;
    height: 48px !important;

    margin-right: 8px !important;
}


/* Arrow */

[data-testid="stChatInputSubmitButton"] svg {
    color: #ffffff !important;
    fill: #ffffff !important;
}

/* ==========================================================
   SEND BUTTON
========================================================== */

[data-testid="stChatInputSubmitButton"] {

    background:
        #111111 !important;

    color:
        #ffffff !important;

    border-radius:
        9px !important;
}


[data-testid="stChatInputSubmitButton"] svg {

    color:
        #ffffff !important;

    fill:
        #ffffff !important;
}


/* ==========================================================
   DIVIDERS
========================================================== */

hr {

    border-color:
        #e5e7eb !important;
}


/* ==========================================================
   HIDE STREAMLIT BRANDING
========================================================== */

#MainMenu {

    visibility:
        hidden;
}


footer {

    visibility:
        hidden;
}


</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    # --------------------------------------------------------
    # Brand
    # --------------------------------------------------------

    st.markdown(
        """<div class="sidebar-logo">College AI</div>
<div class="sidebar-subtitle">Intelligent campus assistant</div>""",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Programme
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-label">Your Programme</div>',
        unsafe_allow_html=True
    )


    student_programme = st.selectbox(
        "Programme",
        options=[
            "BCA",
            "BBA",
            "B.Com (H)"
        ],
        index=0,
        label_visibility="collapsed"
    )


    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    st.markdown(
        f"""<div class="status-card">
<div class="status-row">
<span class="status-dot"></span>
<span class="status-text">Assistant Online</span>
</div>
<div class="status-sub">Configured for {student_programme}</div>
</div>""",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Knowledge Sources
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-label">Knowledge Sources</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="source-card">Academic Handbook</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="source-card">Fee Structure</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="source-card">General Assistant</div>',
        unsafe_allow_html=True
    )


    st.markdown("---")


    # --------------------------------------------------------
    # Clear conversation
    # --------------------------------------------------------

    if st.button(
        "Clear conversation",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.session_state.lc_messages = []
        st.rerun()


    st.caption("LangGraph / RAG / Groq")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """<div class="app-header">
<div class="logo-box">CA</div>
<div class="app-title">College Assistant</div>
<div class="app-subtitle">Ask about academics, fees and college policies.</div>
</div>""",
    unsafe_allow_html=True
)


# ============================================================
# BUTTON STATES
# ============================================================

academic_button = False
fee_button = False
grading_button = False
exam_button = False


# ============================================================
# WELCOME SCREEN
# ============================================================

if len(st.session_state.messages) == 0:


    st.markdown(
        f"""<div class="welcome-card">
<div class="welcome-title">Hello, {student_programme} student</div>
<div class="welcome-description">
Ask questions about attendance, examinations, academic rules,
fees, course requirements or other college-related topics.
The assistant will automatically route your question to the
appropriate knowledge source.
</div>
</div>""",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Suggested Questions
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-label">Suggested Questions</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(
        2,
        gap="small"
    )


    with col1:


        academic_button = st.button(
            "Minimum attendance requirement",
            use_container_width=True
        )


        grading_button = st.button(
            "Explain the grading system",
            use_container_width=True
        )


    with col2:


        fee_button = st.button(
            "Programme fee structure",
            use_container_width=True
        )


        exam_button = st.button(
            "Examination rules",
            use_container_width=True
        )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for msg in st.session_state.messages:
    with st.chat_message(
        msg["role"]
    ):
        # ----------------------------------------------------
        # Routing badge
        # ----------------------------------------------------
        if (
            msg["role"] == "assistant"
            and msg.get("query_type")
        ):
            query_type = msg["query_type"]

            st.markdown(
                f'<span class="query-badge badge-{query_type}">'
                f'{query_type.upper()}'
                f'</span>',
                unsafe_allow_html=True
            )
        # ----------------------------------------------------
        # Message
        # ----------------------------------------------------
        st.markdown(msg["content"])


# ============================================================
# CHAT INPUT
# ============================================================

typed_query = st.chat_input(f"Ask a question as a {student_programme} student...")
# ============================================================
# DETERMINE QUERY
# ============================================================

user_query = None


if academic_button:
    user_query = ("What is the minimum attendance requirement?")


elif fee_button:
    user_query = (f"What is the fee structure for {student_programme}?")


elif grading_button:
    user_query = ("Explain the grading system.")


elif exam_button:
    user_query = ("What are the examination rules?")


elif typed_query:
    user_query = typed_query


# ============================================================
# PROCESS QUERY
# ============================================================

if user_query:
    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------
    st.session_state.messages.append(
        {
            "role":
                "user",

            "content":
                user_query
        }
    )


    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_query
        )


    # --------------------------------------------------------
    # Add message to LangGraph history
    # --------------------------------------------------------

    st.session_state.lc_messages.append(
        (
            "human",
            user_query
        )
    )


    # ========================================================
    # ASSISTANT RESPONSE
    # ========================================================

    with st.chat_message(
        "assistant"
    ):


        with st.spinner(
            "Searching college knowledge..."
        ):


            try:


                result = app.invoke(
                    {

                        "programme":
                            student_programme,

                        "messages":
                            st.session_state.lc_messages

                    }
                )


            except Exception as error:


                st.error(
                    "Something went wrong while processing your question."
                )


                st.exception(
                    error
                )


                st.stop()


        # ----------------------------------------------------
        # AI response
        # ----------------------------------------------------

        ai_response = (
            result["messages"][-1]
            .content
            .strip()
        )


        # ----------------------------------------------------
        # Query classification
        # ----------------------------------------------------

        query_type = result.get(
            "query_type",
            "general"
        )


        # ----------------------------------------------------
        # Routing badge
        # ----------------------------------------------------

        st.markdown(
            f'<span class="query-badge badge-{query_type}">'
            f'{query_type.upper()}'
            f'</span>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # Display response
        # ----------------------------------------------------

        st.markdown(ai_response)


    # ========================================================
    # UPDATE HISTORY
    # ========================================================

    st.session_state.lc_messages = (result["messages"])


    st.session_state.messages.append(
        {

            "role":
                "assistant",

            "content":
                ai_response,

            "query_type":
                query_type

        }
    )


    # --------------------------------------------------------
    # Refresh after suggested question
    # --------------------------------------------------------

    if (
        academic_button
        or fee_button
        or grading_button
        or exam_button
    ):

        st.rerun()
