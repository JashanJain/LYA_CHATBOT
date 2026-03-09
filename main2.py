import os
import re
import speech_recognition as sr
import pyttsx3
import threading

from dotenv import load_dotenv

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_core.prompts import ChatPromptTemplate


# ---------------------------
# Load Environment
# ---------------------------

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in .env file.")


# ---------------------------
# Initialize LLM
# ---------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_output_tokens=700,
    google_api_key=GOOGLE_API_KEY,
)


# ---------------------------
# Conversation Memory (fixed)
# ---------------------------

# Simple in-memory list — works reliably across all langchain versions
conversation_history = []  # list of {"role": "user"/"assistant", "content": "..."}

MAX_HISTORY = 10  # keep last 10 turns to avoid token overflow


def get_history_text():
    """Format conversation history as plain text for prompt injection."""
    if not conversation_history:
        return "No previous conversation."
    lines = []
    for msg in conversation_history[-MAX_HISTORY:]:
        role = "User" if msg["role"] == "user" else "LYA"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def add_to_history(user_msg, assistant_msg):
    conversation_history.append({"role": "user", "content": user_msg})
    conversation_history.append({"role": "assistant", "content": assistant_msg})


# ---------------------------
# Speech Systems (CLI only)
# ---------------------------

recognizer = sr.Recognizer()
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True

# Global flag to stop speech
_stop_speaking = threading.Event()
_tts_thread = None


def speak_cli(text):
    """Speak in a background thread so user can stop it."""
    global _tts_thread
    _stop_speaking.clear()

    def _speak():
        engine = pyttsx3.init()
        engine.setProperty("rate", 180)

        # Split into sentences so we can check stop flag between them
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            if _stop_speaking.is_set():
                break
            engine.say(sentence)
            engine.runAndWait()

        engine.stop()

    _tts_thread = threading.Thread(target=_speak, daemon=True)
    _tts_thread.start()


def stop_speaking():
    """Signal the TTS thread to stop."""
    _stop_speaking.set()
    if _tts_thread:
        _tts_thread.join(timeout=2)


def listen_cli():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening...")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

    try:
        text = recognizer.recognize_google(audio)
        print("You:", text)
        return text
    except:
        return ""


# ---------------------------
# Query Normalization
# ---------------------------

def normalize_query(text):
    replacements = {
        "infra fact": "infra pack",
        "infra park": "infra pack",
        "infra back": "infra pack",
    }
    text = text.lower()
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    return text


# ---------------------------
# Load CRM Document
# ---------------------------

if not os.path.exists("knowledge.docx"):
    raise FileNotFoundError("knowledge.docx missing")

loader = Docx2txtLoader("knowledge.docx")
documents = loader.load()


# ---------------------------
# Split Document
# ---------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150
)

docs = splitter.split_documents(documents)


# ---------------------------
# Embeddings
# ---------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)


# ---------------------------
# Vector DB
# ---------------------------

FAISS_PATH = "faiss_index"

if os.path.exists(FAISS_PATH):
    vector_db = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    vector_db = FAISS.from_documents(docs, embeddings)
    vector_db.save_local(FAISS_PATH)

retriever = vector_db.as_retriever(search_kwargs={"k": 6})


# ---------------------------
# Hybrid RAG Prompt (with memory)
# ---------------------------

rag_prompt = ChatPromptTemplate.from_template(
"""
You are LYA, AI assistant for Laayn CRM.

Previous conversation:
{history}

Use the context below to answer CRM-related questions.
If the context has the answer, use it.
If not, use your general knowledge but stay helpful and on-topic.

Context:
{context}

Question:
{input}

Answer:
"""
)


# ---------------------------
# General Prompt (with memory)
# ---------------------------

general_prompt = ChatPromptTemplate.from_template(
"""
You are LYA, a helpful AI assistant for Laayn CRM.
You can answer general questions as well as CRM-related questions.

Previous conversation:
{history}

User: {input}
LYA:
"""
)


# ---------------------------
# RAG Pipeline
# ---------------------------

doc_chain = create_stuff_documents_chain(llm, rag_prompt)

rag_chain = create_retrieval_chain(retriever, doc_chain)


# ---------------------------
# Intent Classification
# ---------------------------

def classify_intent(question):
    prompt = f"""
Classify the user question as CRM or GENERAL.

CRM = anything about Laayn CRM features, packages, pricing, modules, leads, clients, pipelines, reports, integrations.
GENERAL = everything else (greetings, math, coding, general knowledge).

Return ONLY one word: CRM or GENERAL

Question: {question}
"""
    result = llm.invoke(prompt).content.strip().upper()
    return "CRM" if "CRM" in result else "GENERAL"


# ---------------------------
# Chat Function (fixed memory + hybrid)
# ---------------------------

def chat(question):

    if not question.strip():
        return "Please ask something."

    question = normalize_query(question)
    history = get_history_text()
    intent = classify_intent(question)

    try:

        if intent == "CRM":
            # Try RAG first
            response = rag_chain.invoke({
                "input": question,
                "history": history
            })
            answer = response.get("answer", "").strip()

            # If RAG returns empty or unhelpful, fall through to general
            if answer and len(answer) > 10 and "don't know" not in answer.lower():
                add_to_history(question, answer)
                return answer

        # General (or RAG fallback)
        chain = general_prompt | llm
        result = chain.invoke({
            "input": question,
            "history": history
        })
        answer = result.content.strip()
        add_to_history(question, answer)
        return answer

    except Exception as e:
        return f"Error occurred: {str(e)}"


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":

    print("\nLYA Assistant Ready")
    print("Commands: 'exit' to quit, 'stop' to stop speaking\n")

    while True:

        mode = input("Input (text/voice): ").strip().lower()

        if mode == "voice":
            q = listen_cli()
        else:
            q = input("You: ").strip()

        if q.lower() == "exit":
            break

        if q.lower() == "stop":
            stop_speaking()
            print("Speech stopped.")
            continue

        ans = chat(q)

        out = input("Output (text/speech/both): ").strip().lower()

        if out == "speech":
            speak_cli(ans)
            print("(Speaking... type 'stop' and press Enter to interrupt)")
        elif out == "both":
            print("LYA:", ans)
            speak_cli(ans)
            print("(Speaking... type 'stop' and press Enter to interrupt)")
        else:
            print("LYA:", ans)