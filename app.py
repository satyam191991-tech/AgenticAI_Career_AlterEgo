from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
import sqlite3
import math
from pathlib import Path
from pypdf import PdfReader
import gradio as gr
from typing import Optional, Tuple


load_dotenv(override=True)

# -----------------------------
# Utilities: SQLite Q&A store
# -----------------------------

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "faq.db"

def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_db_connection():
    ensure_data_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS qa (
                id INTEGER PRIMARY KEY,
                question TEXT UNIQUE,
                answer TEXT,
                embedding TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS unknown_questions (
                id INTEGER PRIMARY KEY,
                question TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

def compute_embedding_text_list(openai_client: OpenAI, text: str):
    # Returns list[float] or None if embedding call fails
    try:
        resp = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return resp.data[0].embedding
    except Exception:
        return None

def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        ai = a[i]
        bi = b[i]
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    denom = math.sqrt(na) * math.sqrt(nb)
    return (dot / denom) if denom else 0.0

def upsert_qa_entry(question: str, answer: str):
    """Insert or update a Q&A entry, persisting an embedding if possible."""
    ensure_data_dir()
    client = OpenAI()
    embedding = compute_embedding_text_list(client, question)
    embedding_json = json.dumps(embedding) if embedding else None
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO qa (question, answer, embedding)
            VALUES (?, ?, ?)
            ON CONFLICT(question) DO UPDATE SET
                answer=excluded.answer,
                embedding=COALESCE(excluded.embedding, qa.embedding)
            """,
            (question, answer, embedding_json),
        )
        conn.commit()
        return {"status": "ok", "updated": True}
    finally:
        conn.close()

def query_qa(question: str, top_k: int = 3, min_score: float = 0.7):
    """Return best matching answers from the Q&A store using embeddings when available, else fallback to LIKE."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, question, answer, embedding FROM qa")
        rows = cur.fetchall()
        results = []

        client = OpenAI()
        query_emb = compute_embedding_text_list(client, question)

        if query_emb:
            for row in rows:
                emb = None
                if row[3]:
                    try:
                        emb = json.loads(row[3])
                    except Exception:
                        emb = None
                score = cosine_similarity(query_emb, emb) if emb else 0.0
                results.append({
                    "id": row[0],
                    "question": row[1],
                    "answer": row[2],
                    "score": score,
                })
            results.sort(key=lambda r: r["score"], reverse=True)
            filtered = [r for r in results if r["score"] >= min_score]
            top = (filtered or results)[: max(1, min(top_k, len(results)))]
            return {"matches": top}
        else:
            # Fallback simple LIKE search when embeddings unavailable
            cur.execute(
                "SELECT id, question, answer FROM qa WHERE question LIKE ? ORDER BY id DESC LIMIT ?",
                (f"%{question}%", top_k),
            )
            like_rows = cur.fetchall()
            return {
                "matches": [
                    {"id": r[0], "question": r[1], "answer": r[2], "score": None}
                    for r in like_rows
                ]
            }
    finally:
        conn.close()

def list_recent_qas(limit: int = 10):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, question, answer FROM qa ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        return {"items": [{"id": r[0], "question": r[1], "answer": r[2]} for r in rows]}
    finally:
        conn.close()

# -----------------------------
# Pushover + existing tools
# -----------------------------
def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    # Also persist locally
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO unknown_questions (question) VALUES (?)", (question,))
        conn.commit()
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

# -----------------------------
# New tools: Q&A DB
# -----------------------------

upsert_qa_entry_json = {
    "name": "upsert_qa_entry",
    "description": "Add or update a common Q&A entry the assistant can reuse later.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The canonical question."},
            "answer": {"type": "string", "description": "The best answer for this question."}
        },
        "required": ["question", "answer"],
        "additionalProperties": False
    }
}

query_qa_json = {
    "name": "query_qa",
    "description": "Retrieve best matching answers from the stored Q&A for a user question.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The user's question to match against the Q&A store."},
            "top_k": {"type": "integer", "description": "Number of matches to return (default 3)."},
            "min_score": {"type": "number", "description": "Minimum similarity score 0-1 (default 0.7)."}
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

list_recent_qas_json = {
    "name": "list_recent_qas",
    "description": "List recent Q&A entries for reference or to decide whether to add a new one.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "How many to return (default 10)."}
        },
        "required": [],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
    {"type": "function", "function": upsert_qa_entry_json},
    {"type": "function", "function": query_qa_json},
    {"type": "function", "function": list_recent_qas_json}
]


class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.name = "Satyam Pandit"
        # Initialize local SQLite DB
        init_db()
        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()


    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. \
Always consider the following agentic steps before replying: \
1) Search the Q&A store by calling query_qa with the user's question. \
2) If a good match is found, ground your answer in that content; otherwise, proceed with your best knowledge and consider saving a new canonical Q&A via upsert_qa_entry when appropriate. \
3) Be concise, cite sources when relevant, and avoid speculation."

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                # Got a candidate assistant answer; optionally evaluate and refine
                content = response.choices[0].message.content or ""
                evaluated_content = self._maybe_reflect_and_revise(user_message=message, candidate_answer=content)
                done = True
        return evaluated_content

    def _maybe_reflect_and_revise(self, user_message: str, candidate_answer: str) -> str:
        if not os.getenv("ENABLE_EVALUATOR", "true").lower() in ("1", "true", "yes"):  # feature flag
            print("[Evaluator] Disabled via ENABLE_EVALUATOR flag.", flush=True)
            return candidate_answer
        print("[Evaluator] Starting evaluation of candidate answer...", flush=True)
        score, feedback = self._evaluate_answer_free(user_message, candidate_answer)
        if score is None:
            print("[Evaluator] Skipped (no local evaluator available or error).", flush=True)
            return candidate_answer
        # If poor score, request a single revision using the critique
        threshold = float(os.getenv("EVALUATOR_MIN_SCORE", "0.6"))
        print(f"[Evaluator] Score: {score:.2f} | Threshold: {threshold:.2f}", flush=True)
        if score >= threshold:
            print("[Evaluator] Score meets threshold; using original answer.", flush=True)
            return candidate_answer
        print("[Evaluator] Score below threshold; attempting single-pass revision...", flush=True)
        revision_system = (
            "You received the following critique about your previous answer. "
            "Revise the answer to better address the user's question, grounding in stored Q&A if available, "
            "avoiding speculation, and keeping it concise."
        )
        revision_messages = [
            {"role": "system", "content": revision_system},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": candidate_answer},
            {"role": "user", "content": f"Critique:\n{feedback}\n\nPlease provide an improved final answer only."},
        ]
        try:
            revised = self.openai.chat.completions.create(
                model="gpt-4o-mini", messages=revision_messages, tools=tools
            )
            new_answer = revised.choices[0].message.content or candidate_answer
            print("[Evaluator] Revision produced an updated answer.", flush=True)
            return new_answer
        except Exception:
            print("[Evaluator] Revision failed; falling back to original answer.", flush=True)
            return candidate_answer

    def _evaluate_answer_free(self, question: str, answer: str) -> Tuple[Optional[float], Optional[str]]:
        """Evaluate using OpenAI with an evaluator model (default gpt-4o-mini); returns (score 0-1, feedback)."""
        evaluator_model = os.getenv("EVALUATOR_MODEL", "gpt-4o-mini")
        prompt = (
            "You are an impartial evaluator. "
            "Score the assistant's answer for correctness, helpfulness, and grounding on a 0.0-1.0 scale. "
            "Then provide a short critique with concrete suggestions. "
            "Respond in JSON with keys: score (float 0-1), feedback (string).\n\n"
            f"Question: {question}\n\nAnswer: {answer}\n\nOutput:"
        )
        try:
            print(f"[Evaluator] Using OpenAI evaluator model: {evaluator_model}", flush=True)
            res = self.openai.chat.completions.create(
                model=evaluator_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = res.choices[0].message.content or ""
            data = None
            try:
                data = json.loads(content)
            except Exception:
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1:
                    data = json.loads(content[start:end+1])
            if isinstance(data, dict) and "score" in data:
                score = float(data.get("score", 0))
                feedback = str(data.get("feedback", ""))
                score = max(0.0, min(1.0, score))
                print("[Evaluator] Received OpenAI evaluation.", flush=True)
                return score, feedback
        except Exception as e:
            print(f"[Evaluator] OpenAI evaluation error: {e}", flush=True)
        return None, None
    

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
    