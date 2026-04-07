#MULTI AGENT — Qualtrics-ready edition
#
# Key additions vs. original:
#   1. Flask sessions + userID capture from ?userID=XXX query param
#   2. Azure Blob Storage session logging (ported from single-agent)
#   3. /done route to redirect participants back to Qualtrics
#   4. Every /ask interaction is stamped with userID in the log
#   5. startup.py / gunicorn-compatible (no debug reloader)

from flask import Flask, render_template, request, jsonify, session, redirect
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os, json, re, time, uuid
from datetime import datetime, timezone

# ---------- PDF citation / search ----------
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery

# ---------- Azure Blob (session logging) ----------
from azure.storage.blob import BlobServiceClient, ContentSettings

# ============================================================
#  ENV + FLASK SETUP
# ============================================================
load_dotenv()

app = Flask(__name__)
# Use a strong random secret in production (set via env var)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")

# ------------------ Azure Chat Config ------------------
AZURE_DEPLOYMENT   = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
AZURE_API_VERSION  = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_ENDPOINT     = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY      = os.getenv("AZURE_OPENAI_API_KEY")

# ------------------ Azure Embedding + Search ------------------
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT_NAME")
SEARCH_ENDPOINT      = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_INDEX         = os.getenv("AZURE_SEARCH_INDEX_NAME")
SEARCH_API_KEY       = os.getenv("AZURE_SEARCH_API_KEY")

# ------------------ Azure Blob (analytics) ------------------
BLOB_CONN_STR       = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
USER_LOGS_CONTAINER = os.getenv("USER_LOGS_CONTAINER", "user-logs")

# ------------------ Qualtrics return URL ------------------
# Set this in your .env to the Qualtrics "next page" URL if you want
# students to be sent back after interacting.
QUALTRICS_RETURN_URL = os.getenv("QUALTRICS_RETURN_URL", "")

# ============================================================
#  AZURE CLIENT FACTORIES
# ============================================================
def llm_zero_temp():
    return AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT, api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT, api_key=AZURE_API_KEY, temperature=0,
    )

def llm_brief():
    return AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT, api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT, api_key=AZURE_API_KEY, temperature=0.1,
    )

def llm_deep():
    return AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT, api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT, api_key=AZURE_API_KEY, temperature=0.1,
    )

openai_client = None
if AZURE_API_KEY and AZURE_ENDPOINT and AZURE_API_VERSION:
    openai_client = AzureOpenAI(
        api_key=AZURE_API_KEY, api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
    )

search_client = None
if SEARCH_ENDPOINT and SEARCH_INDEX and SEARCH_API_KEY:
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT, index_name=SEARCH_INDEX,
        credential=AzureKeyCredential(SEARCH_API_KEY),
    )

# ============================================================
#  AZURE BLOB HELPERS  (session analytics)
# ============================================================
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _blob_service() -> BlobServiceClient:
    if not BLOB_CONN_STR:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING not set")
    return BlobServiceClient.from_connection_string(BLOB_CONN_STR)

def _ensure_container(svc: BlobServiceClient, name: str):
    c = svc.get_container_client(name)
    try:
        c.create_container()
    except Exception:
        pass
    return c

def _upload_json(container, blob_path: str, data: dict):
    payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    container.get_blob_client(blob_path).upload_blob(
        payload, overwrite=True,
        content_settings=ContentSettings(content_type="application/json"),
    )

def _checkpoint_session():
    """Write current session log to Blob Storage (best-effort)."""
    if not BLOB_CONN_STR:
        return
    try:
        user_id    = session.get("user_id", "anonymous")
        session_id = session.get("session_id", "unknown")
        svc        = _blob_service()
        container  = _ensure_container(svc, USER_LOGS_CONTAINER)
        payload = {
            "user_id":        user_id,
            "session_id":     session_id,
            "start_time_utc": session.get("start_time_utc"),
            "messages":       session.get("chat_log", []),
        }
        base = f"{user_id}/{session_id}/"
        _upload_json(container, base + "checkpoint.json", payload)
        _upload_json(container, f"{user_id}/latest.json", payload)
    except Exception as e:
        app.logger.warning(f"[BLOB] Checkpoint failed: {e}")

def _init_session(user_id: str):
    """Initialise a fresh analytics session for this participant."""
    session["user_id"]       = user_id
    session["logged_in"]     = True
    start                    = datetime.now(timezone.utc)
    session["start_time_utc"] = start.isoformat().replace("+00:00", "Z")
    session["session_id"]    = f"{start.strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4()}"
    session["chat_log"]      = []
    _checkpoint_session()   # write initial record immediately

def _append_log(role: str, text: str, grade: str = ""):
    entry = {
        "ts_utc":  _utc_now_iso(),
        "user_id": session.get("user_id", "anonymous"),
        "role":    role,
        "text":    text,
    }
    if grade:
        entry["grade"] = grade
    session.setdefault("chat_log", []).append(entry)

# ============================================================
#  GENERIC HELPERS
# ============================================================
def timestamp():
    return datetime.now(timezone.utc).isoformat()

def extract_json(text: str):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"[extract_json] {e}")
    return None

def is_greeting(text: str) -> bool:
    q = re.sub(r"[^\w\s]", "", text.lower()).strip()
    if not q:
        return False
    GREETINGS = {
        "hi","hello","hey","yo","sup",
        "good morning","good afternoon","good evening",
        "hi there","hey there",
    }
    return any(q == g or q.startswith(g + " ") for g in GREETINGS)

# ============================================================
#  GLOBAL IN-MEMORY STATE (per-process; use Redis for multi-worker)
# ============================================================
# Keyed by session_id so each participant gets their own state.
_SESSIONS: dict = {}

def _get_state() -> dict:
    sid = session.get("session_id", "__anon__")
    if sid not in _SESSIONS:
        _SESSIONS[sid] = {
            "history": [],
            "events":  [],
            "quiz_state": None,
            "last_quiz_strong_count": 0,
        }
    return _SESSIONS[sid]


def log_event(state: dict, kind: str, detail: dict):
    state["events"].append({"ts": timestamp(), "type": kind, "detail": detail})
    print(f"[EVENT] {kind}: {detail}")

# ============================================================
#  PDF / REFERENCE HELPERS
# ============================================================
def generate_embedding(text: str):
    if not openai_client or not EMBEDDING_DEPLOYMENT:
        raise RuntimeError("Embedding client/deployment not configured.")
    resp = openai_client.embeddings.create(input=text, model=EMBEDDING_DEPLOYMENT)
    return resp.data[0].embedding

def search_matching_documents(embedding, threshold: float = 0.5):
    if not search_client:
        return []
    vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=5, fields="embedding")
    results = search_client.search(
        search_text=None, vector_queries=[vector_query],
        select=["document_name", "page_number", "sas_url"], top=5,
    )
    matches = []
    for r in results:
        score = r.get("@search.score", 0)
        if score >= threshold:
            matches.append({
                "document_name": r["document_name"],
                "page_number":   r["page_number"],
                "sas_url":       r.get("sas_url"),
                "score":         float(score),
            })
    print(f"[SEARCH] {len(matches)} matches (>= {threshold}).")
    return matches

def generate_refined_response_with_refs(user_prompt: str, matching_documents: list) -> str:
    ref_lines = []
    for doc in matching_documents:
        url = doc.get("sas_url")
        link = f"[{doc['document_name']} p.{doc['page_number']}]({url})" if url \
               else f"{doc['document_name']} p.{doc['page_number']}"
        ref_lines.append(f"- {link}")

    system_prompt = """
You are an AI tutor.
STRICT RULES:
1. Cite docs inline as (Document Name, p.X).
2. End with a "References" section using ONLY [text](url) markdown links.
3. Never output a raw URL without [text](url) wrapping.
4. Keep explanations clear and student-friendly.
"""
    user_message = (
        f"User question:\n{user_prompt}\n\n"
        f"Relevant references:\n" + "\n".join(ref_lines)
    )
    if not openai_client:
        raise RuntimeError("openai_client not configured.")
    resp = openai_client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user",   "content": user_message.strip()},
        ],
        temperature=0.1,
        max_tokens=2000,
    )
    return resp.choices[0].message.content

# ============================================================
#  TOOLS
# ============================================================
@tool
def deep_answer(question: str) -> str:
    """Detailed, comprehensive answer with optional PDF citations."""
    print(f"[deep_answer] {question!r}")
    if is_greeting(question):
        return "Hi! How can I help you today? 😊"

    def _fallback():
        return llm_deep().invoke(f"Give a deep, detailed explanation for: {question}").content

    if not (openai_client and EMBEDDING_DEPLOYMENT and search_client):
        return _fallback()
    try:
        embedding = generate_embedding(question)
    except Exception as e:
        print(f"[deep_answer] Embedding error: {e}")
        return _fallback()
    try:
        matches = search_matching_documents(embedding, threshold=0.5)
    except Exception as e:
        print(f"[deep_answer] Search error: {e}")
        return _fallback()
    if not matches:
        return _fallback()
    try:
        return generate_refined_response_with_refs(question, matches)
    except Exception as e:
        print(f"[deep_answer] Refine error: {e}")
        return _fallback()

@tool
def brief_answer(question: str) -> str:
    """Short, concise answer (2–3 sentences)."""
    print(f"[brief_answer] {question!r}")
    if is_greeting(question):
        return "Hi! I'm your AI tutor. Ask me anything about the course. 👋"
    return llm_brief().invoke(f"Answer briefly (2-3 sentences): {question}").content

# ============================================================
#  GRADER / QUIZ / VERIFIER
# ============================================================
def grade_question_with_llm(question: str, history: list):
    if is_greeting(question):
        return {"grade": "Normal", "reason": "Greeting."}

    last_q, last_a = "", ""
    has_history = False
    for item in reversed(history):
        if "question" in item and "answer" in item:
            last_q, last_a = item["question"], item["answer"]
            has_history = True
            break

    history_context = (
        f"Last Q: {last_q}\nLast A: {last_a}"
        if has_history
        else "NO PRIOR CHAT HISTORY EXISTS."
    )

    sys = """
You are a grader. Classify the user's question into exactly ONE of the following categories:

- "Normal": A foundational, basic, or definitional question. Assign this for broad overviews, basic facts, or casual chit-chat. CRITICAL: If a question introduces a new topic but only asks for a basic definition or overview, it must be "Normal", NOT "StrongIntro".
- "Strong": A conceptual, explanatory, or advanced question showing deep understanding (e.g., "How", "Why", integration mechanisms). Assign this when the user explores or expands the CURRENT overarching topic/use case. This INCLUDES cross-phase applications (e.g., shifting from construction to facility management) and integrating new tools into established workflows. DO NOT ASSIGN STRONG IF THERE IS NO PREVIOUS CHAT HISTORY.
- "StrongIntro": A conceptual, explanatory, or advanced question that introduces a completely NEW domain-level topic or use case. HARD CONSTRAINTS: 1) The overarching USE CASE completely changes. 2) The question MUST be conceptual/advanced. If the question is advanced but there is NO chat history, assign StrongIntro. DO NOT use StrongIntro for transitioning between project lifecycle phases (like construction to operations). MUST BE COMPLETELEY UNRELATED TOPIC. IF THERE IS NO CHAT HISTORY
- "CounterCue": A question that challenges a premise, tests boundaries/limitations, or presents a misconception based on prior context. DO NOT assign this simply because a question is a "follow-up" to previous turns. Use this ONLY for boundary-testing, limits, corrective scenarios, OR A VERY LIGHT FOLLOW UP. 

Return STRICT JSON:
{
  "grade": "Strong" | "Normal" | "CounterCue" | "StrongIntro",
  "reason": "short reason"
}
"""

    resp = llm_zero_temp().invoke(
        sys + f"\n{history_context}\nUser Question: {question}"
    ).content.strip()

    parsed = extract_json(resp)
    if not parsed or "grade" not in parsed:
        return {"grade": "Normal", "reason": "Parse fallback."}

    grade = parsed.get("grade", "Normal")

    if grade == "Strong" and not has_history:
        print(f"[GRADER] Overriding Strong → StrongIntro (no prior history)")
        grade = "StrongIntro"

    if is_greeting(question) and grade != "Normal":
        grade = "Normal"

    print(f"[GRADER] {grade!r} — {parsed.get('reason', '')}")
    return {"grade": grade, "reason": parsed.get("reason", "")}

def should_launch_quiz(history, state: dict) -> bool:
    if state.get("quiz_state"):
        return False
    strong = sum(1 for h in history if h.get("grade") in {"Strong", "StrongIntro"})
    return strong >= state.get("last_quiz_strong_count", 0) + 3

def generate_quiz_from_strong(history):
    items = [h for h in history if h.get("grade") in {"Strong", "StrongIntro"}][-3:]
    if not items:
        return None
    context = "\n".join(f"- {it['question']}" for it in items)
    sys = f"""
You are a tutor. Create ONE short quiz question from:
{context}
Return STRICT JSON:
{{"question":"...","expected_answer":"...","explanation":"2-3 sentences"}}
"""
    resp = llm_zero_temp().invoke(sys).content.strip()
    data = extract_json(resp) or {}
    return {
        "question":    data.get("question", "Quiz generation failed."),
        "answer":      data.get("expected_answer", ""),
        "explanation": data.get("explanation", ""),
    }

def evaluate_quiz_answer(user_answer, expected_answer):
    sys = f"""
Evaluate a student's short answer. Expected: {expected_answer}
Return STRICT JSON: {{"verdict":"Satisfactory"|"Unsatisfactory","reason":"short reason"}}
"""
    resp = llm_zero_temp().invoke(sys + "\nStudent Answer: " + user_answer).content.strip()
    return extract_json(resp) or {"verdict": "Unsatisfactory", "reason": "Parse fallback."}

def verify_conceptual_relevance(question, answer, quiz=None):
    quiz_text = ""
    if quiz:
        quiz_text = f"QuizQ: {quiz.get('question','')}\nExpected: {quiz.get('answer','')}"
    sys = """
Determine if the Answer conceptually aligns with the Question.
Return STRICT JSON: {"relevant":true|false,"notes":"short reason"}
"""
    user = f"Question: {question}\nAnswer: {answer}\n{quiz_text}"
    resp = llm_zero_temp().invoke(sys + "\n" + user).content.strip()
    return extract_json(resp) or {"relevant": True, "notes": "Default true."}

# ============================================================
#  QUALTRICS SESSION CAPTURE — helper
# ============================================================
def _capture_user_id_from_params() -> str | None:
    """
    Read ?userID=XXX (or ?user_id=XXX) from the request.
    Returns the id string or None.
    """
    uid = (
        request.args.get("userID")
        or request.args.get("user_id")
        or request.args.get("userid")
        or ""
    ).strip()
    return uid if uid else None

# ============================================================
#  FLASK ROUTES
# ============================================================

@app.route("/")
def home():
    """
    Entry point.  Qualtrics links look like:
        https://your-app.azurewebsites.net/?userID=${e://Field/ResponseID}

    If a userID is present we initialise a session for them immediately
    and redirect to the chat UI so the URL stays clean.
    """
    uid = _capture_user_id_from_params()
    if uid:
        _init_session(uid)
        print(f"[SESSION] New participant: {uid}")
        return redirect("/chat_ui")

    # No userID — show a simple login/entry form so the app is still
    # usable outside Qualtrics (e.g. for direct testing).
    return render_template("chat.html")


@app.route("/chat_ui")
def chat_ui():
    """The main chat interface (served after userID is captured)."""
    if not session.get("logged_in"):
        return redirect("/")
    return render_template("chat.html")


@app.route("/done")
def done():
    """
    Students click 'Done' (or are redirected here) when finished.
    We write a final session record and redirect them back to Qualtrics.
    """
    _finalize_session()
    if QUALTRICS_RETURN_URL:
        return redirect(QUALTRICS_RETURN_URL)
    return "<h2>Thank you! You may now close this window and return to the survey.</h2>", 200


@app.route("/reset", methods=["POST"])
def reset():
    state = _get_state()
    state["history"].clear()
    state["events"].clear()
    state["quiz_state"] = None
    state["last_quiz_strong_count"] = 0
    print("[RESET] Memory cleared.")
    return jsonify({"message": "Memory cleared ✅"})


@app.route("/ask", methods=["POST"])
def ask():
    # ── Require a session (participant must arrive via Qualtrics link) ──
    if not session.get("logged_in"):
        # Grace: also accept ?userID= on the /ask call itself (unusual but safe)
        uid = _capture_user_id_from_params()
        if uid:
            _init_session(uid)
        else:
            return jsonify({"error": "No participant session. Please access via the Qualtrics link."}), 401

    user_text = request.form.get("question", "").strip()
    print(f"\n[ASK] user_id={session.get('user_id')!r}  q={user_text!r}")

    if not user_text:
        return jsonify({"error": "Please enter a question"}), 400

    state      = _get_state()
    history    = state["history"]
    quiz_state = state.get("quiz_state")

    # ── Log the user message ──
    _append_log("user", user_text)

    # ====================================================
    # 1) QUIZ FLOW
    # ====================================================
    if quiz_state:
        print("[FLOW] Quiz mode active.")
        quiz_state["attempts"] += 1
        verdict = evaluate_quiz_answer(user_text, quiz_state["answer"])
        log_event(state, "quiz_evaluate", {"attempt": quiz_state["attempts"], **verdict})

        if verdict["verdict"] == "Satisfactory":
            msg = f"✅ Correct! {verdict['reason']}"
            history.append({
                "type": "quiz_attempt", "question": quiz_state["question"],
                "user_answer": user_text, "verdict": "Satisfactory", "ts": timestamp(),
            })
            state["quiz_state"] = None
            _append_log("assistant", msg)
            _checkpoint_session()
            return jsonify({"response": msg})

        if quiz_state["attempts"] < quiz_state["max_attempts"]:
            state["quiz_state"] = quiz_state
            _append_log("assistant", f"❌ {verdict['reason']} Try again.")
            _checkpoint_session()
            return jsonify({"response": f"❌ Not quite: {verdict['reason']} Try again."})

        msg = (
            f"❌ Incorrect again.\n\n✅ **Correct:** {quiz_state['answer']}\n"
            f"💡 **Explanation:** {quiz_state['explanation']}"
        )
        history.append({
            "type": "quiz_reveal", "question": quiz_state["question"],
            "user_answer": user_text, "correct_answer": quiz_state["answer"],
            "explanation": quiz_state["explanation"], "ts": timestamp(),
        })
        state["quiz_state"] = None
        _append_log("assistant", msg)
        _checkpoint_session()
        return jsonify({"response": msg})

    # ====================================================
    # 2) NORMAL QA FLOW
    # ====================================================
    grade_info   = grade_question_with_llm(user_text, history)
    grade        = grade_info["grade"]
    grade_reason = grade_info["reason"]
    log_event(state, "question_graded", {"grade": grade, "reason": grade_reason})

    tool_obj = deep_answer if grade in ("Strong", "CounterCue") else brief_answer
    print(f"[ROUTER] grade={grade} → tool={tool_obj.name}")

    try:
        start  = time.time()
        answer = tool_obj.func(user_text)
        duration = round(time.time() - start, 2)
        log_event(state, "tool_used", {"name": tool_obj.name, "duration_s": duration})
    except Exception as e:
        log_event(state, "tool_error", {"name": tool_obj.name, "error": str(e)})
        return jsonify({"error": str(e)}), 500

    verify = verify_conceptual_relevance(user_text, answer)
    log_event(state, "verify", verify)

    history.append({
        "type": "qa", "question": user_text, "answer": answer,
        "grade": grade, "grade_reason": grade_reason,
        "verification": verify, "tools_used": [tool_obj.name], "ts": timestamp(),
    })

    # ── Log assistant reply (with grade for analysis) ──
    _append_log("assistant", answer, grade=grade)
    _checkpoint_session()

    # ====================================================
    # 3) QUIZ TRIGGER
    # ====================================================
    if should_launch_quiz(history, state):
        strong = sum(1 for h in history if h.get("grade") in {"Strong", "StrongIntro"})
        quiz = generate_quiz_from_strong(history)
        if quiz:
            state["quiz_state"] = {
                "question":     quiz["question"],
                "answer":       quiz["answer"],
                "explanation":  quiz["explanation"],
                "attempts":     0,
                "max_attempts": 2,
            }
            state["last_quiz_strong_count"] = strong
            log_event(state, "quiz_launched", {"question": quiz["question"]})
            _checkpoint_session()
            combined = (
                f"{answer}\n\n---\n🧪 **Quick Check:** {quiz['question']}\n"
                "Reply with your answer."
            )
            return jsonify({"response": combined})

    return jsonify({"response": answer})


# ============================================================
#  SESSION FINALISATION
# ============================================================
def _finalize_session():
    """Write a final session.json to Blob with duration + full transcript."""
    if not BLOB_CONN_STR:
        return
    try:
        user_id     = session.get("user_id", "anonymous")
        session_id  = session.get("session_id", str(uuid.uuid4()))
        start_iso   = session.get("start_time_utc")
        end_iso     = _utc_now_iso()

        def _parse(ts):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))

        try:
            duration = int((_parse(end_iso) - _parse(start_iso)).total_seconds()) if start_iso else None
        except Exception:
            duration = None

        messages = session.get("chat_log", [])
        final = {
            "user_id":          user_id,
            "session_id":       session_id,
            "start_time_utc":   start_iso,
            "end_time_utc":     end_iso,
            "duration_seconds": duration,
            "message_count":    len(messages),
            "messages":         messages,
        }
        svc       = _blob_service()
        container = _ensure_container(svc, USER_LOGS_CONTAINER)
        base      = f"{user_id}/{session_id}/"
        _upload_json(container, base + "session.json", final)
        _upload_json(container, f"{user_id}/latest.json", final)
        print(f"[BLOB] Final session saved for {user_id}.")
    except Exception as e:
        app.logger.warning(f"[BLOB] Finalize failed: {e}")

# ============================================================
#  ENTRY POINT  (local dev only — use gunicorn in production)
# ============================================================
if __name__ == "__main__":
    print("[START] AI Tutor (Qualtrics edition) on http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False, port=5000)