from flask import Flask, render_template, request, jsonify
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os, json, re, time
from datetime import datetime, timezone

# ------------------ Load Environment ------------------
load_dotenv()

app = Flask(__name__)
app.secret_key = "supersecretkey"  # rotate in prod

# ------------------ Azure Config ------------------
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# ------------------ Utilities ------------------
def timestamp():
    """UTC timestamp with timezone (avoids deprecation warning)."""
    return datetime.now(timezone.utc).isoformat()

def llm_zero_temp():
    return AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        temperature=0,
    )

def llm_brief():
    return AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        temperature=0.3,
    )

def llm_deep():
    return AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        temperature=0.7,
    )

def extract_json(text: str):
    """Extract first JSON object from LLM response."""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return None

# ------------------ GLOBAL STATE ------------------
GLOBAL_STATE = {
    "history": [],
    "events": [],
    "quiz_state": None,
}

def persist_log():
    """Save GLOBAL_STATE to JSON file."""
    payload = {
        "updated_at": timestamp(),
        "history": GLOBAL_STATE["history"],
        "events": GLOBAL_STATE["events"],
        "quiz": GLOBAL_STATE["quiz_state"],
    }
    with open("chat_log.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def log_event(kind: str, detail: dict):
    """Log event into global events and persist."""
    entry = {"ts": timestamp(), "type": kind, "detail": detail}
    GLOBAL_STATE["events"].append(entry)
    persist_log()

# Load previous memory if exists
if os.path.exists("chat_log.json"):
    with open("chat_log.json", "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            GLOBAL_STATE["history"] = data.get("history", [])
            GLOBAL_STATE["events"] = data.get("events", [])
            GLOBAL_STATE["quiz_state"] = data.get("quiz", None)
        except Exception:
            pass

# ------------------ Tools ------------------
@tool
def deep_answer(question: str) -> str:
    """Detailed, comprehensive answer."""
    prompt = f"Give a deep, detailed explanation for: {question}"
    resp = llm_deep().invoke(prompt)
    return resp.content

@tool
def brief_answer(question: str) -> str:
    """Short, concise answer."""
    prompt = f"Answer briefly (2-3 sentences): {question}"
    resp = llm_brief().invoke(prompt)
    return resp.content

# ------------------ LLM Controllers ------------------
def grade_question_with_llm(question: str, history: list):
    """Grade into Strong / Normal / CounterCue."""
    sys = """
You are a grader. Classify the user's question into exactly one of:
- "Strong": conceptual/advanced question showing deep understanding
- "Normal": foundational or basic question
- "CounterCue": follow-up question based on prior context

Return STRICT JSON:
{
  "grade": "Strong" | "Normal" | "CounterCue",
  "reason": "short reason"
}
"""
    last_q = history[-1]["question"] if history else ""
    last_a = history[-1]["answer"] if history else ""
    user = f"Last Q: {last_q}\nLast A: {last_a}\nUser Question: {question}"
    resp = llm_zero_temp().invoke(sys + "\n" + user).content.strip()
    return extract_json(resp) or {"grade": "Normal", "reason": "Parse fallback."}

def should_launch_quiz(history):
    """Launch quiz if ≥3 Strong questions and no quiz active."""
    if GLOBAL_STATE.get("quiz_state"):
        return False
    strong = sum(1 for h in history if h.get("grade") == "Strong")
    return strong >= 3

def generate_quiz_from_strong(history):
    """Make quiz question based on last 3 Strong."""
    strong_items = [h for h in history if h.get("grade") == "Strong"][-3:]
    if not strong_items:
        return None
    context = "\n".join(f"- {it['question']}" for it in strong_items)
    sys = f"""
You are a tutor. Create ONE short quiz question from these strong questions:
{context}
Return STRICT JSON:
{{
  "question": "quiz question",
  "expected_answer": "concise correct answer",
  "explanation": "2-3 sentence explanation"
}}
"""
    resp = llm_zero_temp().invoke(sys).content.strip()
    data = extract_json(resp) or {}
    return {
        "question": data.get("question", "Quiz generation failed."),
        "answer": data.get("expected_answer", ""),
        "explanation": data.get("explanation", "")
    }

def evaluate_quiz_answer(user_answer, expected_answer):
    sys = f"""
You evaluate a student's short answer.
Expected: {expected_answer}
Return STRICT JSON:
{{
  "verdict": "Satisfactory" | "Unsatisfactory",
  "reason": "short reason"
}}
"""
    resp = llm_zero_temp().invoke(sys + "\nStudent Answer: " + user_answer).content.strip()
    return extract_json(resp) or {"verdict": "Unsatisfactory", "reason": "Parse fallback."}

def verify_conceptual_relevance(question, answer, quiz=None):
    quiz_text = ""
    if quiz:
        quiz_text = f"QuizQ: {quiz.get('question','')}\nExpected: {quiz.get('answer','')}"
    sys = """
You are a verifier. Determine if the Answer conceptually aligns with the Question.
If quiz info is present, ensure they are about the same concept.
Return STRICT JSON:
{
  "relevant": true|false,
  "notes": "short reason"
}
"""
    user = f"Question: {question}\nAnswer: {answer}\n{quiz_text}"
    resp = llm_zero_temp().invoke(sys + "\n" + user).content.strip()
    return extract_json(resp) or {"relevant": True, "notes": "Default true."}

# ------------------ Routes ------------------
@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/reset", methods=["POST"])
def reset():
    GLOBAL_STATE["history"].clear()
    GLOBAL_STATE["events"].clear()
    GLOBAL_STATE["quiz_state"] = None
    persist_log()
    return jsonify({"message": "Memory cleared ✅"})

@app.route("/ask", methods=["POST"])
def ask():
    user_text = request.form.get("question", "").strip()
    if not user_text:
        return jsonify({"error": "Please enter a question"}), 400

    history = GLOBAL_STATE["history"]
    quiz_state = GLOBAL_STATE.get("quiz_state")

    # ---- If quiz active ----
    if quiz_state:
        quiz_state["attempts"] += 1
        verdict = evaluate_quiz_answer(user_text, quiz_state["answer"])
        log_event("quiz_evaluate", {"attempt": quiz_state["attempts"], **verdict})

        if verdict["verdict"] == "Satisfactory":
            msg = f"✅ Correct! {verdict['reason']}"
            history.append({
                "type": "quiz_attempt",
                "question": quiz_state["question"],
                "user_answer": user_text,
                "verdict": "Satisfactory",
                "ts": timestamp()
            })
            GLOBAL_STATE["quiz_state"] = None
            persist_log()
            return jsonify({"response": msg})

        if quiz_state["attempts"] < quiz_state["max_attempts"]:
            GLOBAL_STATE["quiz_state"] = quiz_state
            persist_log()
            return jsonify({"response": f"❌ Not quite: {verdict['reason']} Try again."})

        msg = (
            f"❌ Incorrect again.\n\n✅ **Correct:** {quiz_state['answer']}\n"
            f"💡 **Explanation:** {quiz_state['explanation']}"
        )
        history.append({
            "type": "quiz_reveal",
            "question": quiz_state["question"],
            "user_answer": user_text,
            "correct_answer": quiz_state["answer"],
            "explanation": quiz_state["explanation"],
            "ts": timestamp()
        })
        GLOBAL_STATE["quiz_state"] = None
        persist_log()
        return jsonify({"response": msg})

    # ---- Normal QA ----
    grade_info = grade_question_with_llm(user_text, history)
    grade = grade_info["grade"]
    grade_reason = grade_info["reason"]
    log_event("question_graded", {"grade": grade, "reason": grade_reason})

    tool = deep_answer if grade in ("Strong", "CounterCue") else brief_answer

    try:
        start = time.time()
        answer = tool.func(user_text)
        duration = round(time.time() - start, 2)
        log_event("tool_used", {"name": tool.name, "duration_s": duration})
    except Exception as e:
        log_event("tool_error", {"name": tool.name, "error": str(e)})
        return jsonify({"error": str(e)}), 500

    verify = verify_conceptual_relevance(user_text, answer, None)
    log_event("verify", verify)

    history.append({
        "type": "qa",
        "question": user_text,
        "answer": answer,
        "grade": grade,
        "grade_reason": grade_reason,
        "verification": verify,
        "tools_used": [tool.name],
        "ts": timestamp()
    })

    # ---- Quiz trigger ----
    if should_launch_quiz(history):
        quiz = generate_quiz_from_strong(history)
        GLOBAL_STATE["quiz_state"] = {
            "question": quiz["question"],
            "answer": quiz["answer"],
            "explanation": quiz["explanation"],
            "attempts": 0,
            "max_attempts": 2
        }
        log_event("quiz_launched", {"question": quiz["question"]})
        persist_log()
        combined = (
            f"{answer}\n\n---\n🧪 **Quick Check:** {quiz['question']}\n"
            "Reply with your answer."
        )
        return jsonify({"response": combined})

    persist_log()
    return jsonify({"response": answer})

# ------------------ Run ------------------
if __name__ == "__main__":
    app.run(debug=True)