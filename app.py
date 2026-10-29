"""Habit Streak Counter+ : track one habit, play quizzes, earn XP and badges.

UI: Streamlit (cute pastel theme, readable in phone dark mode too)
Storage: local JSON | AI (quizzes, tips, messages): Groq (optional)
Run with:  streamlit run app.py
"""

import copy
import html
import json
import os
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import streamlit as st

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_FILE = Path(__file__).with_name("habit_data.json")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

XP_PER_LEVEL = 100
QUIZ_LENGTH = 5
XP_PER_CORRECT = 5
TOPIC_HABIT = "About my habit"
TOPIC_SCIENCE = "Habit science"

DEFAULT_DATA = {
    "habit_name": "",
    "current_streak": 0,
    "best_streak": 0,
    "last_done": None,        # ISO date string
    "history": [],            # ISO dates when the habit was completed
    "xp": 0,
    "quizzes_taken": 0,
    "perfect_quizzes": 0,
    "last_quiz_xp_date": None,  # only the first quiz of each day earns XP
    "notes": {},              # {ISO date: journal note}
    "badges": [],
}


# ----------------------------------------------------------------------------
# Data handling (load / save)
# ----------------------------------------------------------------------------
def _valid_date(value) -> Optional[str]:
    """Return the value if it is a valid ISO date string, else None."""
    try:
        date.fromisoformat(value)
        return value
    except (TypeError, ValueError):
        return None


def load_data() -> dict:
    """Load saved data. Fall back to defaults if the file is missing or corrupted."""
    data = copy.deepcopy(DEFAULT_DATA)
    try:
        raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return data
        data["habit_name"] = str(raw.get("habit_name", ""))[:40]
        for key in ("current_streak", "best_streak", "xp", "quizzes_taken", "perfect_quizzes"):
            data[key] = max(0, int(raw.get(key, 0)))
        data["last_done"] = _valid_date(raw.get("last_done"))
        data["last_quiz_xp_date"] = _valid_date(raw.get("last_quiz_xp_date"))
        history = raw.get("history", [])
        data["history"] = sorted({d for d in history if _valid_date(d)}) if isinstance(history, list) else []
        notes = raw.get("notes", {})
        if isinstance(notes, dict):
            data["notes"] = {k: str(v) for k, v in notes.items() if _valid_date(k)}
        badges = raw.get("badges", [])
        data["badges"] = [b for b in badges if b in BADGES] if isinstance(badges, list) else []
        return data
    except (OSError, ValueError, TypeError, AttributeError):
        return copy.deepcopy(DEFAULT_DATA)


def save_data(data: dict) -> None:
    """Write data to the local JSON file."""
    DATA_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------
# Streak, XP and badge logic
# ----------------------------------------------------------------------------
def apply_streak_reset(data: dict, today: date) -> bool:
    """Reset the streak to 0 if more than one day has passed. Returns True if changed."""
    if data["last_done"] is None:
        return False
    days_gap = (today - date.fromisoformat(data["last_done"])).days
    if days_gap > 1 and data["current_streak"] != 0:
        data["current_streak"] = 0
        return True
    return False


def is_done_today(data: dict, today: date) -> bool:
    return data["last_done"] == today.isoformat()


def get_level(xp: int) -> int:
    return xp // XP_PER_LEVEL + 1


def mark_done(data: dict, today: date) -> int:
    """Increase streak by 1, log the date, award XP. Returns the XP gained."""
    data["current_streak"] += 1
    data["best_streak"] = max(data["best_streak"], data["current_streak"])
    data["last_done"] = today.isoformat()
    if today.isoformat() not in data["history"]:
        data["history"].append(today.isoformat())
    gained = 10 + min(data["current_streak"], 10)  # small streak bonus
    data["xp"] += gained
    return gained


# badge id -> (name, description, unlock condition)
BADGES = {
    "first_step": ("🌱 First Step", "Complete your habit once", lambda d: len(d["history"]) >= 1),
    "streak_3": ("🔥 3-Day Spark", "Reach a 3-day streak", lambda d: d["best_streak"] >= 3),
    "streak_7": ("⚡ Weekly Warrior", "Reach a 7-day streak", lambda d: d["best_streak"] >= 7),
    "streak_30": ("🏆 Monthly Master", "Reach a 30-day streak", lambda d: d["best_streak"] >= 30),
    "quiz_rookie": ("🧠 Quiz Rookie", "Finish your first quiz", lambda d: d["quizzes_taken"] >= 1),
    "quiz_ace": ("🎯 Quiz Ace", "Get a perfect quiz score", lambda d: d["perfect_quizzes"] >= 1),
    "level_3": ("⭐ Rising Star", "Reach level 3", lambda d: get_level(d["xp"]) >= 3),
    "journaler": ("📓 Journaler", "Write 3 journal notes", lambda d: len(d["notes"]) >= 3),
}


def check_badges(data: dict) -> list:
    """Unlock any newly earned badges. Returns their names."""
    unlocked = []
    for badge_id, (name, _desc, condition) in BADGES.items():
        if badge_id not in data["badges"] and condition(data):
            data["badges"].append(badge_id)
            unlocked.append(name)
    return unlocked


def mascot_for(streak: int):
    """Your cute buddy grows as the streak grows. Returns (emoji, caption)."""
    if streak == 0:
        return "🥚", "Your buddy is waiting to hatch"
    if streak < 3:
        return "🐣", "Your buddy just hatched!"
    if streak < 7:
        return "🐥", "Your buddy is growing!"
    if streak < 30:
        return "🦊", "Your buddy is thriving!"
    return "🦄", "Your buddy is legendary!"


def flash(message: str) -> None:
    """Queue a toast message to show after the next rerun."""
    st.session_state.setdefault("flash", []).append(message)


# ----------------------------------------------------------------------------
# Groq helpers (all optional: the app works offline with fallbacks)
# ----------------------------------------------------------------------------
def get_groq_key() -> Optional[str]:
    """Read the API key from env var or Streamlit secrets (never hard-code it)."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GROQ_API_KEY")
    except Exception:  # no secrets file present
        return None


def ask_groq(system: str, user: str, json_mode: bool = False, max_tokens: int = 100) -> Optional[str]:
    """Send one prompt to Groq. Returns the text, or None on any failure."""
    api_key = get_groq_key()
    if not api_key:
        return None
    try:
        from groq import Groq

        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        response = Groq(api_key=api_key).chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            **extra,
        )
        return (response.choices[0].message.content or "").strip() or None
    except Exception:
        return None


def fallback_message(streak: int) -> str:
    if streak == 0:
        return "Every journey starts with one small step 🌱"
    if streak < 3:
        return "Great start! Keep the momentum going 💪"
    if streak < 7:
        return "You're building a real habit now! 🔥"
    if streak < 30:
        return "You're on fire! Don't break the chain 🚀"
    return "Unstoppable! This habit is part of you now 🏆"


def get_motivation(habit: str, streak: int) -> str:
    """Short motivational line, cached per (habit, streak)."""
    cache = st.session_state.setdefault("msg_cache", {})
    key = (habit, streak)
    if key not in cache:
        cache[key] = ask_groq(
            "You are a warm, upbeat habit coach. Reply with ONE short motivational "
            "sentence (max 15 words) and one emoji. No quotes, no hashtags.",
            f"Habit: {habit}. Current streak: {streak} days.",
            max_tokens=50,
        ) or fallback_message(streak)
    return cache[key]


FALLBACK_TIPS = [
    "Do it at the same time every day so it becomes automatic.",
    "Make it tiny today: even two minutes counts as a win.",
    "Prepare what you need the night before to remove friction.",
    "Pair it with something you already do daily, like morning tea.",
    "Reward yourself right after you finish. Celebrate the small win!",
]


def get_daily_tip(habit: str, today: date) -> str:
    """One practical tip per day for this habit, cached for the session."""
    cache = st.session_state.setdefault("tip_cache", {})
    key = (habit, today.isoformat())
    if key not in cache:
        cache[key] = ask_groq(
            "You are a friendly habit coach. Reply with ONE practical, specific tip "
            "(max 30 words) for doing this habit well today. No preamble.",
            f"Habit: {habit}",
            max_tokens=80,
        ) or random.choice(FALLBACK_TIPS)
    return cache[key]


# ----------------------------------------------------------------------------
# Quiz generation
# ----------------------------------------------------------------------------
FALLBACK_QUIZ = [
    {
        "question": "On average, about how long does it take to form a new habit?",
        "options": ["7 days", "21 days", "66 days", "365 days"],
        "answer_index": 2,
        "explanation": "A well-known study found an average of about 66 days, with wide variation.",
    },
    {
        "question": "What does 'habit stacking' mean?",
        "options": [
            "Doing many new habits at once",
            "Attaching a new habit to an existing routine",
            "Tracking habits on paper",
            "Rewarding yourself with snacks",
        ],
        "answer_index": 1,
        "explanation": "Linking a new habit to something you already do makes it easier to remember.",
    },
    {
        "question": "You missed a day. What is the best move?",
        "options": [
            "Give up, the streak is ruined",
            "Double the effort tomorrow",
            "Never miss twice: get back on track right away",
            "Wait until next Monday",
        ],
        "answer_index": 2,
        "explanation": "One miss is normal. Missing twice is when a new pattern starts.",
    },
    {
        "question": "Which three parts make up the classic 'habit loop'?",
        "options": [
            "Plan, Action, Result",
            "Cue, Routine, Reward",
            "Goal, Effort, Prize",
            "Start, Middle, End",
        ],
        "answer_index": 1,
        "explanation": "A cue triggers a routine, and the reward makes your brain want to repeat it.",
    },
    {
        "question": "Which is better when starting a new habit?",
        "options": [
            "Start huge to build momentum",
            "Start tiny and build up gradually",
            "Wait until you feel motivated",
            "Change many habits at once",
        ],
        "answer_index": 1,
        "explanation": "Tiny starts are easier to repeat, and consistency matters more than intensity.",
    },
    {
        "question": "Why does tracking a streak help?",
        "options": [
            "It makes progress visible and motivating",
            "It removes the need for effort",
            "It guarantees success",
            "It burns calories",
        ],
        "answer_index": 0,
        "explanation": "Seeing your progress builds momentum and makes you not want to break the chain.",
    },
]


def parse_quiz(raw: Optional[str]) -> Optional[list]:
    """Validate the JSON returned by the LLM. Returns None if it is unusable."""
    if not raw:
        return None
    try:
        items = json.loads(raw).get("questions", [])
    except (ValueError, AttributeError):
        return None
    questions = []
    for item in items:
        try:
            options = [str(o).strip() for o in item["options"]]
            answer_index = int(item["answer_index"])
            if len(options) == 4 and len(set(options)) == 4 and 0 <= answer_index < 4:
                questions.append({
                    "question": str(item["question"]).strip(),
                    "options": options,
                    "answer_index": answer_index,
                    "explanation": str(item.get("explanation", "")).strip(),
                })
        except (KeyError, TypeError, ValueError):
            continue
    return questions[:QUIZ_LENGTH] if len(questions) >= 3 else None


def shuffle_options(question: dict) -> dict:
    """Return a copy of the question with shuffled options (answer index updated)."""
    correct = question["options"][question["answer_index"]]
    options = question["options"][:]
    random.shuffle(options)
    return {**question, "options": options, "answer_index": options.index(correct)}


def generate_quiz(habit: str, topic: str):
    """Return (questions, used_ai). Falls back to a built-in quiz if the AI fails."""
    if topic == TOPIC_HABIT:
        subject = f'the habit "{habit}" (benefits, science, common myths and practical tips)'
    else:
        subject = "how habits are formed and how to stay consistent"

    raw = ask_groq(
        "You write fun, accurate multiple-choice quizzes. Reply with JSON only.",
        f"Create {QUIZ_LENGTH} beginner-friendly multiple-choice questions about {subject}. "
        "Each question has exactly 4 distinct options and exactly one correct answer. "
        'Return JSON: {"questions": [{"question": "...", "options": ["...", "...", "...", "..."], '
        '"answer_index": 0, "explanation": "one short sentence"}]}',
        json_mode=True,
        max_tokens=1400,
    )
    questions = parse_quiz(raw)
    if questions:
        return [shuffle_options(q) for q in questions], True
    picked = random.sample(FALLBACK_QUIZ, QUIZ_LENGTH)
    return [shuffle_options(q) for q in picked], False


# ----------------------------------------------------------------------------
# Styling: cute pastel look. Every text/background colour is forced so the app
# stays readable even when the phone or browser is in dark mode.
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&display=swap');

:root {
    color-scheme: light;
    --ink: #3D3355;
    --muted: #6E6490;
    --pink: #FF8FB1;
    --purple: #7B6CF6;
    --line: #EADFFF;
}

/* ---- Background (forced light, even in phone dark mode) ---- */
html, body, .stApp, [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: linear-gradient(160deg, #FFF0F6 0%, #F3EEFF 55%, #E9F6FF 100%) !important;
    background-attachment: fixed !important;
}
#MainMenu, footer, [data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"] {display: none !important;}
.block-container {max-width: 520px; padding: 1.2rem 1rem 3rem !important;}

/* ---- Text colours and font ---- */
.stApp, .stApp .stMarkdown, .stApp p, .stApp li, .stApp label,
.stApp h1, .stApp h2, .stApp h3 {color: var(--ink) !important;}
.stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp textarea {
    font-family: 'Fredoka', 'Nunito', 'Segoe UI', system-ui, sans-serif !important;
}
[data-testid="stSpinner"] * {color: var(--ink) !important;}

/* ---- Cards ---- */
.card {background: #FFFFFF; border: 2px solid #F0E8FF; border-radius: 24px;
       padding: 1.2rem 1rem; margin-bottom: 0.9rem;
       box-shadow: 0 8px 24px rgba(123, 108, 246, 0.12);}
.hero {text-align: center; background: linear-gradient(180deg, #FFFFFF 0%, #FFF5FA 100%);}
.mascot {font-size: 3.6rem; line-height: 1; display: inline-block;
         animation: bob 2.4s ease-in-out infinite;}
@keyframes bob {0%, 100% {transform: translateY(0);} 50% {transform: translateY(-8px);}}
.buddy {font-size: 0.8rem; color: var(--muted); margin-top: 0.2rem;}
.streak-number {font-size: clamp(4.5rem, 22vw, 6rem); font-weight: 700; line-height: 1.05;
                color: var(--purple); margin-top: 0.4rem;}
.streak-label {font-size: 1.1rem; color: var(--muted); margin-bottom: 0.6rem;}
.message {font-size: 1rem; font-weight: 500; color: var(--ink); min-height: 2.4rem;}

/* ---- Top bar + XP bar ---- */
.topbar {display: flex; align-items: center; justify-content: space-between; gap: 0.6rem;
         margin-bottom: 0.5rem;}
.habit-name {font-size: 1.4rem; font-weight: 700; color: var(--ink);}
.level-pill {background: linear-gradient(90deg, #E8629A, #7B6CF6); color: #FFFFFF;
             border-radius: 999px; padding: 0.25rem 0.85rem; font-size: 0.85rem;
             font-weight: 600; white-space: nowrap;}
.xp-bar {height: 12px; background: #FFFFFF; border: 2px solid var(--line);
         border-radius: 999px; overflow: hidden;}
.xp-fill {height: 100%; background: linear-gradient(90deg, #FFB3CB, #9D8CFF);
          border-radius: 999px;}
.xp-text {font-size: 0.75rem; color: var(--muted); text-align: right;
          margin: 0.2rem 0 0.8rem;}

/* ---- Chips, tip, notices ---- */
.chip-row {text-align: center; margin: 0.8rem 0;}
.chip {display: inline-block; background: #FFFFFF; border: 2px solid var(--line);
       color: var(--ink); border-radius: 999px; padding: 0.3rem 0.9rem; font-size: 0.9rem;}
.tip {background: #FFF8DC; border: 2px solid #FFE9A8; border-radius: 20px;
      padding: 0.8rem 1rem; color: var(--ink); font-size: 0.95rem;}
.notice {border-radius: 16px; padding: 0.75rem 1rem; margin: 0.6rem 0; font-size: 0.95rem;
         color: var(--ink); background: #FFFFFF; border: 2px solid var(--line);}
.notice.info {background: #EAF3FF; border-color: #CFE3FF;}
.notice.warn {background: #FFF4DB; border-color: #FFE3A3;}
.section-title {font-size: 1.25rem; font-weight: 700; color: var(--ink); margin: 0.4rem 0 0.5rem;}
.hint {font-size: 0.85rem; color: var(--muted); text-align: center; margin: 0.3rem 0 0.8rem;}

/* ---- Buttons ---- */
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    width: 100%; padding: 0.8rem 0; border: none; border-radius: 999px;
    background: linear-gradient(90deg, #E8629A, #7B6CF6) !important; color: #FFFFFF !important;
    font-weight: 600; box-shadow: 0 6px 16px rgba(123, 108, 246, 0.3);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div.stButton > button p, div[data-testid="stFormSubmitButton"] > button p {
    color: #FFFFFF !important; font-size: 1.05rem; font-weight: 600; margin: 0;
}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px) scale(1.01); box-shadow: 0 10px 22px rgba(123, 108, 246, 0.4);
}
div.stButton > button:disabled {
    background: #DDF6E8 !important; box-shadow: none; transform: none; cursor: not-allowed;
}
div.stButton > button:disabled p {color: #2F7D55 !important;}

/* ---- Tabs as pills ---- */
div[data-baseweb="tab-list"] {gap: 6px; background: rgba(255, 255, 255, 0.75);
    padding: 6px; border-radius: 20px;}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {display: none !important;}
button[data-baseweb="tab"] {flex: 1; height: auto; padding: 0.5rem 0.2rem;
    border-radius: 14px; background: transparent !important;}
button[data-baseweb="tab"] p {color: var(--muted) !important; font-weight: 600; font-size: 0.85rem;}
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(90deg, #FF8FB1, #9D8CFF) !important;}
button[data-baseweb="tab"][aria-selected="true"] p {color: #FFFFFF !important;}

/* ---- Quiz: each question is a card, each option is a pill ---- */
div[data-testid="stRadio"] {background: #FFFFFF; border: 2px solid var(--line);
    border-radius: 22px; padding: 0.9rem 0.9rem 0.6rem; margin-bottom: 0.8rem;
    box-shadow: 0 6px 18px rgba(123, 108, 246, 0.08);}
div[data-testid="stRadio"] [data-testid="stWidgetLabel"] p {
    font-weight: 700; font-size: 1.05rem; color: var(--ink) !important;}
div[role="radiogroup"] label {background: #F8F4FF; border: 2px solid var(--line);
    border-radius: 14px; padding: 0.5rem 0.75rem; margin: 0.2rem 0;
    transition: all 0.12s ease;}
div[role="radiogroup"] label:hover {border-color: var(--pink);}
div[role="radiogroup"] label:has(input:checked) {background: #FFE6EF; border-color: var(--pink);}
div[role="radiogroup"] label p {color: var(--ink) !important; font-size: 0.98rem;}
[data-testid="stForm"] {border: none !important; padding: 0 !important;
    background: transparent !important;}

.banner {background: linear-gradient(135deg, #FFE1EC, #E6DEFF); border-radius: 22px;
         padding: 1rem; text-align: center; margin-bottom: 0.8rem; color: var(--ink);}
.banner b {font-size: 1.3rem; color: var(--ink);}
.score {text-align: center; background: linear-gradient(135deg, #FFE1EC, #DDEBFF);
        border-radius: 24px; padding: 1.1rem; margin-bottom: 0.8rem; color: var(--ink);}
.score-big {font-size: 2.8rem; font-weight: 700; color: var(--purple); line-height: 1.1;}
.res {border-radius: 18px; padding: 0.8rem 0.9rem; margin-bottom: 0.6rem; color: var(--ink);}
.res.ok {background: #E4F8EC; border: 2px solid #BDEBD0;}
.res.bad {background: #FFE9EE; border: 2px solid #FFC9D6;}
.res-q {font-weight: 600; color: var(--ink);}
.res-a {font-size: 0.88rem; margin-top: 0.3rem; color: var(--ink);}

/* ---- Inputs ---- */
div[data-baseweb="input"], div[data-baseweb="textarea"] {
    border: 2px solid var(--line) !important; border-radius: 14px !important;}
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="textarea"] {
    background: #FFFFFF !important;}
.stApp input, .stApp textarea {background: #FFFFFF !important; color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important; caret-color: var(--ink);}
.stApp input::placeholder, .stApp textarea::placeholder {
    color: #A79DC4 !important; -webkit-text-fill-color: #A79DC4 !important;}

/* ---- Toast ---- */
[data-testid="stToast"] {background: #FFFFFF !important; border: 2px solid var(--line);
    border-radius: 16px;}
[data-testid="stToast"] * {color: var(--ink) !important;}

/* ---- Stats, calendar, notes, badges ---- */
.stats {display: flex; gap: 0.6rem; margin-bottom: 0.8rem;}
.stat {flex: 1; background: #FFFFFF; border: 2px solid var(--line); border-radius: 18px;
       padding: 0.7rem 0.3rem; text-align: center;}
.stat b {display: block; font-size: 1.5rem; color: var(--purple);}
.stat span {font-size: 0.75rem; color: var(--muted);}

.cal {display: grid; grid-template-columns: repeat(7, 1fr); gap: 6px; margin: 0.4rem 0 1rem;}
.cal-cell {aspect-ratio: 1; display: flex; align-items: center; justify-content: center;
           border-radius: 10px; background: #FFFFFF; border: 2px solid var(--line);
           color: var(--muted); font-size: 0.75rem;}
.cal-cell.done {background: linear-gradient(135deg, #FF8FB1, #9D8CFF); border-color: transparent;
                color: #FFFFFF; font-weight: 700;}
.cal-cell.today {outline: 2px solid var(--ink); outline-offset: 1px;}

.note {background: #FFFFFF; border: 2px solid var(--line); border-radius: 16px;
       padding: 0.6rem 0.9rem; margin-bottom: 0.5rem; color: var(--ink); font-size: 0.92rem;}
.note small {display: block; color: var(--muted); margin-bottom: 0.1rem;}

.badge {border-radius: 18px; padding: 0.75rem 0.8rem; margin-bottom: 0.6rem;
        min-height: 4.4rem; font-size: 0.92rem;}
.badge small {display: block; margin-top: 0.2rem; font-size: 0.78rem;}
.badge.earned {background: #FFFFFF; color: var(--ink); border: 2px solid var(--pink);
               box-shadow: 0 6px 16px rgba(255, 143, 177, 0.25);}
.badge.earned small {color: var(--muted);}
.badge.locked {background: #EFEAF8; color: #7D7399; border: 2px dashed #D4C9EE;}
</style>
"""


def notice(text: str, kind: str = "info") -> None:
    """A small rounded message box (kind: info or warn)."""
    st.markdown(f'<div class="notice {kind}">{html.escape(text)}</div>', unsafe_allow_html=True)


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{html.escape(text)}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# UI sections
# ----------------------------------------------------------------------------
def render_setup(data: dict) -> None:
    """First launch: ask for the habit name (set once)."""
    st.markdown(
        '<div class="card hero"><div class="mascot">🥚</div>'
        '<div class="section-title">Hatch a new habit!</div>'
        '<div class="message">Pick ONE habit to track. Build streaks, play quizzes '
        "and earn badges ✨</div></div>",
        unsafe_allow_html=True,
    )
    name = st.text_input("Habit name", placeholder="e.g. Drink Water", label_visibility="collapsed")
    if st.button("Start tracking 🚀"):
        if name.strip():
            data["habit_name"] = name.strip()[:40]
            save_data(data)
            flash("Habit saved! Try your first quiz in the Quiz tab 🧠")
            st.rerun()
        else:
            notice("Please enter a habit name first 🙂", "warn")


def render_header(data: dict) -> None:
    xp_in_level = data["xp"] % XP_PER_LEVEL
    st.markdown(
        f'<div class="topbar"><div class="habit-name">{html.escape(data["habit_name"])}</div>'
        f'<div class="level-pill">⭐ Level {get_level(data["xp"])}</div></div>'
        f'<div class="xp-bar"><div class="xp-fill" style="width:{xp_in_level}%"></div></div>'
        f'<div class="xp-text">{xp_in_level}/{XP_PER_LEVEL} XP</div>',
        unsafe_allow_html=True,
    )


def render_home(data: dict, today: date) -> None:
    done_today = is_done_today(data, today)
    emoji, buddy_text = mascot_for(data["current_streak"])
    message = html.escape(get_motivation(data["habit_name"], data["current_streak"]))

    st.markdown(
        f'<div class="card hero"><div class="mascot">{emoji}</div>'
        f'<div class="buddy">{buddy_text}</div>'
        f'<div class="streak-number">{data["current_streak"]}</div>'
        f'<div class="streak-label">day streak</div>'
        f'<div class="message">{message}</div></div>',
        unsafe_allow_html=True,
    )

    label = "Done for today ✓" if done_today else "Done Today 🎉"
    if st.button(label, disabled=done_today):
        gained = mark_done(data, today)
        new_badges = check_badges(data)
        save_data(data)
        flash(f"+{gained} XP! Streak is now {data['current_streak']} 🔥")
        for name in new_badges:
            flash(f"New badge unlocked: {name}")
        st.rerun()

    st.markdown(
        f'<div class="chip-row"><span class="chip">🏅 Best streak: {data["best_streak"]} days</span></div>',
        unsafe_allow_html=True,
    )

    tip = html.escape(get_daily_tip(data["habit_name"], today))
    st.markdown(f'<div class="tip">💡 <b>Today\'s tip:</b> {tip}</div>', unsafe_allow_html=True)

    if done_today:
        notice("Nice work! Keep your brain busy with a quick quiz 🧠", "info")


def grade_quiz(data: dict, quiz: dict, answers: list, today: date) -> dict:
    """Score the quiz, update XP/badges, and return the result."""
    details = []
    score = 0
    for question, chosen in zip(quiz["questions"], answers):
        correct_text = question["options"][question["answer_index"]]
        is_correct = chosen == correct_text
        score += is_correct
        details.append({
            "question": question["question"],
            "chosen": chosen,
            "correct": correct_text,
            "is_correct": is_correct,
            "explanation": question["explanation"],
        })

    total = len(quiz["questions"])
    earns_xp = data["last_quiz_xp_date"] != today.isoformat()
    xp_gained = score * XP_PER_CORRECT if earns_xp else 0

    data["quizzes_taken"] += 1
    if score == total:
        data["perfect_quizzes"] += 1
    if earns_xp:
        data["xp"] += xp_gained
        data["last_quiz_xp_date"] = today.isoformat()

    for name in check_badges(data):
        flash(f"New badge unlocked: {name}")
    save_data(data)
    return {"score": score, "total": total, "xp": xp_gained, "earns_xp": earns_xp, "details": details}


def render_quiz_result(result: dict) -> None:
    score, total = result["score"], result["total"]
    if score == total:
        face, line = "🎉", "Perfect score! You're a genius!"
    elif score >= total / 2:
        face, line = "🌟", "Great job! Keep it up!"
    else:
        face, line = "💪", "Nice try! You'll get it next time!"
    xp_text = f"+{result['xp']} XP earned ✨" if result["earns_xp"] else "Practice round (no XP)"

    st.markdown(
        f'<div class="score"><div style="font-size:2.4rem">{face}</div>'
        f'<div class="score-big">{score}/{total}</div>'
        f'<div style="font-weight:600">{line}</div>'
        f'<div class="hint" style="margin-bottom:0">{xp_text}</div></div>',
        unsafe_allow_html=True,
    )
    if score == total:
        st.balloons()

    for i, d in enumerate(result["details"], start=1):
        css, icon = ("ok", "✅") if d["is_correct"] else ("bad", "❌")
        wrong = (
            f'<div class="res-a">Your answer: {html.escape(d["chosen"])}</div>'
            if not d["is_correct"] else ""
        )
        st.markdown(
            f'<div class="res {css}"><div class="res-q">{icon} {i}. {html.escape(d["question"])}</div>'
            f"{wrong}"
            f'<div class="res-a">💡 {html.escape(d["correct"])}. {html.escape(d["explanation"])}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="hint">Tap “New quiz” above to play again 🎲</div>', unsafe_allow_html=True)


def render_quiz(data: dict, today: date) -> None:
    st.markdown(
        '<div class="banner"><div style="font-size:2rem">🧠✨</div><b>Brain Break</b>'
        '<div class="hint" style="margin-bottom:0">Quick quiz time! Pick a topic and play.</div></div>',
        unsafe_allow_html=True,
    )
    topic = st.radio("Topic", [TOPIC_HABIT, TOPIC_SCIENCE], horizontal=True, label_visibility="collapsed")
    st.markdown(
        f'<div class="hint">First quiz of the day earns +{XP_PER_CORRECT} XP per correct answer. '
        "More quizzes are free practice.</div>",
        unsafe_allow_html=True,
    )

    if st.button("🎲 New quiz"):
        with st.spinner("Preparing your quiz..."):
            questions, used_ai = generate_quiz(data["habit_name"], topic)
        st.session_state["quiz_counter"] = st.session_state.get("quiz_counter", 0) + 1
        st.session_state["quiz"] = {
            "id": st.session_state["quiz_counter"],
            "questions": questions,
            "used_ai": used_ai,
        }
        st.session_state["quiz_result"] = None

    quiz = st.session_state.get("quiz")
    if not quiz:
        notice("Press “New quiz” to start 🎲", "info")
        return
    if not quiz["used_ai"]:
        st.markdown(
            '<div class="hint">AI quiz unavailable right now, so here is a general habit quiz 📚</div>',
            unsafe_allow_html=True,
        )

    result = st.session_state.get("quiz_result")
    if result is not None:
        render_quiz_result(result)
        return

    with st.form(f"quiz_form_{quiz['id']}"):
        answers = [
            st.radio(f"{i + 1}. {q['question']}", q["options"], index=None, key=f"q{quiz['id']}_{i}")
            for i, q in enumerate(quiz["questions"])
        ]
        submitted = st.form_submit_button("Submit answers ✅")
    if submitted:
        if any(a is None for a in answers):
            notice("Please answer every question first 🙂", "warn")
        else:
            st.session_state["quiz_result"] = grade_quiz(data, quiz, answers, today)
            st.rerun()


def calendar_html(history: list, today: date) -> str:
    """Last 28 days as a small grid. Filled cells = habit done."""
    done = set(history)
    cells = []
    for offset in range(27, -1, -1):
        day = today - timedelta(days=offset)
        css = "cal-cell"
        if day.isoformat() in done:
            css += " done"
        if day == today:
            css += " today"
        cells.append(f'<div class="{css}">{day.day}</div>')
    return f'<div class="cal">{"".join(cells)}</div>'


def render_progress(data: dict, today: date) -> None:
    section_title("📅 Your Progress")
    st.markdown(
        f'<div class="stats">'
        f'<div class="stat"><b>{len(data["history"])}</b><span>Total days</span></div>'
        f'<div class="stat"><b>{data["best_streak"]}</b><span>Best streak</span></div>'
        f'<div class="stat"><b>{data["quizzes_taken"]}</b><span>Quizzes</span></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hint" style="text-align:left">Last 28 days (colored = done 💜)</div>', unsafe_allow_html=True)
    st.markdown(calendar_html(data["history"], today), unsafe_allow_html=True)

    section_title("📓 Journal")
    today_key = today.isoformat()
    note = st.text_area(
        "How did it go today?",
        value=data["notes"].get(today_key, ""),
        placeholder="How did it go today?",
        label_visibility="collapsed",
        max_chars=280,
        height=90,
    )
    if st.button("Save note 💌"):
        if note.strip():
            is_new = today_key not in data["notes"]
            data["notes"][today_key] = note.strip()
            if is_new:
                data["xp"] += 3
                flash("+3 XP for journaling 📓")
            for name in check_badges(data):
                flash(f"New badge unlocked: {name}")
            save_data(data)
            st.rerun()
        else:
            notice("Write something first 🙂", "warn")

    for day_key, text in sorted(data["notes"].items(), reverse=True)[:5]:
        st.markdown(
            f'<div class="note"><small>{html.escape(day_key)}</small>{html.escape(text)}</div>',
            unsafe_allow_html=True,
        )

    section_title("⚠️ Danger zone")
    confirm = st.checkbox("Delete all progress and start a new habit")
    if st.button("Reset habit", disabled=not confirm):
        save_data(copy.deepcopy(DEFAULT_DATA))
        st.session_state.clear()
        st.rerun()


def render_badges(data: dict) -> None:
    earned = set(data["badges"])
    section_title("🏅 Badges")
    st.markdown(
        f'<div class="hint" style="text-align:left">{len(earned)}/{len(BADGES)} unlocked</div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(2)
    for index, (badge_id, (name, description, _)) in enumerate(BADGES.items()):
        css = "badge earned" if badge_id in earned else "badge locked"
        columns[index % 2].markdown(
            f'<div class="{css}"><b>{name}</b><small>{description}</small></div>',
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Habit Streak Counter+", page_icon="🔥", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    for message in st.session_state.pop("flash", []):
        st.toast(message)

    today = date.today()
    data = load_data()
    if apply_streak_reset(data, today):
        save_data(data)

    if not data["habit_name"]:
        render_setup(data)
        return

    render_header(data)
    home_tab, quiz_tab, progress_tab, badges_tab = st.tabs(
        ["🏠 Home", "🧠 Quiz", "📅 Stats", "🏅 Badges"]
    )
    with home_tab:
        render_home(data, today)
    with quiz_tab:
        render_quiz(data, today)
    with progress_tab:
        render_progress(data, today)
    with badges_tab:
        render_badges(data)


if __name__ == "__main__":
    main()
