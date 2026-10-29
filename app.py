"""Habit Streak Counter+ : track one habit, play quizzes, earn XP and badges.

UI: Streamlit | Storage: local JSON | AI (quizzes, tips, messages): Groq (optional)
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
TOPIC_SCIENCE = "Habit-building science"

DEFAULT_DATA = {
    "habit_name": "",
    "current_streak": 0,
    "best_streak": 0,
    "last_done": None,        # ISO date string
    "history": [],            # list of ISO dates when the habit was completed
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
        return "Every journey starts with one small step. 🌱"
    if streak < 3:
        return "Great start! Keep the momentum going. 💪"
    if streak < 7:
        return "You're building a real habit now! 🔥"
    if streak < 30:
        return "You're on fire! Don't break the chain. 🚀"
    return "Unstoppable! This habit is part of you now. 🏆"


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
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp {background: #F6F7FB;}
.block-container {max-width: 520px; padding-top: 2rem;}

.habit-name {text-align: center; font-size: 1.5rem; font-weight: 700; color: #2D3142;}
.level-line {text-align: center; font-size: 0.85rem; color: #6C63FF; font-weight: 600;
             margin-bottom: 0.3rem;}
.streak-number {text-align: center; font-size: 6.5rem; font-weight: 800; line-height: 1;
                color: #6C63FF; margin: 1rem 0 0 0;}
.streak-label {text-align: center; font-size: 1.1rem; color: #4F5D75; margin-bottom: 0.8rem;}
.message {text-align: center; font-size: 1rem; color: #4F5D75; min-height: 2.6rem;
          margin-bottom: 0.8rem;}
.best {text-align: center; font-size: 0.85rem; color: #9AA0B4; margin: 1rem 0;}
.tip {background: #FFFFFF; border-left: 4px solid #6C63FF; border-radius: 10px;
      padding: 0.8rem 1rem; color: #2D3142; font-size: 0.95rem;}

div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    width: 100%; padding: 0.8rem 0; font-size: 1.05rem; font-weight: 600;
    color: #FFFFFF; background: #6C63FF; border: none; border-radius: 14px;
    transition: all 0.15s ease;
}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
    background: #5750D9; color: #FFFFFF; transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(108, 99, 255, 0.35);
}
div.stButton > button:disabled {
    background: #D9DBE9; color: #8A8FA8; box-shadow: none; transform: none;
    cursor: not-allowed;
}

.cal {display: grid; grid-template-columns: repeat(7, 1fr); gap: 6px; margin: 0.5rem 0 1rem;}
.cal-cell {aspect-ratio: 1; display: flex; align-items: center; justify-content: center;
           border-radius: 8px; background: #E6E8F2; color: #9AA0B4; font-size: 0.75rem;}
.cal-cell.done {background: #6C63FF; color: #FFFFFF; font-weight: 700;}
.cal-cell.today {outline: 2px solid #2D3142;}

.badge {border-radius: 12px; padding: 0.7rem 0.8rem; margin-bottom: 0.6rem;
        min-height: 4.2rem; font-size: 0.9rem;}
.badge small {display: block; margin-top: 0.2rem;}
.badge.earned {background: #FFFFFF; color: #2D3142; border: 2px solid #6C63FF;}
.badge.locked {background: #E6E8F2; color: #9AA0B4; filter: grayscale(1);}
</style>
"""


# ----------------------------------------------------------------------------
# UI sections
# ----------------------------------------------------------------------------
def render_setup(data: dict) -> None:
    """First launch: ask for the habit name (set once)."""
    st.markdown('<div class="habit-name">🔥 Habit Streak Counter+</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="message">Pick ONE habit to track. Then build streaks, play quizzes '
        "and earn badges!</div>",
        unsafe_allow_html=True,
    )
    name = st.text_input("Habit name", placeholder="e.g. Drink Water", label_visibility="collapsed")
    if st.button("Start tracking"):
        if name.strip():
            data["habit_name"] = name.strip()[:40]
            save_data(data)
            flash("Habit saved! Try your first quiz in the Quiz tab 🧠")
            st.rerun()
        else:
            st.warning("Please enter a habit name.")


def render_header(data: dict) -> None:
    in_level = data["xp"] % XP_PER_LEVEL
    st.markdown(
        f'<div class="habit-name">{html.escape(data["habit_name"])}</div>'
        f'<div class="level-line">Level {get_level(data["xp"])} · {in_level}/{XP_PER_LEVEL} XP</div>',
        unsafe_allow_html=True,
    )
    st.progress(in_level / XP_PER_LEVEL)


def render_home(data: dict, today: date) -> None:
    done_today = is_done_today(data, today)

    st.markdown(f'<div class="streak-number">{data["current_streak"]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="streak-label">day streak</div>', unsafe_allow_html=True)

    message = html.escape(get_motivation(data["habit_name"], data["current_streak"]))
    st.markdown(f'<div class="message">{message}</div>', unsafe_allow_html=True)

    label = "Done for today ✓" if done_today else "Done Today"
    if st.button(label, disabled=done_today):
        gained = mark_done(data, today)
        new_badges = check_badges(data)
        save_data(data)
        flash(f"+{gained} XP! Streak is now {data['current_streak']} 🔥")
        for name in new_badges:
            flash(f"New badge unlocked: {name}")
        st.rerun()

    st.markdown(
        f'<div class="best">Best streak: {data["best_streak"]} days</div>',
        unsafe_allow_html=True,
    )

    tip = html.escape(get_daily_tip(data["habit_name"], today))
    st.markdown(f'<div class="tip">💡 <b>Today\'s tip:</b> {tip}</div>', unsafe_allow_html=True)

    if done_today:
        st.info("Nice work! Keep your brain busy with a quick quiz in the Quiz tab 🧠")


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


def render_quiz(data: dict, today: date) -> None:
    st.subheader("🧠 Brain Break")
    topic = st.radio("Topic", [TOPIC_HABIT, TOPIC_SCIENCE], horizontal=True)
    st.caption(
        f"First quiz of the day earns +{XP_PER_CORRECT} XP per correct answer. "
        "More quizzes are free practice."
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
        st.info("Press **New quiz** to start.")
        return
    if not quiz["used_ai"]:
        st.caption("AI quiz unavailable right now, so here's a general habit quiz.")

    result = st.session_state.get("quiz_result")
    if result is None:
        with st.form(f"quiz_form_{quiz['id']}"):
            answers = [
                st.radio(f"{i + 1}. {q['question']}", q["options"], index=None, key=f"q{quiz['id']}_{i}")
                for i, q in enumerate(quiz["questions"])
            ]
            submitted = st.form_submit_button("Submit answers")
        if submitted:
            if any(a is None for a in answers):
                st.warning("Please answer every question first.")
            else:
                st.session_state["quiz_result"] = grade_quiz(data, quiz, answers, today)
                st.rerun()
        return

    # Show results
    suffix = f"+{result['xp']} XP" if result["earns_xp"] else "practice round, no XP"
    st.success(f"You scored {result['score']}/{result['total']} ({suffix})")
    if result["score"] == result["total"]:
        st.balloons()
    for i, d in enumerate(result["details"], start=1):
        icon = "✅" if d["is_correct"] else "❌"
        st.markdown(f"**{icon} {i}. {d['question']}**")
        if not d["is_correct"]:
            st.caption(f"Your answer: {d['chosen']}")
        st.caption(f"Correct answer: {d['correct']}. {d['explanation']}")


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
    st.subheader("📅 Your Progress")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total days", len(data["history"]))
    col2.metric("Best streak", data["best_streak"])
    col3.metric("Quizzes", data["quizzes_taken"])

    st.caption("Last 28 days")
    st.markdown(calendar_html(data["history"], today), unsafe_allow_html=True)

    st.subheader("📓 Journal")
    today_key = today.isoformat()
    note = st.text_area(
        "How did it go today?",
        value=data["notes"].get(today_key, ""),
        max_chars=280,
        height=90,
    )
    if st.button("Save note"):
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
            st.warning("Write something first.")

    recent = sorted(data["notes"].items(), reverse=True)[:5]
    for day_key, text in recent:
        st.markdown(f"**{day_key}**: {text}")

    with st.expander("⚠️ Danger zone"):
        confirm = st.checkbox("I want to delete all progress and start a new habit")
        if st.button("Reset habit", disabled=not confirm):
            save_data(copy.deepcopy(DEFAULT_DATA))
            st.session_state.clear()
            st.rerun()


def render_badges(data: dict) -> None:
    earned = set(data["badges"])
    st.subheader("🏅 Badges")
    st.caption(f"{len(earned)}/{len(BADGES)} unlocked")
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
        ["🏠 Home", "🧠 Quiz", "📅 Progress", "🏅 Badges"]
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
