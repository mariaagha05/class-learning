"""Habit Streak Counter - a simple single-habit tracker.

UI: Streamlit | Storage: local JSON file | AI messages: Groq (optional)
Run with:  streamlit run app.py
"""

import json
import os
from datetime import date
from pathlib import Path

import streamlit as st

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_FILE = Path(__file__).with_name("habit_data.json")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

DEFAULT_DATA = {
    "habit_name": "",
    "current_streak": 0,
    "best_streak": 0,
    "last_done": None,  # ISO date string, e.g. "2026-10-29"
}


# ----------------------------------------------------------------------------
# Data handling (load / save)
# ----------------------------------------------------------------------------
def load_data() -> dict:
    """Load saved data. Fall back to defaults if file is missing or corrupted."""
    try:
        raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        data = {
            "habit_name": str(raw.get("habit_name", "")),
            "current_streak": max(0, int(raw.get("current_streak", 0))),
            "best_streak": max(0, int(raw.get("best_streak", 0))),
            "last_done": raw.get("last_done"),
        }
        if data["last_done"] is not None:
            date.fromisoformat(data["last_done"])  # validate format
        return data
    except (OSError, ValueError, TypeError, AttributeError):
        return DEFAULT_DATA.copy()


def save_data(data: dict) -> None:
    """Write data to the local JSON file."""
    DATA_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------
# Streak logic
# ----------------------------------------------------------------------------
def apply_streak_reset(data: dict, today: date) -> bool:
    """Reset streak to 0 if more than one day has passed. Returns True if changed."""
    if data["last_done"] is None:
        return False
    days_gap = (today - date.fromisoformat(data["last_done"])).days
    if days_gap > 1 and data["current_streak"] != 0:
        data["current_streak"] = 0
        return True
    return False


def is_done_today(data: dict, today: date) -> bool:
    return data["last_done"] == today.isoformat()


def mark_done(data: dict, today: date) -> None:
    """Increase the streak by 1 and remember today's date."""
    data["current_streak"] += 1
    data["best_streak"] = max(data["best_streak"], data["current_streak"])
    data["last_done"] = today.isoformat()


# ----------------------------------------------------------------------------
# Motivational messages (Groq with a safe offline fallback)
# ----------------------------------------------------------------------------
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


def get_groq_key() -> str | None:
    """Read the API key from env var or Streamlit secrets (never hard-code it)."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GROQ_API_KEY")
    except Exception:  # no secrets file present
        return None


def get_motivation(habit: str, streak: int) -> str:
    """Ask Groq for a short message; cache per (habit, streak); fall back on error."""
    cache = st.session_state.setdefault("msg_cache", {})
    cache_key = (habit, streak)
    if cache_key in cache:
        return cache[cache_key]

    message = fallback_message(streak)
    api_key = get_groq_key()
    if api_key:
        try:
            from groq import Groq

            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a warm, upbeat habit coach. Reply with ONE short "
                            "motivational sentence (max 15 words) and one emoji. "
                            "No quotes, no hashtags."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Habit: {habit}. Current streak: {streak} days.",
                    },
                ],
                max_tokens=50,
                temperature=0.8,
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                message = text
        except Exception:
            pass  # keep the fallback message

    cache[cache_key] = message
    return message


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
CSS = """
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp {background: #F6F7FB;}
.block-container {max-width: 420px; padding-top: 3rem;}

.habit-name {text-align: center; font-size: 1.5rem; font-weight: 600;
             color: #2D3142; margin-bottom: 0.5rem;}
.streak-number {text-align: center; font-size: 7rem; font-weight: 800;
                line-height: 1; color: #6C63FF; margin: 0.5rem 0 0 0;}
.streak-label {text-align: center; font-size: 1.1rem; color: #4F5D75;
               margin-bottom: 1rem;}
.message {text-align: center; font-size: 1rem; color: #4F5D75;
          min-height: 3rem; margin-bottom: 1.2rem;}
.best {text-align: center; font-size: 0.85rem; color: #9AA0B4; margin-top: 1.2rem;}

div.stButton > button {
    width: 100%; padding: 0.9rem 0; font-size: 1.15rem; font-weight: 600;
    color: #FFFFFF; background: #6C63FF; border: none; border-radius: 14px;
    transition: all 0.15s ease;
}
div.stButton > button:hover {
    background: #5750D9; color: #FFFFFF; transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(108, 99, 255, 0.35);
}
div.stButton > button:disabled {
    background: #D9DBE9; color: #8A8FA8; box-shadow: none;
    transform: none; cursor: not-allowed;
}
</style>
"""


def render_setup(data: dict) -> None:
    """First launch: ask for the habit name (set once)."""
    st.markdown('<div class="habit-name">🔥 Habit Streak</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="message">What habit do you want to track?</div>',
        unsafe_allow_html=True,
    )
    name = st.text_input(
        "Habit name", placeholder="e.g. Drink Water", label_visibility="collapsed"
    )
    if st.button("Start tracking"):
        if name.strip():
            data["habit_name"] = name.strip()[:40]
            save_data(data)
            st.rerun()
        else:
            st.warning("Please enter a habit name.")


def render_main(data: dict, today: date) -> None:
    done_today = is_done_today(data, today)

    st.markdown(f'<div class="habit-name">{data["habit_name"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="streak-number">{data["current_streak"]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="streak-label">day streak</div>', unsafe_allow_html=True)

    message = get_motivation(data["habit_name"], data["current_streak"])
    st.markdown(f'<div class="message">{message}</div>', unsafe_allow_html=True)

    label = "Done for today ✓" if done_today else "Done Today"
    if st.button(label, disabled=done_today):
        mark_done(data, today)
        save_data(data)
        st.rerun()

    st.markdown(
        f'<div class="best">Best streak: {data["best_streak"]} days</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Habit Streak Counter", page_icon="🔥", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    today = date.today()
    data = load_data()
    if apply_streak_reset(data, today):
        save_data(data)

    if not data["habit_name"]:
        render_setup(data)
    else:
        render_main(data, today)


if __name__ == "__main__":
    main()
