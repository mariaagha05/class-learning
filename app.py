"""
AI Expense Analyzer
-------------------
Streamlit app that lets you log expenses (with descriptions), visualises them
with charts and trend views, and uses a Groq-hosted LLM to find spending
patterns and recommend strategies.

Run:
    pip install -r requirements.txt
    streamlit run app.py

API key (any ONE of these):
    1) Paste it in the sidebar
    2) export GROQ_API_KEY="gsk_..."            (Linux/Mac)
       setx GROQ_API_KEY "gsk_..."              (Windows)
    3) .streamlit/secrets.toml  ->  GROQ_API_KEY = "gsk_..."
"""

import json
import os
import random
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Expense Analyzer", page_icon="💸", layout="wide")

DATA_FILE = "expenses.csv"
COLUMNS = ["date", "amount", "category", "description"]
CATEGORIES = [
    "Food & Dining", "Groceries", "Transport", "Rent & Housing", "Utilities",
    "Shopping", "Entertainment", "Health", "Education", "Travel",
    "Subscriptions", "Personal Care", "Gifts & Donations", "Other",
]
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]
STATUS_COLORS = {"High spending": "#e74c3c", "Normal": "#2ecc71", "No spending": "#bdc3c7"}


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            return clean_df(df)
        except Exception:
            pass
    return pd.DataFrame({
        "date": pd.Series(dtype="datetime64[ns]"),
        "amount": pd.Series(dtype="float"),
        "category": pd.Series(dtype="object"),
        "description": pd.Series(dtype="object"),
    })


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["category"] = df["category"].fillna("Other").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df = df.dropna(subset=["date", "amount"])
    df = df[df["amount"] > 0]
    return df.sort_values("date").reset_index(drop=True)


def save_data(df: pd.DataFrame) -> None:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(DATA_FILE, index=False)


def fmt(x: float, cur: str) -> str:
    return f"{cur} {x:,.0f}"


def generate_sample_data(days: int = 120) -> pd.DataFrame:
    random.seed(7)
    rows = []
    today = date.today()
    items = {
        "Food & Dining": ["Lunch at cafe", "Dinner with friends", "Pizza delivery", "Biryani takeaway", "Coffee"],
        "Groceries": ["Weekly groceries", "Fruits and vegetables", "Milk and bread"],
        "Transport": ["Uber ride", "Fuel", "Bus pass", "Careem ride"],
        "Shopping": ["New shoes", "Online order", "Clothes sale", "Headphones"],
        "Entertainment": ["Movie tickets", "Gaming purchase", "Concert"],
        "Utilities": ["Electricity bill", "Internet bill", "Gas bill"],
        "Subscriptions": ["Netflix", "Spotify", "Cloud storage"],
        "Health": ["Pharmacy", "Doctor visit"],
    }
    price = {
        "Food & Dining": (300, 2500), "Groceries": (800, 6000), "Transport": (150, 3000),
        "Shopping": (1500, 15000), "Entertainment": (500, 5000), "Utilities": (2000, 9000),
        "Subscriptions": (300, 1500), "Health": (500, 6000),
    }
    for i in range(days):
        d = today - timedelta(days=days - i)
        n = random.choice([0, 1, 1, 2, 2, 3])
        if d.weekday() >= 5:          # weekends = more spending
            n += random.choice([1, 2])
        for _ in range(n):
            cat = random.choices(
                list(items.keys()), weights=[30, 15, 15, 12, 10, 3, 5, 5])[0]
            lo, hi = price[cat]
            rows.append([d, round(random.uniform(lo, hi), -1), cat, random.choice(items[cat])])
        if d.day == 1:
            rows.append([d, 45000, "Rent & Housing", "Monthly rent"])
    return clean_df(pd.DataFrame(rows, columns=COLUMNS))


# ----------------------------------------------------------------------------
# Groq / LLM helpers
# ----------------------------------------------------------------------------
def get_api_key(sidebar_value: str) -> str:
    if sidebar_value.strip():
        return sidebar_value.strip()
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    try:
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        return ""


def ask_groq(api_key: str, model: str, system: str, user: str, temperature: float = 0.4) -> str:
    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=2000,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq API error: {e}"


def ai_categorize(api_key: str, model: str, description: str) -> str:
    system = (
        "You classify expense descriptions into exactly one category. "
        f"Allowed categories: {', '.join(CATEGORIES)}. "
        "Reply with ONLY the category name, nothing else."
    )
    answer = ask_groq(api_key, model, system, description, temperature=0).strip()
    for c in CATEGORIES:
        if c.lower() in answer.lower():
            return c
    return "Other"


def build_summary(df: pd.DataFrame, cur: str, threshold: float) -> dict:
    """Compact, LLM-friendly summary of the spending data."""
    d = df.copy()
    d["month"] = d["date"].dt.to_period("M").astype(str)
    d["weekday"] = d["date"].dt.day_name()
    daily = d.groupby(d["date"].dt.normalize())["amount"].sum()
    days_span = max((d["date"].max() - d["date"].min()).days + 1, 1)

    high_days = daily[daily > threshold].sort_values(ascending=False).head(8)
    high_days_info = []
    for day, total in high_days.items():
        items = d[d["date"].dt.normalize() == day][["category", "description", "amount"]]
        high_days_info.append({
            "date": day.strftime("%Y-%m-%d"),
            "total": round(float(total), 2),
            "items": items.head(5).to_dict("records"),
        })

    return {
        "currency": cur,
        "period": f"{d['date'].min().date()} to {d['date'].max().date()}",
        "total_spent": round(float(d["amount"].sum()), 2),
        "num_transactions": int(len(d)),
        "avg_per_day": round(float(d["amount"].sum() / days_span), 2),
        "avg_per_transaction": round(float(d["amount"].mean()), 2),
        "by_category": d.groupby("category")["amount"].sum().round(2).sort_values(ascending=False).to_dict(),
        "by_month": d.groupby("month")["amount"].sum().round(2).to_dict(),
        "avg_by_weekday": d.groupby("weekday")["amount"].sum().div(
            d.groupby("weekday")["date"].nunique()).round(2).to_dict(),
        "high_spend_threshold_per_day": round(float(threshold), 2),
        "high_spend_days": high_days_info,
        "top_10_expenses": d.nlargest(10, "amount")[["date", "category", "description", "amount"]]
                            .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_dict("records"),
        "recent_30_expenses": d.tail(30)[["date", "category", "description", "amount"]]
                               .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_dict("records"),
    }


SYSTEM_PROMPT = """You are an expert personal-finance coach and data analyst.
You receive a JSON summary of a person's expenses (with descriptions) and must:
1. Identify spending PATTERNS (categories, weekdays, months, impulse/recurring behaviour,
   what the descriptions reveal, e.g. frequent food delivery, subscriptions, late-month spikes).
2. Explain WHEN they overspend and what triggered those high-spend days.
3. Point out any wasteful, duplicate or avoidable spending.
4. Give concrete, prioritised STRATEGIES with estimated monthly savings using the user's
   currency and real numbers from the data (e.g. 50/30/20 rule, category caps, 24-hour rule,
   no-spend days, meal planning, subscription audit, envelope budgeting).
5. Suggest a realistic monthly budget per category.
Be specific, friendly and concise. Use markdown with clear headings and bullet points.
Never invent data that is not in the summary."""


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = load_data()
if "ai_report" not in st.session_state:
    st.session_state.ai_report = ""

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    key_input = st.text_input("Groq API Key", type="password", placeholder="gsk_...",
                              help="Or set GROQ_API_KEY env var / Streamlit secrets.")
    api_key = get_api_key(key_input)
    model = st.selectbox("Groq model", GROQ_MODELS)
    currency = st.text_input("Currency symbol", value="Rs.")
    monthly_budget = st.number_input("Monthly budget (0 = off)", min_value=0.0, value=0.0, step=1000.0)

    st.markdown("---")
    st.subheader("🔎 Filters")
    df_all = st.session_state.df
    if not df_all.empty:
        min_d, max_d = df_all["date"].min().date(), df_all["date"].max().date()
        date_range = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        cats_available = sorted(df_all["category"].unique())
        sel_cats = st.multiselect("Categories", cats_available, default=cats_available)
    else:
        date_range, sel_cats = None, []

    sensitivity = st.slider(
        "High-spend sensitivity", 0.0, 3.0, 1.0, 0.25,
        help="A day is 'high spending' if it exceeds average daily spend + (this value × std deviation). "
             "Lower = stricter.")

    st.markdown("---")
    if st.button("🧪 Load sample data"):
        st.session_state.df = generate_sample_data()
        save_data(st.session_state.df)
        st.rerun()

# Apply filters
df = st.session_state.df.copy()
if not df.empty and date_range is not None and len(date_range) == 2:
    df = df[(df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])]
    df = df[df["category"].isin(sel_cats)]

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("💸 AI Expense Analyzer")
st.caption("Log expenses, see where your money goes, and get AI-powered strategies to spend smarter.")

tab_add, tab_dash, tab_trend, tab_ai, tab_data = st.tabs(
    ["➕ Add Expense", "📊 Dashboard", "📈 Trends", "🤖 AI Insights", "🗂️ Manage Data"])

# ----------------------------------------------------------------------------
# TAB 1 - Add expense
# ----------------------------------------------------------------------------
with tab_add:
    st.subheader("Add a new expense")
    with st.form("add_expense", clear_on_submit=True):
        c1, c2 = st.columns(2)
        e_date = c1.date_input("Date", value=date.today())
        e_amount = c2.number_input(f"Amount ({currency})", min_value=0.0, step=10.0, format="%.2f")
        c3, c4 = st.columns(2)
        e_cat = c3.selectbox("Category", ["✨ Auto (AI decides)"] + CATEGORIES)
        e_desc = c4.text_input("Description", placeholder="e.g. Dinner at KFC with friends")
        submitted = st.form_submit_button("Add expense", type="primary")

    if submitted:
        if e_amount <= 0:
            st.error("Please enter an amount greater than 0.")
        else:
            category = e_cat
            if e_cat.startswith("✨"):
                if api_key and e_desc.strip():
                    with st.spinner("AI is categorising..."):
                        category = ai_categorize(api_key, model, e_desc)
                else:
                    category = "Other"
                    st.info("Add an API key and a description for AI auto-categorisation. Saved as 'Other'.")
            new_row = pd.DataFrame([[pd.to_datetime(e_date), e_amount, category, e_desc.strip()]], columns=COLUMNS)
            st.session_state.df = clean_df(pd.concat([st.session_state.df, new_row], ignore_index=True))
            save_data(st.session_state.df)
            st.success(f"Added {fmt(e_amount, currency)} under **{category}**.")
            st.rerun() if False else None

    st.markdown("##### Latest entries")
    if st.session_state.df.empty:
        st.info("No expenses yet. Add one above or load sample data from the sidebar.")
    else:
        recent = st.session_state.df.sort_values("date", ascending=False).head(8).copy()
        recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(recent, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# Shared computations
# ----------------------------------------------------------------------------
daily = pd.DataFrame()
threshold = 0.0
if not df.empty:
    daily_s = df.groupby(df["date"].dt.normalize())["amount"].sum()
    full_range = pd.date_range(daily_s.index.min(), daily_s.index.max())
    daily = daily_s.reindex(full_range, fill_value=0).rename_axis("date").reset_index()
    spend_days = daily.loc[daily["amount"] > 0, "amount"]
    threshold = float(spend_days.mean() + sensitivity * (spend_days.std() if len(spend_days) > 1 else 0))

    def _status(a):
        if a == 0:
            return "No spending"
        return "High spending" if a > threshold else "Normal"

    daily["status"] = daily["amount"].apply(_status)
    daily["rolling_7d"] = daily["amount"].rolling(7, min_periods=1).mean()

# ----------------------------------------------------------------------------
# TAB 2 - Dashboard
# ----------------------------------------------------------------------------
with tab_dash:
    if df.empty:
        st.info("No data to show yet.")
    else:
        total = df["amount"].sum()
        span = max((df["date"].max() - df["date"].min()).days + 1, 1)
        top_cat = df.groupby("category")["amount"].sum().idxmax()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total spent", fmt(total, currency))
        m2.metric("Avg / day", fmt(total / span, currency))
        m3.metric("Transactions", f"{len(df):,}")
        m4.metric("Top category", top_cat)

        if monthly_budget > 0:
            this_month = date.today().strftime("%Y-%m")
            spent_m = df[df["date"].dt.strftime("%Y-%m") == this_month]["amount"].sum()
            pct = spent_m / monthly_budget
            st.progress(min(pct, 1.0),
                        text=f"This month: {fmt(spent_m, currency)} of {fmt(monthly_budget, currency)} ({pct:.0%})")
            if pct > 1:
                st.error("You have exceeded this month's budget!")
            elif pct > 0.8:
                st.warning("You've used over 80% of this month's budget.")

        left, right = st.columns(2)
        by_cat = df.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)

        with left:
            st.markdown("##### Spending by category (pie)")
            fig = px.pie(by_cat, names="category", values="amount", hole=0.45)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("##### Spending by category (bar)")
            fig = px.bar(by_cat, x="amount", y="category", orientation="h", color="category", text_auto=".2s")
            fig.update_layout(showlegend=False, yaxis=dict(autorange="reversed"), margin=dict(t=10, b=10),
                              xaxis_title=f"Amount ({currency})", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Monthly spending by category")
        mdf = df.copy()
        mdf["month"] = mdf["date"].dt.to_period("M").astype(str)
        mc = mdf.groupby(["month", "category"], as_index=False)["amount"].sum()
        fig = px.bar(mc, x="month", y="amount", color="category", barmode="stack")
        fig.update_layout(xaxis_title="", yaxis_title=f"Amount ({currency})", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Biggest single expenses")
        big = df.nlargest(10, "amount").copy()
        big["date"] = big["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(big, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# TAB 3 - Trends
# ----------------------------------------------------------------------------
with tab_trend:
    if df.empty:
        st.info("No data to show yet.")
    else:
        st.markdown("##### Daily spending – when did you overspend?")
        st.caption(f"Red = high-spending day (above {fmt(threshold, currency)}), "
                   "green = normal, grey = no spending. The line is the 7-day moving average.")
        fig = px.bar(daily, x="date", y="amount", color="status", color_discrete_map=STATUS_COLORS,
                     category_orders={"status": ["High spending", "Normal", "No spending"]})
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["rolling_7d"], mode="lines",
                                 name="7-day average", line=dict(color="#2c3e50", width=2)))
        fig.add_hline(y=threshold, line_dash="dash", line_color="#e74c3c",
                      annotation_text="High-spend threshold", annotation_position="top left")
        fig.update_layout(xaxis_title="", yaxis_title=f"Amount ({currency})", legend_title="", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

        n_high = int((daily["status"] == "High spending").sum())
        n_norm = int((daily["status"] == "Normal").sum())
        n_none = int((daily["status"] == "No spending").sum())
        s1, s2, s3 = st.columns(3)
        s1.metric("🔴 High-spend days", n_high)
        s2.metric("🟢 Normal days", n_norm)
        s3.metric("⚪ No-spend days", n_none)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Average spend by weekday")
            wd = daily.copy()
            wd["weekday"] = wd["date"].dt.day_name()
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            wd = wd.groupby("weekday", as_index=False)["amount"].mean()
            wd["weekday"] = pd.Categorical(wd["weekday"], categories=order, ordered=True)
            wd = wd.sort_values("weekday")
            fig = px.bar(wd, x="weekday", y="amount", color="amount", color_continuous_scale="RdYlGn_r")
            fig.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title=f"Avg ({currency})",
                              margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("##### Weekly totals")
            wk = daily.set_index("date")["amount"].resample("W").sum().reset_index()
            wk_avg = wk["amount"].mean()
            wk["status"] = wk["amount"].apply(lambda a: "High spending" if a > wk_avg * 1.25
                                              else ("Normal" if a > 0 else "No spending"))
            fig = px.bar(wk, x="date", y="amount", color="status", color_discrete_map=STATUS_COLORS)
            fig.add_hline(y=wk_avg, line_dash="dot", annotation_text="Weekly average")
            fig.update_layout(xaxis_title="", yaxis_title=f"Amount ({currency})", showlegend=False,
                              margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### 🔴 Your highest-spend days and what caused them")
        hi = daily[daily["status"] == "High spending"].sort_values("amount", ascending=False).head(10)
        if hi.empty:
            st.success("No days exceeded the threshold. Great control!")
        else:
            rows = []
            for _, r in hi.iterrows():
                day_items = df[df["date"].dt.normalize() == r["date"]]
                what = "; ".join(f"{x.description or x.category} ({x.amount:,.0f})"
                                 for x in day_items.itertuples())
                rows.append({"Date": r["date"].strftime("%a, %d %b %Y"),
                             "Total": fmt(r["amount"], currency), "What you bought": what})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("##### 🟢 Your calmest weeks")
        calm = wk[wk["amount"] > 0].nsmallest(3, "amount")
        for _, r in calm.iterrows():
            st.write(f"• Week ending **{r['date'].strftime('%d %b %Y')}** – {fmt(r['amount'], currency)}")

# ----------------------------------------------------------------------------
# TAB 4 - AI insights
# ----------------------------------------------------------------------------
with tab_ai:
    st.subheader("🤖 AI pattern analysis & strategies")
    if df.empty:
        st.info("Add some expenses first.")
    elif not api_key:
        st.warning("Enter your Groq API key in the sidebar (or set GROQ_API_KEY) to use AI insights.")
    else:
        focus = st.text_input("Optional focus (e.g. 'I want to save 20% a month' or 'reduce food spending')")
        if st.button("Analyze my expenses", type="primary"):
            summary = build_summary(df, currency, threshold)
            user_msg = "Here is my expense summary as JSON:\n" + json.dumps(summary, default=str)
            if focus.strip():
                user_msg += f"\n\nMy goal / focus: {focus.strip()}"
            with st.spinner("Groq is analysing your spending..."):
                st.session_state.ai_report = ask_groq(api_key, model, SYSTEM_PROMPT, user_msg)

        if st.session_state.ai_report:
            st.markdown(st.session_state.ai_report)
            st.download_button("Download report", st.session_state.ai_report,
                               file_name="expense_report.md", mime="text/markdown")

        st.markdown("---")
        st.markdown("##### 💬 Ask a question about your spending")
        q = st.text_input("e.g. 'Why was my spending so high in the last week?' or 'Where can I cut Rs. 10,000?'")
        if st.button("Ask") and q.strip():
            summary = build_summary(df, currency, threshold)
            msg = ("Expense summary JSON:\n" + json.dumps(summary, default=str) +
                   f"\n\nQuestion: {q.strip()}\nAnswer using only this data.")
            with st.spinner("Thinking..."):
                st.markdown(ask_groq(api_key, model, SYSTEM_PROMPT, msg))

# ----------------------------------------------------------------------------
# TAB 5 - Manage data
# ----------------------------------------------------------------------------
with tab_data:
    st.subheader("Edit, delete, import or export")
    if st.session_state.df.empty:
        st.info("No data yet.")
    else:
        st.caption("Edit cells directly, add rows at the bottom, or select rows and press Delete. "
                   "Click **Save changes** when done.")
        edited = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "amount": st.column_config.NumberColumn("Amount", min_value=0.0, format="%.2f"),
                "category": st.column_config.SelectboxColumn("Category", options=CATEGORIES),
                "description": st.column_config.TextColumn("Description"),
            },
            key="editor",
        )
        b1, b2, _ = st.columns([1, 1, 3])
        if b1.button("💾 Save changes", type="primary"):
            st.session_state.df = clean_df(pd.DataFrame(edited))
            save_data(st.session_state.df)
            st.success("Saved.")
            st.rerun()
        b2.download_button("⬇️ Export CSV", st.session_state.df.to_csv(index=False),
                           file_name="expenses.csv", mime="text/csv")

    st.markdown("---")
    up = st.file_uploader("Import expenses from CSV (columns: date, amount, category, description)", type="csv")
    if up is not None and st.button("Import file"):
        try:
            imported = clean_df(pd.read_csv(up))
            st.session_state.df = clean_df(pd.concat([st.session_state.df, imported], ignore_index=True))
            save_data(st.session_state.df)
            st.success(f"Imported {len(imported)} rows.")
            st.rerun()
        except Exception as e:
            st.error(f"Could not import: {e}")

    with st.expander("⚠️ Danger zone"):
        if st.button("Delete ALL expenses"):
            st.session_state.df = clean_df(pd.DataFrame(columns=COLUMNS))
            save_data(st.session_state.df)
            st.session_state.ai_report = ""
            st.rerun()
