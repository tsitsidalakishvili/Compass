import base64
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Political Compass Party Matcher",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESPONSE_LABELS = [
    "Strongly disagree",
    "Disagree",
    "Neutral / unsure",
    "Agree",
    "Strongly agree",
]

RESPONSE_SCORES = {
    "Strongly disagree": -2,
    "Disagree": -1,
    "Neutral / unsure": 0,
    "Agree": 1,
    "Strongly agree": 2,
}

# Anchor points roughly map common ideological families on a -10..10 grid.
DATA_PATH = Path(__file__).parent / "data" / "policy_topics.json"


def load_policy_topics() -> List[Dict]:
    try:
        with DATA_PATH.open("r", encoding="utf-8") as source:
            return json.load(source)
    except FileNotFoundError:
        return []


LOGO_PATH = Path(__file__).parent / "logos"


@lru_cache(maxsize=None)
def load_logo_data(filename: str | None) -> str | None:
    if not filename:
        return None
    path = LOGO_PATH / filename
    if not path.exists():
        return None
    mime = "image/" + path.suffix.lower().replace(".", "")
    with path.open("rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def get_party_logo(name: str) -> str | None:
    for party in PARTY_PROFILES:
        if party["name"].lower() == name.lower():
            return party.get("logo")
    return None


def get_logo_path(filename: str | None) -> str | None:
    if not filename:
        return None
    path = LOGO_PATH / filename
    return str(path) if path.exists() else None


CHATBOT_LOGO_PATH = get_logo_path("bot.png")


PARTY_PROFILES = [
    {
        "name": "Girchi",
        "econ_anchor": 7,
        "soc_anchor": -6,
        "summary": "Libertarian, pro-market platform focused on deregulation and civil liberties.",
        "priority": "Low taxes, minimal state, ending victimless-crime prosecutions.",
        "color": "#f39c12",
        "logo": "girchi.jpg",
    },
    {
        "name": "Coalition for Change - Ahali, Droa, Girchi-More Freedom",
        "econ_anchor": -1,
        "soc_anchor": -2,
        "summary": "Liberal reformers pushing for transparent institutions and local empowerment.",
        "priority": "Judicial reform, decentralization, accountable governance.",
        "color": "#1abc9c",
        "logo": "droa.jpg",
    },
    {
        "name": "Strong Georgia–Lelo, for People, for Liberty",
        "econ_anchor": -4,
        "soc_anchor": -2,
        "summary": "Social-liberal bloc combining market dynamism with welfare safeguards.",
        "priority": "Public service modernization, anti-corruption, inclusive development.",
        "color": "#2980b9",
        "logo": "lelo.jpg",
    },
    {
        "name": "Georgian Dream",
        "econ_anchor": 3,
        "soc_anchor": 3,
        "summary": "Incumbent coalition favoring gradual reform, conservative social order.",
        "priority": "Fiscal stability, service expansion, traditional institutions.",
        "color": "#8e44ad",
        "logo": "georgian dream.jpg",
    },
    {
        "name": "For Georgia",
        "econ_anchor": 1,
        "soc_anchor": 1,
        "summary": "Technocratic centrists emphasizing anti-corruption and balanced governance.",
        "priority": "Independent institutions, anti-corruption drive, EU alignment.",
        "color": "#c0392b",
        "logo": "for georgia.png",
    },
]

POLICY_TOPICS: List[Dict] = load_policy_topics()

MAX_RESPONSE_VALUE = 2
AXES = ("economic", "social")
AXIS_NORMALIZERS = {
    axis: max(
        1,
        sum(
            abs(topic["axes"].get(axis, 0)) * MAX_RESPONSE_VALUE
            for topic in POLICY_TOPICS
        ),
    )
    for axis in AXES
}


def score_responses(raw_scores: Dict[str, int]) -> Dict[str, float]:
    """Convert accumulated axis scores to a -10..10 range."""
    normalized = {}
    for axis in AXES:
        normalized[axis] = 10 * raw_scores[axis] / AXIS_NORMALIZERS[axis]
    return normalized


def classify_quadrant(economic_position: float, social_position: float) -> str:
    if economic_position < 0 and social_position < 0:
        return "Progressive Libertarian"
    if economic_position < 0 and social_position >= 0:
        return "Progressive Communitarian"
    if economic_position >= 0 and social_position < 0:
        return "Market Libertarian"
    return "Market Communitarian"


def rank_parties(economic_position: float, social_position: float):
    def distance(party: Dict) -> float:
        return math.sqrt(
            (economic_position - party["econ_anchor"]) ** 2
            + (social_position - party["soc_anchor"]) ** 2
        )

    ranked = []
    for profile in PARTY_PROFILES:
        dist = distance(profile)
        ranked.append({**profile, "distance": dist})
    ranked.sort(key=lambda item: item["distance"])
    return ranked


def build_compass_chart(economic_position: float, social_position: float):
    fig = go.Figure()

    fig.add_hline(y=0, line_width=1, line_color="#bbbbbb")
    fig.add_vline(x=0, line_width=1, line_color="#bbbbbb")

    fig.add_trace(
        go.Scatter(
            x=[party["econ_anchor"] for party in PARTY_PROFILES],
            y=[party["soc_anchor"] for party in PARTY_PROFILES],
            mode="markers",
            marker=dict(
                size=10,
                color=[party["color"] for party in PARTY_PROFILES],
                symbol="diamond",
                line=dict(color="#222", width=1),
            ),
            hovertemplate="<b>%{customdata}</b><br>Economic %{x:.1f}<br>Social %{y:.1f}<extra></extra>",
            customdata=[party["name"] for party in PARTY_PROFILES],
            name="Parties",
        )
    )

    for party in PARTY_PROFILES:
        logo_src = load_logo_data(party.get("logo"))
        if not logo_src:
            continue
        fig.add_layout_image(
            dict(
                source=logo_src,
                x=party["econ_anchor"],
                y=party["soc_anchor"],
                sizex=1.8,
                sizey=1.8,
                xanchor="center",
                yanchor="middle",
                xref="x",
                yref="y",
                layer="above",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[economic_position],
            y=[social_position],
            mode="markers+text",
            text=["You"],
            textposition="top center",
            marker=dict(size=16, color="#ff4b4b"),
            name="You",
        )
    )

    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=[-10, 10], title="Economic (Left  ⟶  Right)"),
        yaxis=dict(range=[-10, 10], title="Social (Libertarian  ⟶  Authoritarian)"),
        showlegend=False,
        plot_bgcolor="#fbfbfb",
    )

    return fig


def move_topic(delta: int):
    """Shift the focused topic/question while keeping the index in range."""
    if "topic_picker" not in st.session_state:
        st.session_state["topic_picker"] = 0
    new_idx = st.session_state["topic_picker"] + delta
    st.session_state["topic_picker"] = min(
        len(POLICY_TOPICS) - 1, max(0, new_idx)
    )


def answer_with_program_facts(question: str) -> str:
    """Very lightweight retrieval over the in-app program snippets."""
    if not question.strip():
        return "Please enter a question about a party or policy topic."

    query = question.lower()
    hits = []

    for topic in POLICY_TOPICS:
        topic_score = 1 if topic["theme"].lower() in query else 0
        for party in topic["parties"]:
            text = party["position"]
            score = topic_score
            if party["name"].lower() in query:
                score += 2
            if any(keyword in text.lower() for keyword in query.split()):
                score += 1
            if score > 0:
                hits.append(
                    {
                        "score": score,
                        "party": party["name"],
                        "topic": topic["theme"],
                        "question": topic["question"],
                        "snippet": text,
                    }
                )

    if not hits:
        return (
            "I couldn't find a matching snippet in the loaded party programs. "
            "Try mentioning a specific party name or topic."
        )

    hits.sort(key=lambda item: item["score"], reverse=True)
    top_hits = hits[:2]
    response_lines = []
    for hit in top_hits:
        response_lines.append(
            f"**{hit['party']}** on *{hit['topic']}* — {hit['snippet']}"
        )
    if len(hits) > len(top_hits):
        response_lines.append(
            f"...and {len(hits) - len(top_hits)} more snippet(s) found. "
            "Ask again with more specific keywords to narrow it down."
        )
    return "\n\n".join(response_lines)


def summarize_text(text: str, limit: int = 220) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    truncated = stripped[:limit].rsplit(" ", 1)[0]
    return f"{truncated}…"


def main():
    st.title("Political Compass Party Matcher")
    st.caption(
        "Start by browsing real party program text, then record how much you agree. "
        "Your live position on the compass reflects each answer."
    )

    if not POLICY_TOPICS:
        st.error(
            "No policy topics found. Make sure `data/policy_topics.json` exists "
            "and restart the app."
        )
        return

    if "answers" not in st.session_state:
        st.session_state["answers"] = [RESPONSE_LABELS[2]] * len(POLICY_TOPICS)
    elif len(st.session_state["answers"]) != len(POLICY_TOPICS):
        existing = st.session_state["answers"]
        resized = (existing + [RESPONSE_LABELS[2]] * len(POLICY_TOPICS))[: len(POLICY_TOPICS)]
        st.session_state["answers"] = resized
    if "topic_picker" not in st.session_state:
        st.session_state["topic_picker"] = 0

    st.subheader("Question & stance")
    topic_idx = st.selectbox(
        "Choose a topic",
        options=list(range(len(POLICY_TOPICS))),
        format_func=lambda idx: POLICY_TOPICS[idx]["theme"],
        index=st.session_state["topic_picker"],
    )
    if topic_idx != st.session_state["topic_picker"]:
        st.session_state["topic_picker"] = topic_idx
    current_topic = POLICY_TOPICS[st.session_state["topic_picker"]]

    current_answer = st.session_state["answers"][st.session_state["topic_picker"]]
    st.markdown(
        f"<div style='font-size:1.2rem;font-weight:600'>{current_topic['prompt']}</div>",
        unsafe_allow_html=True,
    )
    stance_choice = st.radio(
        "Select your stance",
        RESPONSE_LABELS,
        index=RESPONSE_LABELS.index(current_answer),
        horizontal=True,
        help="Selections default to neutral until you change them.",
        key=f"stance_{st.session_state['topic_picker']}",
    )
    st.session_state["answers"][st.session_state["topic_picker"]] = stance_choice

    nav_prev, nav_next = st.columns(2)
    nav_prev.button(
        "← Prev topic",
        use_container_width=True,
        disabled=st.session_state["topic_picker"] == 0,
        on_click=move_topic,
        kwargs={"delta": -1},
    )
    nav_next.button(
        "Next topic →",
        use_container_width=True,
        disabled=st.session_state["topic_picker"] == len(POLICY_TOPICS) - 1,
        on_click=move_topic,
        kwargs={"delta": 1},
    )

    answered_count = sum(
        response != RESPONSE_LABELS[2] for response in st.session_state["answers"]
    )
    st.progress(
        answered_count / len(POLICY_TOPICS),
        text=f"{answered_count}/{len(POLICY_TOPICS)} answered",
    )
    st.caption("Complete every topic to unlock the most accurate compass reading.")


    raw_scores = {axis: 0 for axis in AXES}
    for response, policy_topic in zip(st.session_state["answers"], POLICY_TOPICS):
        strength = RESPONSE_SCORES[response]
        for axis, weight in policy_topic["axes"].items():
            raw_scores[axis] += strength * weight

    normalized = score_responses(raw_scores)
    quadrant = classify_quadrant(normalized["economic"], normalized["social"])
    party_matches = rank_parties(normalized["economic"], normalized["social"])
    top_party = party_matches[0]

    st.sidebar.header("How scoring works", divider="gray")
    st.sidebar.write(
        "- Economic axis: redistribution & markets\n"
        "- Social axis: authority vs. liberty\n"
        "Agreeing with a prompt nudges you toward the direction encoded for that statement."
    )
    st.sidebar.plotly_chart(
        build_compass_chart(
            normalized["economic"], normalized["social"]
        ),
        use_container_width=True,
    )

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    bot_expander_label = "Party-program chatbot"
    bot_logo_data = load_logo_data("bot.png")
    if bot_logo_data:
        bot_expander_label = (
            f"<span style='display:flex;align-items:center;font-weight:700;font-size:1.05rem;'>"
            f"<img src='{bot_logo_data}' width='32' style='margin-right:0.5rem;'/>Chatbot"
            f"</span>"
        )

    with st.sidebar.expander(bot_expander_label, expanded=False):
        st.caption(
            "Ask about a party or topic. Answers quote the program text above."
        )
        if st.session_state["chat_messages"]:
            for message in st.session_state["chat_messages"]:
                role_label = "You" if message["role"] == "user" else "Assistant"
                st.markdown(f"**{role_label}:** {message['content']}")

        with st.form("sidebar_chat_form", clear_on_submit=True):
            user_prompt = st.text_input(
                "Your question",
                placeholder="e.g. What does Girchi promise about public service reform?",
                key="sidebar_chat_input",
            )
            submitted = st.form_submit_button("Send")

        if submitted and user_prompt:
            st.session_state["chat_messages"].append(
                {"role": "user", "content": user_prompt}
            )
            response = answer_with_program_facts(user_prompt)
            st.session_state["chat_messages"].append(
                {"role": "assistant", "content": response}
            )
            st.rerun()

    current_topic = POLICY_TOPICS[st.session_state["topic_picker"]]

    st.subheader(f"Party views on {current_topic['theme']}")
    st.markdown(f"**{current_topic['question']}**")
    st.caption(f"[Source link]({current_topic['source']})")
    party_names = [party["name"] for party in current_topic["parties"]]
    filter_key = f"party_filter_{current_topic.get('slug', st.session_state['topic_picker'])}"
    selected_parties = st.multiselect(
        "Choose which parties to display",
        party_names,
        default=party_names,
        key=filter_key,
    )

    party_count = len(selected_parties)
    if party_count == 0:
        st.info("Select at least one party to see their program summary.")
    else:
        filtered_parties = [
            party for party in current_topic["parties"] if party["name"] in selected_parties
        ]
        cols = st.columns(min(5, len(filtered_parties)))
        for idx, party in enumerate(filtered_parties):
            with cols[idx % len(cols)]:
                logo_name = get_party_logo(party["name"])
                logo_src = load_logo_data(logo_name) if logo_name else None
                logo_path = get_logo_path(logo_name)
                if logo_src:
                    st.image(logo_path or logo_src, width=120, caption=party["name"])
                else:
                    st.markdown(f"**{party['name']}**")
                st.write(summarize_text(party["position"]))
                with st.expander("Full statement"):
                    st.write(party["position"])

if __name__ == "__main__":
    main()

