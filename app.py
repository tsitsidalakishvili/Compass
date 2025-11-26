import json
import math
from pathlib import Path
from typing import Dict, List

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Political Compass Party Matcher",
    page_icon="🧭",
    layout="wide",
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


PARTY_PROFILES = [
    {
        "name": "Progressive Greens",
        "econ_anchor": -7,
        "soc_anchor": -7,
        "summary": "Deep redistribution, global cooperation, and expansive civil liberties.",
        "priority": "Climate justice, social safety nets, participatory democracy.",
        "color": "#2ca02c",
    },
    {
        "name": "Social Democrats",
        "econ_anchor": -5,
        "soc_anchor": -1,
        "summary": "Tax-funded welfare state with modest institutional guardrails.",
        "priority": "Universal services, worker power, pro-democracy alliances.",
        "color": "#1f77b4",
    },
    {
        "name": "Liberal Internationalists",
        "econ_anchor": -1,
        "soc_anchor": -4,
        "summary": "Mixed economy pragmatists who emphasize rights and open societies.",
        "priority": "Regulated markets, civil rights, international institutions.",
        "color": "#ff7f0e",
    },
    {
        "name": "Market Libertarians",
        "econ_anchor": 7,
        "soc_anchor": -6,
        "summary": "Minimal state in both markets and personal life.",
        "priority": "Low taxes, deregulation, maximal personal freedom.",
        "color": "#d62728",
    },
    {
        "name": "Center Conservatives",
        "econ_anchor": 4,
        "soc_anchor": 3,
        "summary": "Pro-market with an emphasis on order and traditional institutions.",
        "priority": "Business-friendly reforms, cautious social change, strong defense.",
        "color": "#9467bd",
    },
    {
        "name": "National Populists",
        "econ_anchor": -1,
        "soc_anchor": 7,
        "summary": "Economically mixed but favor a strong cultural state.",
        "priority": "Borders, national industry, majoritarian values.",
        "color": "#8c564b",
    },
    {
        "name": "Technocratic Centrists",
        "econ_anchor": 2,
        "soc_anchor": -1,
        "summary": "Data-first moderates balancing markets with targeted regulation.",
        "priority": "Evidence-based policy, innovation, limited but effective government.",
        "color": "#17becf",
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

    col_programs, col_compass = st.columns((1.05, 1))

    with col_programs:
        st.subheader("Party programs by topic")
        topic_idx = st.selectbox(
            "Choose a topic",
            options=list(range(len(POLICY_TOPICS))),
            format_func=lambda idx: POLICY_TOPICS[idx]["theme"],
            key="topic_picker",
        )
        topic = POLICY_TOPICS[topic_idx]
        st.markdown(f"**Question:** {topic['question']}")
        st.caption(f"Source: {topic['source']}")
        party_options = [party["name"] for party in topic["parties"]]
        selected_parties = st.multiselect(
            "Select parties to compare",
            options=party_options,
            default=party_options,
            key=f"party_selector_topic_{topic_idx}",
        )
        if not selected_parties:
            st.info("Pick at least one party above to read their detailed position.")
        else:
            for party in topic["parties"]:
                if party["name"] in selected_parties:
                    with st.expander(party["name"], expanded=True):
                        st.write(party["position"])

    with col_compass:
        st.subheader(
            "Compass view",
            help="The chart expands to fill the page so you can focus on how the red dot moves.",
        )
        compass_chart_placeholder = st.empty()
        compass_caption_placeholder = st.empty()

    st.sidebar.header("How scoring works")
    st.sidebar.write(
        "- Economic axis: redistribution & markets\n"
        "- Social axis: authority vs. liberty\n"
        "Agreeing with a prompt nudges you toward the direction encoded for that statement."
    )
   
    if "answers" not in st.session_state:
        st.session_state["answers"] = [RESPONSE_LABELS[2]] * len(POLICY_TOPICS)
    elif len(st.session_state["answers"]) != len(POLICY_TOPICS):
        # Resize stored answers if new topics were added to the dataset.
        existing = st.session_state["answers"]
        resized = (existing + [RESPONSE_LABELS[2]] * len(POLICY_TOPICS))[: len(POLICY_TOPICS)]
        st.session_state["answers"] = resized
    if "topic_picker" not in st.session_state:
        st.session_state["topic_picker"] = 0

    st.sidebar.header(
        "Question navigator",
        help=(
            "Pick a prompt, set your stance, then jump with the arrows below. "
            "The compass updates instantly."
        ),
    )
    question_idx = st.session_state["topic_picker"]
    current_topic = POLICY_TOPICS[question_idx]
    st.sidebar.caption(f"Currently scoring: {current_topic['theme']}")

    current_answer = st.session_state["answers"][question_idx]
    stance_choice = st.sidebar.radio(
        current_topic["prompt"],
        RESPONSE_LABELS,
        index=RESPONSE_LABELS.index(current_answer),
        help="Selections default to neutral until you change them.",
        key=f"stance_{question_idx}",
    )
    st.session_state["answers"][question_idx] = stance_choice

    nav_prev, nav_next = st.sidebar.columns(2)
    nav_prev.button(
        "← Prev",
        use_container_width=True,
        disabled=question_idx == 0,
        on_click=move_topic,
        kwargs={"delta": -1},
    )
    nav_next.button(
        "Next →",
        use_container_width=True,
        disabled=question_idx == len(POLICY_TOPICS) - 1,
        on_click=move_topic,
        kwargs={"delta": 1},
    )

    answered_count = sum(
        response != RESPONSE_LABELS[2] for response in st.session_state["answers"]
    )
    st.sidebar.progress(
        answered_count / len(POLICY_TOPICS),
        text=f"{answered_count}/{len(POLICY_TOPICS)} answered",
    )

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    with st.sidebar.expander("💬 Party-program chatbot", expanded=False):
        st.caption(
            "Ask about a party or topic. Answers quote the program text above."
        )
        if not st.session_state["chat_messages"]:
            st.info("No chat history yet. Ask the first question!")
        else:
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

    raw_scores = {axis: 0 for axis in AXES}
    answer_map = []
    for response, policy_topic in zip(st.session_state["answers"], POLICY_TOPICS):
        strength = RESPONSE_SCORES[response]
        for axis, weight in policy_topic["axes"].items():
            raw_scores[axis] += strength * weight
        answer_map.append(
            {
                "prompt": policy_topic["prompt"],
                "response": response,
                "delta": strength,
                "axes": policy_topic["axes"],
            }
        )

    normalized = score_responses(raw_scores)
    quadrant = classify_quadrant(normalized["economic"], normalized["social"])
    party_matches = rank_parties(normalized["economic"], normalized["social"])
    top_party = party_matches[0]

    chart = build_compass_chart(
        normalized["economic"], normalized["social"]
    )
    compass_chart_placeholder.plotly_chart(chart, use_container_width=True)
    compass_caption_placeholder.caption(
        "Quadrants: ↙️ Progressive Libertarian · ↖️ Progressive Communitarian · "
        "↗️ Market Communitarian · ↘️ Market Libertarian"
    )

if __name__ == "__main__":
    main()

