"""Look up one Jira bug, recommend a next step, and post it after approval."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from jira_triage.bootstrap import build_defect_retriever, build_triage_advisor
from jira_triage.config.settings import Settings
from jira_triage.errors import TriageError
from jira_triage.jira.client import JiraCloudClient
from jira_triage.rag.augmentation import CLEAR_MATCH_SIMILARITY, TriageRecommendation
from jira_triage.rag.retrieval import RetrievalResult

st.title("Triage a defect")
st.caption(
    "Enter a Jira bug key. A stored bug at "
    f"{CLEAR_MATCH_SIMILARITY * 100:.0f}% similarity or higher can supply the root cause. "
    "A weaker match gets investigation steps instead. "
    "The comment is posted only after you approve it."
)


@st.cache_resource
def _services():
    settings = Settings.from_env(ROOT / ".env")
    return (
        build_defect_retriever(settings, ROOT / ".chroma"),
        build_triage_advisor(settings),
        JiraCloudClient(settings),
    )


def _show_result(result: RetrievalResult) -> None:
    defect = result.defect
    st.subheader(defect.defect_id)
    st.write(defect.summary)
    priority, project, matches = st.columns(3)
    priority.metric("Priority", defect.priority or "None")
    project.metric("Project", defect.project_key)
    matches.metric("Similar defects", len(result.matches))
    st.markdown("**Description**")
    st.write(defect.description or "No description.")
    st.markdown("**Root cause**")
    st.write(defect.root_cause or "No root cause section.")

    st.subheader("Similar defects")
    if not result.matches:
        st.info("No other defects are stored yet. Run `python index_defects.py` first.")
        return
    st.caption("Ranked by the summary and description. The defect you entered is left out of this list.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "defect_id": match.defect_id,
                    "similarity": f"{match.similarity * 100:.1f}%",
                    "root_cause": match.root_cause,
                    "document": match.document,
                }
                for match in result.matches
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def _show_recommendation(recommendation: TriageRecommendation) -> None:
    st.subheader("Recommendation")
    if recommendation.notice:
        st.warning(recommendation.notice)
    if recommendation.clear_match:
        st.success(recommendation.headline)
    else:
        st.warning(recommendation.headline)
    st.markdown("**Reasoning**")
    st.write(recommendation.reasoning)
    st.markdown("**What to do**")
    for index, step in enumerate(recommendation.instructions, start=1):
        st.write(f"{index}. {step}")


with st.form("triage_form"):
    defect_id = st.text_input("Defect id", placeholder="FDE-63")
    submitted = st.form_submit_button("Triage")

if submitted:
    if not defect_id.strip():
        st.session_state.triage_error = "Enter a Jira defect id such as FDE-63."
        st.session_state.triage_result = None
        st.session_state.triage_recommendation = None
    else:
        try:
            retriever, advisor, _client = _services()
            with st.spinner("Loading the defect and preparing a recommendation..."):
                result = retriever.retrieve(defect_id)
                st.session_state.triage_recommendation = advisor.advise(result)
            st.session_state.triage_result = result
            st.session_state.triage_error = None
            st.session_state.triage_posted = None
            st.session_state.triage_post_error = None
        except TriageError as error:
            st.session_state.triage_error = str(error)
            st.session_state.triage_result = None
            st.session_state.triage_recommendation = None

error = st.session_state.get("triage_error")
if error:
    st.error(error)

result = st.session_state.get("triage_result")
recommendation = st.session_state.get("triage_recommendation")
if isinstance(result, RetrievalResult):
    _show_result(result)
if isinstance(recommendation, TriageRecommendation):
    _show_recommendation(recommendation)
    posted = st.session_state.get("triage_posted") == recommendation.defect_id
    post_error = st.session_state.get("triage_post_error")
    if posted:
        st.success(f"Posted the recommendation as a comment on {recommendation.defect_id}.")
    else:
        if post_error:
            st.error(post_error)
        if st.button("Approve and post to Jira", type="primary"):
            _retriever, _advisor, client = _services()
            try:
                client.add_comment(recommendation.defect_id, recommendation.comment_text())
                st.session_state.triage_posted = recommendation.defect_id
                st.session_state.triage_post_error = None
                st.rerun()
            except TriageError as error:
                st.session_state.triage_post_error = str(error)
                st.rerun()
