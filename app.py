import os
from typing import List

import streamlit as st
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Answer Sheet Evaluator",
    page_icon="📝",
    layout="wide"
)

st.title("📝 AI Answer Sheet Evaluator for Teachers")
st.markdown(
    "Automate answer evaluation against required **Skill Keywords** "
    "and concept criteria."
)


# ---------------------------------------------------------------------------
# Sidebar for API Configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    # API key is securely stored in Streamlit Secrets
    api_key = st.secrets["OPENAI_API_KEY"]

    selected_model = st.selectbox(
        "Select Model",
        ["gpt-4o", "gpt-4o-mini"],
        index=0
    )

    st.markdown("---")
    st.info(
        "Tip: Enter all required skill keywords separated by commas."
    )

# ---------------------------------------------------------------------------
# Pydantic Output Schema
# ---------------------------------------------------------------------------

class KeywordEvaluation(BaseModel):
    keyword: str = Field(
        description="The target skill or keyword evaluated"
    )

    present: bool = Field(
        description=(
            "True if the keyword or its underlying concept "
            "is present"
        )
    )

    evidence: str = Field(
        description="Brief explanation of presence or absence in answer"
    )


class AnswerEvaluationResult(BaseModel):
    score_obtained: float = Field(
        description="Total score granted"
    )

    max_score: float = Field(
        description="Maximum possible score"
    )

    percentage: float = Field(
        description="Score expressed as percentage (0-100)"
    )

    matched_skills: List[str] = Field(
        description="Matched keywords/skills"
    )

    missing_skills: List[str] = Field(
        description="Missing keywords/skills"
    )

    keyword_breakdown: List[KeywordEvaluation] = Field(
        description="Keyword details"
    )

    detailed_feedback: str = Field(
        description="Constructive student feedback"
    )


# ---------------------------------------------------------------------------
# LangChain Evaluation Chain
# ---------------------------------------------------------------------------

def evaluate_answer(
    api_key: str,
    model_name: str,
    payload: dict
) -> AnswerEvaluationResult:

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key
    )

    structured_llm = llm.with_structured_output(
        AnswerEvaluationResult
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an academic examiner evaluating student answers.

Analyze the student's submission against the required skill
keywords and the model reference answer.

Rules:

1. Mark a keyword as present if:
   - It is explicitly stated by the student, OR
   - The underlying concept is accurately explained even if
     different terminology is used.

2. Do not mark a keyword as present merely because the student
   mentions an unrelated word.

3. Evaluate the student's conceptual understanding, not just
   exact keyword matching.

4. Calculate score_obtained based on the overall correctness,
   completeness, and coverage of the required concepts.

5. score_obtained must never be greater than max_score.

6. Calculate:
   percentage = (score_obtained / max_score) * 100

7. The percentage must be between 0 and 100.

8. matched_skills must contain the keywords whose concepts are
   adequately demonstrated.

9. missing_skills must contain keywords whose concepts are absent
   or inadequately explained.

10. keyword_breakdown must contain one evaluation for every
    supplied skill keyword.

11. Provide constructive teacher-style feedback explaining:
    - What the student did correctly
    - What concepts are missing
    - How the answer could be improved

Be fair and academically reasonable.
"""
            ),
            (
                "human",
                """
Question:
{question}

Maximum Score:
{max_score}

Required Skill Keywords:
{skill_keywords}

Model Reference Answer:
{model_answer}

Student Answer:
{student_answer}
"""
            )
        ]
    )

    chain = prompt | structured_llm

    return chain.invoke(payload)


# ---------------------------------------------------------------------------
# Streamlit Interface
# ---------------------------------------------------------------------------

col1, col2 = st.columns(
    [1, 1],
    gap="medium"
)


# ---------------------------------------------------------------------------
# LEFT COLUMN - INPUT
# ---------------------------------------------------------------------------

with col1:

    st.subheader("1. Setup Assignment Criteria")

    question = st.text_area(
        "Question",
        value=(
            "Explain how backpropagation works in artificial "
            "neural networks."
        ),
        height=100
    )

    max_score = st.number_input(
        "Maximum Score",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=1.0
    )

    keywords_input = st.text_input(
        "Skill Keywords (comma-separated)",
        value=(
            "Chain Rule, Gradient Descent, Loss Function, "
            "Weight Updates, Forward Pass"
        ),
        help="Separate keywords using commas"
    )

    model_answer = st.text_area(
        "Model Answer / Key Points",
        value=(
            "Backpropagation begins with a forward pass where "
            "inputs produce a prediction and calculate error "
            "using a loss function. During the backward pass, "
            "error gradients are computed using the calculus "
            "chain rule. Optimization algorithms like gradient "
            "descent use these gradients to perform weight "
            "updates."
        ),
        height=150
    )

    st.subheader("2. Student Submission")

    student_answer = st.text_area(
        "Paste Student Answer Sheet",
        value=(
            "In neural networks, the input passes through layers "
            "to generate output in the forward pass. Then we "
            "calculate the error with a loss function. Backprop "
            "takes this error backwards to update weights so the "
            "model performs better next time."
        ),
        height=180
    )

    evaluate_btn = st.button(
        "🚀 Evaluate Answer Sheet",
        type="primary",
        use_container_width=True
    )


# ---------------------------------------------------------------------------
# RIGHT COLUMN - RESULTS
# ---------------------------------------------------------------------------

with col2:

    st.subheader("3. Evaluation Results")

    if evaluate_btn:

        if not api_key:
            st.error(
                "Please enter your OpenAI API key in the sidebar."
            )

        elif not student_answer.strip():
            st.warning(
                "Please enter a student answer to evaluate."
            )

        elif not keywords_input.strip():
            st.warning(
                "Please provide at least one required skill keyword."
            )

        else:

            with st.spinner(
                "Analyzing answer against skill keywords..."
            ):

                keywords_list = [
                    keyword.strip()
                    for keyword in keywords_input.split(",")
                    if keyword.strip()
                ]

                payload = {
                    "question": question,
                    "max_score": max_score,
                    "skill_keywords": ", ".join(keywords_list),
                    "model_answer": model_answer,
                    "student_answer": student_answer
                }

                try:

                    result: AnswerEvaluationResult = evaluate_answer(
                        api_key,
                        selected_model,
                        payload
                    )

                    # -------------------------------------------------------
                    # Metrics
                    # -------------------------------------------------------

                    m1, m2, m3 = st.columns(3)

                    m1.metric(
                        "Percentage Score",
                        f"{result.percentage:.1f}%"
                    )

                    m2.metric(
                        "Points Granted",
                        f"{result.score_obtained:g} / "
                        f"{result.max_score:g}"
                    )

                    m3.metric(
                        "Matched Keywords",
                        f"{len(result.matched_skills)} / "
                        f"{len(keywords_list)}"
                    )

                    # Clamp percentage for Streamlit progress bar
                    progress_value = max(
                        0.0,
                        min(result.percentage / 100.0, 1.0)
                    )

                    st.progress(progress_value)

                    # -------------------------------------------------------
                    # Skill Analysis
                    # -------------------------------------------------------

                    st.markdown("### 🎯 Skill Keywords Analysis")

                    st.write("**Matched Skills:**")

                    if result.matched_skills:
                        st.success(
                            ", ".join(
                                f"`{skill}`"
                                for skill in result.matched_skills
                            )
                        )
                    else:
                        st.write("None")

                    st.write("**Missing Skills:**")

                    if result.missing_skills:
                        st.error(
                            ", ".join(
                                f"`{skill}`"
                                for skill in result.missing_skills
                            )
                        )
                    else:
                        st.write("None")

                    # -------------------------------------------------------
                    # Keyword Evidence Breakdown
                    # -------------------------------------------------------

                    st.markdown(
                        "### 🔍 Keyword Evidence Breakdown"
                    )

                    table_data = []

                    for item in result.keyword_breakdown:

                        table_data.append(
                            {
                                "Status": (
                                    "✅ Present"
                                    if item.present
                                    else "❌ Missing"
                                ),
                                "Skill Keyword": item.keyword,
                                "Analysis / Evidence": item.evidence
                            }
                        )

                    st.dataframe(
                        table_data,
                        use_container_width=True,
                        hide_index=True
                    )

                    # -------------------------------------------------------
                    # Teacher Feedback
                    # -------------------------------------------------------

                    st.markdown("### 👨‍🏫 Teacher Feedback")

                    st.info(result.detailed_feedback)

                except Exception as e:

                    st.error(
                        f"Evaluation Error: {str(e)}"
                    )

    else:

        st.info(
            "Fill out the assignment criteria and click "
            "**Evaluate Answer Sheet** to get the score and percentage."
        )

