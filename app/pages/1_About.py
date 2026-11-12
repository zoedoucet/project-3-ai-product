import streamlit as st

st.set_page_config(
    page_title="About | Tree Health Investigator")

st.title("About Tree Health Investigator")

st.write(
    "Tree Health Investigator is an AI-powered assistant that helps "
    "garden and small forest owners explore possible insect-related "
    "causes of tree health problems.")

st.subheader("How it works")

st.write(
    "The assistant uses Retrieval-Augmented Generation (RAG) to "
    "retrieve relevant passages from a forest entomology textbook "
    "and generate answers grounded in those sources.")

st.subheader("Textbook and Sources")

st.write(
    "Tree Health Investigator uses *Forest Entomology and Pathology, "
    "Volume 1: Entomology* (2023) as its primary knowledge source. "
    "This open-access textbook covers forest insects, their ecological "
    "roles, different feeding groups, insect-related tree damage, "
    "and approaches to forest pest management."
)

st.write(
    "The assistant searches relevant passages from this textbook "
    "and uses them to generate evidence-based explanations. "
    "Page references are included in its answers so users can "
    "consult the original source."
)

st.markdown(
    "Allison, J. D., Paine, T. D., Slippers, B., & Wingfield, M. J. "
    "(Eds.). (2023). *Forest Entomology and Pathology: Volume 1: "
    "Entomology*. Springer.")

st.link_button(
    "Read the open-access textbook",
    "https://doi.org/10.1007/978-3-031-11553-0")

st.subheader("What to expect")

st.write(
    "Describe the symptoms you observe on your tree. "
    "The assistant may ask follow-up questions to narrow down "
    "possible causes and provides textbook page references "
    "to support its explanations."
)

st.subheader("Important limitations")

st.info(
    "Tree Health Investigator provides educational guidance, "
    "not a confirmed diagnosis. Insect damage can resemble damage "
    "caused by diseases, weather, or other environmental factors. "
    "For serious concerns, consult a qualified tree care professional.")