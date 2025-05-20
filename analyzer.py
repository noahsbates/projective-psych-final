from llm import llm
from langchain.schema import SystemMessage, HumanMessage

print("== Initializing analyzer ... ==")

message_pre = """
You are a skilled clinical psychologist speaking directly to the person in front of you. When analyzing multiple projective-test interpretations, you should:

- Keep your feedback concise—between 5 and 12 sentences.
- Address the respondent as “you” and “your,” as if guiding them in session.
- Briefly describe what you see in their combined responses.
- Highlight broad feelings or themes that emerge across the tests.
- Relate those themes to how you might help them understand their experience.
- End with a supportive, next-step suggestion for the user (this should not be a subsequent conversation to be had with this information bot).

Speak in a warm, empathetic tone and avoid jargon.
"""

def query_gpt(interpretations: list[str], kinds: list[str]) -> str:
    """
    Performs an integrated analysis on a variable number of projective tests.
    :param interpretations: List of strings describing each interpretation.
    :param kinds: List of test types corresponding to each interpretation.
    :return: A concise, second-person analysis addressing broad themes across responses.
    """
    combined_kinds = ", ".join(kinds)
    interp_lines = "\n".join(f"- {interp}" for interp in interpretations)
    user_content = (
        f"Test Types: {combined_kinds}\n"
        "Your Interpretations:\n"
        f"{interp_lines}\n\n"
        "Please provide an integrated analysis that addresses broad themes across these responses."
    )

    messages = [
        SystemMessage(content=message_pre),
        HumanMessage(content=user_content)
    ]
    result = llm.invoke(messages)
    return result.content

print("== Analyzer initialized successfully. ==")

if __name__ == "__main__":
    # Example usage with multiple tests
    kinds = ["inkblot", "waterspill"]
    interpretations = [
        "I see an angry dog or a pair of legs",
        "A mother comforting a child under a tree"
    ]
    analysis = query_gpt(interpretations, kinds)
    print(analysis)
