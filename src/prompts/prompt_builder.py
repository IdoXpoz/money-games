instruction_prefix = "Answer with exactly one word: 'betray' or 'silent'. OUTPUT ONLY THE WORD."


def construct_prompt(prefix, decision_task):
    # Compose prompt parts cleanly
    parts = [instruction_prefix]
    if prefix and prefix.strip():
        parts.append(prefix.strip())
    parts.append(decision_task.strip())
    # Ensure the next token after the prompt is the answer token
    return "\n\n".join(parts) + "\n"
