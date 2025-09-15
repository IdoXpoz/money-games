instruction_prefix_game1 = "Answer with one word only: betray or silent"
instruction_prefix_game2 = "Answer with one word only: volunteer or wait"


def construct_prompt(prefix, decision_task):
    # Compose prompt parts cleanly
    parts = [instruction_prefix_game2]
    if prefix and prefix.strip():
        parts.append(prefix.strip())
    parts.append(decision_task.strip())
    # Ensure the next token after the prompt is the answer token
    return "\n\n".join(parts) + "\n"
