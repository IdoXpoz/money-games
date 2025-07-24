instruction_prefix = "Answer with one word only: betray, silent, cooperate, or defect."


def construct_prompt(prefix, decision_task):
    # Compose prompt parts cleanly
    parts = [instruction_prefix]
    if prefix and prefix.strip():
        parts.append(prefix.strip())
    parts.append(decision_task.strip())
    return "\n\n".join(parts)
