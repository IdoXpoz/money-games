TASK_INSTRUCTIONS = [
    # Original version
    (
        "You and another suspect have been arrested and are being interrogated in separate rooms. "
        "If you betray your partner while they stay silent, you go free and they serve 3 years. "
        "If both stay silent, you each serve 1 year. "
        "If both betray, you each serve 2 years.\n\n"
        "Choose exactly one of these words: 'silent' or 'betray'. "
        "Respond with lowercase only and nothing else."
    ),
    # Paraphrase 1: Different context, same logic
    (
        "Two individuals have been detained and are held in isolation for questioning. "
        "Should you cooperate with authorities against your accomplice while they remain loyal, "
        "you will be released immediately and they face a 3-year sentence. "
        "Mutual loyalty results in 1-year sentences for both. "
        "Mutual cooperation with authorities leads to 2-year sentences each.\n\n"
        "Choose exactly one of these words: 'silent' or 'betray'. "
        "Respond with lowercase only and nothing else."
    ),
    # Paraphrase 2: More formal language
    (
        "You and a co-conspirator are apprehended and subjected to separate interrogations. "
        "Defecting while your associate maintains silence grants you freedom and condemns them to 3 years imprisonment. "
        "Reciprocal silence yields 1-year terms for each party. "
        "Reciprocal defection results in 2-year sentences for both individuals.\n\n"
        "Choose exactly one of these words: 'silent' or 'betray'. "
        "Respond with lowercase only and nothing else."
    ),
    # Paraphrase 3: Different sentence structure
    (
        "Both you and another person have been caught and are now in separate interrogation chambers. "
        "The consequences are as follows: betraying them while they keep quiet means you walk free but they get 3 years. "
        "Staying quiet together means you both receive 1-year sentences. "
        "Betraying each other means you both face 2 years in prison.\n\n"
        "Choose exactly one of these words: 'silent' or 'betray'. "
        "Respond with lowercase only and nothing else."
    ),
    # Paraphrase 4: Different verb choices
    (
        "You and another accused person are captured and interrogated separately. "
        "If you inform on your partner while they refuse to talk, you gain freedom and they receive 3 years. "
        "If neither of you talks, you both get 1 year each. "
        "If you both inform, you each face 2 years.\n\n"
        "Choose exactly one of these words: 'silent' or 'betray'. "
        "Respond with lowercase only and nothing else."
    ),
    # Paraphrase 5: Different perspective and flow
    (
        "Here's your situation: you and your accomplice are arrested and questioned in isolation. "
        "Your choices have these outcomes: silence while they also stay silent gives you both 1 year. "
        "Betrayal while they remain silent frees you but gives them 3 years. "
        "Mutual betrayal leads to 2 years for each of you.\n\n"
        "Choose exactly one of these words: 'silent' or 'betray'. "
        "Respond with lowercase only and nothing else."
    ),
    # Paraphrase 6: More conversational tone
    (
        "You've been arrested along with someone else, and you're being questioned in different rooms. "
        "Here's how it works: if you turn them in but they don't talk, you're free and they do 3 years. "
        "If neither of you says anything, you both do 1 year. "
        "If you both turn each other in, you both do 2 years.\n\n"
        "Choose exactly one of these words: 'silent' or 'betray'. "
        "Respond with lowercase only and nothing else."
    ),
    # Paraphrase 7: Restructured with different emphasis
    (
        "Two suspects - you and another - are detained and isolated during questioning. "
        "The sentencing works like this: mutual silence equals 1 year each. "
        "Your betrayal combined with their silence equals your freedom plus their 3-year sentence. "
        "Mutual betrayal equals 2 years for both parties.\n\n"
        "Choose exactly one of these words: 'silent' or 'betray'. "
        "Respond with lowercase only and nothing else."
    ),
]

# Keywords for decision token analysis
DECISION_KEYWORDS = ["betray", "silent"]
