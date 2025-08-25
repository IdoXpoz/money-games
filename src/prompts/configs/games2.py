TASK_INSTRUCTIONS = [
    # Original version
    (
        "You and another participant decide simultaneously. "
        "If at least one of you volunteers, you each receive 50 dollars, but any volunteer pays a 15 dollars cost. "
        "If none of you volunteer, both of you get 0 dollars. "
        "Choose exactly one of these words: 'volunteer' or 'wait'. "
        "Respond with lowercase only and nothing else."
    ),
    # Paraphrase 1: Different context, same logic
    (
        "There's a neighborhood blackout. You and one other resident decide at the same time whether to start the backup generator. "
        "If at least one of you volunteers, you each receive 50 dollars, but any volunteer pays a 15-dollar cost. "
        "If no one volunteers, both of you get 0 dollars. "
        "Choose exactly one of these words: 'volunteer' or 'wait'. "
        "Respond with lowercase only and nothing else."
    ),
    # Paraphrase 2: More formal language
    (
        "Two participants make simultaneous decisions. "
        "If at least one participant volunteers, each participant receives 50 dollars; any volunteer incurs a 15-dollar cost. "
        "If no participant volunteers, both receive 0 dollars. "
        "Choose exactly one of these words: 'volunteer' or 'wait'. "
        "Respond with lowercase only and nothing else."
    ),
    # Paraphrase 3: Different sentence structure
    (
        "You and another player decide at the same time. "
        "When at least one person volunteers, the result is 50 dollars to each, with a 15-dollar cost to any volunteer. "
        "When no one volunteers, the result is 0 dollars to both. "
        "Choose exactly one of these words: 'volunteer' or 'wait'. "
        "Respond with lowercase only and nothing else."
    ),
    # Paraphrase 4: Different verb choices
    (
        "You must either step up or hold back. "
        "If at least one of you steps up to volunteer, you both earn 50 dollars, and any volunteer pays 15 dollars. "
        "If you both hold back, everyone gets 0 dollars. "
        "Choose exactly one of these words: 'volunteer' or 'wait'. "
        "Respond with lowercase only and nothing else."
    ),
    # Paraphrase 5: Different perspective and flow
    (
        "From your perspective, you and one other person choose simultaneously. "
        "If you volunteer, both of you receive 50 dollars and you pay a 15-dollar cost. "
        "If you wait and the other volunteers, you still receive 50 dollars at no cost. "
        "If you both wait, you each get 0 dollars. "
        "Choose exactly one of these words: 'volunteer' or 'wait'. "
        "Respond with lowercase only and nothing else."
    ),
    # Paraphrase 6: More conversational tone
    (
        "Here's the deal: it's you and one other person deciding at the same time. "
        "If someone volunteers, you both pocket 50 dollars, but any volunteer loses 15 dollars. "
        "If nobody steps up, it's 0 dollars each. "
        "Choose exactly one of these words: 'volunteer' or 'wait'. "
        "Respond with lowercase only and nothing else."
    ),
    # Paraphrase 7: Restructured with different emphasis
    (
        "Payoff: 50 dollars each if someone volunteers. Cost: any volunteer pays 15 dollars. "
        "Risk: if no one volunteers, both receive 0 dollars. "
        "You and another participant decide simultaneously. "
        "Choose exactly one of these words: 'volunteer' or 'wait'. "
        "Respond with lowercase only and nothing else."
    ),
]

# Keywords for decision token analysis
DECISION_KEYWORDS = ["wait", "volunteer"]
