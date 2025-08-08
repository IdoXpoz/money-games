def find_position_of_end_thinking_tag(sequence: list) -> int:
    """Return index of last occurrence of value in sequence or None if not found."""
    try:
        thinking_tag_id = 151668
        print(f"searching for </think> in {sequence}")
        return len(sequence) - 1 - sequence[::-1].index(thinking_tag_id)
    except ValueError:
        return 0
