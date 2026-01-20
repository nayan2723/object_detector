"""
Caption generation from inferred human–object interactions.

This module converts structured `Interaction` instances into concise
natural-language descriptions suitable for real-time overlays.
"""

from __future__ import annotations

from typing import List

from .interaction import Interaction


def _article_for(word: str) -> str:
    """
    Return an appropriate article ("a" / "an") for a given word.
    """
    if not word:
        return "a"
    return "an" if word[0].lower() in {"a", "e", "i", "o", "u"} else "a"


def generate_caption(interactions: List[Interaction]) -> str:
    """
    Generate a short, human-readable caption from interactions.

    Parameters
    ----------
    interactions : List[Interaction]
        List of inferred interactions.

    Returns
    -------
    str
        Natural-language caption.
    """
    if not interactions:
        return "A scene with no clear human–object interaction"

    # Aggregate by interaction type for simplicity
    phrases: List[str] = []

    for interaction in interactions:
        obj_label = interaction.object_label
        obj_label_lower = obj_label.lower()

        # Normalize common labels (e.g., "cell phone" vs "mobile phone")
        normalized = obj_label_lower.replace("_", " ")
        article = _article_for(normalized)

        phrase = f"a person {interaction.interaction_type} {article} {normalized}"
        phrases.append(phrase)

    # Deduplicate phrases while preserving order
    seen = set()
    unique_phrases: List[str] = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            unique_phrases.append(p)

    if len(unique_phrases) == 1:
        # Capitalize first letter
        s = unique_phrases[0]
        return s[0].upper() + s[1:]

    # For multiple interactions, join with commas and "and" for readability
    if len(unique_phrases) == 2:
        caption = " and ".join(unique_phrases)
    else:
        caption = ", ".join(unique_phrases[:-1]) + " and " + unique_phrases[-1]

    caption = caption[0].upper() + caption[1:]
    return caption


__all__ = ["generate_caption"]


