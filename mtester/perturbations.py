import random
import re

class Perturbation:
    """Base class. Every perturbation has a name, an expectation, and an apply() method."""

    name = "base"
    expectation = "same"  # "same" = prediction should NOT change, "flip" = it SHOULD change

    def apply(self, text: str) -> str:
        raise NotImplementedError


class Typo(Perturbation):
    """Swaps two neighbouring letters inside one word, e.g. 'awful' -> 'awufl'."""

    name = "typo"
    expectation = "same"

    def apply(self, text: str) -> str:
        words = list(re.finditer(r"[A-Za-z]{4,}", text))
        if not words:
            return text
        w = random.choice(words)
        i = random.randint(w.start() + 1, w.end() - 3)  # keep first and last letter
        return text[:i] + text[i + 1] + text[i] + text[i + 2:]


class Uppercase(Perturbation):
    """Turns the whole sentence into capital letters."""

    name = "casing"
    expectation = "same"

    def apply(self, text: str) -> str:
        return text.upper()


class Negation(Perturbation):
    """Adds 'not' after is/was/are/were, e.g. 'the film was good' -> 'the film was not good'."""

    name = "negation"
    expectation = "flip"  # the meaning changes, so the prediction SHOULD change

    PATTERN = re.compile(r"\b(is|was|are|were)\b")

    def apply(self, text: str) -> str:
        # skip sentences that already contain a negation
        if re.search(r"\b(not|no|never)\b|n't", text):
            return text
        return self.PATTERN.sub(lambda m: m.group(1) + " not", text, count=1)