import pandas as pd


def run_tests(model, sentences, perturbations):
    """Apply every perturbation to every sentence and record pass/fail."""
    rows = []
    for text in sentences:
        before = model.predict(text)
        for p in perturbations:
            perturbed = p.apply(text)
            
            after = model.predict(perturbed)

            if perturbed == text:      # perturbation didn't apply, skip it
                            continue
            if p.expectation == "same":
                passed = before == after
            else:  # "flip"
                passed = before != after

            rows.append({
                "original": text,
                "perturbed": perturbed,
                "test": p.name,
                "pred_before": before,
                "pred_after": after,
                "passed": passed,
            })
    return pd.DataFrame(rows)