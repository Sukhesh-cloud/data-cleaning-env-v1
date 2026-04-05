def get_hard_dataset():
    return {
        "missing_ratio": 0.3,
        "outlier_ratio": 0.2,
        "skewness": [2.5, -2.2, 1.8],
    }


def grade_hard(env):
    score = 0

    if env.dataset["missing_ratio"] < 0.1:
        score += 0.3

    if env.dataset["outlier_ratio"] < 0.05:
        score += 0.3

    if all(abs(s) < 1 for s in env.dataset["skewness"]):
        score += 0.4

    return min(score, 1.0)