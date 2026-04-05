def get_medium_dataset():
    return {
        "missing_ratio": 0.2,
        "outlier_ratio": 0.15,
        "skewness": [1.5, -1.2, 0.8],
    }


def grade_medium(env):
    score = 0

    if env.dataset["missing_ratio"] < 0.1:
        score += 0.4
    if env.dataset["outlier_ratio"] < 0.05:
        score += 0.6
    return min(score, 1.0)