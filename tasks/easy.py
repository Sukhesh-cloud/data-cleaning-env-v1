def get_easy_dataset():
    return {
        "missing_ratio": 0.25,
        "outlier_ratio": 0.0,
        "skewness": [0.2, 0.1, -0.3],
    }


def grade_easy(env):
    score = 0

    if env.dataset["missing_ratio"] < 0.1:
        score += 1.0

    return score