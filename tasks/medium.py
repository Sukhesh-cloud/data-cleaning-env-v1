def get_medium_dataset():
    return {
        "missing_ratio": 0.2,
        "outlier_ratio": 0.15,
        "skewness": [1.5, -1.2, 0.8],
    }


def grade_medium(env):
    obs = env.last_observation

    missing = obs.missing_ratio
    outliers = obs.outlier_ratio

    score = 0
    score += max(0, (0.3 - missing)) * 1.5
    score += max(0, (0.2 - outliers)) * 2

    return round(min(1.0, score), 2)