def get_hard_dataset():
    return {
        "missing_ratio": 0.35,
        "outlier_ratio": 0.25,
        "skewness": [3.0, -2.5, 2.2]
    }

def grade_hard(env):
    obs = env.last_observation

    missing = obs.missing_ratio
    outliers = obs.outlier_ratio
    skew = sum(abs(s) for s in obs.skewness) / len(obs.skewness)

    score = 0
    score += max(0, (0.3 - missing)) * 1.2
    score += max(0, (0.2 - outliers)) * 1.5
    score += max(0, (1.0 - skew)) * 1.3

    return round(min(1.0, score), 2)