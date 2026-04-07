def get_easy_dataset():
    return {
        "missing_ratio": 0.25,
        "outlier_ratio": 0.0,
        "skewness": [0.2, 0.1, -0.3],
    }


def grade_easy(env):
    obs = env.last_observation   
    missing = obs.missing_ratio

    score = max(0, (0.3 - missing)) * 3
    return round(min(1.0, score), 2)