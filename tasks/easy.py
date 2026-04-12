def get_easy_dataset():
    return {
        "missing_ratio": 0.25,
        "outlier_ratio": 0.0,
        "skewness": [0.2, 0.1, -0.3],
    }


def grade_easy(env):
    """Grade easy task: Improve missing value ratio."""
    try:
        obs = env.last_observation
        if not obs:
            return 0.0
        
        missing = float(obs.missing_ratio)
        # Score formula: max(0, (0.3 - missing)) * 3
        score = max(0.0, (0.3 - missing)) * 3.0
        return round(min(1.0, score), 2)
    except Exception as e:
        print(f"[DEBUG] grade_easy error: {e}")
        return 0.0