def get_medium_dataset():
    return {
        "missing_ratio": 0.2,
        "outlier_ratio": 0.15,
        "skewness": [1.5, -1.2, 0.8],
    }


def grade_medium(env):
    """Grade medium task: Improve missing values and outliers."""
    try:
        obs = env.last_observation
        if not obs:
            return 0.0
        
        missing = float(obs.missing_ratio)
        outliers = float(obs.outlier_ratio)

        score = 0.0
        score += max(0.0, (0.3 - missing)) * 1.5
        score += max(0.0, (0.2 - outliers)) * 2.0

        return round(min(1.0, score), 2)
    except Exception as e:
        print(f"[DEBUG] grade_medium error: {e}")
        return 0.0