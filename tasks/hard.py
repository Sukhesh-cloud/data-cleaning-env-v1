def get_hard_dataset():
    return {
        "missing_ratio": 0.35,
        "outlier_ratio": 0.25,
        "skewness": [3.0, -2.5, 2.2]
    }

def grade_hard(env):
    """Grade hard task: Improve missing values, outliers, and skewness."""
    try:
        obs = env.last_observation
        if not obs:
            return 0.0
        
        missing = float(obs.missing_ratio)
        outliers = float(obs.outlier_ratio)
        
        # Calculate average absolute skewness
        skewness_values = obs.skewness
        if skewness_values and len(skewness_values) > 0:
            skew = sum(abs(float(s)) for s in skewness_values) / len(skewness_values)
        else:
            skew = 0.0

        score = 0.0
        score += max(0.0, (0.3 - missing)) * 1.2
        score += max(0.0, (0.2 - outliers)) * 1.5
        score += max(0.0, (1.0 - skew)) * 1.3

        return round(min(1.0, score), 2)
    except Exception as e:
        print(f"[DEBUG] grade_hard error: {e}")
        return 0.0