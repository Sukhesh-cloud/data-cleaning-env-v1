from tasks.easy import grade_easy
from tasks.medium import grade_medium
from tasks.hard import grade_hard


def grade(task_name, env):
    """
    Grade a task based on final observation.
    
    Args:
        task_name: 'easy', 'medium', or 'hard'
        env: Environment with last_observation attribute
    
    Returns:
        score: Float in [0.0, 1.0]
    """
    try:
        if not hasattr(env, 'last_observation'):
            return 0.0
        
        if task_name == "easy":
            return grade_easy(env)
        elif task_name == "medium":
            return grade_medium(env)
        elif task_name == "hard":
            return grade_hard(env)
        else:
            return 0.0
    except Exception as e:
        print(f"[DEBUG] Grading error for {task_name}: {e}")
        return 0.0