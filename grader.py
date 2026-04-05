from tasks.easy import grade_easy
from tasks.medium import grade_medium
from tasks.hard import grade_hard


def grade(task_name, env):
    if task_name == "easy":
        return grade_easy(env)
    elif task_name == "medium":
        return grade_medium(env)
    else:
        return grade_hard(env)