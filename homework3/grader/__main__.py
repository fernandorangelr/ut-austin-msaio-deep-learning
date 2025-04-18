from . import tests  # noqa

try:
    # For test grading only, don't worry about it being missing
    from .safe_grader import run

    print("Test grader loaded.")
except ImportError:
    from .grader import run

    print("Public grader loaded.")

run()
