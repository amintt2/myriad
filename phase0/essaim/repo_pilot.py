"""Real-repository printer pilot, using the shared agent/campaign accounting without model calls on import."""
from essaim import code_pilot, repo_tasks


def cli_main(argv=None) -> int:
    return code_pilot.cli_main(argv, backend=repo_tasks)
