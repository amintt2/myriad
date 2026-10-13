"""Explicit frozen final worker, sharing the audited bootstrap/watchdog/inventory lifecycle."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import e17_campaign_remote as shared


def worker(root):
    def preflight(root, request):
        from essaim.ephemeral_vm import require_colab
        from e17_final_freeze import verify_vm_sources
        require_colab()
        verify_vm_sources(request, root / "phase0")
    names = ("RUNTIME_PROJECT", "CONTROLLER_SCRIPT", "PROTOCOL_MODULE", "WORKER_PREFLIGHT")
    previous = {name: getattr(shared, name) for name in names}
    try:
        shared.RUNTIME_PROJECT = "bigcode_pilot"
        shared.CONTROLLER_SCRIPT = "e17_final_vm.py"
        shared.PROTOCOL_MODULE = "e17_final_freeze"
        shared.WORKER_PREFLIGHT = preflight
        return shared.worker(root)
    finally:
        for name, value in previous.items():
            setattr(shared, name, value)


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--worker":
    worker(Path(sys.argv[2]))
