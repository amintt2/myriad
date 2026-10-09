"""Entry point of the packaged Myriad app (PyInstaller needs a script, not a module)."""
import multiprocessing
import sys

from myriad.desktop import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
