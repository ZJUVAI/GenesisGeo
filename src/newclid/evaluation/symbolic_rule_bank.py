"""Compile a frozen symbolic rule package in the evaluation forkserver."""

import hashlib
import os
from pathlib import Path

RULE_BANK = None
RULE_NAMES = frozenset()
COMPILE_PID = None

if filename := os.environ.get("NEWCLID_SYMBOLIC_RULE_BANK"):
    from newclid.api import DDAR

    payload = Path(filename).read_bytes()
    if (
        hashlib.sha256(payload).hexdigest()
        != os.environ["NEWCLID_SYMBOLIC_RULE_BANK_SHA256"]
    ):
        raise ValueError("Frozen symbolic rule bank changed")
    lines = [line.strip() for line in payload.decode().splitlines() if line.strip()]
    RULE_BANK = DDAR.compile_custom_theorems(lines)
    RULE_NAMES = frozenset(line.split("|", 1)[0] for line in lines)
    COMPILE_PID = os.getpid()
