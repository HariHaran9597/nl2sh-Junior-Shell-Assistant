"""Risk, explanation, and dry-run helpers for generated shell commands."""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass

from .safety import Level, check


HIGH = {
    "rm", "mkfs", "dd", "shred", "chmod", "chown", "iptables", "crontab",
    "kill", "pkill", "reboot", "shutdown", "poweroff", "halt", "fdisk",
    "wipefs", "git", "docker", "nc", "ncat", "netcat",
}
MED = {"find", "chmod", "chown", "tar", "rsync", "mv", "cp", "truncate", "wipefs"}
LOW = {"ls", "cat", "grep", "head", "tail", "wc", "du", "df", "ps", "echo", "pwd", "whoami"}

HIGH_PATTERNS = [
    (r"\brm\s+.*-rf\s+/", "recursive delete of root"),
    (r"\bmkfs\b", "format filesystem"),
    (r"\bdd\s+.*of=/dev/", "raw device write"),
    (r"\b(?:curl|wget)\b.*\|\s*(?:bash|sh)", "download piped to shell"),
    (r":\(\)\s*\{", "fork bomb"),
]

FLAG_EXPLAIN = {
    "tar -xzf": "-x extract, -z gzip, -f file",
    "tar -czf": "-c create, -z gzip, -f file",
    "find -delete": "-delete removes matched files permanently",
    "find -exec": "-exec runs command per match",
    "rm -rf": "-r recursive, -f force (no confirm)",
    "chmod -R": "-R recursive permission change",
    "du -sh": "-s summarize, -h human-readable",
    "ps aux": "a all users, u user format, x incl. no-tty",
}


@dataclass
class RiskResult:
    level: str
    reason: str
    explain: str
    dry_run_cmd: str | None = None


def _first_tool(command: str) -> str:
    parts = command.split()
    while parts and ("=" in parts[0] and not parts[0].startswith("=")):
        parts.pop(0)
    if parts and parts[0] == "sudo":
        parts.pop(0)
    return parts[0] if parts else ""


def classify(cmd: str) -> RiskResult:
    command = (cmd or "").strip()
    first = _first_tool(command)

    safety_level, findings = check(command)
    if safety_level == Level.DANGER:
        reason = findings[0].detail if findings else "destructive or executable shell pattern"
        return RiskResult("HIGH", reason, explain_cmd(command), dry_run_for(command))

    for pattern, reason in HIGH_PATTERNS:
        if re.search(pattern, command, re.I):
            return RiskResult("HIGH", reason, explain_cmd(command), dry_run_for(command))

    if first in HIGH:
        return RiskResult("HIGH", f"{first} can change system state", explain_cmd(command), dry_run_for(command))
    if first in MED or " -delete" in command or " -exec rm" in command:
        return RiskResult("MED", f"{first or 'command'} modifies files", explain_cmd(command), dry_run_for(command))
    return RiskResult("LOW", "read-only / low risk", explain_cmd(command), None)


def explain_cmd(cmd: str) -> str:
    parts = []
    for key, description in FLAG_EXPLAIN.items():
        if key.split()[0] in cmd and any(flag in cmd for flag in key.split()[1:]):
            parts.append(f"{key}: {description}")
    if not parts:
        tool = _first_tool(cmd)
        snippet = _man_snippet(tool)
        if snippet:
            parts.append(snippet)
    shellcheck = _shellcheck_note(cmd)
    if shellcheck:
        parts.append(shellcheck)
    return " | ".join(parts) if parts else "no extra notes"


def _man_snippet(tool: str) -> str | None:
    if not tool or not shutil.which("man"):
        return None
    try:
        result = subprocess.run(["man", tool], capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                line = line.strip()
                if len(line) > 20 and not line.startswith("NAME"):
                    return line[:120]
    except Exception:
        pass
    return None


def _shellcheck_note(cmd: str) -> str | None:
    shellcheck = shutil.which("shellcheck")
    if not shellcheck:
        return None
    try:
        result = subprocess.run([shellcheck, "-S", "error", "-"], input=cmd,
                                capture_output=True, text=True, timeout=5)
        if result.stdout.strip():
            return f"shellcheck: {result.stdout.strip()[:120]}"
    except Exception:
        pass
    return None


def dry_run_for(cmd: str) -> str | None:
    command = (cmd or "").strip()
    if "find" in command and "-delete" in command:
        return command.replace("-delete", "-print") + "  # dry-run: lists matched paths"
    if "find" in command and "-exec rm" in command:
        return command.replace("-exec rm", "-print") + "  # dry-run: lists matched paths"
    # Do not fabricate a file count for arbitrary rm or append unsupported
    # --dry-run flags to chmod/chown. A misleading preview creates confidence.
    return None


def render_risk(cmd: str) -> str:
    result = classify(cmd)
    lines = [f"[{result.level}] {cmd}", f"Explain: {result.explain}"]
    if result.dry_run_cmd:
        lines.append(f"Dry-run: {result.dry_run_cmd}  -> confirm? y/n")
    else:
        lines.append(f"Reason: {result.reason}")
    return "\n".join(lines)
