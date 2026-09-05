"""Risk classification for the standalone Space app."""
from __future__ import annotations

import re
import safety


HIGH = {"rm", "mkfs", "dd", "shred", "chmod", "chown", "iptables", "crontab", "kill", "pkill",
        "reboot", "shutdown", "poweroff", "halt", "fdisk", "wipefs", "git", "docker", "nc", "ncat", "netcat"}
MED = {"find", "chmod", "chown", "tar", "rsync", "mv", "cp", "truncate", "wipefs"}
FLAG_EXPLAIN = {
    "tar -xzf": "-x extract, -z gzip, -f file",
    "tar -czf": "-c create, -z gzip, -f file",
    "find -delete": "-delete removes matched files permanently",
    "find -exec": "-exec runs a command per match",
    "rm -rf": "-r recursive, -f force (no confirm)",
    "chmod -R": "-R recursive permission change",
    "du -sh": "-s summarize, -h human-readable",
    "ps aux": "a all users, u user format, x incl. no-tty",
}


class RiskResult:
    def __init__(self, level: str, reason: str, explain: str, dry_run_cmd: str | None = None):
        self.level = level
        self.reason = reason
        self.explain = explain
        self.dry_run_cmd = dry_run_cmd


def _first_tool(command: str) -> str:
    parts = command.split()
    while parts and "=" in parts[0] and not parts[0].startswith("="):
        parts.pop(0)
    if parts and parts[0] == "sudo":
        parts.pop(0)
    return parts[0] if parts else ""


def classify(command: str) -> RiskResult:
    command = (command or "").strip()
    first = _first_tool(command)
    level, findings = safety.check(command)
    if level == safety.Level.DANGER:
        reason = findings[0].detail if findings else "destructive or executable shell pattern"
        return RiskResult("HIGH", reason, explain_cmd(command), dry_run_for(command))
    if first in HIGH:
        return RiskResult("HIGH", f"{first} can change system state", explain_cmd(command), dry_run_for(command))
    if first in MED or " -delete" in command or " -exec rm" in command:
        return RiskResult("MED", f"{first or 'command'} modifies files", explain_cmd(command), dry_run_for(command))
    return RiskResult("LOW", "read-only / low risk", explain_cmd(command), None)


def explain_cmd(command: str) -> str:
    parts = [f"{key}: {value}" for key, value in FLAG_EXPLAIN.items()
             if key.split()[0] in command and any(flag in command for flag in key.split()[1:])]
    return " | ".join(parts) if parts else "no extra notes"


def dry_run_for(command: str) -> str | None:
    if "find" in command and "-delete" in command:
        return command.replace("-delete", "-print") + "  # dry-run: lists matched paths"
    if "find" in command and "-exec rm" in command:
        return command.replace("-exec rm", "-print") + "  # dry-run: lists matched paths"
    return None


def render_risk(command: str) -> str:
    result = classify(command)
    return f"[{result.level}] {command}\nExplain: {result.explain}\nReason: {result.reason}"
