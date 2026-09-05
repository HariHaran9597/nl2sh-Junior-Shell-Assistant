"""Conservative shell safety checks for the standalone Space app."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Level(Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    DANGER = "DANGER"

    @property
    def rank(self):
        return {"SAFE": 0, "CAUTION": 1, "DANGER": 2}[self.value]


@dataclass(frozen=True)
class Finding:
    level: Level
    rule: str
    detail: str


_RULES = [
    ("shell-wrapper", r"\b(?:bash|sh|zsh|dash)\s+(?:-[A-Za-z]*c\b|--command\b)", "embedded shell execution"),
    ("script-wrapper", r"\b(?:python|python3|perl|ruby|node)\s+(?:-[A-Za-z]*c\b|--eval\b)", "embedded script execution"),
    ("recursive-force-delete", r"\brm\b(?=[^;\n|]*\s-[A-Za-z]*r)(?=[^;\n|]*\s-[A-Za-z]*f)[^;\n|]*", "recursive force delete"),
    ("recursive-delete-critical", r"\brm\b[^;\n|]*\s(?:/|~|\.\.?)(?:\s|/|$)|\brm\b[^;\n|]*\s/(?:etc|var|usr|bin|sbin|boot|root|lib|dev|home)(?:/|\s|$)", "recursive or critical-path delete"),
    ("force-delete", r"\brm\b[^;\n|]*\s-[A-Za-z]*f[A-Za-z]*\s+", "force rm without recursive flag"),
    ("recursive-delete", r"\brm\b[^;\n|]*\s-[A-Za-z]*r[A-Za-z]*\s+", "recursive delete without force flag"),
    ("find-delete", r"\bfind\b[^;\n|]*\s-delete\b", "find permanently deletes matched paths"),
    ("find-exec-delete", r"\bfind\b[^;\n|]*\s-exec\s+(?:[^;\n|]*\s)?(?:rm|shred|unlink)\b", "find executes a destructive command per match"),
    ("device-write", r"\b(?:dd|mkfs(?:\.[A-Za-z0-9_-]+)?|fdisk|wipefs|shred)\b[^;\n|]*(?:of=\s*|/dev/)", "raw device or filesystem destruction"),
    ("download-pipe-shell", r"\b(?:curl|wget)\b[^;\n]*\|\s*(?:sudo\s+)?(?:ba|z|fi)?sh\b", "piping an untrusted download into a shell"),
    ("fork-bomb", r":\s*\(\s*\)\s*\{[^}]*\|[^}]*&|:\{\s*\|\s*:\&\};", "fork bomb"),
    ("firewall-flush", r"\b(?:iptables|ufw|nft)\b[^;\n|]*(?:-F|-X|\bflush\b)", "flushing firewall rules"),
    ("crontab-wipe", r"\bcrontab\s+(?:-r|-i\s+-r)\b", "wiping scheduled jobs"),
    ("system-power", r"\b(?:reboot|shutdown|poweroff|halt)\b(?:\s|$)", "system power state change"),
    ("privileged-system-mutation", r"\b(?:chmod|chown)\b[^;\n|]*\s-R[^;\n|]*(?:/etc|/usr|/boot|/root|/var|/home)(?:/|\s|$)", "recursive privileged system-path mutation"),
    ("sensitive-redirection", r"(?:>>?|\|\s*tee)\s*(?:/etc|/usr|/boot|/dev/\S+|/root|/home)(?:/|\s|$)", "writing to a sensitive system path"),
    ("git-clean", r"\bgit\s+clean\b[^;\n|]*\s-[A-Za-z]*f", "permanently removing untracked Git files"),
    ("docker-prune", r"\bdocker\s+(?:system|volume|builder)\s+prune\b[^;\n|]*\s-[A-Za-z]*f", "pruning Docker data"),
    ("process-kill-init", r"\b(?:kill|pkill)\b[^;\n|]*-9[^;\n|]*\b1\b", "killing the init process"),
    ("netcat-exec", r"\b(?:nc|ncat|netcat)\b[^;\n|]*\s-e\s", "network shell execution"),
]
_COMPILED = [(name, re.compile(pattern, re.I), detail) for name, pattern, detail in _RULES]
_SEG = re.compile(r"(;|\|\||&&|;|\\n)")


def split_compound(command: str) -> list[str]:
    parts = [part.strip() for part in _SEG.split(command)]
    return [part for part in parts if part and part not in (";", "&&", "||", "|")] or [command.strip()]


def check(command: str):
    command = (command or "").strip()
    if not command:
        return Level.SAFE, []
    worst, findings, seen = Level.SAFE, [], set()
    for segment in split_compound(command):
        for name, regex, detail in _COMPILED:
            key = (name, segment)
            if regex.search(re.sub(r"\s+", " ", segment)) and key not in seen:
                seen.add(key)
                level = Level.DANGER if name not in {"force-delete", "recursive-delete"} else Level.CAUTION
                if name == "find-delete" and not re.search(r"\bfind\s+(?:/|~)(?:\s|$)", segment):
                    level = Level.CAUTION
                finding = Finding(level, name, detail)
                findings.append(finding)
                if finding.level.rank > worst.rank:
                    worst = finding.level
    return worst, findings


def is_dangerous(command: str) -> bool:
    return check(command)[0] == Level.DANGER


def render(command: str) -> str:
    level, findings = check(command)
    return command if level == Level.SAFE else "\n".join([command] + [f"  !! {f.level.value}  {f.detail}" for f in findings])
