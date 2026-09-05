"""Conservative shell safety checks.

These regexes are a guardrail, not a shell parser. They intentionally prefer
false positives over allowing a destructive command to look safe.
"""
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

    def __str__(self):
        return f"[{self.level.value}] {self.rule}: {self.detail}"


_RULES = [
    ("shell-wrapper", r"\b(?:bash|sh|zsh|dash)\s+(?:-[A-Za-z]*c\b|--command\b)",
     Level.DANGER, "embedded shell execution"),
    ("script-wrapper", r"\b(?:python|python3|perl|ruby|node)\s+(?:-[A-Za-z]*c\b|--eval\b)",
     Level.DANGER, "embedded script execution"),
    ("recursive-force-delete",
     r"\brm\b(?=[^;\n|]*\s-[A-Za-z]*r)(?=[^;\n|]*\s-[A-Za-z]*f)[^;\n|]*",
     Level.DANGER, "recursive force delete"),
    ("recursive-delete-critical",
     r"\brm\b[^;\n|]*\s(?:/|~|\.\.?)(?:\s|/|$)|\brm\b[^;\n|]*\s/(?:etc|var|usr|bin|sbin|boot|root|lib|dev|home)(?:/|\s|$)",
     Level.DANGER, "recursive or critical-path delete"),
    ("force-delete", r"\brm\b[^;\n|]*\s-[A-Za-z]*f[A-Za-z]*\s+",
     Level.CAUTION, "force rm without recursive flag"),
    ("recursive-delete", r"\brm\b[^;\n|]*\s-[A-Za-z]*r[A-Za-z]*\s+",
     Level.CAUTION, "recursive delete without force flag"),
    ("find-delete", r"\bfind\b[^;\n|]*\s-delete\b",
     Level.DANGER, "find permanently deletes matched paths"),
    ("find-exec-delete", r"\bfind\b[^;\n|]*\s-exec\s+(?:[^;\n|]*\s)?(?:rm|shred|unlink)\b",
     Level.DANGER, "find executes a destructive command per match"),
    ("device-write",
     r"\b(?:dd|mkfs(?:\.[A-Za-z0-9_-]+)?|fdisk|wipefs|shred)\b[^;\n|]*(?:of=\s*|/dev/)",
     Level.DANGER, "raw device or filesystem destruction"),
    ("download-pipe-shell",
     r"\b(?:curl|wget)\b[^;\n]*\|\s*(?:sudo\s+)?(?:ba|z|fi)?sh\b",
     Level.DANGER, "piping an untrusted download into a shell"),
    ("fork-bomb", r":\s*\(\s*\)\s*\{[^}]*\|[^}]*&|:\{\s*\|\s*:\&\};",
     Level.DANGER, "fork bomb"),
    ("firewall-flush", r"\b(?:iptables|ufw|nft)\b[^;\n|]*(?:-F|-X|\bflush\b)",
     Level.DANGER, "flushing firewall rules"),
    ("crontab-wipe", r"\bcrontab\s+(?:-r|-i\s+-r)\b",
     Level.DANGER, "wiping scheduled jobs"),
    ("system-power", r"\b(?:reboot|shutdown|poweroff|halt)\b(?:\s|$)",
     Level.DANGER, "system power state change"),
    ("privileged-system-mutation",
     r"\b(?:chmod|chown)\b[^;\n|]*\s-R[^;\n|]*(?:/etc|/usr|/boot|/root|/var|/home)(?:/|\s|$)",
     Level.DANGER, "recursive privileged system-path mutation"),
    ("sensitive-redirection",
     r"(?:>>?|\|\s*tee)\s*(?:/etc|/usr|/boot|/dev/\S+|/root|/home)(?:/|\s|$)",
     Level.DANGER, "writing to a sensitive system path"),
    ("git-clean", r"\bgit\s+clean\b[^;\n|]*\s-[A-Za-z]*f",
     Level.DANGER, "permanently removing untracked Git files"),
    ("docker-prune", r"\bdocker\s+(?:system|volume|builder)\s+prune\b[^;\n|]*\s-[A-Za-z]*f",
     Level.DANGER, "pruning Docker data"),
    ("process-kill-init", r"\b(?:kill|pkill)\b[^;\n|]*-9[^;\n|]*\b1\b",
     Level.DANGER, "killing the init process"),
    ("netcat-exec", r"\b(?:nc|ncat|netcat)\b[^;\n|]*\s-e\s",
     Level.DANGER, "network shell execution"),
]
_COMPILED = [(name, re.compile(pattern, re.I), level, detail)
             for name, pattern, level, detail in _RULES]
_SEG = re.compile(r"(;|\|\||&&|;|\\n)")


def split_compound(c: str) -> list[str]:
    segments = []
    for part in _SEG.split(c):
        part = part.strip()
        if part and part not in (";", "&&", "||", "|"):
            segments.append(part)
    return segments or [c.strip()]


def check(cmd: str):
    cmd = (cmd or "").strip()
    if not cmd:
        return Level.SAFE, []
    worst = Level.SAFE
    findings = []
    seen = set()
    for segment in split_compound(cmd):
        canonical = re.sub(r"\s+", " ", segment.strip())
        for name, regex, level, detail in _COMPILED:
            if regex.search(canonical):
                key = (name, segment)
                if key in seen:
                    continue
                seen.add(key)
                # Constrained relative-path cleanup is risky but reviewable;
                # an unbounded root/home cleanup is a hard stop.
                if name == "find-delete" and not re.search(r"\bfind\s+(?:/|~)(?:\s|$)", canonical):
                    level = Level.CAUTION
                findings.append(Finding(level, name, detail))
                if level.rank > worst.rank:
                    worst = level
    return worst, findings


def is_dangerous(cmd: str) -> bool:
    return check(cmd)[0] == Level.DANGER


def render(cmd: str) -> str:
    level, findings = check(cmd)
    if level == Level.SAFE:
        return cmd
    lines = [cmd] + [f"  !! {finding.level.value}  {finding.detail}" for finding in findings]
    return "\n".join(lines)
