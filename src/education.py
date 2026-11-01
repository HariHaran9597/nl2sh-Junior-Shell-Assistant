"""Short, beginner-friendly explanations for common shell commands."""
from __future__ import annotations

import re


COMMANDS = {
    "ls": "lists files and directories",
    "find": "searches a directory tree for files matching conditions",
    "grep": "searches text for lines matching a pattern",
    "rg": "ripgrep searches files for matching text",
    "fd": "searches for files and directories by name",
    "du": "estimates disk space used by files or directories",
    "df": "shows free and used space on mounted filesystems",
    "ps": "shows processes running on the system",
    "wc": "counts lines, words, or bytes",
    "head": "prints the beginning of a file or input",
    "tail": "prints the end of a file or input",
    "sort": "sorts lines of text",
    "uniq": "filters or counts adjacent duplicate lines",
    "tar": "creates or extracts an archive",
    "gzip": "compresses a file",
    "ssh": "connects to a remote machine over SSH",
    "curl": "sends a request to a URL or transfers data",
    "chmod": "changes file permissions",
    "chown": "changes file ownership",
    "rm": "removes files or directories",
    "cp": "copies files or directories",
    "mv": "moves or renames files",
    "mkdir": "creates directories",
    "cat": "prints or joins file contents",
    "sed": "transforms text, commonly by replacing matching text",
    "awk": "selects or transforms text by fields and patterns",
    "cut": "extracts selected fields or character ranges from each line",
    "xargs": "builds command arguments from standard input",
    "git": "runs a Git version-control operation",
    "docker": "runs a Docker container or management operation",
    "python": "runs Python code or a Python script",
    "python3": "runs Python 3 code or a Python 3 script",
}

FLAGS = {
    "-a": "include hidden entries",
    "-l": "use a detailed listing",
    "-r": "process directories recursively (meaning depends on the command)",
    "-f": "force the operation or suppress prompts (meaning depends on the command)",
    "-i": "ask before overwriting or removing (meaning depends on the command)",
    "-n": "limit a count or line range (meaning depends on the command)",
    "-type f": "match regular files",
    "-type d": "match directories",
    "-mtime": "filter by file modification age",
    "-size": "filter by file size",
    "-delete": "delete each matching path",
    "-exec": "run another command for each matched path",
    "-maxdepth": "limit how many directory levels to search",
    "-iname": "match a name without case sensitivity",
    "-name": "match a file name using a pattern",
    "-print": "print each matching path",
    "-p": "create parent directories as needed (mkdir)",
    "-xzf": "extract a gzip-compressed tar archive",
    "-czf": "create a gzip-compressed tar archive",
}


def explain_command(command: str) -> str:
    """Explain recognized commands and flags without claiming full semantics."""
    text = (command or "").strip()
    if not text:
        return "No command was generated."
    first = re.split(r"\s|[|;&]", text, maxsplit=1)[0]
    first = first.rsplit("/", 1)[-1]
    explanation = COMMANDS.get(first, f"runs `{first}` with the supplied arguments")
    notes = []
    if "-h" in text and first in {"ls", "du", "df"}:
        notes.append("-h: show sizes in a human-readable format")
    for flag, meaning in FLAGS.items():
        if re.search(r"(?<!\w)" + re.escape(flag) + r"(?!\w)", text):
            # Keep the tar short-option check from matching unrelated words.
            if flag in {"-xzf", "-czf"} and not re.search(r"\btar\b", text):
                continue
            notes.append(f"{flag}: {meaning}")
    if notes:
        explanation += "; " + "; ".join(notes[:3])
    if re.search(r"\|", text):
        explanation += "; `|` passes one command's output to the next command"
    if re.search(r"&&", text):
        explanation += "; `&&` runs the next command only if the previous one succeeds"
    return explanation[0].upper() + explanation[1:] + "."
