import os
import subprocess


def diff_versus_commit(git_dname, commit):
    """Return tracked and untracked changes versus a base commit."""
    result = subprocess.run(
        ["git", "-C", git_dname, "diff", commit],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    diff_output = result.stdout.decode("utf-8", errors="replace")

    result = subprocess.run(
        ["git", "-C", git_dname, "ls-files", "--others", "--exclude-standard"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    untracked_files = result.stdout.decode("utf-8", errors="replace").splitlines()

    for file_name in untracked_files:
        file_path = os.path.join(git_dname, file_name)
        result = subprocess.run(
            ["git", "-C", git_dname, "diff", "--no-index", "/dev/null", file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=git_dname,
            check=False,
        )
        if os.path.exists(file_path):
            diff_output += result.stdout.decode("utf-8", errors="replace")

    return diff_output
