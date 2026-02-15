#!/usr/bin/env python3
"""Git operations helper to work around shell issues."""

import os
import subprocess
import sys


def run_git_command(cmd):
    """Run a git command and return result."""
    try:
        result = subprocess.run(
            ["git"] + cmd.split()[1:] if cmd.startswith("git ") else cmd.split(),
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def main():
    """Check git status and stage auto-instrumentation files."""
    print("🔍 Checking Git status...")

    # Check git status
    retcode, stdout, stderr = run_git_command("git status --porcelain")
    if retcode != 0:
        print(f"❌ Git status failed: {stderr}")
        return 1

    print("📁 Current changes:")
    if stdout.strip():
        for line in stdout.strip().split("\n"):
            print(f"  {line}")
    else:
        print("  No changes detected")

    # Add auto-instrumentation files
    files_to_add = [
        "src/genops/auto_instrumentation.py",
        "src/genops/__init__.py",
        "src/genops/cli/main.py",
        "examples/auto_instrumentation.py",
        "test_auto_init.py",
    ]

    print(f"\n📦 Staging {len(files_to_add)} auto-instrumentation files...")

    for file in files_to_add:
        if os.path.exists(file):
            retcode, stdout, stderr = run_git_command(f"git add {file}")
            if retcode == 0:
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file}: {stderr}")
        else:
            print(f"  ⚠️ {file}: File not found")

    # Check status after adding
    print("\n🔍 Status after staging:")
    retcode, stdout, stderr = run_git_command("git status --porcelain")
    if retcode == 0 and stdout.strip():
        for line in stdout.strip().split("\n"):
            print(f"  {line}")

    # Create commit
    commit_msg = """Add auto-instrumentation system inspired by OpenLLMetry

- Implement GenOpsInstrumentor class with singleton pattern
- Add genops.init() for one-line setup similar to OpenLLMetry
- Auto-detect and instrument available providers (OpenAI, Anthropic)
- Support configurable defaults for team/project governance attributes
- Add comprehensive examples and CLI integration
- Enable uninstrumentation and status checking

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"""

    print("\n📝 Creating commit...")
    retcode, stdout, stderr = run_git_command(f'git commit -m "{commit_msg}"')

    if retcode == 0:
        print("✅ Commit created successfully!")
        print(f"Output: {stdout}")

        # Try to push
        print("\n🚀 Pushing to GitHub...")
        retcode, stdout, stderr = run_git_command("git push origin main")
        if retcode == 0:
            print("✅ Successfully pushed to GitHub!")
            return 0
        else:
            print(f"❌ Push failed: {stderr}")
            return 1
    else:
        print(f"❌ Commit failed: {stderr}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
