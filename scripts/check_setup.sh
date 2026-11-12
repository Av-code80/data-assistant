#!/usr/bin/env bash
# Check that a machine is ready to run the Data Assistant.
# Usage (from anywhere): bash scripts/check_setup.sh
# Exit code 0 when every required check passes, 1 otherwise (usable in CI).

set -euo pipefail
cd "$(dirname "$0")/.."

failures=0
ok()   { printf '  OK    %s\n' "$1"; }
warn() { printf '  WARN  %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; failures=$((failures + 1)); }

echo "Data Assistant setup check"

if command -v uv > /dev/null; then
  ok "uv $(uv --version | cut -d' ' -f2)"
else
  fail "uv is not installed: see https://docs.astral.sh/uv/"
fi

if uv run python -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null; then
  ok "Python $(uv run python -c 'import platform; print(platform.python_version())')"
else
  fail "Python 3.12+ is required: run 'uv sync'"
fi

if [[ ! -f .env ]]; then
  fail ".env is missing: run 'cp .env.example .env', then set ANTHROPIC_API_KEY"
else
  if grep -qE '^ANTHROPIC_API_KEY *= *["'\'']?sk-ant-' .env; then   # quotes are optional
    ok "API key is set (value not shown)"
  else
    fail "ANTHROPIC_API_KEY is missing or invalid in .env"
  fi
  permissions=$(stat -c %a .env 2>/dev/null || stat -f %Lp .env)   # Linux, then macOS
  if [[ "$permissions" == 600 ]]; then
    ok ".env is readable by its owner only"
  else
    warn ".env permissions are $permissions: run 'chmod 600 .env'"
  fi
fi

if [[ -f data/shop.duckdb ]]; then
  ok "database $(du -h data/shop.duckdb | cut -f1)"
else
  fail "database is missing: run 'uv run python scripts/generate_data.py'"
fi

docs=$(find knowledge -name '*.md' 2>/dev/null | wc -l | tr -d ' ') || true
if (( docs > 0 )); then ok "$docs knowledge documents"; else fail "no knowledge/*.md documents"; fi

if [[ -d .cache/models ]]; then
  ok "embedding model is cached"
else
  warn "embedding model is not cached: the first start downloads ~220 MB"
fi

if (( failures > 0 )); then
  echo "$failures problem(s) to fix."
  exit 1
fi
echo "Ready."
