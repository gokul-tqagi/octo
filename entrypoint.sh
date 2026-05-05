#!/bin/bash
# entrypoint.sh: Install mounted project as editable package, then run the command.

if [ -f /octo/setup.py ] || [ -f /octo/pyproject.toml ]; then
    pip install -e /octo -q 2>/dev/null
fi

exec "$@"
