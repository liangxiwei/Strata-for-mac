#!/bin/sh
# Strata for Linux (and the macOS development setup): the first run installs everything and starts the model;
# later runs just start it.  Needs only an NVIDIA driver (or, for an AMD Radeon card, the kernel's amdgpu
# driver: see docs/AMD_HIP.md).  On a Mac the engine cannot run (it needs an NVIDIA or AMD card); setup.py
# says so and points at docs/MAC.md, which has the server-and-tests steps that DO work there.
# Python (with venv) is installed through apt/dnf/pacman/brew if it is missing (asks for sudo where it needs it).
cd "$(dirname "$0")" || exit 1
# Python 3.10+ that can make a venv WITH pip: Debian/Ubuntu ship `venv` without `ensurepip` (that is the separate
# python3-venv package), and a venv made without it has no pip
ok_py() { "$1" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; }
# a .venv from an earlier run that failed half-way has a python but no pip: start it again
if [ -x .venv/bin/python ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
  PY=""
  for c in python3 python python3.13 python3.12 python3.11 python3.10; do
    if command -v $c >/dev/null 2>&1 && ok_py $c; then
      PY=$c; break
    fi
  done
  if [ -z "$PY" ]; then
    echo "Python 3.10+ with venv is needed; installing it (sudo will ask for your password) ..."
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
    elif command -v dnf >/dev/null 2>&1; then
      sudo dnf install -y python3 python3-pip
    elif command -v pacman >/dev/null 2>&1; then
      sudo pacman -S --noconfirm python python-pip
    elif command -v brew >/dev/null 2>&1; then
      brew install python
    fi
    PY=python3
    if ! ok_py $PY; then
      echo "Please install Python 3.10 or newer with venv, then run this again:"
      echo "  Ubuntu/Debian: sudo apt install python3-venv   Fedora: sudo dnf install python3-pip"
      echo "  macOS: brew install python                     Arch: sudo pacman -S python-pip"
      exit 1
    fi
  fi
  # a private environment inside this folder (system Python stays untouched; newer distros refuse global pip)
  $PY -m venv .venv || { rm -rf .venv; echo "could not create .venv: sudo apt install python3-venv"; exit 1; }
fi
exec .venv/bin/python setup.py "$@"
