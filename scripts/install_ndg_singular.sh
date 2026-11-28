#!/bin/sh
# Ubuntu/Debian local install; no root or system package mutations.
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ndg_prefix=${NDG_SINGULAR_PREFIX:-"$repo_root/.venv/share/ndg-singular"}
ndg_packages=$(mktemp -d)
trap 'rm -rf "$ndg_packages"' EXIT HUP INT TERM
mkdir -p "$ndg_prefix"
cd "$ndg_packages"
# apt resolves the distribution's versions of the native dependencies.
apt-get download singular-ui singular-data singular-modules libsingular4m1 libflint-2.5.2 libntl43 libcdd0d libgf2x3
for ndg_package in ./*.deb; do
    dpkg-deb -x "$ndg_package" "$ndg_prefix"
done
printf 'Native Singular installed in %s\n' "$ndg_prefix"
