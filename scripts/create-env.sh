#!/usr/bin/env sh
set -eu
umask 077
printf 'API_KEY=%s\n' "$(openssl rand -hex 32)" > .env
