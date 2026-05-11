#!/usr/bin/env bash
# Bring up podman + the runtime fixes harbor needs.
#
# Usage:
#   source ./setup_podman_harbor.sh        # starts podman, exports DOCKER_HOST
#   ./setup_podman_harbor.sh               # same, but DOCKER_HOST is only set for the spawned shell
#
# What it does:
#   1. Creates /dev/net/tun (needed by podman's pasta/netavark even with netns=host)
#   2. Creates /run/containers/networks/aardvark-dns (netavark DNS runtime dir)
#   3. Starts `podman system service` on unix:///tmp/podman.sock if not already up
#   4. Exports DOCKER_HOST=unix:///tmp/podman.sock
#
# Note: this script does NOT patch harbor's vendored docker-compose-base.yaml.
# Those edits (network_mode: host, :U on bind mounts) live in the installed package:
#   .venv/lib/python3.12/site-packages/harbor/environments/docker/docker-compose-base.yaml
# Re-running `uv sync` or reinstalling harbor will undo them — re-apply if that happens.

set -euo pipefail

SOCKET="${PODMAN_SOCKET:-/tmp/podman.sock}"

# 1. /dev/net/tun — pasta/netavark need this even when containers.conf has netns="host".
if [ ! -e /dev/net/tun ]; then
  mkdir -p /dev/net
  mknod /dev/net/tun c 10 200
  chmod 600 /dev/net/tun
fi

# 2. aardvark-dns runtime dir — netavark fails to create it on first use.
mkdir -p /run/containers/networks/aardvark-dns

# 3. Start podman service if the socket isn't already live.
if [ -S "$SOCKET" ] && DOCKER_HOST="unix://$SOCKET" docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
  echo "podman service already responsive at $SOCKET"
else
  rm -f "$SOCKET"
  echo "starting podman system service at $SOCKET..."
  nohup podman system service --time=0 "unix://$SOCKET" >/tmp/podman-service.log 2>&1 &
  for _ in $(seq 1 50); do
    [ -S "$SOCKET" ] && break
    sleep 0.2
  done
  if [ ! -S "$SOCKET" ]; then
    echo "podman service failed to come up — see /tmp/podman-service.log" >&2
    exit 1
  fi
fi

export DOCKER_HOST="unix://$SOCKET"
echo "DOCKER_HOST=$DOCKER_HOST"
echo "ready — run harbor with: uv run harbor run --env docker --dataset ... --agent ..."
