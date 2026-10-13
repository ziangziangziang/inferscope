#!/bin/bash
# Tunnel watchdog for InferScope.
#
# The rc-tunnel's frpc client periodically loses its broker credentials
# ("invalid_client"), which leaves a hostname that returns frp's 404 page even
# though InferScope is healthy on loopback. This re-issues the tunnel whenever
# the broker reports no active tunnel for this pod.
#
# Deliberately conservative:
#   * it only *checks and re-issues* the tunnel — it never touches the app
#   * it asks the broker (`rc-tunnel status`), which is the authority, rather
#     than HTTP-probing the public URL: the pod has no hairpin route to its own
#     public host, so such a probe always fails
#   * a lockfile prevents two watchdogs racing
#   * repeated failures back off, so a broker outage does not become a loop
set -uo pipefail

PORT="${INFERSCOPE_PORT:-8090}"
# A fixed public prefix. Without it every re-issue mints a new random hostname,
# so any URL handed to a user or bookmarked is dead within hours. With it the
# address is constant and re-issuing is invisible.
PREFIX="${INFERSCOPE_TUNNEL_PREFIX:-inferscope}"
CHECK_EVERY="${WATCHDOG_CHECK_S:-60}"
BACKOFF_MAX="${WATCHDOG_BACKOFF_MAX_S:-600}"
LOG=/workspace/inferscope/tunnel-watchdog.log
PIDFILE=/tmp/tunnel-watchdog.pid
RC=/root/.local/bin/rc-tunnel

log() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$LOG"; }

# Single instance via a pidfile rather than flock.
#
# flock was the first attempt and it deadlocked: a SIGKILLed watchdog leaves a
# zombie whose open file descriptor keeps the advisory lock held, so the next
# start refused with "another watchdog already running" while nothing was
# actually running. A pidfile checked against /proc/<pid>/cmdline is immune to
# that, and it also rejects a recycled PID that is some other program.
claim_pidfile() {
  if [ -f "$PIDFILE" ]; then
    local old
    old=$(cat "$PIDFILE" 2>/dev/null)
    if [ -n "$old" ] && kill -0 "$old" 2>/dev/null \
       && grep -qa tunnel-watchdog "/proc/$old/cmdline" 2>/dev/null; then
      log "another watchdog already running (pid $old); exiting"
      return 1
    fi
  fi
  echo $$ > "$PIDFILE" || return 1
  trap 'rm -f "$PIDFILE"' EXIT
  return 0
}

# The broker env lives in PID 1's environment; a non-interactive shell does not
# inherit it, which is why rc-tunnel fails with a bare invocation.
#
# Each assignment must be exported separately: `export "$kv"` on a multi-line
# string exports one (invalid) name and leaves FRP_BROKER_URL unset — which
# makes `expose` fail *after* `stop` has already torn the tunnel down.
load_env() {
  local line
  while IFS= read -r line; do
    case "$line" in
      *=*) export "${line?}" || return 1 ;;
    esac
  done < <(tr '\0' '\n' < /proc/1/environ 2>/dev/null \
           | grep -E '^(FRP_BROKER_URL|FRP_BROKER_SERVICE_HOST|FRP_BROKER_SERVICE_PORT|FRP_BROKER_TLS_SERVER_NAME|RC_TUNNEL_CA_FILE)=')
  [ -n "${FRP_BROKER_URL:-}${FRP_BROKER_SERVICE_HOST:-}" ]
}

tunnel_ok() {
  # exit 0 == the broker still knows about an active tunnel for this pod
  "$RC" status >/dev/null 2>&1
}

reissue() {
  # Refuse to touch a working tunnel unless we can actually rebuild it.
  if [ -z "${FRP_BROKER_URL:-}${FRP_BROKER_SERVICE_HOST:-}" ]; then
    log "ABORT: broker env missing; refusing to stop a working tunnel"
    return 1
  fi
  log "re-issuing tunnel on port $PORT"
  "$RC" stop  >/dev/null 2>&1
  sleep 3
  local out
  out=$("$RC" expose --port "$PORT" --prefix "$PREFIX" 2>&1)
  local url
  url=$(printf '%s' "$out" | grep -o 'https://[^ ]*' | head -1)
  if [ -n "$url" ]; then
    log "OK: tunnel is $url"
  else
    log "FAILED to re-issue: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-200)"
    return 1
  fi
  # Confirm the broker agrees before we call it done.
  sleep 8
  if tunnel_ok; then log "broker confirms tunnel active"; else log "WARN: broker still reports no active tunnel"; return 1; fi
}

claim_pidfile || exit 0
log "watchdog started (pid=$$ port=$PORT check=${CHECK_EVERY}s)"

if [ ! -x "$RC" ]; then
  log "ERROR: $RC not found or not executable; exiting"
  exit 1
fi

if ! load_env; then
  log "ERROR: could not read FRP_BROKER_* from /proc/1/environ; exiting"
  exit 1
fi
log "broker env loaded (FRP_BROKER_URL=${FRP_BROKER_URL:-<service host>})"

backoff=0
while true; do
  if tunnel_ok; then
    [ "$backoff" -ne 0 ] && log "tunnel recovered; backoff reset"
    backoff=0
  else
    if reissue; then
      backoff=0
    else
      # 60s, 120s, 240s ... capped at BACKOFF_MAX
      backoff=$(( backoff == 0 ? 30 : backoff * 2 ))
      [ "$backoff" -gt "$BACKOFF_MAX" ] && backoff="$BACKOFF_MAX"
      log "retrying in ${backoff}s"
      sleep "$backoff"
      continue
    fi
  fi
  sleep "$CHECK_EVERY"
done