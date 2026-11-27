#!/usr/bin/env bash
# Run as root on Linux; code/inference run as the existing unprivileged account.
set -euo pipefail
service_user=
repo=
while (($#)); do
  case "$1" in
    --user) service_user=${2:?Missing service user}; shift 2 ;;
    --repo-dir) repo=${2:?Missing repository directory}; shift 2 ;;
    -h|--help) echo 'Usage: sudo bash deploy/install-system-service.sh --user <account> --repo-dir <absolute-checkout-path>'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ $EUID == 0 ]] || { echo 'Run this installer with sudo.' >&2; exit 1; }
[[ "$service_user" =~ ^[a-z_][a-z0-9_-]*$ && "$service_user" != root ]] || { echo 'Specify a non-root --user.' >&2; exit 2; }
service_home=$(getent passwd "$service_user" | cut -d: -f6)
[[ "$service_home" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'Unsupported home directory.' >&2; exit 2; }
service_group=$(id -gn "$service_user")
[[ "$repo" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'Specify an absolute --repo-dir.' >&2; exit 2; }
private="$service_home/.config/ha-jev"
caddy=$(command -v caddy) || { echo "Install Caddy on PATH first." >&2; exit 1; }
test -x "$repo/.venv/bin/ha-jev-server"
test -x "$caddy" || { echo "Install Caddy at $caddy first." >&2; exit 1; }
test -s "$private/service.env"
test -s "$private/users.json"
# Every config file is optional (docs/configuration.md); an empty directory runs on Home Assistant's own setup.
test -d "$private/config" || install -d -m 700 -o "$service_user" -g "$service_group" "$private/config"
if grep -q REPLACE_WITH "$private/service.env" "$private/users.json"; then
  echo 'Provision service.env and users.json before activation.' >&2; exit 1
fi
chmod 700 "$private"
chmod 600 "$private/service.env" "$private/users.json"
# Do not source secret configuration as shell code. Validate required names and
# the actual HA identity again in the bridge at startup.
for name in HA_URL HA_JEV_BRIDGE_KEY TYPESAFE_API_KEY; do
  grep -q "^${name}=.\+" "$private/service.env" || { echo "Missing $name" >&2; exit 1; }
done
install -d -m 755 /etc/ha-jev
cat > /etc/ha-jev/Caddyfile <<'CADDY'
{
    admin off
    auto_https disable_redirects
    skip_install_trust
}
# HA uses host networking on the same machine: no LAN bridge port is needed.
https://127.0.0.1:8444 {
    bind 127.0.0.1
    tls internal
    reverse_proxy 127.0.0.1:8766
}
CADDY
"$caddy" validate --config /etc/ha-jev/Caddyfile --adapter caddyfile
cat > /etc/systemd/system/ha-jev.service <<UNIT
[Unit]
Description=HA Jev authenticated conversation bridge
Wants=network-online.target
After=network-online.target docker.service

[Service]
Type=simple
User=$service_user
Group=$service_group
WorkingDirectory=/var/lib/ha-jev
Environment=HA_JEV_FALLBACK=luna
EnvironmentFile=$private/service.env
Environment=HA_JEV_DATA_DIR=/var/lib/ha-jev
ExecStart=$repo/.venv/bin/ha-jev-server --users-file $private/users.json --config-dir $private/config --fallback \${HA_JEV_FALLBACK}
Restart=on-failure
RestartSec=5
TimeoutStopSec=30
StateDirectory=ha-jev
StateDirectoryMode=0700
UMask=0077
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6

[Install]
WantedBy=multi-user.target
UNIT
cat > /etc/systemd/system/ha-jev-proxy.service <<UNIT
[Unit]
Description=Private TLS proxy for HA Jev
Wants=ha-jev.service
After=network-online.target ha-jev.service

[Service]
Type=simple
User=$service_user
Group=$service_group
Environment=XDG_DATA_HOME=/var/lib/ha-jev-proxy
Environment=XDG_CONFIG_HOME=/var/lib/ha-jev-proxy
ExecStart=$caddy run --config /etc/ha-jev/Caddyfile --adapter caddyfile
Restart=on-failure
RestartSec=5
StateDirectory=ha-jev-proxy
StateDirectoryMode=0700
UMask=0077
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6

[Install]
WantedBy=multi-user.target
UNIT
systemd-analyze verify /etc/systemd/system/ha-jev.service /etc/systemd/system/ha-jev-proxy.service
systemctl daemon-reload
systemctl enable ha-jev.service ha-jev-proxy.service
systemctl restart ha-jev.service ha-jev-proxy.service
# The bridge binds only after user-token checks and HA registry initialization.
# Verify the proxy with its public CA, including IP SAN, and require auth to be
# enforced. Never put the bridge bearer key in argv or deployment output.
"$repo/.venv/bin/python" - <<'PY'
import ssl, time, urllib.error, urllib.request
from pathlib import Path
ca = Path('/var/lib/ha-jev-proxy/caddy/pki/authorities/local/root.crt')
for attempt in range(30):
    try:
        ctx = ssl.create_default_context(cafile=str(ca))
        urllib.request.urlopen('https://127.0.0.1:8444/v1/models', context=ctx, timeout=2)
    except urllib.error.HTTPError as error:
        if error.code == 401:
            print('Verified TLS proxy and authenticated bridge listener ready.')
            break
    except (OSError, ValueError):
        pass
    time.sleep(1)
else:
    raise SystemExit('Bridge did not become ready; inspect systemctl status ha-jev ha-jev-proxy.')
PY
systemctl is-active --quiet ha-jev.service ha-jev-proxy.service
echo 'Enabled system services: ha-jev and ha-jev-proxy (boot persistent).'
echo 'HA core URL: https://127.0.0.1:8444'
echo 'Public CA: /var/lib/ha-jev-proxy/caddy/pki/authorities/local/root.crt'
echo 'Copy that public certificate to HA /config/certs/jev-root.crt.'
