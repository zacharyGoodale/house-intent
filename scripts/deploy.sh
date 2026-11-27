#!/usr/bin/env bash
# Upload this checkout, sync its locked dependencies, then install boot services.
set -euo pipefail
host=
stage_only=false
while (($#)); do
  case "$1" in
    --host) host=${2:?Missing SSH host}; shift 2 ;;
    --stage-only) stage_only=true; shift ;;
    -h|--help) echo 'Usage: scripts/deploy.sh --host user@example.test [--stage-only]'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$host" =~ ^[a-zA-Z0-9_.@-]+$ && "$host" != -* ]] || { echo 'Specify a valid --host' >&2; exit 2; }
repo=$(cd "$(dirname "$0")/.." && pwd)
git -C "$repo" diff --quiet
git -C "$repo" diff --cached --quiet
revision=$(git -C "$repo" rev-parse HEAD)
# Fixed paths are relative to the remote SSH account's home. Secrets and live
# data are outside the upload; never use --delete against a remote checkout.
ssh "$host" 'set -e; mkdir -p "$HOME/.local/share/ha-jev/release" "$HOME/.config/ha-jev"; chmod 700 "$HOME/.config/ha-jev"; test -x "$HOME/.local/bin/uv"'
git -C "$repo" archive HEAD | ssh "$host" 'tar -xf - -C "$HOME/.local/share/ha-jev/release"'
printf '%s\n' "$revision" | ssh "$host" 'cat > "$HOME/.local/share/ha-jev/release/.deployed-revision"'
ssh "$host" 'set -e; cd "$HOME/.local/share/ha-jev/release"; "$HOME/.local/bin/uv" sync --locked --no-dev; test -f "$HOME/.config/ha-jev/service.env" || { cp deploy/service.env.example "$HOME/.config/ha-jev/service.env"; chmod 600 "$HOME/.config/ha-jev/service.env"; }; test -f "$HOME/.config/ha-jev/users.json" || { cp deploy/users.json.example "$HOME/.config/ha-jev/users.json"; chmod 600 "$HOME/.config/ha-jev/users.json"; }'
echo "Staged revision $revision on $host: ~/.local/share/ha-jev/release"
echo 'Credentials: ~/.config/ha-jev/service.env and ~/.config/ha-jev/users.json'
if "$stage_only"; then
  echo "Activate after provisioning private config. See docs/home-assistant.md."
else
  ssh -t "$host" 'sudo bash "$HOME/.local/share/ha-jev/release/deploy/install-system-service.sh" --user "$(id -un)" --repo-dir "$HOME/.local/share/ha-jev/release"'
fi
