# Home Assistant setup

## Configure a real home

1. In Home Assistant, put your devices in areas (and areas on floors), and expose to Assist the entities voice may control. That is all House Intent needs. Optional overrides go in `~/.config/ha-jev/config/` or the directory in `HA_JEV_CONFIG_DIR`; see the [configuration guide](configuration.md).
2. For direct CLI access, set `HA_URL`, `HA_TOKEN`, `HA_JEV_ALLOW_REAL=1`, and `TYPESAFE_API_KEY` in your private environment. Start the CLI with `--ha real` after checking scope and safety.
3. To load variables from a local file for development, set `HA_JEV_DEV_ENV` to its path. Keep provider keys and Home Assistant tokens in that file or your normal secret store.

## Install the voice bridge

Copy `custom_components/ha_jev` into your Home Assistant `custom_components` directory and restart Home Assistant. The integration needs the authenticated core bridge URL and, for STT and TTS, a compatible HTTPS speech service. Configure service addresses and keys in the integration flow. Use a trusted certificate on each endpoint.

For a same-host Linux deployment, install the project's dependencies and make Caddy available on `PATH`. Copy the files in `deploy/*.example` for the service environment and user mapping to `~/.config/ha-jev/service.env` and `~/.config/ha-jev/users.json`. Put your private TOML files in `~/.config/ha-jev/config/`. In the user mapping, replace the example key with the Home Assistant user ID and set the named environment variable to that user's HA token. Choose a separate bridge key of at least 32 characters. The example enables the OpenAI fallback. Adjust it if you use another backend.

Run the systemd installer as an administrator, with the account that owns the checkout and the absolute checkout path.

    sudo bash deploy/install-system-service.sh --user hauser --repo-dir /srv/ha-jev

The bridge listens on loopback. Put the Caddy CA certificate reported by the installer into the trusted CA configuration for the HA integration. For a manual deployment, run `uv run ha-jev-server --help` and provide the private user map and configuration directory. Terminate TLS before accepting requests from another machine.
