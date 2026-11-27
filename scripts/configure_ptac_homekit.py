#!/usr/bin/env python3
"""Replace legacy PTAC HomeKit controls through HA's supported options flow.

HA_URL and HA_TOKEN must be supplied in the environment. Never edits .storage,
resets the bridge, or prints credentials. Existing unrelated accessories and
bridge options are preserved. Requires an admin HA identity.
"""
import argparse
import asyncio
import os

import aiohttp

THERMOSTATS = ("climate.bedroom_demo_thermostat", "climate.living_room_demo_thermostat")
LEGACY_PREFIXES = ("example_bedroom_ptac", "example_living_room_ptac")


def legacy_entities(prefixes):
    return frozenset(
        f"{domain}.{prefix}_{suffix}"
        for prefix in prefixes
        for domain, suffix in (("switch", "power"), ("number", "target_temperature"),
                               ("button", "temperature_up"), ("button", "temperature_down"),
                               ("select", "operating_mode"))
    )


LEGACY = legacy_entities(LEGACY_PREFIXES)


def defaults(flow):
    return {field["name"]: field["default"] for field in flow.get("data_schema", []) if "default" in field}


def thermostat_entities(entities, thermostats=THERMOSTATS, legacy=LEGACY):
    return list(dict.fromkeys([entity for entity in entities if entity not in legacy] + list(thermostats)))


async def configure(entry_id, thermostats=THERMOSTATS, legacy_prefixes=LEGACY_PREFIXES):
    legacy = legacy_entities(legacy_prefixes)
    base = os.environ["HA_URL"].rstrip("/")
    headers = {"Authorization": "Bearer " + os.environ["HA_TOKEN"]}
    async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as session:
        async def request(method, path, data=None):
            async with session.request(method, base + path, json=data) as response:
                if response.status >= 400:
                    raise RuntimeError(f"Home Assistant {path} returned HTTP {response.status}")
                return await response.json() if response.content_type == "application/json" else None

        states = await request("GET", "/api/states")
        present = {state["entity_id"] for state in states}
        available = [entity for entity in thermostats if entity in present]
        if not available:
            raise RuntimeError("No PTAC thermostats exist yet; no bridge changes made")
        path = "/api/config/config_entries/options/flow"
        # Read the current include selection from a draft flow without committing it.
        flow = await request("POST", path, {"handler": entry_id})
        draft = flow["flow_id"]
        try:
            initial = defaults(flow)
            if flow.get("step_id") != "init" or initial.get("mode") != "bridge" or initial.get("include_exclude_mode") != "include":
                raise RuntimeError("Expected an existing UI-managed bridge with explicit entity inclusions")
            flow = await request("POST", path + "/" + draft, initial)
            if flow.get("step_id") != "include":
                raise RuntimeError("Expected the bridge entity inclusion form")
            old = defaults(flow).get("entities", [])
            new = thermostat_entities(old, available, legacy)
        finally:
            await request("DELETE", path + "/" + draft)
        # Derive domains from explicit entities. A domain with no selected entities
        # would expose its entire domain, including unrelated home controls.
        initial["domains"] = sorted({entity.split(".", 1)[0] for entity in new})
        flow = await request("POST", path, {"handler": entry_id})
        active = flow["flow_id"]
        try:
            flow = await request("POST", path + "/" + active, initial)
            flow = await request("POST", path + "/" + active, {"entities": new})
            while flow.get("type") == "form":
                data = defaults(flow)
                if flow.get("step_id") == "climate":
                    for field in flow.get("data_schema", []):
                        if any(entity in field["name"] for entity in available):
                            data[field["name"]] = "thermostat"
                elif flow.get("step_id") != "bridged_device_triggers":
                    raise RuntimeError(f"Unexpected options step: {flow.get('step_id')}; no changes committed")
                flow = await request("POST", path + "/" + active, data)
            if flow.get("type") != "create_entry":
                raise RuntimeError("HomeKit options were not saved")
            # HA strips saved options from the HTTP response. Read them back
            # through a second uncommitted form instead of relying on internals.
            verify = await request("POST", path, {"handler": entry_id})
            verify_id = verify["flow_id"]
            try:
                saved = defaults(verify)
                assert set(saved["domains"]) == set(initial["domains"])
                verify = await request("POST", path + "/" + verify_id, saved)
                assert set(defaults(verify)["entities"]) == set(new)
                assert set(old) - legacy <= set(new)
            finally:
                await request("DELETE", path + "/" + verify_id)
        except BaseException:
            if flow.get("type") != "create_entry":
                await request("DELETE", path + "/" + active)
            raise
        print(f"Saved existing HomeKit bridge {entry_id}: removed {sorted(set(old) & legacy)}; included {available}; preserved {len(set(old) - legacy - set(available))} unrelated accessories.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-id", required=True)
    parser.add_argument("--thermostat", action="append", help="Climate entity to expose. Repeat for each room.")
    parser.add_argument("--legacy-prefix", action="append", help="Source entity prefix to replace. Repeat for each room.")
    args = parser.parse_args()
    thermostats = args.thermostat or THERMOSTATS
    if not all(entity.startswith("climate.") for entity in thermostats):
        parser.error("--thermostat expects a climate entity ID")
    asyncio.run(configure(args.entry_id, thermostats, args.legacy_prefix or LEGACY_PREFIXES))
