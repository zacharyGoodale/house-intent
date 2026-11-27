# Configuring your home

House Intent learns your home from Home Assistant. It needs no config files to run. You add files only to change something.

## What comes from Home Assistant

- **Rooms** are your HA areas, with their HA aliases. You can have any number of rooms: three bedrooms, a studio, a garage workshop. Each request carries the area of the voice satellite that heard it, so "the lights" means the lights in that room.
- **Floors** are your HA floors. A floor's name and aliases count as names for every area on it, so "turn off the upstairs lights" covers all upstairs rooms.
- **Devices** are the entities you expose to Assist (Settings → Voice assistants → Expose). Newly exposed entities are picked up as soon as the registry changes. A device's name, aliases, area and supported features all come from HA.
- **Device kinds** are built in: lights, switches, automations, covers, media players, thermostats, scenes, scripts, to-do lists, plus read-only sensors, locks, weather, sun and people. Jev is only asked about kinds your home has. A home with no blinds never gets a blinds question.

## Where config goes

The directory is the first of:

1. `--config-dir` (the `ha-jev-server` option),
2. the `HA_JEV_CONFIG_DIR` environment variable,
3. `~/.config/ha-jev/config/`, if it exists.

Every file in it is optional, and each file holds only the keys you change. A directory you name that doesn't exist is an error, so a typo never quietly falls back to the defaults. `demo/config/` is a complete worked example for the fictional demo home.

| File | Purpose |
| --- | --- |
| `scope.toml` | Which entities voice can reach, extra names, per-room defaults, presets |
| `safety.toml` | Additions to the denylist |
| `people.toml` | Household members, their wake words, units and calendars |
| `capabilities.toml` | New device kinds without code |
| `ha_jev.toml` | Model, backend, policy and memory settings |

## scope.toml

```toml
include_exposed = true             # every Assist-exposed entity (the default)
include = ["script.movie_night"]   # more entities, globs allowed
exclude = ["switch.*_child_lock", "calendar.*"]

[aliases]                          # extra names; keys are entity globs or "area:<area_id>"
"light.kids_bedroom_*" = ["Ava's lights"]
"area:primary_bedroom" = ["our room", "master bedroom"]

[defaults]
list = "todo.shopping_list"        # where "add milk" goes
transport_player = { primary_bedroom = "media_player.bedroom_speaker", kitchen = "media_player.kitchen" }
high_stakes = ["switch.space_heater"]   # needs a higher confidence before switching on
```

Keep `calendar.*` excluded. Calendars are read through `people.toml`, and calendar states carry event text.

If your dashboard scripts wrap an integration's scenes, say `qualified_scene_platforms = ["hue"]` at the top of the file. Those scenes are then offered only when the integration is named ("Hue scene relax"), so they don't crowd out the scripts. By default, every scene is offered by its own name.

`[[presets]]` defines multi-call composites that have no HA script. See `demo/config/scope.toml`. `[music]` turns on free-form "play …" through Music Assistant (`integration = "music_assistant"`).

## safety.toml

These rules are built in: lock, alarm and button domains are denied; garage, gate and door covers are refused; and covers, switches, scripts, scenes and automations with security words in their names are denied (lock, deadbolt, gate, door, entry, and others). A cover in a garage, gate, driveway or carport area is refused too. Scripts and automations whose definitions touch a denied entity are refused as well.

Most homes only add entities:

```toml
denied_entities = ["switch.front_gate_relay"]
allowed_entities = ["cover.patio_door_blind"]   # skips the name/area heuristics; never the hard denies
```

Setting a field replaces its default. If you set `denied_domains`, list every domain you want denied. The defaults are in `SafetyConfig` in `src/ha_jev/config.py`.

## people.toml

```toml
[people.sam]
name = "Sam"
temperature_unit = "C"
agents = ["conversation.jev_sam"]        # the HA conversation agent of Sam's wake word pipeline
calendars = ["calendar.sam*"]
calendar_share = "household"             # or "titles_private" / "private"
```

Without people the router still works. Requests are anonymous, and memory is household-only.

## ha_jev.toml: handoffs

A handoff adds an intent that the Hermes agent answers, for agents with their own tools, such as a restaurant-booking agent. Jev sees `what` and `not_for` as a new intent option. Any of the `keywords` (regex word patterns) in an utterance routes it straight to the agent. Homes without handoffs never see these options.

```toml
[[handoffs]]
id = "reservations"
what = "Anything about restaurants, dining reservations or booking a table."
not_for = "ordinary calendar appointments"
keywords = ["reservations?", "restaurants?"]
```

## capabilities.toml

A new domain is one `[[domains]]` row: what it is, which actions it has, and the HA service for each action. A row with the same `domain` as a built-in replaces the built-in. `demo/config/capabilities.toml` adds the demo home's AC fan-speed selectors, and `tests/unit/test_extensibility.py` shows a `fan` row.
