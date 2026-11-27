"""Persistent one-shot schedules, timestamp signals, and due events owned by HA."""
import asyncio
from datetime import datetime, timedelta, timezone
import uuid
import logging
from pathlib import Path

import voluptuous as vol
from homeassistant.core import SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.event import async_track_point_in_utc_time
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

DOMAIN = 'ha_jev'
ALERT_URL = '/api/ha_jev/alert.wav'
_LOGGER = logging.getLogger(__name__)
SCHEMA = vol.Schema({
    vol.Required('op'): vol.In(('set', 'replace', 'cancel', 'query', 'create', 'list', 'update', 'delete')),
    vol.Required('kind'): vol.In(('timer', 'alarm')),
    vol.Optional('id'): vol.All(str, vol.Length(min=1, max=64)),
    vol.Optional('new_label'): vol.All(str, vol.Length(max=64)),
    vol.Optional('at'): str,
    vol.Optional('label', default=''): vol.All(str, vol.Length(max=64)),
    vol.Optional('area_id', default=''): vol.All(str, vol.Length(max=128)),
    vol.Optional('device_id'): vol.All(str, vol.Length(min=1, max=128)),
    vol.Optional('seconds'): vol.All(vol.Coerce(float), vol.Range(min=0.001, max=604800)),
    vol.Optional('hour'): vol.All(int, vol.Range(min=0, max=23)),
    vol.Optional('minute', default=0): vol.All(int, vol.Range(min=0, max=59)),
    vol.Optional('tomorrow', default=False): bool,
    vol.Optional('all', default=False): bool,
})


def format_time(at):
    clock = str(at.hour % 12 or 12)
    if at.minute or at.second:
        clock += f":{at.minute:02d}"
    if at.second:
        clock += f":{at.second:02d}"
    return f"{clock} {at:%p}"


async def async_setup(hass):
    from homeassistant.components.http import StaticPathConfig
    await hass.http.async_register_static_paths([
        StaticPathConfig(ALERT_URL, str(Path(__file__).with_name('alert.wav')), cache_headers=True)
    ])
    manager = ScheduleManager(hass)
    await manager.load()
    hass.data.setdefault(DOMAIN, {})['schedules'] = manager
    hass.services.async_register(DOMAIN, 'manage_schedule', manager.manage,
                                 schema=SCHEMA, supports_response=SupportsResponse.OPTIONAL)


class ScheduleManager:
    def __init__(self, hass):
        self.hass = hass
        self.store = Store(hass, 1, 'ha_jev.schedules')
        self.items = []
        self.lock = asyncio.Lock()
        self.unsubscribe = None

    async def load(self):
        self.items = (await self.store.async_load() or {}).get('items', [])
        self.publish()
        self.arm()

    def publish(self):
        for kind in ('alarm', 'timer'):
            items = sorted((i for i in self.items if i['kind'] == kind), key=lambda i: i['at'])
            self.hass.states.async_set(f'sensor.ha_jev_next_{kind}', items[0]['at'] if items else 'unknown', {
                'friendly_name': f'Jev next {kind}', 'device_class': 'timestamp',
                'count': len(items), 'schedules': items,
                **({k: items[0][k] for k in ('id', 'user_id', 'area_id', 'label')} if items else {}),
            })

    def arm(self):
        if self.unsubscribe:
            self.unsubscribe()
            self.unsubscribe = None
        if self.items:
            at = min(datetime.fromisoformat(i['at']) for i in self.items)
            self.unsubscribe = async_track_point_in_utc_time(self.hass, self.due, at)

    async def commit(self, items):
        # Persist before acknowledging a voice command or publishing a changed timestamp.
        await self.store.async_save({'items': items})
        self.items = items
        self.publish()
        self.arm()

    async def manage(self, call):
        user_id = call.context.user_id
        if not user_id:
            raise HomeAssistantError('A signed-in HA user is required')
        async with self.lock:
            result = await self._manage(call.data, user_id)
        if call.return_response:
            return result
        # Hermes's native HA tool uses plain REST calls without response data.
        # A no-op must be a real service error rather than a false success.
        if call.data['op'] not in ('query', 'list') and not result['changed']:
            raise HomeAssistantError(result['reply'])
        return None

    async def _manage(self, data, user_id):
        kind = data['kind']
        op = {'create': 'set', 'update': 'replace', 'delete': 'cancel'}.get(data['op'], data['op'])
        if data.get('id') and op == 'set':
            raise HomeAssistantError('IDs are assigned by Home Assistant when creating schedules')
        label = data.get('label', '').strip()
        own = sorted((i for i in self.items if i['user_id'] == user_id and i['kind'] == kind
                      and (not data.get('id') or i['id'] == data['id'])
                      and (not label or i['label'].casefold() == label.casefold())), key=lambda i: i['at'])
        if op == 'cancel' and kind == 'alarm' and 'hour' in data:
            # A spoken clock time selects stored local times, not today's date or a label.
            tomorrow = dt_util.now().date() + timedelta(days=1)
            own = [i for i in own
                   if (at := dt_util.as_local(datetime.fromisoformat(i['at']))).hour == data['hour']
                   and at.minute == data.get('minute', 0)
                   and (not data.get('tomorrow') or at.date() == tomorrow)]
        if op == 'list':
            summary = '; '.join(f"{i['label'] or kind.capitalize()} at {format_time(dt_util.as_local(datetime.fromisoformat(i['at'])))}" for i in own)
            return {'reply': f'You have {len(own)} matching {kind}s.' + (f' {summary}.' if summary else ''),
                    'changed': False, 'schedules': own}
        if op == 'query':
            if not own:
                return {'reply': f"You don't have a Jev {kind} set.", 'changed': False, 'schedules': []}
            at = dt_util.as_local(datetime.fromisoformat(own[0]['at']))
            remaining = max(0, int((at.timestamp() - dt_util.now().timestamp())))
            reply = (f'Your next timer has {remaining // 60} minutes and {remaining % 60} seconds left.' if kind == 'timer'
                     else f"Your next alarm is {at:%A} at {format_time(at)}.")
            return {'reply': reply, 'next_at': own[0]['at'], 'changed': False, 'schedules': own}
        if op in ('cancel', 'replace'):
            if not own:
                return {'reply': f"I couldn't find that {kind}.", 'changed': False, 'schedules': []}
            if len(own) > 1 and not (op == 'cancel' and data.get('all')):
                labels = ', '.join(i['label'] or format_time(dt_util.as_local(datetime.fromisoformat(i['at']))) for i in own)
                return {'reply': f'You have multiple {kind}s: {labels}. Give a name or ID, or say cancel all {kind}s.', 'changed': False, 'schedules': own}
        if op == 'cancel':
            await self.commit([i for i in self.items if i not in own])
            return {'reply': f'Cancelled {len(own)} {kind}' + ('s.' if len(own) != 1 else '.'), 'changed': True, 'schedules': own}
        previous = own[0] if op == 'replace' else None
        # Resolve while accepting the command: never acknowledge a silent schedule.
        target = (previous.get('alert_entity_id') if previous and not (data.get('device_id') or data.get('area_id'))
                  else self.resolve_target(data))
        if target is None:
            return {'reply': 'I need a voice device with a speaker to set that. Please ask on your Voice device.', 'changed': False}
        user = await self.hass.auth.async_get_user(user_id)
        from homeassistant.auth.permissions.const import POLICY_CONTROL
        if user is None or not user.permissions.check_entity(target, POLICY_CONTROL):
            raise HomeAssistantError('This user cannot play alerts on that device')
        now = dt_util.now()
        if sum(key in data for key in ('at', 'seconds', 'hour')) > 1:
            raise HomeAssistantError('Choose exactly one time: at, seconds, or hour')
        if 'at' in data:
            try:
                at = datetime.fromisoformat(data['at'])
                if at.tzinfo is None or at.timestamp() <= now.timestamp():
                    raise ValueError('Expected a future timestamp with timezone')
            except ValueError as exc:
                raise HomeAssistantError('at must be a future ISO timestamp with timezone') from exc
        elif 'seconds' in data:
            at = datetime.fromtimestamp(now.timestamp() + data['seconds'], timezone.utc)
        elif kind == 'alarm' and 'hour' in data:
            day = now.date() + timedelta(days=int(data.get('tomorrow', False)))
            at = datetime.combine(day, datetime.min.time(), tzinfo=now.tzinfo).replace(hour=data['hour'], minute=data.get('minute', 0))
            if at.timestamp() <= now.timestamp():
                at += timedelta(days=1)
            # Spring-forward gaps must not silently become a different wakeup time.
            if datetime.fromtimestamp(at.timestamp(), now.tzinfo).replace(tzinfo=None) != at.replace(tzinfo=None):
                return {'reply': "That time doesn't exist because the clocks change. Choose another time.", 'changed': False}
        elif 'hour' in data:
            raise HomeAssistantError('Clock times are only supported for alarms')
        elif previous:
            at = datetime.fromisoformat(previous['at'])
        else:
            raise HomeAssistantError('Provide a duration, timestamp, or an alarm hour and minute')
        if len(self.items) >= 100 and op != 'replace':
            raise HomeAssistantError('Cancel a schedule before adding more')
        item = {'id': previous['id'] if previous else uuid.uuid4().hex, 'kind': kind,
                'label': data.get('new_label', previous['label'] if previous else label).strip(), 'user_id': user_id,
                'area_id': data.get('area_id') or (previous['area_id'] if previous else ''),
                'device_id': data.get('device_id') or (previous.get('device_id') if previous and not data.get('area_id') else None),
                'alert_entity_id': target, 'at': at.astimezone(timezone.utc).isoformat()}
        await self.commit([i for i in self.items if op != 'replace' or i not in own] + [item])
        local = dt_util.as_local(at)
        return {'reply': f"{item['label'] + ' ' if item['label'] else ''}{kind.capitalize()} set for {local:%A} at {format_time(local)}.",
                'changed': True, 'next_at': item['at'], 'schedule': item}

    async def due(self, now):
        async with self.lock:
            due = [i for i in self.items if datetime.fromisoformat(i['at']).timestamp() <= now.timestamp()]
            try:
                await self.commit([i for i in self.items if i not in due])
            except Exception:
                # Keep pending schedules on disk and retry rather than silently losing them.
                self.unsubscribe = async_track_point_in_utc_time(self.hass, self.due, now + timedelta(seconds=5))
                raise
        await asyncio.gather(*(self.deliver(item, now) for item in due))

    async def deliver(self, item, now):
        # Expired schedules restored after extended downtime never start a stale wakeup routine.
        missed = now.timestamp() - datetime.fromisoformat(item['at']).timestamp() > 300
        self.hass.bus.async_fire('ha_jev_schedule_missed' if missed else f"ha_jev_{item['kind']}_finished", item)
        from homeassistant.components import persistent_notification
        persistent_notification.async_create(self.hass,
            f"{item['label'] or item['kind'].capitalize()} {'was missed while HA was offline' if missed else 'finished'}.",
            title='Jev', notification_id='ha_jev_' + item['id'])

        if not missed:
            await self.play_alert(item)

    def resolve_target(self, data):
        from homeassistant.helpers import entity_registry, device_registry
        from homeassistant.components.assist_satellite import AssistSatelliteEntityFeature
        registry = entity_registry.async_get(self.hass)
        devices = device_registry.async_get(self.hass)
        candidates = []
        for entity in registry.entities.values():
            if entity.domain != 'assist_satellite' or entity.disabled_by:
                continue
            if data.get('device_id'):
                if entity.device_id != data['device_id']:
                    continue
            else:
                device = devices.async_get(entity.device_id)
                area = entity.area_id or (device.area_id if device else None)
                if not data.get('area_id') or area != data['area_id']:
                    continue
            state = self.hass.states.get(entity.entity_id)
            if state is not None and state.state not in ('unknown', 'unavailable') and (
                state.attributes.get('supported_features', 0) & AssistSatelliteEntityFeature.ANNOUNCE
            ):
                candidates.append(entity.entity_id)
        return candidates[0] if len(candidates) == 1 else None

    async def play_alert(self, item):
        from homeassistant.core import Context
        # Stored targets preserve the requesting device across moves/restarts.
        target = item.get('alert_entity_id') or self.resolve_target(item)
        error = None
        if target:
            for attempt in range(3):
                try:
                    async with asyncio.timeout(30):
                        await self.hass.services.async_call('assist_satellite', 'announce', {
                            'entity_id': target, 'media_id': ALERT_URL, 'preannounce': False,
                        }, blocking=True, context=Context(user_id=item['user_id']))
                    return
                except Exception as exc:
                    error = exc
                    if attempt < 2:
                        await asyncio.sleep(2)
        _LOGGER.warning('Could not play %s alert on its voice device (%s)', item['kind'], type(error).__name__)
        self.hass.bus.async_fire('ha_jev_alert_failed', {**item, 'reason': 'playback_unavailable'})
        from homeassistant.components import persistent_notification
        persistent_notification.async_create(self.hass,
            'The timer or alarm finished, but its voice device could not play the alert.',
            title='Jev alert failed', notification_id='ha_jev_alert_failed_' + item['id'])
