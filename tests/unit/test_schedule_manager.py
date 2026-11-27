"""Run in HA's Python runtime; no services, sounds, or real stores are touched."""
import asyncio
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('homeassistant')


async def exercise(monkeypatch):
    path = Path(__file__).parents[2] / 'custom_components/ha_jev/schedules.py'
    spec = importlib.util.spec_from_file_location('schedule_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    now = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(module.dt_util, 'now', lambda: now)
    timers, cancelled, states, events = [], [], {}, []
    def track(hass, callback, at):
        timers.append((callback, at))
        return lambda: cancelled.append(at)
    monkeypatch.setattr(module, 'async_track_point_in_utc_time', track)
    class Store:
        saved = None
        fail = False
        async def async_save(self, value):
            if self.fail:
                raise OSError('disk unavailable')
            self.saved = value
        async def async_load(self):
            return self.saved
    store = Store()
    monkeypatch.setattr(module, 'Store', lambda *args: store)
    async def user(uid):
        return SimpleNamespace(permissions=SimpleNamespace(check_entity=lambda *args: True))
    hass = SimpleNamespace(
        states=SimpleNamespace(async_set=lambda eid, state, attrs: states.update({eid: (state, attrs)})),
        auth=SimpleNamespace(async_get_user=user),
        bus=SimpleNamespace(async_fire=lambda event, data: events.append((event, data))))
    manager = module.ScheduleManager(hass)
    monkeypatch.setattr(manager, 'resolve_target', lambda data: 'assist_satellite.nabu' if data.get('device_id') else None)
    async def call(op, kind='alarm', uid='owner', **fields):
        return await manager.manage(SimpleNamespace(
            data=module.SCHEMA({'op': op, 'kind': kind, **fields}), context=SimpleNamespace(user_id=uid), return_response=True))
    first = await call('create', label='Wake up', hour=7, minute=30, device_id='nabu', area_id='bedroom')
    original = first['schedule']
    sid = original['id']
    assert original['alert_entity_id'] == 'assist_satellite.nabu'
    assert states['sensor.ha_jev_next_alarm'][1]['schedules'] == [original]
    assert (await call('list'))['schedules'] == [original]
    assert (await call('query'))['schedules'] == [original]
    # Native Hermes uses REST without return_response. No-op writes must raise.
    async def plain(op, **fields):
        return await manager.manage(SimpleNamespace(data=module.SCHEMA({'op': op, 'kind': 'alarm', **fields}),
            context=SimpleNamespace(user_id='owner'), return_response=False))
    assert await plain('update', id=sid, new_label='Wake up') is None
    with pytest.raises(module.HomeAssistantError, match="couldn't find"):
        await plain('delete', id='missing')
    # Other token owners cannot list, change, or delete this user's alarms, even with the ID.
    assert (await call('list', uid='other'))['schedules'] == []
    assert not (await call('update', id=sid, seconds=60, uid='other'))['changed']
    assert not (await call('delete', id=sid, uid='other'))['changed']
    with pytest.raises(module.HomeAssistantError, match='signed-in'):
        await call('list', uid=None)
    updated = await call('update', id=sid, seconds=90, new_label='Later')
    assert updated['schedule'] == {**original, 'label': 'Later', 'at': (now + timedelta(seconds=90)).isoformat()}
    assert len(manager.items) == 1 and cancelled
    # Reload preserves identity and delivery target and arms the stored deadline.
    restored = module.ScheduleManager(hass)
    await restored.load()
    assert restored.items == manager.items
    assert timers[-1][1] == now + timedelta(seconds=90)
    before = list(manager.items)
    for at in ('2037-10-16T07:00:00', '2020-01-01T00:00:00+00:00', 'garbage'):
        with pytest.raises(module.HomeAssistantError, match='future ISO'):
            await call('update', id=sid, at=at)
        assert manager.items == before
    with pytest.raises(module.HomeAssistantError, match='exactly one'):
        await call('update', id=sid, seconds=30, hour=8)
    # Failure to persist leaves the acknowledged state and scheduled callback intact.
    store.fail = True
    with pytest.raises(OSError):
        await call('delete', id=sid)
    assert manager.items == before
    store.fail = False
    second = (await call('create', seconds=300, label='Other', device_id='nabu'))['schedule']
    ambiguous = await call('update', hour=8)
    assert not ambiguous['changed'] and len(ambiguous['schedules']) == 2
    assert len(manager.items) == 2
    assert (await call('delete', id=second['id']))['changed']
    # Rename only and legacy replace preserve the same record ID and speaker.
    assert (await call('update', id=sid, new_label='Morning'))['schedule']['at'] == before[0]['at']
    assert (await call('replace', label='Morning', seconds=30))['schedule']['id'] == sid
    played = []
    async def deliver(item, at):
        played.append(item['id'])
    monkeypatch.setattr(manager, 'deliver', deliver)
    await manager.due(now + timedelta(seconds=29))
    assert played == []
    await manager.due(now + timedelta(seconds=30))
    assert played == [sid] and manager.items == []
    timer = await call('set', kind='timer', seconds=120, device_id='nabu')
    assert (await call('cancel', kind='timer', id=timer['schedule']['id']))['changed']
    assert states['sensor.ha_jev_next_timer'][1]['count'] == 0

    # Voice clock selectors must match the stored local time, including tomorrow.
    from zoneinfo import ZoneInfo
    from ha_jev.schedules import parse
    zone = ZoneInfo('America/Barbados')
    now = datetime(2026, 10, 15, 18, tzinfo=zone)
    monkeypatch.setattr(module.dt_util, 'as_local', lambda at: at.astimezone(zone))
    for room in ('living_room', 'bedroom'):
        alarm = await call(**parse('set an alarm for 8 AM'), device_id=room, area_id=room)
        sid = alarm['schedule']['id']
        assert datetime.fromisoformat(alarm['next_at']).astimezone(zone).day == 16
        result = await call(**parse('cancel 8 a.m. alarm'))
        assert result['changed'], result['reply']
        assert result['schedules'][0]['id'] == sid
        assert manager.items == []

    morning = (await call('create', hour=8, device_id='nabu'))['schedule']
    evening = (await call('create', hour=20, device_id='nabu'))['schedule']
    later = (await call('create', hour=8, minute=15, device_id='nabu'))['schedule']
    other = (await call('create', hour=8, device_id='nabu', uid='other'))['schedule']
    # A matching clock never bypasses ownership or silently cancels duplicates.
    duplicate = (await call('create', hour=8, device_id='nabu',
                            label='second morning'))['schedule']
    assert not (await call(**parse('cancel 8 AM alarm')))['changed']
    assert (await call('delete', id=duplicate['id']))['changed']
    assert (await call(**parse('cancel 8 AM alarm')))['schedules'] == [morning]
    assert {i['id'] for i in manager.items} == {evening['id'], later['id'], other['id']}
    # An explicit tomorrow restricts the date; a bare clock works before 8 AM too.
    now = datetime(2026, 10, 16, 6, tzinfo=zone)
    today = (await call('create', hour=8, device_id='nabu'))['schedule']
    tomorrow = (await call('create', hour=8, tomorrow=True, device_id='nabu'))['schedule']
    assert not (await call(**parse('cancel 8 AM alarm')))['changed']
    assert (await call(**parse('cancel tomorrow at 8 AM alarm')))['schedules'] == [tomorrow]
    assert (await call(**parse('cancel 8 AM alarm')))['schedules'] == [today]


def test_schedule_crud(monkeypatch):
    asyncio.run(exercise(monkeypatch))
