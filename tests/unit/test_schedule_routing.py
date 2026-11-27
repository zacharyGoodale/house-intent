from tests.unit.test_hermes_session import setup_core


async def test_explicit_hermes_bypasses_classifier_anywhere():
    for text in ('Hermes, change my alarm', 'ask HERMES to delete my timer', 'turn off Alder, Hermes'):
        core, jev, hermes, _, _, ha = await setup_core()
        result = await core.handle(text, session_id='s')
        assert result.debug.route == 'hermes/explicit'
        assert jev.calls == ha.calls == []
        assert len(hermes.calls) == 1


async def test_schedule_commands_reach_ha_and_bypass_sticky_hermes():
    core, jev, hermes, _, _, ha = await setup_core()
    await core.handle('Hello', session_id='s')
    jev._answers = {'intent': 'timer_alarm_set'}
    calls = []
    async def service(domain, service, entities, data, return_response=False):
        calls.append((domain, service, data, return_response))
        return {'reply': 'Updated.', 'changed': data['op'] != 'list'}
    ha.call_service = service
    for text, operation in (
        ('set a timer for five minutes', {'op': 'set', 'kind': 'timer', 'label': '', 'seconds': 300}),
        ('update my breakfast alarm to 8 AM', {'op': 'replace', 'kind': 'alarm', 'label': 'breakfast', 'hour': 8, 'minute': 0, 'tomorrow': False}),
        ('cancel 8 a.m. alarm', {'op': 'cancel', 'kind': 'alarm', 'label': '', 'all': False, 'hour': 8, 'minute': 0, 'tomorrow': False}),
        ('delete my breakfast alarm', {'op': 'cancel', 'kind': 'alarm', 'label': 'breakfast', 'all': False}),
        ('list my alarms', {'op': 'list', 'kind': 'alarm'}),
    ):
        result = await core.handle(text, session_id='s', device_id='nabu')
        assert result.reply == 'Updated.'
        assert result.outcome == ('answered' if operation['op'] == 'list' else 'executed')
        context = {'area_id': '', 'device_id': 'nabu'} if operation['op'] == 'set' else {}
        assert calls[-1] == ('ha_jev', 'manage_schedule', {**operation, **context}, True)
    assert len(hermes.calls) == 1


async def test_unparsed_schedule_request_falls_back_to_hermes():
    core, jev, hermes, _, _, _ = await setup_core()
    jev._answers = {'intent': 'timer_alarm_set'}
    result = await core.handle('move the alarm I set yesterday to an hour later')
    assert result.debug.route == 'hermes/schedules'
    assert len(hermes.calls) == 1


async def test_hermes_word_boundary_and_disabled_backend():
    from tests.fakes import FakeLuna
    core, jev, hermes, _, _, _ = await setup_core()
    result = await core.handle('What is a hermesite?')
    assert result.debug.route != 'hermes/explicit'
    core.luna = FakeLuna()
    jev.calls.clear()
    result = await core.handle('Hermes, delete my alarm')
    assert result.outcome == 'unsupported'
    assert 'not enabled' in result.reply
    assert jev.calls == []


def test_alarm_clock_cancellation_variants():
    from ha_jev.schedules import parse
    for text in ('cancel 8 AM alarm', 'cancel my eight a.m. alarm',
                 'delete the alarm at 8 a.m.', 'stop the alarm for 8 AM'):
        assert parse(text) == {'op': 'cancel', 'kind': 'alarm', 'label': '', 'all': False,
                               'hour': 8, 'minute': 0, 'tomorrow': False}
    assert parse('cancel 8:15 p.m. alarm')['hour'] == 20
    assert parse('cancel 8:15 p.m. alarm')['minute'] == 15
    assert parse('cancel tomorrow at 8 AM alarm')['tomorrow'] is True
    assert parse('cancel all 8 AM alarms')['all'] is True
    assert parse('delete my breakfast alarm')['label'] == 'breakfast'
