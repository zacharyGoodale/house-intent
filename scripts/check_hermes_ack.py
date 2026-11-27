"""Check the deployed Hermes acknowledgement adapter using installed HA types.

Run inside the HA container, passing the deployed adapter directory as argv[1].
Services and devices are fakes: this check never plays audio or calls Hermes.
"""
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from homeassistant.components.conversation import ConversationInput
from homeassistant.components.conversation.chat_log import ChatLog
from homeassistant.components.media_player import MediaPlayerEntityFeature
from homeassistant.core import Context
from homeassistant.exceptions import HomeAssistantError

ROOT=Path(sys.argv[1]) if len(sys.argv) == 2 else Path('/config/custom_components/ha_jev')
pkg=ModuleType('_ack_check');pkg.__path__=[str(ROOT)];sys.modules[pkg.__name__]=pkg
spec=importlib.util.spec_from_file_location('_ack_check.conversation',ROOT/'conversation.py')
adapter=importlib.util.module_from_spec(spec);sys.modules[spec.name]=adapter;spec.loader.exec_module(adapter)

async def check(agent,device='nabu',service_error=False):
    release,reached,called,announce_release=asyncio.Event(),asyncio.Event(),asyncio.Event(),asyncio.Event()
    calls,tasks=[],[]
    class Core:
        async def converse_stream(self,payload):
            assert payload['routing_events'] is True
            yield {'type':'agent','agent':agent,'session_mode':'expert' if agent!='jev' else 'simple'}
            if agent=='hermes':
                yield {'type':'agent','agent':agent,'session_mode':'expert'}
            reached.set();await release.wait()
            yield {'type':'text','text':'Paris.'}
            yield {'type':'result','result':{'reply':'Paris.','request_id':'check','outcome':'answered','response_mode':'speech'}}
    async def service(domain,name,data,**options):
        assert domain=='tts' and name=='speak'
        assert data['media_player_entity_id']=='media_player.nabu'
        assert data['message'] in adapter.ACKNOWLEDGEMENTS
        assert data['cache'] is True and options['target']=={'entity_id':'tts.mac_mini_tts'}
        calls.append(data);called.set()
        await announce_release.wait()
        if service_error:raise HomeAssistantError('speaker unavailable')
    def schedule(hass,coroutine,name):
        task=asyncio.create_task(coroutine);tasks.append(task);return task
    adapter.device_registry.async_get=lambda hass:SimpleNamespace(async_get=lambda device_id:SimpleNamespace(area_id='living_room'))
    adapter.entity_registry.async_get=lambda hass:SimpleNamespace(async_get_entity_id=lambda *args:'tts.mac_mini_tts')
    adapter.entity_registry.async_entries_for_device=lambda registry,device_id:[SimpleNamespace(entity_id='media_player.nabu',disabled_by=None)]
    hass=SimpleNamespace(data={'ha_jev':{}},services=SimpleNamespace(async_call=service),states=SimpleNamespace(get=lambda entity_id:SimpleNamespace(state='idle',attributes={'supported_features':MediaPlayerEntityFeature.MEDIA_ANNOUNCE})))
    entry=SimpleNamespace(entry_id='check',runtime_data={'core':Core(),'streaming_enabled':True,'session_supported':True},async_create_background_task=schedule)
    entity=adapter.JevConversation(entry);entity.hass=hass;entity.entity_id='conversation.jev'
    user=ConversationInput(text='What is the capital of France?',context=Context(user_id='check'),conversation_id='check',device_id=device,satellite_id=None,language='en',agent_id='conversation.jev')
    log=ChatLog(hass,'check')
    task=asyncio.create_task(entity._async_handle_message(user,log))
    try:
        await asyncio.wait_for(reached.wait(),2)
        expected=agent=='hermes' and device is not None
        if expected:
            await asyncio.wait_for(called.wait(),2)
            assert not task.done() and len(calls)==1
        else:assert not calls and not tasks
        release.set()
        result=await asyncio.wait_for(task,2)
        # Playback runs independently: a slow or failed announcement cannot delay the answer.
        assert result.response.speech['plain']['speech']=='Paris.'
        assert log.content[-1].content=='Paris.'
        announce_release.set();await asyncio.gather(*tasks)
        print(agent,device,'service_error='+str(service_error)+': PASS')
    finally:
        release.set();announce_release.set();await task;await asyncio.gather(*tasks)

async def main():
    await check('hermes')
    await check('jev')
    await check('luna')
    await check('hermes',device=None)
    await check('hermes',service_error=True)
asyncio.run(main())
