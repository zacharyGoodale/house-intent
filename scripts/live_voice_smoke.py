"""HA 2026.8.3 live voice smoke. Reads credentials from env, never prints them.
Modes inspect (read-only), conversation, intent, tts, wav. Speech tests consume
inference and read-only spoken questions; arbitrary command args can change HA.
Run on the deployment host with an HTTPS HA URL or a loopback SSH tunnel.
"""
import argparse
import asyncio
import io
import json
import os
from pathlib import Path
import ssl
import time
from urllib.parse import urlsplit, urljoin
import uuid
import wave

import aiohttp


def context(url, ca):
    parsed = urlsplit(url)
    if parsed.scheme != 'https' and not (parsed.scheme == 'http' and parsed.hostname in {'127.0.0.1', 'localhost', '::1'}):
        raise ValueError('Use HTTPS HA_URL or a loopback SSH tunnel')
    return ssl.create_default_context(cafile=ca or None)


async def main(args):
    base = os.environ['HA_URL'].rstrip('/')
    tls = context(base, os.environ.get('HA_CA_FILE'))
    timeout = aiohttp.ClientTimeout(total=args.timeout + 20)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        wsurl = base.replace('https://', 'wss://', 1).replace('http://', 'ws://', 1) + '/api/websocket'
        async with http.ws_connect(wsurl, ssl=tls) as ws:
            greeting = await ws.receive_json()
            assert greeting['type'] == 'auth_required'
            print('HA version:', greeting.get('ha_version'))
            await ws.send_json({'type': 'auth', 'access_token': os.environ['HA_TOKEN']})
            assert (await ws.receive_json())['type'] == 'auth_ok', 'HA token rejected'
            seq = 0
            async def command(kind, **fields):
                nonlocal seq
                seq += 1
                await ws.send_json({'id': seq, 'type': kind, **fields})
                while True:
                    result = await ws.receive_json()
                    if result.get('id') == seq and result['type'] == 'result':
                        if not result.get('success'):
                            raise RuntimeError(str(result.get('error')))
                        return result.get('result')
            who = await command('auth/current_user')
            print('Token owner:', who['id'], 'admin:', who.get('is_admin'))
            if args.mode == 'inspect':
                entries = await command('config_entries/get', domain='ha_jev')
                print('Jev config entries:', [{k: e.get(k) for k in ('entry_id', 'state', 'title')} for e in entries])
                states = await command('get_states')
                print('Voice entities:', [{k: e.get(k) for k in ('entity_id', 'state')} for e in states
                      if e['entity_id'].startswith(('conversation.', 'stt.', 'tts.'))])
                print('Pipelines:', json.dumps(await command('assist_pipeline/pipeline/list'), indent=2))
                return
            if args.mode == 'conversation':
                assert args.agent, '--agent is required'
                output = await command('conversation/process', agent_id=args.agent,
                    text=args.text, language='en', conversation_id=args.conversation_id)
                response = output['response']
                metadata = response.get('speech', {}).get('plain', {}).get('extra_data', {}).get('ha_jev')
                assert metadata and metadata['version'] == 1, 'Missing Jev outcome metadata'
                print('Conversation:', json.dumps(output, indent=2))
                return
            assert args.pipeline, '--pipeline is required'
            pipeline = await command('assist_pipeline/pipeline/get', pipeline_id=args.pipeline)
            assert pipeline['conversation_engine'].startswith('conversation.'), 'Expected Jev conversation entity'
            assert pipeline.get('prefer_local_intents') is False, 'Disable prefer_local_intents first'
            pcm = None
            if args.mode == 'wav':
                assert args.wav, '--wav fixture required'
                with wave.open(args.wav, 'rb') as audio:
                    assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) == (1, 2, 16000, 'NONE')
                    assert 0 < audio.getnframes() <= 60 * 16000
                    pcm = audio.readframes(audio.getnframes())
                start, input_data = 'stt', {'sample_rate': 16000, 'no_vad': True}
            elif args.mode == 'tts':
                start, input_data = 'tts', {'text': args.tts_text}
            else:
                start, input_data = 'intent', {'text': args.text}
            seq += 1
            run_id = seq
            await ws.send_json({'id':run_id, 'type':'assist_pipeline/run', 'pipeline': args.pipeline,
                'start_stage':start, 'end_stage':'tts', 'input':input_data,
                'conversation_id':args.conversation_id, 'timeout':args.timeout})
            events = []
            event_times = {}
            sent = False
            binary_id = None
            tts_output = None
            async with asyncio.timeout(args.timeout + 10):
                while True:
                    message = await ws.receive_json()
                    if message.get('id') != run_id:
                        continue
                    if message['type'] == 'result':
                        if not message.get('success'):
                            raise RuntimeError(str(message.get('error')))
                        continue
                    if message['type'] != 'event':
                        continue
                    event = message['event']
                    kind, data = event['type'], event.get('data', {})
                    events.append(kind)
                    event_times[kind] = time.monotonic()
                    print('Event:', kind)
                    if kind == 'run-start' and pcm is not None:
                        binary_id = data['runner_data']['stt_binary_handler_id']
                    if kind == 'stt-start' and pcm is not None and not sent:
                        assert isinstance(binary_id, int) and 1 <= binary_id <= 255, 'Invalid STT binary handler'
                        prefix = bytes([binary_id])
                        for offset in range(0, len(pcm), 4096):
                            await ws.send_bytes(prefix + pcm[offset:offset+4096])
                        await ws.send_bytes(prefix)
                        sent = True
                    if kind == 'stt-end':
                        print('Transcript:', data.get('stt_output'))
                    if kind == 'intent-end':
                        assert data.get('processed_locally') is False, 'Request bypassed Jev via native intent or sentence trigger'
                        response = data['intent_output']['response']
                        metadata = response.get('speech', {}).get('plain', {}).get('extra_data', {}).get('ha_jev')
                        assert metadata and metadata['version'] == 1, 'Missing Jev response metadata'
                        print('Jev outcome:', metadata['outcome'], metadata['response_mode'])
                        print('Reply:', response.get('speech', {}).get('plain', {}).get('speech'))
                    if kind == 'tts-end':
                        tts_output = data['tts_output']
                    if kind == 'error':
                        raise RuntimeError(str(data))
                    if kind == 'run-end':
                        break
            assert tts_output, 'No TTS output'
            url = urljoin(base + '/', tts_output['url'])
            if urlsplit(url).netloc != urlsplit(base).netloc:
                raise ValueError('Refusing to forward HA credential to another audio host')
            async with http.get(url, ssl=tls, headers={'Authorization':'Bearer ' + os.environ['HA_TOKEN']}, allow_redirects=False) as result:
                assert result.status == 200, 'Could not fetch HA TTS audio'
                audio = bytearray()
                async for chunk in result.content.iter_chunked(65536):
                    audio.extend(chunk)
                    assert len(audio) <= 8 * 1024 * 1024, 'Oversized HA audio'
            assert audio, 'Empty TTS audio'
            output = Path(args.output)
            suffix = {'audio/mpeg': '.mp3', 'audio/wav': '.wav', 'audio/x-wav': '.wav', 'audio/ogg': '.ogg'}.get(result.content_type)
            if suffix:
                output = output.with_suffix(suffix)
            output.write_bytes(audio)
            print('Audio saved:', output, 'bytes:', len(audio), 'content-type:', result.content_type)
            print('Stages passed:', ', '.join(events))
            completed = time.monotonic()
            timings = {stage: round(event_times[stage + '-end'] - event_times[stage + '-start'], 3)
                       for stage in ('stt', 'intent') if stage + '-end' in event_times}
            timings['tts_to_audio_download'] = round(completed - event_times['tts-start'], 3)
            timings['pipeline_to_audio_download'] = round(completed - event_times['run-start'], 3)
            print('Measured seconds:', json.dumps(timings))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['inspect', 'conversation', 'intent', 'tts', 'wav'])
    parser.add_argument('--agent')
    parser.add_argument('--pipeline')
    parser.add_argument('--text', default='What is the weather today?')
    parser.add_argument('--tts-text', default='The local voice service is ready.')
    parser.add_argument('--wav')
    parser.add_argument('--conversation-id', default='e2e-' + uuid.uuid4().hex)
    parser.add_argument('--timeout', type=int, default=240)
    parser.add_argument('--output', default='/tmp/ha-jev-e2e-audio.wav')
    asyncio.run(main(parser.parse_args()))
