"""Strict code-side timer/alarm extraction; HA owns clocks, storage and delivery."""
import re
from ha_jev.extract import _WORD_NUMBER, _word_value, normalize

def _clock(value: str) -> dict | None:
    value = re.sub(r'^tomorrow (?:at )?', '', value) + (' tomorrow' if value.startswith('tomorrow ') else '')
    clock = re.fullmatch(r'(\d{1,2})(?::| )?(\d{2})?\s*(am|pm)( tomorrow)?', value)
    if clock:
        hour, minute, period, tomorrow = clock.groups()
        if 1 <= int(hour) <= 12 and 0 <= int(minute or 0) <= 59:
            return {'hour': int(hour) % 12 + (12 if period == 'pm' else 0),
                    'minute': int(minute or 0), 'tomorrow': bool(tomorrow)}
    return None


def parse(text: str) -> dict | None:
    text = normalize(text).casefold().rstrip('?')
    # Full matches keep device/routine names and reservation requests out of this path.
    text = re.sub(r'^wake me up (?:at|for) ', 'set an alarm for ', text)
    # Also accept the common adjective form, e.g. "set a five minute timer".
    text = re.sub(r'^(set|start) (?:a |an )?(.+?) (timer|alarm)$', r'\1 a \3 for \2', text)
    text = _WORD_NUMBER.sub(lambda m: str(_word_value(m.group(0))), text)
    text = re.sub(r'\b([ap])\.?\s*m\b\.?', r'\1m', text)
    listing = re.fullmatch(r'(?:list|show|read)(?: me)? (?:all )?(?:my |the )?(alarms|timers)', text)
    if listing:
        return {'op': 'list', 'kind': listing[1].rstrip('s')}
    query = re.fullmatch(r"(?:when(?:'s| is)?|what(?:'s| is)?|how long(?: is)?)(?: left (?:on|for))? (?:my |the |a )?(?:next )?(alarm|timer)(?: set for| remaining)?", text)
    if query:
        return {'op': 'query', 'kind': query[1]}
    held = re.fullmatch(r"(?:do i have|have i got|is there|are there) (?:an? |any )?(alarm|timer)s?(?: set| running| on)?", text)
    if held:
        return {'op': 'query', 'kind': held[1]}
    text = re.sub(r'^(cancel|delete|remove|stop) (?:my |the )?(alarms?) (?:at|for) (.+)$', r'\1 \3 \2', text)
    cancel = re.fullmatch(r'(?:cancel|delete|remove|stop) (?:my |the )?(?:(all)(?: my| the)? )?(?:(.+?) )?(timers?|alarms?)', text)
    if cancel:
        result = {'op': 'cancel', 'kind': cancel[3].rstrip('s'), 'all': bool(cancel[1]), 'label': cancel[2] or ''}
        clock = _clock(result['label']) if result['kind'] == 'alarm' else None
        return {**result, 'label': '', **clock} if clock else result
    match = re.fullmatch(r'(set|start|create|change|reset|update|move|reschedule) (?:a |an |my |the )?(?:(.+?) )?(timer|alarm) (?:for|in|to|at) (.+)', text)
    if not match:
        return None
    verb, label, kind, value = match.groups()
    named = re.fullmatch(r'(.+?) (?:called|named) ([\w -]{1,64})', value)
    if named:
        value, label = named.groups()
    result = {'op': 'replace' if verb in ('change', 'reset', 'update', 'move', 'reschedule') else 'set', 'kind': kind, 'label': label or ''}
    value = re.sub(r'\b(?:a|an) (second|minute|hour)\b', r'1 \1', value)
    units = {'second': 1, 'minute': 60, 'hour': 3600}
    duration = re.fullmatch(r'(?:\d+(?:\.\d+)? (?:seconds?|minutes?|hours?))(?: (?:and )?\d+(?:\.\d+)? (?:seconds?|minutes?|hours?))*', value)
    if duration:
        seconds = sum(float(n) * units[u.rstrip('s')] for n, u in re.findall(r'(\d+(?:\.\d+)?) (seconds?|minutes?|hours?)', value))
        if 0 < seconds <= 604800:
            return {**result, 'seconds': seconds}
        return None
    if kind == 'alarm':
        clock = _clock(value)
        if clock:
            return {**result, **clock}
    return None
