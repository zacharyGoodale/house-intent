"""Conversation adapter using the authenticated HA context as identity."""
import uuid
from homeassistant.components import conversation
from homeassistant.helpers import intent, device_registry
from .client import ServiceError


async def async_setup_entry(hass, entry, async_add_entities):
    async_add_entities([JevConversation(entry)])


class JevConversation(conversation.ConversationEntity):
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, entry):
        self.entry = entry
        self.client = entry.runtime_data["core"]
        self._attr_name = "Jev"
        self._attr_unique_id = entry.entry_id + "-conversation"

    @property
    def supported_languages(self):
        return ["en"]

    async def async_process(self, user_input):
        cid = user_input.conversation_id or uuid.uuid4().hex
        response = intent.IntentResponse(language=user_input.language)
        try:
            if not user_input.context.user_id:
                raise ServiceError("This voice request has no authenticated Home Assistant user")
            area = None
            if user_input.device_id:
                device = device_registry.async_get(self.hass).async_get(user_input.device_id)
                area = device.area_id if device else None
            result = await self.client.request("/v1/conversation", json={
                "text": user_input.text, "user_id": user_input.context.user_id,
                "conversation_id": cid, "speaker_area": area,
                **({"device_id": user_input.device_id} if user_input.device_id else {}),
            })
            if result["outcome"] in ("error", "refused", "unsupported"):
                response.async_set_error(intent.IntentResponseErrorCode.FAILED_TO_HANDLE, result["reply"])
            else:
                response.response_type = (intent.IntentResponseType.ACTION_DONE if result["outcome"] == "executed"
                                          else intent.IntentResponseType.QUERY_ANSWER)
            response.async_set_speech(result["reply"], extra_data={"ha_jev": {
                "version": 1, "outcome": result["outcome"], "response_mode": result["response_mode"],
                "request_id": result["request_id"],
            }})
            return conversation.ConversationResult(response, cid, result["outcome"] == "clarify")
        except (ServiceError, KeyError, TypeError):
            response.async_set_error(intent.IntentResponseErrorCode.FAILED_TO_HANDLE, "The Jev voice service is unavailable or this user is not configured.")
            return conversation.ConversationResult(response, cid)
