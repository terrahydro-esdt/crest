import json
import requests
import logging

log = logging.getLogger("crest")

TEAMS_WEBHOOK = ***REMOVED***

def send_teams(message: str):
    """Send a notification to Microsoft Teams via Incoming Webhook."""
    if not TEAMS_WEBHOOK:
        log.warning("Teams webhook not configured; skipping send_teams",
                    extra={"kind": "notify", "stage": "teams"})
        return

    payload = {"text": message}

    try:
        resp = requests.post(
            TEAMS_WEBHOOK,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.status_code >= 400:
            log.error(
                "Teams notification failed",
                extra={
                    "kind": "notify",
                    "stage": "teams_error",
                    "status_code": resp.status_code,
                    "response": resp.text,
                },
            )
        else:
            log.info("Teams notification sent",
                     extra={"kind": "notify", "stage": "teams"})
    except Exception:
        log.exception("Failed to send Teams notification",
                      extra={"kind": "notify", "stage": "teams_error"})

def notify(subject: str, body: str):
    text = f"**{subject}**\n\n{body}"
    send_teams(text)    