from core.runtime_streams import EventBroadcaster


def test_event_broadcaster_publish_and_iter():
    bus = EventBroadcaster(max_queue_size=2)
    sid = bus.subscribe()
    bus.publish({"type": "x", "payload": {"v": 1}})
    events = bus.iter_events(sid, heartbeat_seconds=0.01)
    first = next(events)
    assert '"type": "x"' in first
    bus.unsubscribe(sid)
