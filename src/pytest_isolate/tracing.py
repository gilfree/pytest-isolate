def create_event(worker_name, test_name, category, start_time, end_time, **kwargs):
    start_event = {
        "name": test_name.rsplit("/")[-1],
        "cat": category,
        "ph": "B",
        "pid": worker_name,
        "tid": 0,
        "ts": start_time * (1000**2),
        "args": kwargs,
    }
    end_event = {
        "name": test_name.rsplit("/")[-1],
        "cat": "pipeline",
        "ph": "E",
        "pid": worker_name,
        "tid": 0,
        "ts": end_time * (1000**2),
    }
    return start_event, end_event
