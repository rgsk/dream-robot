"""Any rotation function resolves, and raises if called (see __init__.py)."""


def __getattr__(name):
    def unavailable(*args, **kwargs):
        raise NotImplementedError(f"pytorch3d.transforms.{name}: stubbed; set abs_action: false")
    return unavailable
