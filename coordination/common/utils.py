from datetime import datetime


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, datetime):
        return obj.isoformat()

    raise TypeError (f"Type {type(obj)} is not serializable")