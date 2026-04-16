"""
api/limiter.py
--------------
Shared rate limiter instance — imported by both main.py and routes.
Defined here to avoid circular imports.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

# 10 requests/minute per IP — enough for demo, prevents API abuse.
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
