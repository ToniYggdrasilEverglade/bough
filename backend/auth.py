"""Request authentication.

auth_mode=off      -> every caller is the single LOCAL owner (the original loopback-only prototype).
auth_mode=supabase -> callers must present a Supabase access token (RS/ES-signed JWT issued by the
                      project's GoTrue). Tokens are verified locally against the project's JWKS
                      (`/auth/v1/.well-known/jwks.json`): signature, expiry, issuer and the
                      `authenticated` audience. No network call per request; keys are cached.

The verifier never trusts unsigned or HS256 tokens, and the publishable key is not a credential -
it only identifies the project.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from .config import Settings
from .errors import AuthRequired
from .memory.base import LOCAL, Owner

log = logging.getLogger("brain.auth")
ALGORITHMS = ("ES256", "RS256")


@dataclass(frozen=True)
class Principal:
    user_id: str
    email: Optional[str]
    token: str
    expires_at: float

    def owner(self) -> Owner:
        return Owner(self.user_id, self.token)

    def __repr__(self) -> str:                 # tokens must never reach logs
        return f"Principal({self.user_id!r}, {self.email!r})"


class Authenticator:
    def __init__(self, settings: Settings):
        self.mode = settings.auth_mode
        self._jwks = None
        if self.mode == "supabase":
            try:
                import jwt  # PyJWT
                from jwt import PyJWKClient
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("BRAIN_AUTH_MODE=supabase needs 'PyJWT[crypto]' (see requirements.txt)") from e
            self._jwt = jwt
            self.issuer = settings.supabase_url + "/auth/v1"
            self._jwks = PyJWKClient(self.issuer + "/.well-known/jwks.json", cache_keys=True,
                                     lifespan=int(settings.jwks_cache_s), timeout=10)

    @property
    def required(self) -> bool:
        return self.mode != "off"

    # ---------------------------------------------------------------- API
    async def authenticate(self, token: Optional[str]) -> Optional[Principal]:
        """Return the Principal for a bearer token, None when auth is off, or raise AuthRequired."""
        if not self.required:
            return None
        if not token:
            raise AuthRequired("sign in required")
        return await asyncio.to_thread(self._verify, token)

    @staticmethod
    def owner_of(principal: Optional[Principal]) -> Owner:
        return principal.owner() if principal else LOCAL

    @staticmethod
    def bearer(header: Optional[str]) -> Optional[str]:
        if not header:
            return None
        scheme, _, value = header.strip().partition(" ")
        return value.strip() if scheme.lower() == "bearer" and value.strip() else None

    # ---------------------------------------------------------------- internals
    def _verify(self, token: str) -> Principal:
        jwt = self._jwt
        try:
            key = self._jwks.get_signing_key_from_jwt(token)
            claims = jwt.decode(token, key.key, algorithms=list(ALGORITHMS), audience="authenticated",
                                issuer=self.issuer, options={"require": ["exp", "sub", "iss", "aud"]})
        except jwt.ExpiredSignatureError:
            raise AuthRequired("session expired, sign in again") from None
        except jwt.PyJWKClientError as e:
            log.warning("JWKS lookup failed: %s", e)
            raise AuthRequired("could not verify credentials (key lookup failed)") from None
        except jwt.InvalidTokenError as e:
            log.info("rejected token: %s", type(e).__name__)
            raise AuthRequired("invalid credentials") from None
        sub = str(claims.get("sub", ""))
        if not sub:
            raise AuthRequired("invalid credentials")
        return Principal(sub, claims.get("email"), token, float(claims.get("exp", time.time())))
