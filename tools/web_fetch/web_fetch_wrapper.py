"""Permission-checked async wrapper. Configuration is trusted application state."""
from __future__ import annotations
import asyncio
import ipaddress
import json
import os
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

STDOUT_MAX = 131072
STDERR_MAX = 16384
ERROR_CODES = {
    "INVALID_REQUEST", "PERMISSION_DENIED", "URL_BLOCKED", "DNS_ERROR",
    "NETWORK_ERROR", "TIMEOUT", "TOO_MANY_REDIRECTS", "HTTP_ERROR",
    "UNSUPPORTED_CONTENT_TYPE", "RESPONSE_TOO_LARGE", "INTERNAL_ERROR",
}

class ToolError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message

def failure(code, message):
    return {"version": 1, "ok": False, "data": None,
            "error": {"code": code, "message": message}}

def hostname(value):
    if not isinstance(value, str) or not value or not value.isascii() or value.endswith('.'):
        raise ToolError("URL_BLOCKED", "Use an ASCII hostname without a trailing dot")
    value = value.lower()
    if len(value) > 253 or not all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", x) for x in value.split('.')):
        raise ToolError("URL_BLOCKED", "Invalid hostname")
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return value
    raise ToolError("URL_BLOCKED", "IP literals are not allowed")

@dataclass(frozen=True)
class WebFetchConfig:
    executable: Path
    enabled: bool = False
    allowed_hosts: tuple[str, ...] = ()
    timeout_s: float = 20.0

    def __post_init__(self):
        if not isinstance(self.executable, Path) or not self.executable.is_absolute():
            raise ValueError("executable must be a trusted absolute Path")
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if not 0 < self.timeout_s <= 20:
            raise ValueError("timeout_s must be in (0, 20]")
        if not isinstance(self.allowed_hosts, tuple) or len(self.allowed_hosts) > 100:
            raise ValueError("allowed_hosts must be a tuple of at most 100 hosts")
        for h in self.allowed_hosts:
            hostname(h)

BLOCKED_V4 = tuple(ipaddress.ip_network(x) for x in (
    '0.0.0.0/8','10.0.0.0/8','100.64.0.0/10','127.0.0.0/8','169.254.0.0/16',
    '172.16.0.0/12','192.0.0.0/24','192.0.2.0/24','192.88.99.0/24','192.168.0.0/16',
    '198.18.0.0/15','198.51.100.0/24','203.0.113.0/24','224.0.0.0/3'))
BLOCKED_V6 = tuple(ipaddress.ip_network(x) for x in ('2001::/23','2001:db8::/32','2002::/16','3fff::/20'))
def public_ip(raw):
    ip = ipaddress.ip_address(raw)
    if ip.version == 4:
        return not any(ip in n for n in BLOCKED_V4)
    return ip in ipaddress.ip_network('2000::/3') and not any(ip in n for n in BLOCKED_V6)

def validate_url(url, allowed):
    if not isinstance(url, str) or not url or len(url.encode('utf-8')) > 4096:
        raise ToolError('INVALID_REQUEST', 'Invalid URL type or length')
    if '\\' in url or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in url):
        raise ToolError('INVALID_REQUEST', 'Invalid URL characters')
    try:
        u = urlsplit(url)
        if u.scheme not in ('http', 'https') or u.username is not None or u.password is not None:
            raise ToolError('URL_BLOCKED', 'Unsupported scheme or URL credentials')
        host = hostname(u.hostname)
        port = 443 if u.scheme == 'https' else 80
        if u.port is not None and u.port != port:
            raise ToolError('URL_BLOCKED', 'Nonstandard port')
        if host not in allowed:
            raise ToolError('URL_BLOCKED', 'Hostname not allowlisted')
        return host, port
    except (ValueError, UnicodeError) as e:
        raise ToolError('INVALID_REQUEST', 'Malformed URL') from e

async def _public_dns(host, port):
    try:
        rows = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        raise ToolError('DNS_ERROR', 'DNS resolution failed') from e
    if not rows:
        raise ToolError('DNS_ERROR', 'No destination addresses')
    if any(not public_ip(row[4][0]) for row in rows):
        raise ToolError('URL_BLOCKED', 'DNS returned a non-public address')

def _object(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError('Duplicate JSON field')
        out[k] = v
    return out

def _invalid_constant(value):
    raise ValueError('Invalid JSON constant')

def validate_response(raw, request_url, allowed):
    try:
        r = json.loads(raw.decode('utf-8'), object_pairs_hook=_object, parse_constant=_invalid_constant)
        if not (isinstance(r, dict) and set(r) == {'version','ok','data','error'}):
            raise ValueError("Invalid response schema")
        if not (type(r['version']) is int and r['version'] == 1):
            raise ValueError("Invalid response schema")
        if not (type(r['ok']) is bool):
            raise ValueError("Invalid response schema")
        if r['ok']:
            d = r['data']
            if not (r['error'] is None and isinstance(d, dict)):
                raise ValueError("Invalid response schema")
            if not (set(d) == {'requested_url','final_url','status','content_type','title','text','truncated'}):
                raise ValueError("Invalid response schema")
            if not (d['requested_url'] == request_url):
                raise ValueError("Invalid response schema")
            if not (type(d['status']) is int and 200 <= d['status'] < 300):
                raise ValueError("Invalid response schema")
            if not (d['content_type'] in ('text/html','text/plain','application/json')):
                raise ValueError("Invalid response schema")
            if not (isinstance(d['text'], str) and len(d['text'].encode()) <= 65536):
                raise ValueError("Invalid response schema")
            if not (d['title'] is None or (isinstance(d['title'], str) and len(d['title'].encode()) <= 1024)):
                raise ValueError("Invalid response schema")
            if not (type(d['truncated']) is bool):
                raise ValueError("Invalid response schema")
            validate_url(d['final_url'], allowed)
        else:
            e = r['error']
            if not (r['data'] is None and isinstance(e, dict) and set(e) == {'code','message'}):
                raise ValueError("Invalid response schema")
            if not (isinstance(e['code'], str) and e['code'] in ERROR_CODES):
                raise ValueError("Invalid response schema")
            if not (isinstance(e['message'], str) and len(e['message'].encode()) <= 4096):
                raise ValueError("Invalid response schema")
        return r
    except (ValueError, UnicodeError, AssertionError, TypeError, KeyError, ToolError, RecursionError) as e:
        raise ToolError('INVALID_RESPONSE', 'Invalid CLI JSON response') from e

async def _read_stdout(stream):
    out = bytearray()
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            return bytes(out)
        if len(out) + len(chunk) > STDOUT_MAX:
            raise ToolError('RESPONSE_TOO_LARGE', 'CLI stdout exceeds limit')
        out.extend(chunk)

async def _read_stderr(stream):
    out = bytearray()
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            return bytes(out)
        out.extend(chunk[:max(0, STDERR_MAX-len(out))])

async def _discard(stream):
    while await stream.read(4096):
        pass

async def _stop(proc):
    if proc.returncode is None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), 0.5)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
    await proc.wait()

async def web_fetch(url: str, config: WebFetchConfig):
    """Return a protocol response. Cancellation propagates after process cleanup.

    The URL is untrusted. The config must be supplied only by trusted code.
    Never invoke automatically based on words in a model's response.
    """
    proc = None
    readers = []
    try:
        async with asyncio.timeout(config.timeout_s):
            if not config.enabled or not config.allowed_hosts:
                raise ToolError('PERMISSION_DENIED', 'Web fetching is disabled or allowlist is empty')
            allowed = {hostname(x) for x in config.allowed_hosts}
            host, port = validate_url(url, allowed)
            await _public_dns(host, port)
            request = json.dumps({'version':1,'action':'web_fetch','url':url}, ensure_ascii=False).encode()
            if len(request) > 8192:
                raise ToolError('INVALID_REQUEST', 'JSON request exceeds limit')
            env = {'PATH': os.defpath, 'WEB_FETCH_ENABLED':'1',
                   'WEB_FETCH_ALLOWED_HOSTS_JSON':json.dumps(sorted(allowed))}
            proc = await asyncio.create_subprocess_exec(
                str(config.executable), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
            readers = [asyncio.create_task(_read_stdout(proc.stdout)),
                       asyncio.create_task(_read_stderr(proc.stderr))]
            try:
                proc.stdin.write(request)
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                proc.stdin.close()
            raw, _stderr = await asyncio.gather(*readers)
            code = await proc.wait()
            if code not in (0, 1):
                raise ToolError('PROCESS_ERROR', 'CLI exited unexpectedly')
            result = validate_response(raw, url, allowed)
            if (code == 0) != result['ok']:
                raise ToolError('INVALID_RESPONSE', 'CLI exit status contradicts response')
            return result
    except ToolError as e:
        return failure(e.code, e.message)
    except TimeoutError:
        return failure('TIMEOUT', 'Python tool deadline exceeded')
    except OSError:
        return failure('PROCESS_ERROR', 'Cannot launch CLI executable')
    finally:
        # Also runs on task cancellation. Do not swallow CancelledError.
        if proc is not None:
            # A capped/failed reader must not leave the child blocked on a full pipe.
            for index, stream in enumerate((proc.stdout, proc.stderr)):
                if index >= len(readers) or readers[index].done():
                    readers.append(asyncio.create_task(_discard(stream)))
            cleanup = asyncio.create_task(_stop(proc))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
        for task in readers:
            if not task.done():
                task.cancel()
        if readers:
            await asyncio.gather(*readers, return_exceptions=True)

async def dispatch_web_action(action: dict, config: WebFetchConfig):
    """Explicit entry point; permission is always taken from trusted config."""
    if not isinstance(action, dict) or set(action) != {"kind", "url"} or action.get("kind") != "web_fetch":
        return failure("INVALID_REQUEST", "Expected exactly kind=web_fetch and url")
    return await web_fetch(action["url"], config)
