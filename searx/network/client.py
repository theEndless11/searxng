import sys
import asyncio
import logging
import random
from ssl import SSLContext
import threading
from typing import Any, Dict

import httpx
from httpx_socks import AsyncProxyTransport
from python_socks import parse_proxy_url, ProxyConnectionError, ProxyTimeoutError, ProxyError

from searx import logger

# --- Optional uvloop setup ---
if sys.platform != "win32":
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass  # uvloop is not available; fallback to default loop
# -----------------------------

logger = logger.getChild('searx.network.client')
LOOP = None
SSLCONTEXTS: Dict[Any, SSLContext] = {}


def shuffle_ciphers(ssl_context: SSLContext):
    """Shuffle httpx's default ciphers of a SSL context randomly."""
    c_list = [cipher["name"] for cipher in ssl_context.get_ciphers()]
    sc_list, c_list = c_list[:3], c_list[3:]
    random.shuffle(c_list)
    ssl_context.set_ciphers(":".join(sc_list + c_list))


def get_sslcontexts(proxy_url=None, cert=None, verify=True, trust_env=True):
    key = (proxy_url, cert, verify, trust_env)
    if key not in SSLCONTEXTS:
        SSLCONTEXTS[key] = httpx.create_ssl_context(verify, cert, trust_env)
    shuffle_ciphers(SSLCONTEXTS[key])
    return SSLCONTEXTS[key]


class AsyncHTTPTransportNoHttp(httpx.AsyncHTTPTransport):
    def __init__(self, *args, **kwargs):
        pass

    async def handle_async_request(self, request):
        raise httpx.UnsupportedProtocol('HTTP protocol is disabled')

    async def aclose(self) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type=None, exc_value=None, traceback=None) -> None:
        pass


class AsyncProxyTransportFixed(AsyncProxyTransport):
    async def handle_async_request(self, request):
        try:
            return await super().handle_async_request(request)
        except ProxyConnectionError as e:
            raise httpx.ProxyError("ProxyConnectionError: " + e.strerror, request=request) from e
        except ProxyTimeoutError as e:
            raise httpx.ProxyError("ProxyTimeoutError: " + e.args[0], request=request) from e
        except ProxyError as e:
            raise httpx.ProxyError("ProxyError: " + e.args[0], request=request) from e


def get_transport_for_socks_proxy(verify, http2, local_address, proxy_url, limit, retries):
    rdns = False
    socks5h = 'socks5h://'
    if proxy_url.startswith(socks5h):
        proxy_url = 'socks5://' + proxy_url[len(socks5h):]
        rdns = True

    proxy_type, proxy_host, proxy_port, proxy_username, proxy_password = parse_proxy_url(proxy_url)
    verify = get_sslcontexts(proxy_url, None, verify, True) if verify is True else verify
    return AsyncProxyTransportFixed(
        proxy_type=proxy_type,
        proxy_host=proxy_host,
        proxy_port=proxy_port,
        username=proxy_username,
        password=proxy_password,
        rdns=rdns,
        loop=get_loop(),
        verify=verify,
        http2=http2,
        local_address=local_address,
        limits=limit,
        retries=retries,
    )


def get_transport(verify, http2, local_address, proxy_url, limit, retries):
    verify = get_sslcontexts(None, None, verify, True) if verify is True else verify
    return httpx.AsyncHTTPTransport(
        verify=verify,
        http2=http2,
        limits=limit,
        proxy=httpx._config.Proxy(proxy_url) if proxy_url else None,
        local_address=local_address,
        retries=retries,
    )


def new_client(
    enable_http,
    verify,
    enable_http2,
    max_connections,
    max_keepalive_connections,
    keepalive_expiry,
    proxies,
    local_address,
    retries,
    max_redirects,
    hook_log_response,
):
    limit = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        keepalive_expiry=keepalive_expiry,
    )
    mounts = {}
    for pattern, proxy_url in proxies.items():
        if not enable_http and pattern.startswith('http://'):
            continue
        if proxy_url.startswith('socks4://') or proxy_url.startswith('socks5://') or proxy_url.startswith('socks5h://'):
            mounts[pattern] = get_transport_for_socks_proxy(
                verify, enable_http2, local_address, proxy_url, limit, retries
            )
        else:
            mounts[pattern] = get_transport(verify, enable_http2, local_address, proxy_url, limit, retries)

    if not enable_http:
        mounts['http://'] = AsyncHTTPTransportNoHttp()

    transport = get_transport(verify, enable_http2, local_address, None, limit, retries)

    event_hooks = {'response': [hook_log_response]} if hook_log_response else None

    return httpx.AsyncClient(
        transport=transport,
        mounts=mounts,
        max_redirects=max_redirects,
        event_hooks=event_hooks,
    )


def get_loop():
    return LOOP


def init():
    for logger_name in (
        'httpx',
        'httpcore.proxy',
        'httpcore.connection',
        'httpcore.http11',
        'httpcore.http2',
        'hpack.hpack',
        'hpack.table',
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    def loop_thread():
        global LOOP
        LOOP = asyncio.new_event_loop()
        LOOP.run_forever()

    thread = threading.Thread(
        target=loop_thread,
        name='asyncio_loop',
        daemon=True,
    )
    thread.start()

init()

