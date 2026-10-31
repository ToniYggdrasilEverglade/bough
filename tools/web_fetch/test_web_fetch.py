import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from web_fetch_wrapper import (WebFetchConfig, web_fetch, dispatch_web_action,
    public_ip, validate_url, validate_response, ToolError)

FAKE = r"""#!/usr/bin/python3
import sys,json,time,os
r=json.load(sys.stdin)
u=r['url']
if '/sleep' in u: time.sleep(30)
if '/overflow' in u:
    sys.stdout.write('x'*300000);sys.stdout.flush();time.sleep(30)
elif '/malformed' in u: print('not JSON')
elif '/crash' in u: sys.exit(7)
elif '/badversion' in u: print('{"version":true,"ok":false,"data":null,"error":{"code":"TIMEOUT","message":"x"}}');sys.exit(1)
else:
    assert os.environ['WEB_FETCH_ENABLED']=='1'
    assert json.loads(os.environ['WEB_FETCH_ALLOWED_HOSTS_JSON'])==['example.com']
    if '/stderr' in u: sys.stderr.write('x'*100000);sys.stderr.flush()
    print(json.dumps({'version':1,'ok':True,'data':{'requested_url':u,'final_url':u,'status':200,'content_type':'text/plain','title':None,'text':'Hello','truncated':False},'error':None}))
"""

class PolicyTests(unittest.TestCase):
    def test_addresses(self):
        for ip in ('127.0.0.1','10.0.0.1','169.254.169.254','100.64.0.1','192.0.2.1','::1','::ffff:8.8.8.8','fc00::1','2001:db8::1'):
            self.assertFalse(public_ip(ip), ip)
        self.assertTrue(public_ip('8.8.8.8'))
    def test_urls(self):
        for url in ('http://127.0.0.1/','file:///etc/passwd','https://example.com:8000/','https://user@example.com/','https://example.com./','https://evil.com/'):
            with self.assertRaises(ToolError): validate_url(url, {'example.com'})
    def test_schema(self):
        for raw in (b'{}', b'not json',b'{"version":1,"version":1}'):
            with self.assertRaises(ToolError): validate_response(raw,'https://example.com/',{'example.com'})

class ProcessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.exe=Path(self.tmp.name)/'fake'
        self.exe.write_text(FAKE);self.exe.chmod(0o700)
        self.config=WebFetchConfig(self.exe,True,('example.com',),2)
        self.dns=patch('web_fetch_wrapper._public_dns',new_callable=AsyncMock)
        self.dns.start();self.addCleanup(self.dns.stop)
        self.addCleanup(self.tmp.cleanup)
    async def test_success(self):
        r=await web_fetch('https://example.com/',self.config)
        self.assertTrue(r['ok']);self.assertEqual(r['data']['text'],'Hello')
    async def test_denied_does_not_launch(self):
        with patch('asyncio.create_subprocess_exec',new_callable=AsyncMock) as launch:
            r=await web_fetch('https://example.com/',WebFetchConfig(self.exe))
            self.assertEqual(r['error']['code'],'PERMISSION_DENIED');launch.assert_not_called()
    async def test_invalid_url_does_not_launch(self):
        with patch('asyncio.create_subprocess_exec',new_callable=AsyncMock) as launch:
            r=await web_fetch('http://127.0.0.1/',self.config)
            self.assertEqual(r['error']['code'],'URL_BLOCKED');launch.assert_not_called()
    async def test_process_failures(self):
        for path,code in (('malformed','INVALID_RESPONSE'),('badversion','INVALID_RESPONSE'),('crash','PROCESS_ERROR'),('overflow','RESPONSE_TOO_LARGE')):
            r=await asyncio.wait_for(web_fetch('https://example.com/'+path,self.config),4)
            self.assertEqual(r['error']['code'],code,path)
    async def test_stderr_is_drained(self):
        self.assertTrue((await web_fetch('https://example.com/stderr',self.config))['ok'])
    async def test_timeout_reaps(self):
        created=[];real=asyncio.create_subprocess_exec
        async def launch(*a,**kw):
            p=await real(*a,**kw);created.append(p);return p
        with patch('asyncio.create_subprocess_exec',side_effect=launch):
            r=await web_fetch('https://example.com/sleep',WebFetchConfig(self.exe,True,('example.com',),0.15))
        self.assertEqual(r['error']['code'],'TIMEOUT');self.assertIsNotNone(created[0].returncode)
    async def test_cancel_reaps(self):
        created=[];ready=asyncio.Event();real=asyncio.create_subprocess_exec
        async def launch(*a,**kw):
            p=await real(*a,**kw);created.append(p);ready.set();return p
        with patch('asyncio.create_subprocess_exec',side_effect=launch):
            task=asyncio.create_task(web_fetch('https://example.com/sleep',self.config))
            await ready.wait();await asyncio.sleep(.03);task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
        self.assertIsNotNone(created[0].returncode)
    async def test_dispatch_rejects_permission_injection(self):
        r=await dispatch_web_action({'kind':'web_fetch','url':'https://example.com','enabled':True},self.config)
        self.assertEqual(r['error']['code'],'INVALID_REQUEST')

if __name__ == '__main__':unittest.main()
