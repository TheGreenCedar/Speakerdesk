"""Actual local HTTP startup must not depend on reverse name resolution."""
import importlib.util
from pathlib import Path
import socket
import threading
import unittest
import urllib.request
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('loopback_sidecar',
    Path(__file__).resolve().parents[1]/'packaging/sidecar.py')
sidecar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sidecar)


class LoopbackServerTests(unittest.TestCase):
    def test_real_http_and_shutdown_do_not_reverse_resolve_the_local_address(self):
        def application(environ, start_response):
            self.assertEqual(environ['SERVER_NAME'], '127.0.0.1')
            start_response('200 OK', [('Content-Type', 'text/plain')])
            return [b'disposable loopback fixture']

        with patch('socket.getfqdn', side_effect=AssertionError('Reverse lookup must not run')) as lookup:
            server = sidecar.make_loopback_server(application)
            self.assertEqual(server.address_family, socket.AF_INET)
            self.assertEqual(server.server_address[0], '127.0.0.1')
            self.assertGreater(server.server_port, 0)
            self.assertTrue(server.multithread)
            worker = threading.Thread(target=server.serve_forever)
            worker.start()
            try:
                http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with http.open(f'http://127.0.0.1:{server.server_port}', timeout=3) as response:
                    self.assertEqual(response.read(), b'disposable loopback fixture')
                lookup.assert_not_called()
            finally:
                server.shutdown()
                worker.join(timeout=5)
                server.server_close()
            self.assertFalse(worker.is_alive())


if __name__ == '__main__':
    unittest.main()
