"""CARLA sidecar: runs the CARLA client in its own Python, serves it to carla_bridge over localhost.

ROS 2 Jazzy needs Python 3.12, but CARLA 0.9.15 only ships clients up to Python 3.10.
Run this under the CARLA-compatible interpreter (see sim/README.md):

    PYTHONPATH=sim sim/carla_env/bin/python -m netra_sim.carla_sidecar --port 5555

One client at a time; when it disconnects the CARLA world is restored. Frames are
length-prefixed pickles, so the socket is bound to 127.0.0.1 only.
"""
import argparse
import pickle
import signal
import socket
import struct

from netra_sim.carla_backend import CarlaBackend


def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack('!I', len(data)) + data)


def recv_msg(sock):
    head = _recv_exact(sock, 4)
    if head is None:
        return None
    (n,) = struct.unpack('!I', head)
    body = _recv_exact(sock, n)
    return None if body is None else pickle.loads(body)


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def serve(port, carla_module=None, once=False):
    # SIGTERM (e.g. from `ros2 launch` shutting down) must still run the cleanup below,
    # otherwise the server is left in synchronous mode with our actors in it.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', port))
    srv.listen(1)
    print('carla_sidecar listening on 127.0.0.1:%d' % port, flush=True)
    while True:
        conn, _ = srv.accept()
        backend = None
        try:
            while True:
                req = recv_msg(conn)
                if req is None or req[0] == 'close':
                    break
                try:
                    if req[0] == 'open':
                        backend = CarlaBackend(req[1], carla_module)
                        reply = ('ok', backend.open())
                    elif req[0] == 'step':
                        reply = ('ok', backend.step(*req[1]))
                    else:
                        reply = ('error', 'unknown request %r' % (req[0],))
                except Exception as e:                      # reported to the ROS node, which stops
                    reply = ('error', '%s: %s' % (type(e).__name__, e))
                send_msg(conn, reply)
        finally:
            if backend is not None:
                backend.close()                             # restore the server on any exit
            conn.close()
        if once:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=5555)
    serve(ap.parse_args().port)


if __name__ == '__main__':
    main()
