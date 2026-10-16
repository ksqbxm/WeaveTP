"""Real TCPStore + concurrent CPU actors exercise controller gates, never CUDA."""

import json
import queue
import tempfile
import threading
import time
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import common
import controller
import run_probe
import torch.distributed as dist
import worker


class ProtocolTests(unittest.TestCase):
    def test_notification_arrives_while_pg_store_connection_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            tcp_store = dist.TCPStore
            server = tcp_store("127.0.0.1", 0, world_size=None, is_master=True,
                               wait_for_workers=False, timeout=timedelta(seconds=5), use_libuv=False)
            config = {**common.DEFAULTS, "master": "127.0.0.1", "port": server.port,
                      "launch": "cpu_notify", "startup_timeout_s": 5, "launch_timeout_s": 5}
            # Windows CPU test runtime lacks libuv; production connection code is unchanged.
            with patch.object(dist, "TCPStore", side_effect=lambda *a, **k: tcp_store(*a, **k, use_libuv=False)):
                pg_store, _ = common.connect(config)
                control = common.Bus(dist.PrefixStore("control/cpu_notify/", server))
                started, finished = threading.Event(), threading.Event()
                notifications = queue.Queue()
                journal = common.Journal(Path(directory) / "notify.jsonl", "SL3060", "worker")

                def rendezvous():
                    started.set()
                    pg_store.get("pg/missing_rank")
                    finished.set()

                blocked = threading.Thread(target=rendezvous, daemon=True)
                listener = threading.Thread(target=worker.listen,
                                            args=(config, 1, journal, notifications), daemon=True)
                blocked.start()
                try:
                    self.assertTrue(started.wait(1))
                    time.sleep(0.05)
                    listener.start()
                    for epoch in (1, 2):
                        control.put(f"transition/{epoch}", {"epoch": epoch})
                        self.assertEqual(notifications.get(timeout=2)[0], epoch)
                        self.assertFalse(finished.is_set())
                finally:
                    server.set("pg/missing_rank", "release")
                    blocked.join(timeout=5)
                    listener.join(timeout=5)
                self.assertFalse(blocked.is_alive() or listener.is_alive())

    def test_terminal_exchange_propagates_peer_cleanup_failure(self):
        for peer_status in ("passed", "failed"):
            with self.subTest(peer_status=peer_status):
                server = dist.TCPStore("127.0.0.1", 0, world_size=None, is_master=True,
                                      wait_for_workers=False, timeout=timedelta(seconds=5), use_libuv=False)
                client = dist.TCPStore("127.0.0.1", server.port, world_size=None, is_master=False,
                                      timeout=timedelta(seconds=5), use_libuv=False)
                local, remote = common.Bus(server), common.Bus(client)
                errors = []

                def peer():
                    try:
                        run_probe.exchange_terminal({"node": "SL3061"}, remote, {"status": peer_status})
                    except RuntimeError as error:
                        errors.append(str(error))

                thread = threading.Thread(target=peer, daemon=True)
                thread.start()
                if peer_status == "passed":
                    run_probe.exchange_terminal({"node": "SL3060"}, local, {"status": "passed"})
                    self.assertTrue(local.get("terminal_received"))
                else:
                    with self.assertRaisesRegex(RuntimeError, "SL3061 terminal failure"):
                        run_probe.exchange_terminal({"node": "SL3060"}, local, {"status": "passed"})
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
                self.assertEqual(len(errors), int(peer_status == "failed"))

    def test_real_store_prefix_isolation_and_missing_key_timeout(self):
        server = dist.TCPStore("127.0.0.1", 0, world_size=None, is_master=True,
                               wait_for_workers=False, timeout=timedelta(seconds=5), use_libuv=False)
        client = dist.TCPStore("127.0.0.1", server.port, world_size=None, is_master=False,
                               timeout=timedelta(seconds=5), use_libuv=False)
        first = common.Bus(dist.PrefixStore(common.prefix("one", 0), server), timeout=0)
        second = common.Bus(dist.PrefixStore(common.prefix("one", 1), client), timeout=0)
        first.put("ready", {"epoch": 0})
        self.assertFalse(second.has("ready"))
        with self.assertRaises(TimeoutError):
            second.wait(["ready"])
        second.put("ready", {"epoch": 1})
        self.assertEqual(first.get("ready")["epoch"], 0)

    def test_controller_full_lifecycle_over_real_tcpstore(self):
        for kind in ("idle", "inflight"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                self.exercise(kind, directory)

    @patch.object(controller.signal, "SIGKILL", 9, create=True)
    def exercise(self, kind, directory):
        server = dist.TCPStore("127.0.0.1", 0, world_size=None, is_master=True,
                               wait_for_workers=False, timeout=timedelta(seconds=5), use_libuv=False)
        client = dist.TCPStore("127.0.0.1", server.port, world_size=None, is_master=False,
                               timeout=timedelta(seconds=5), use_libuv=False)
        bus = common.Bus(dist.PrefixStore("cpu-control/", server), timeout=5)
        remote = common.Bus(dist.PrefixStore("cpu-control/", client), timeout=5)
        config = {**common.DEFAULTS, "output": directory, "launch": f"cpu_{kind}", "kind": kind,
                  "commit": "cpu-fixture", "port": server.port, "startup_timeout_s": 5}
        inventory = {node: [{"index": i, "uuid": f"GPU-{node}-{i}"} for i in range(8)] for node in common.NODES}
        rows = common.topology(inventory)
        victims = common.failed_slots(rows)
        errors = []

        def event(**extra):
            now = time.monotonic()
            return {"t": now, "t_local": now, **extra}

        def cuda(slot, epoch):
            return event(slot=slot, pid=100 + slot + 1000 * epoch,
                         tensor={"pointer": 10000 + slot + epoch, "reference_sha256": str(slot)},
                         gpu_uuid=rows[slot]["gpu_uuid"])

        def actor():
            try:
                for node in common.NODES:
                    for index in range(5):
                        remote.wait([f"ping/{node}/{index}"])
                        remote.put(f"pong/{node}/{index}", time.monotonic())
                    remote.wait([f"clock/{node}"])
                    remote.put(f"preflight/{node}", {"gpus": inventory[node], "host": node, "config": config})
                remote.wait(["start"])
                initial = {s: cuda(s, 0) for s in range(16)}
                for s, data in initial.items():
                    remote.put(f"cuda/0/{s}", data)
                for node in common.NODES:
                    remote.put(f"verified/0/{node}", {})
                remote.wait(["cuda_verified/0"])

                def ready(epoch, returned):
                    members = common.membership(rows, epoch)
                    for index, slot in enumerate(members):
                        data = returned.get(slot, initial[slot])
                        remote.put(f"ready/{epoch}/{slot}", {**data, **event(), "epoch": epoch,
                                    "world_size": len(members), "new_rank": index,
                                    "identities": [[s, returned.get(s, initial[s])["pid"]] for s in members]})

                ready(0, {})
                remote.wait(["arm"])
                for slot in range(16):
                    remote.put(f"armed/{slot}", {"slot": slot, "pending": kind == "inflight" and slot not in victims})
                remote.wait(["kill"])
                remote.put("kill_sent", {s: event() for s in victims})
                remote.put("kill_exited", [{"slot": s, "pid": initial[s]["pid"], "exit": {"returncode": -9}}
                                            for s in victims])
                remote.wait(["transition/1"])
                for slot in common.membership(rows, 1):
                    remote.put(f"aborted/1/{slot}", {"start": event(), "end": event()})
                remote.wait(["build/1"])
                ready(1, {})
                remote.wait(["rejoin_request"])
                reclaimed = time.monotonic()
                remote.put("reclaimed", {"t_done": reclaimed,
                                         "cards": {rows[s]["gpu_uuid"]: {"confirmed_t": reclaimed,
                                                                          "confirmed_t_local": reclaimed} for s in victims}})
                remote.wait(["spawn"])
                returned = {s: cuda(s, 2) for s in victims}
                for slot, data in returned.items():
                    remote.put(f"cuda/2/{slot}", data)
                for node in common.NODES:
                    remote.put(f"verified/2/{node}", {})
                    remote.put(f"spawned/{node}", [{"slot": s, "start": event()}
                                                   for s in victims] if node == "SL3061" else [])
                remote.wait(["transition/2"])
                for slot in common.membership(rows, 1):
                    remote.put(f"aborted/2/{slot}", {"start": event(), "end": event()})
                remote.wait(["build/2"])
                ready(2, returned)
                remote.wait(["finish"])
                for slot in range(16):
                    remote.put(f"done/{slot}", {})
                for node in common.NODES:
                    remote.put(f"clean/{node}", {})
            except BaseException as error:
                errors.append(error)
                remote.put("failure", str(error))

        thread = threading.Thread(target=actor, daemon=True)
        thread.start()
        journal = common.Journal(Path(directory) / "events.jsonl", "SL3060", "cpu-controller")
        result = controller.control(config, bus, journal, lambda: None)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(result["status"], "passed")
        self.assertEqual([len(result["epochs"][e]) for e in range(3)], [16, 12, 16])
        events = [json.loads(line)["stage"] for line in (Path(directory) / "events.jsonl").read_text().splitlines()]
        for earlier, later in (("kill", "transition/1"), ("reclaim_received", "spawn"), ("transition/2", "build/2")):
            self.assertLess(events.index(earlier), events.index(later))


if __name__ == "__main__":
    unittest.main()
