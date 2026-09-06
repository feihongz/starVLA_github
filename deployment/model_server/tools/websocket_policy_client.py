# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging
import math
import os
import time
from typing import Dict, Optional, Tuple
import websockets.exceptions

import websockets.sync.client
from typing_extensions import override

from . import msgpack_numpy


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = 10093,
        api_key: Optional[str] = None,
        connect_timeout: Optional[float] = None,
        handshake_timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ) -> None:
        # 0.0.0.0 cannot be used as a connection target, here default 127.0.0.1
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws: Optional[websockets.sync.client.ClientConnection] = None
        self._closed = False
        self._connect_timeout = float(
            connect_timeout
            if connect_timeout is not None
            else os.getenv("POLICY_CONNECT_TIMEOUT_SECONDS", "300")
        )
        self._handshake_timeout = float(
            handshake_timeout
            if handshake_timeout is not None
            else os.getenv("POLICY_HANDSHAKE_TIMEOUT_SECONDS", "30")
        )
        self._request_timeout = float(
            request_timeout
            if request_timeout is not None
            else os.getenv("POLICY_REQUEST_TIMEOUT_SECONDS", "600")
        )
        for name, value in (
            ("connect_timeout", self._connect_timeout),
            ("handshake_timeout", self._handshake_timeout),
            ("request_timeout", self._request_timeout),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        self._ws, self._server_metadata = self._wait_for_server(self._connect_timeout)

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self, timeout: float) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None

        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(k, None)

        while time.monotonic() < deadline:
            conn = None
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                remaining = max(0.1, deadline - time.monotonic())
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=min(10.0, remaining),
                    close_timeout=min(5.0, remaining),
                    ping_interval=None,
                    ping_timeout=60,
                    proxy=None,
                )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Connection to {self._uri} used the entire connect timeout"
                    )
                packed_metadata = conn.recv(timeout=min(self._handshake_timeout, remaining))
                try:
                    metadata = msgpack_numpy.unpackb(packed_metadata)
                except Exception as exc:
                    raise RuntimeError(
                        f"Invalid msgpack metadata from {self._uri}: {type(exc).__name__}: {exc}"
                    ) from exc
                if type(metadata) is not dict:
                    raise RuntimeError(
                        f"Invalid policy-server metadata type: {type(metadata).__name__}"
                    )
                return conn, metadata
            except (
                OSError,
                TimeoutError,
                websockets.exceptions.WebSocketException,
                RuntimeError,
            ) as exc:
                last_error = exc
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                logging.info("Still waiting for server %s: %s", self._uri, exc)
                time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
        raise TimeoutError(
            f"Failed to connect and receive valid metadata from {self._uri} "
            f"within {timeout} seconds; last error: {last_error}"
        )

    def close(self) -> None:
        self._closed = True
        ws = self._ws
        self._ws = None
        if ws is None:
            return
        try:
            ws.close()
        except Exception:
            pass

    @override
    def predict_action(self, query_info: Dict) -> Dict:
        if self._closed or self._ws is None:
            raise ConnectionError(f"Policy-server connection at {self._uri} is closed")

        deadline = time.monotonic() + self._request_timeout
        try:
            if type(query_info) is not dict:
                raise TypeError(
                    f"Policy request must be a dict, got {type(query_info).__name__}"
                )
            data = self._packer.pack(query_info)
            self._ws.send(data)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Policy request timed out while sending")
            response = self._ws.recv(timeout=remaining)
            if isinstance(response, str):
                raise RuntimeError(f"Error text frame from inference server:\n{response}")
            try:
                unpacked = msgpack_numpy.unpackb(response)
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid msgpack response from {self._uri}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if type(unpacked) is not dict:
                raise RuntimeError(
                    f"Invalid policy-server response type: {type(unpacked).__name__}"
                )

            # Application/protocol errors aren't safe to retry on the same
            # request stream. Close before returning the structured envelope so
            # callers retain its error code without risking a late response
            # being consumed by a subsequent request.
            if unpacked.get("ok") is False or unpacked.get("status") == "error":
                self.close()
            return unpacked
        except TimeoutError as exc:
            self.close()
            raise TimeoutError(
                f"Policy inference timed out after {self._request_timeout}s at {self._uri}"
            ) from exc
        except websockets.exceptions.ConnectionClosed as exc:
            self.close()
            raise ConnectionError(
                f"Policy server connection closed at {self._uri}: {exc}"
            ) from exc
        except BaseException:
            self.close()
            raise
