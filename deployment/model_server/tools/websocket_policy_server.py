# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import asyncio
import logging
import time

import websockets.asyncio.server
import websockets.exceptions
import websockets.frames

# from openpi_client import base_policy as _base_policy
from . import msgpack_numpy


_PROTOCOL_IO_TIMEOUT_SECONDS = 5.0


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int = -1,  # Idle timeout in seconds, -1 means never auto-close
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy  #
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._idle_timeout = idle_timeout
        self._last_active = time.time()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            close_timeout=_PROTOCOL_IO_TIMEOUT_SECONDS,
        ) as server:
            logging.info(
                "POLICY_SERVER_READY host=%s port=%s metadata=%s",
                self._host,
                self._port,
                self._metadata,
            )
            if self._idle_timeout > 0:
                await self._idle_watchdog(server)
            else:
                await server.serve_forever()

    async def _idle_watchdog(self, server):
        """Monitor idle time and shut down the server on timeout."""
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info(f"Idle timeout ({self._idle_timeout}s) reached, shutting down server.")
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        try:
            await asyncio.wait_for(
                websocket.send(packer.pack(self._metadata)),
                timeout=_PROTOCOL_IO_TIMEOUT_SECONDS,
            )
        except websockets.exceptions.ConnectionClosed:
            logging.info(
                "Connection from %s closed during metadata handshake",
                websocket.remote_address,
            )
            return
        except Exception:
            logging.exception(
                "Failed to send metadata to %s", websocket.remote_address
            )
            await self._safe_close(
                websocket,
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="METADATA_HANDSHAKE_ERROR",
            )
            return

        while True:
            try:
                frame = await websocket.recv()
            except websockets.exceptions.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                return

            self._last_active = time.time()
            try:
                msg = msgpack_numpy.unpackb(frame)
            except Exception as exc:
                logging.warning(
                    "Invalid msgpack request from %s: %s: %s",
                    websocket.remote_address,
                    type(exc).__name__,
                    exc,
                )
                await self._send_error_and_close(
                    websocket,
                    packer,
                    code="INVALID_MSGPACK",
                    error_type="ProtocolError",
                    message="Request must be a valid binary msgpack object",
                    close_code=websockets.frames.CloseCode.INVALID_DATA,
                )
                return

            if type(msg) is not dict:
                await self._send_error_and_close(
                    websocket,
                    packer,
                    code="INVALID_MESSAGE_TYPE",
                    error_type="TypeError",
                    message="Decoded request must be a dict",
                    close_code=websockets.frames.CloseCode.UNSUPPORTED_DATA,
                    details={"message_type": type(msg).__name__},
                )
                return

            try:
                ret = self._route_message(msg)
                packed_ret = packer.pack(ret)
            except Exception:
                logging.exception(
                    "Unexpected request-routing error from %s",
                    websocket.remote_address,
                )
                await self._send_error_and_close(
                    websocket,
                    packer,
                    code="INTERNAL_SERVER_ERROR",
                    error_type="RuntimeError",
                    message="Policy server failed to route or encode the request",
                    close_code=websockets.frames.CloseCode.INTERNAL_ERROR,
                )
                return

            try:
                await websocket.send(packed_ret)
            except websockets.exceptions.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                return
            except Exception:
                logging.exception(
                    "Failed to send response to %s", websocket.remote_address
                )
                await self._safe_close(
                    websocket,
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="RESPONSE_SEND_ERROR",
                )
                return

    async def _send_error_and_close(
        self,
        websocket: websockets.asyncio.server.ServerConnection,
        packer: msgpack_numpy.Packer,
        *,
        code: str,
        error_type: str,
        message: str,
        close_code: websockets.frames.CloseCode,
        details: dict | None = None,
    ) -> None:
        error = {"code": code, "type": error_type, "message": message}
        if details:
            error.update(details)
        response = {
            "status": "error",
            "ok": False,
            "type": "protocol_error",
            "request_id": "default",
            "error": error,
        }
        try:
            await asyncio.wait_for(
                websocket.send(packer.pack(response)),
                timeout=_PROTOCOL_IO_TIMEOUT_SECONDS,
            )
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            logging.exception(
                "Failed to send protocol error %s to %s",
                code,
                websocket.remote_address,
            )
        finally:
            await self._safe_close(websocket, code=close_code, reason=code)

    async def _safe_close(
        self,
        websocket: websockets.asyncio.server.ServerConnection,
        *,
        code: websockets.frames.CloseCode,
        reason: str,
    ) -> None:
        try:
            await asyncio.wait_for(
                websocket.close(code=code, reason=reason),
                timeout=_PROTOCOL_IO_TIMEOUT_SECONDS,
            )
            return
        except websockets.exceptions.ConnectionClosed:
            return
        except Exception:
            logging.exception(
                "Failed to close connection from %s cleanly",
                websocket.remote_address,
            )

        # A peer that doesn't participate in the closing handshake must not
        # keep a server-side transport alive indefinitely.
        transport = getattr(websocket, "transport", None)
        if transport is not None:
            try:
                transport.abort()
            except Exception:
                logging.exception(
                    "Failed to abort connection from %s",
                    websocket.remote_address,
                )

    # route logic: recognize request from client
    def _route_message(self, msg: dict) -> dict:
        """
        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - The handler validates that msg is a dict before calling this method.
        - Policy exceptions are caught and encoded in the response.
        """
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")  # default = infer
        payload = msg.get("payload", msg)  # when no explicit payload, treat top-level as payload

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        # infer --> framework.predict_action
        elif mtype == "infer" or mtype == "predict_action":
            # Basic payload sanity
            if type(payload) is not dict:
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {
                        "code": "INVALID_PAYLOAD",
                        "type": "TypeError",
                        "message": "Payload must be a dict",
                        "payload_type": str(type(payload)),
                    },
                }
            try:
                output_dict = self._policy.predict_action(**payload)
            except Exception as e:
                logging.exception("Policy inference error (request_id=%s)", req_id)

                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {
                        "code": "POLICY_INFERENCE_ERROR",
                        "type": type(e).__name__,
                        "message": str(e),
                    },
                }
            data = output_dict
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": req_id,
                "data": data,
            }

        # unknow request type
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {
                    "code": "UNSUPPORTED_MESSAGE_TYPE",
                    "type": "ValueError",
                    "message": f"Unsupported message type '{mtype}'",
                },
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()
    raise NotImplementedError("This module is not intended to be run directly.")
#
#  Instead, it should be imported and used in a server context.
