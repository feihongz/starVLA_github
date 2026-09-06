from unittest import mock

import pytest

from examples.Robocasa_tabletop.eval_files import run_robocasa_ckpt_eval


def test_wait_for_port_reports_server_exit_with_log_tail(
    tmp_path, monkeypatch
) -> None:
    server_log = tmp_path / "server.log"
    server_log.write_text("loading\nroot cause\n", encoding="utf-8")
    process = mock.Mock()
    process.poll.return_value = 7
    connect = mock.Mock(side_effect=AssertionError("socket must not be called"))
    monkeypatch.setattr(
        run_robocasa_ckpt_eval.socket, "create_connection", connect
    )

    with pytest.raises(run_robocasa_ckpt_eval.EvalFailed) as error:
        run_robocasa_ckpt_eval.wait_for_port(
            "127.0.0.1",
            22000,
            900.0,
            process=process,
            log_path=server_log,
        )

    message = str(error.value)
    assert "exited before opening 127.0.0.1:22000 (rc=7)" in message
    assert str(server_log) in message
    assert "root cause" in message
    connect.assert_not_called()


def test_wait_for_port_detects_exit_after_refused_connection(
    tmp_path, monkeypatch
) -> None:
    server_log = tmp_path / "server.log"
    server_log.write_text("traceback\n", encoding="utf-8")
    process = mock.Mock()
    process.poll.side_effect = [None, 9]
    monkeypatch.setattr(
        run_robocasa_ckpt_eval.socket,
        "create_connection",
        mock.Mock(side_effect=ConnectionRefusedError("not listening")),
    )
    sleep = mock.Mock(side_effect=AssertionError("must fail before sleeping"))
    monkeypatch.setattr(run_robocasa_ckpt_eval.time, "sleep", sleep)

    with pytest.raises(
        run_robocasa_ckpt_eval.EvalFailed, match=r"rc=9"
    ):
        run_robocasa_ckpt_eval.wait_for_port(
            "127.0.0.1",
            22000,
            900.0,
            process=process,
            log_path=server_log,
        )

    sleep.assert_not_called()


def test_wait_for_port_returns_when_server_is_listening(monkeypatch) -> None:
    process = mock.Mock()
    process.poll.return_value = None
    connection = mock.MagicMock()
    connect = mock.Mock(return_value=connection)
    monkeypatch.setattr(
        run_robocasa_ckpt_eval.socket, "create_connection", connect
    )

    run_robocasa_ckpt_eval.wait_for_port(
        "127.0.0.1", 22000, 1.0, process=process
    )

    connect.assert_called_once_with(("127.0.0.1", 22000), timeout=2.0)


def test_wait_for_port_preserves_timeout_error(monkeypatch) -> None:
    monotonic = mock.Mock(side_effect=[10.0, 11.0])
    monkeypatch.setattr(run_robocasa_ckpt_eval.time, "monotonic", monotonic)

    with pytest.raises(
        run_robocasa_ckpt_eval.EvalFailed,
        match=r"did not open 127.0.0.1:22000 within 0.5s",
    ):
        run_robocasa_ckpt_eval.wait_for_port(
            "127.0.0.1", 22000, 0.5
        )
