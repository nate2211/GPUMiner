from __future__ import annotations

from typing import Callable, Optional

from blocknet_client import BlockNetClient
from models import MinerConfig, MiningJob, NonceWindow, SubmitResult, VerifiedShare
from monero_rpc_client import MoneroRpcClient
from solo_zmq import SoloMiningConnection
from stratum_client import StratumClient


class MiningConnection:
    def __init__(
        self,
        config: MinerConfig,
        on_log: Callable[[str], None],
        on_job: Callable[[MiningJob], None],
        on_status: Callable[[str], None],
    ) -> None:
        self.config = config
        self.on_log = on_log
        self.on_job = on_job
        self.on_status = on_status
        self._inner: Optional[object] = None

    def connect(self) -> None:
        backend = self.config.mining_backend_name()

        if backend == "blocknet":
            self._inner = BlockNetClient(
                self.config,
                on_log=self.on_log,
                on_job=self.on_job,
                on_status=self.on_status,
            )
        elif backend == "solo":
            self._inner = SoloMiningConnection(
                self.config,
                on_log=self.on_log,
                on_job=self.on_job,
                on_status=self.on_status,
            )
        elif backend == "monerorpc":
            self._inner = MoneroRpcClient(
                self.config,
                on_log=self.on_log,
                on_job=self.on_job,
                on_status=self.on_status,
            )
        else:
            self._inner = StratumClient(
                self.config,
                on_log=self.on_log,
                on_job=self.on_job,
                on_status=self.on_status,
            )

        self._inner.connect()

    def close(self) -> None:
        if self._inner is not None:
            try:
                self._inner.close()
            finally:
                self._inner = None

    def submit(self, verified: VerifiedShare) -> SubmitResult:
        if self._inner is None:
            return SubmitResult(
                accepted=False,
                status="not_connected",
                error="backend not connected",
                reject_class="backend_error",
                backend_error=True,
            )
        return self._inner.submit(verified)

    def request_scan_window(self, span: int) -> Optional[NonceWindow]:
        inner = self._inner
        if inner is None:
            return None
        fn = getattr(inner, "request_scan_window", None)
        if callable(fn):
            return fn(int(span))
        return None

    @property
    def current_job(self) -> Optional[MiningJob]:
        inner = self._inner
        if inner is None:
            return None
        return getattr(inner, "current_job", None)

    @property
    def session_id(self) -> str:
        inner = self._inner
        if inner is None:
            return ""
        return str(getattr(inner, "session_id", "") or "")