from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QStandardPaths, Qt, QThread, QTimer
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from models import MinerConfig
from worker import MinerWorker


APP_NAME = "GPUMinerGUI"


def app_data_dir(app_name: str = APP_NAME) -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    path = Path(base) / app_name
    path.mkdir(parents=True, exist_ok=True)
    return path


CFG_PATH = app_data_dir() / "gpuminergui_config.json"


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(18, 20, 24))
    pal.setColor(QPalette.WindowText, QColor(232, 236, 241))
    pal.setColor(QPalette.Base, QColor(12, 14, 18))
    pal.setColor(QPalette.AlternateBase, QColor(24, 27, 33))
    pal.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    pal.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    pal.setColor(QPalette.Text, QColor(232, 236, 241))
    pal.setColor(QPalette.Button, QColor(30, 34, 41))
    pal.setColor(QPalette.ButtonText, QColor(232, 236, 241))
    pal.setColor(QPalette.Link, QColor(106, 170, 255))
    pal.setColor(QPalette.Highlight, QColor(60, 122, 255))
    pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)

    app.setStyleSheet(
        """
        QWidget { font-size: 12px; }
        QMainWindow { background: #121418; }
        QGroupBox {
            border: 1px solid #313845;
            border-radius: 14px;
            margin-top: 12px;
            padding: 12px;
            background: #171b21;
            font-weight: 600;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 8px;
            color: #e6ebf2;
        }
        QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            border: 1px solid #384151;
            border-radius: 10px;
            padding: 8px;
            background: #0f1217;
            selection-background-color: #3c7aff;
        }
        QPlainTextEdit { padding: 10px; }
        QPushButton {
            border: 1px solid #3b4454;
            border-radius: 10px;
            padding: 9px 14px;
            background: #232934;
            font-weight: 600;
        }
        QPushButton:hover { background: #2b3240; }
        QPushButton:pressed { background: #1d232d; }
        QPushButton:disabled {
            color: #7d8796;
            background: #1d2128;
            border-color: #2b313b;
        }
        QLabel#Title {
            font-size: 18px;
            font-weight: 700;
            color: #f2f5fa;
        }
        QLabel#SubTitle { color: #aab4c3; }
        QLabel#StatusPill {
            border-radius: 14px;
            padding: 5px 12px;
            font-weight: 700;
        }
        QFrame#StatCard {
            border: 1px solid #313845;
            border-radius: 14px;
            background: #171b21;
        }
        QLabel#StatLabel {
            color: #aab4c3;
            font-size: 11px;
            font-weight: 600;
        }
        QLabel#StatValue {
            color: #f2f5fa;
            font-size: 18px;
            font-weight: 700;
        }
        QLabel#StatSubValue {
            color: #8e99aa;
            font-size: 11px;
            font-weight: 500;
        }
        QTabWidget::pane {
            border: 1px solid #313845;
            border-radius: 14px;
            top: -1px;
            background: #171b21;
        }
        QTabBar::tab {
            background: #232934;
            border: 1px solid #313845;
            padding: 10px 16px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            margin-right: 3px;
            font-weight: 600;
        }
        QTabBar::tab:selected { background: #171b21; }
        QScrollArea { border: none; background: transparent; }
        QSplitter::handle { background: #2e3541; }
        QSplitter::handle:horizontal {
            width: 10px;
            border-radius: 5px;
        }
        """
    )


class StatCard(QFrame):
    def __init__(self, title: str, value: str = "-", subvalue: str = "") -> None:
        super().__init__()
        self.setObjectName("StatCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("StatLabel")

        self.value_lbl = QLabel(value)
        self.value_lbl.setObjectName("StatValue")

        self.subvalue_lbl = QLabel(subvalue)
        self.subvalue_lbl.setObjectName("StatSubValue")
        self.subvalue_lbl.setWordWrap(True)

        layout.addWidget(self.title_lbl)
        layout.addWidget(self.value_lbl)
        layout.addWidget(self.subvalue_lbl)
        layout.addStretch(1)

    def set_value(self, value: str, subvalue: str | None = None) -> None:
        self.value_lbl.setText(value)
        if subvalue is not None:
            self.subvalue_lbl.setText(subvalue)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GPUMiner - OpenCL + Stratum / BlockNet / Solo / MoneroRPC")
        self.resize(1760, 1160)

        self.worker_thread: QThread | None = None
        self.worker: MinerWorker | None = None
        self.started_at: float = 0.0

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self.save_config)

        self._uptime_timer = QTimer(self)
        self._uptime_timer.setInterval(1000)
        self._uptime_timer.timeout.connect(self._update_uptime)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        root.addLayout(self._make_header())
        root.addWidget(self._make_stat_area())

        self.main_split = QSplitter(Qt.Horizontal)
        self.main_split.setChildrenCollapsible(False)
        self.main_split.setHandleWidth(10)
        root.addWidget(self.main_split, 1)

        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.left_scroll.setMinimumWidth(820)

        self.left_container = QWidget()
        self.left_scroll.setWidget(self.left_container)

        self.left_layout = QVBoxLayout(self.left_container)
        self.left_layout.setContentsMargins(0, 0, 8, 0)
        self.left_layout.setSpacing(10)

        self.main_split.addWidget(self.left_scroll)
        self.main_split.addWidget(self._make_right_tabs())
        self.main_split.setSizes([900, 960])

        self.left_layout.addWidget(self._make_connection_group())
        self.left_layout.addWidget(self._make_blocknet_group())
        self.left_layout.addWidget(self._make_monerorpc_group())
        self.left_layout.addWidget(self._make_solo_group())
        self.left_layout.addWidget(self._make_opencl_group())
        self.left_layout.addWidget(self._make_pipeline_group())
        self.left_layout.addWidget(self._make_tuning_group())
        self.left_layout.addWidget(self._make_controls_group())
        self.left_layout.addStretch(1)

        self._connect_autosave_signals()
        self.load_config()
        self._sync_backend_controls()
        self._sync_scan_mode_controls()
        self._sync_verification_controls()
        self._sync_tuning_controls()
        self._sync_adaptive_controls()
        self.set_status("idle")
        self.statusBar().showMessage(f"Config path: {CFG_PATH}")

    @staticmethod
    def _combo_value_or_none(combo: QComboBox) -> str | None:
        text = combo.currentText().strip().lower()
        return text or None

    @staticmethod
    def _set_combo_from_optional(combo: QComboBox, value: Any) -> None:
        text = str(value or "").strip().lower()
        idx = combo.findText(text, Qt.MatchFixedString)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setCurrentIndex(0)

    def _monero_feeder_mode(self) -> str:
        return self.monero_rpc_feeder_mode_combo.currentText().strip().lower()

    def _make_header(self) -> QHBoxLayout:
        layout = QHBoxLayout()

        text_col = QVBoxLayout()
        title = QLabel("GPUMiner GUI")
        title.setObjectName("Title")

        subtitle = QLabel(
            "OpenCL scan + optional CPU verify + adaptive queue pressure control + "
            "rank / threshold / credit / confidence tuning + brokered MoneroRPC feeder modes"
        )
        subtitle.setObjectName("SubTitle")

        text_col.addWidget(title)
        text_col.addWidget(subtitle)

        self.status_pill = QLabel("IDLE")
        self.status_pill.setObjectName("StatusPill")
        self.uptime_lbl = QLabel("Uptime: 00:00:00")
        self.uptime_lbl.setObjectName("SubTitle")

        right = QVBoxLayout()
        right.addWidget(self.status_pill, alignment=Qt.AlignRight)
        right.addWidget(self.uptime_lbl, alignment=Qt.AlignRight)

        layout.addLayout(text_col)
        layout.addStretch(1)
        layout.addLayout(right)
        return layout

    def _make_stat_area(self) -> QWidget:
        wrapper = QWidget()
        grid = QGridLayout(wrapper)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self.card_hashrate = StatCard("Accepted 15m", "0 H/s", "accepted credited work")
        self.card_scan = StatCard("GPU Scan 15m", "0 H/s", "raw scan throughput")
        self.card_verified_rate = StatCard("Verified 15m", "0 H/s", "verified credited work")
        self.card_accepted = StatCard("Accepted", "0", "")
        self.card_rejected = StatCard("Rejected", "0", "")
        self.card_candidates = StatCard("Candidates", "0", "")
        self.card_verified = StatCard("Verified", "0", "")
        self.card_verify_yield = StatCard("Verify Yield", "0.000", "verified / candidates")
        self.card_accept_yield = StatCard("Accept Yield", "0.000", "accepted / verified")
        self.card_job = StatCard("Job", "-", "")
        self.card_height = StatCard("Height", "-", "")
        self.card_queues = StatCard("Queues", "0 / 0", "verify / submit")
        self.card_backend = StatCard("Backend / Mode", "-", "")
        self.card_effective_target = StatCard("Effective Target", "-", "dynamic candidate target")
        self.card_effective_work = StatCard("Effective Work", "-", "dynamic work window")

        cards = [
            self.card_hashrate,
            self.card_scan,
            self.card_verified_rate,
            self.card_accepted,
            self.card_rejected,
            self.card_candidates,
            self.card_verified,
            self.card_verify_yield,
            self.card_accept_yield,
            self.card_job,
            self.card_height,
            self.card_queues,
            self.card_backend,
            self.card_effective_target,
            self.card_effective_work,
        ]

        for i, card in enumerate(cards):
            grid.addWidget(card, i // 5, i % 5)

        return wrapper

    def _make_connection_group(self) -> QGroupBox:
        box = QGroupBox("Backend / Pool")
        form = QFormLayout(box)
        form.setSpacing(10)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["stratum", "blocknet", "solo", "monerorpc"])
        self.backend_combo.currentTextChanged.connect(self._sync_backend_controls)

        self.host_edit = QLineEdit("127.0.0.1")
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(3333)

        self.login_edit = QLineEdit("x")
        self.pass_edit = QLineEdit("x")
        self.agent_edit = QLineEdit("GPUMiner-PyQt5/0.2")
        self.tls_check = QCheckBox("Use TLS for direct stratum")

        form.addRow("Backend", self.backend_combo)
        form.addRow("Host", self.host_edit)
        form.addRow("Port", self.port_spin)
        form.addRow("Login / Wallet", self.login_edit)
        form.addRow("Password", self.pass_edit)
        form.addRow("Agent", self.agent_edit)
        form.addRow("", self.tls_check)
        return box

    def _make_blocknet_group(self) -> QGroupBox:
        box = QGroupBox("BlockNet Relay / P2Pool")
        grid = QGridLayout(box)

        self.blocknet_relay_edit = QLineEdit("")
        self.blocknet_token_edit = QLineEdit("")
        self.blocknet_prefix_edit = QLineEdit("/v1")

        self.blocknet_force_scheme_combo = QComboBox()
        self.blocknet_force_scheme_combo.addItems(["", "http", "https"])

        self.blocknet_verify_tls_check = QCheckBox("Verify HTTPS certificate")
        self.blocknet_timeout_spin = QSpinBox()
        self.blocknet_timeout_spin.setRange(1, 600)
        self.blocknet_timeout_spin.setValue(30)

        self.blocknet_poll_ms_spin = QSpinBox()
        self.blocknet_poll_ms_spin.setRange(20, 5000)
        self.blocknet_poll_ms_spin.setValue(50)

        self.blocknet_max_msgs_spin = QSpinBox()
        self.blocknet_max_msgs_spin.setRange(1, 1024)
        self.blocknet_max_msgs_spin.setValue(32)

        grid.addWidget(QLabel("Relay host[:port] or URL"), 0, 0)
        grid.addWidget(self.blocknet_relay_edit, 0, 1, 1, 3)

        grid.addWidget(QLabel("Token"), 1, 0)
        grid.addWidget(self.blocknet_token_edit, 1, 1, 1, 3)

        grid.addWidget(QLabel("API prefix"), 2, 0)
        grid.addWidget(self.blocknet_prefix_edit, 2, 1)
        grid.addWidget(QLabel("Force scheme"), 2, 2)
        grid.addWidget(self.blocknet_force_scheme_combo, 2, 3)

        grid.addWidget(self.blocknet_verify_tls_check, 3, 0, 1, 2)
        grid.addWidget(QLabel("HTTP timeout (s)"), 3, 2)
        grid.addWidget(self.blocknet_timeout_spin, 3, 3)

        grid.addWidget(QLabel("Poll interval (ms)"), 4, 0)
        grid.addWidget(self.blocknet_poll_ms_spin, 4, 1)
        grid.addWidget(QLabel("Poll max msgs"), 4, 2)
        grid.addWidget(self.blocknet_max_msgs_spin, 4, 3)

        return box

    def _make_monerorpc_group(self) -> QGroupBox:
        box = QGroupBox("MoneroRPC Broker")
        grid = QGridLayout(box)

        self.monero_rpc_url_edit = QLineEdit("")
        self.monero_rpc_token_edit = QLineEdit("")
        self.monero_rpc_prefix_edit = QLineEdit("/v1")
        self.monero_rpc_force_scheme_combo = QComboBox()
        self.monero_rpc_force_scheme_combo.addItems(["", "http", "https"])

        self.monero_rpc_verify_tls_check = QCheckBox("Verify HTTPS certificate")
        self.monero_rpc_timeout_spin = QSpinBox()
        self.monero_rpc_timeout_spin.setRange(1, 600)
        self.monero_rpc_timeout_spin.setValue(30)

        self.monero_rpc_poll_ms_spin = QSpinBox()
        self.monero_rpc_poll_ms_spin.setRange(20, 10000)
        self.monero_rpc_poll_ms_spin.setValue(250)

        self.monero_rpc_client_id_edit = QLineEdit("")
        self.monero_rpc_lease_size_spin = QSpinBox()
        self.monero_rpc_lease_size_spin.setRange(1, 1_000_000_000)
        self.monero_rpc_lease_size_spin.setValue(8_388_608)

        self.monero_rpc_require_leases_check = QCheckBox("Require backend nonce leases (no local fallback)")

        self.monero_rpc_feeder_mode_combo = QComboBox()
        self.monero_rpc_feeder_mode_combo.addItems(["none", "solo", "blocknet"])
        self.monero_rpc_feeder_mode_combo.currentTextChanged.connect(self._sync_backend_controls)

        self.monero_rpc_feeder_poll_ms_spin = QSpinBox()
        self.monero_rpc_feeder_poll_ms_spin.setRange(100, 60_000)
        self.monero_rpc_feeder_poll_ms_spin.setValue(1000)

        grid.addWidget(QLabel("Broker URL or host[:port]"), 0, 0)
        grid.addWidget(self.monero_rpc_url_edit, 0, 1, 1, 3)

        grid.addWidget(QLabel("Token"), 1, 0)
        grid.addWidget(self.monero_rpc_token_edit, 1, 1, 1, 3)

        grid.addWidget(QLabel("API prefix"), 2, 0)
        grid.addWidget(self.monero_rpc_prefix_edit, 2, 1)
        grid.addWidget(QLabel("Force scheme"), 2, 2)
        grid.addWidget(self.monero_rpc_force_scheme_combo, 2, 3)

        grid.addWidget(self.monero_rpc_verify_tls_check, 3, 0, 1, 2)
        grid.addWidget(QLabel("HTTP timeout (s)"), 3, 2)
        grid.addWidget(self.monero_rpc_timeout_spin, 3, 3)

        grid.addWidget(QLabel("Poll interval (ms)"), 4, 0)
        grid.addWidget(self.monero_rpc_poll_ms_spin, 4, 1)
        grid.addWidget(QLabel("Client ID"), 4, 2)
        grid.addWidget(self.monero_rpc_client_id_edit, 4, 3)

        grid.addWidget(QLabel("Lease size"), 5, 0)
        grid.addWidget(self.monero_rpc_lease_size_spin, 5, 1)
        grid.addWidget(self.monero_rpc_require_leases_check, 5, 2, 1, 2)

        grid.addWidget(QLabel("Embedded feeder"), 6, 0)
        grid.addWidget(self.monero_rpc_feeder_mode_combo, 6, 1)
        grid.addWidget(QLabel("Feeder poll (ms)"), 6, 2)
        grid.addWidget(self.monero_rpc_feeder_poll_ms_spin, 6, 3)

        note = QLabel("Feeder=solo uses Solo/monerod settings below. Feeder=blocknet uses BlockNet relay settings above.")
        note.setObjectName("SubTitle")
        note.setWordWrap(True)
        grid.addWidget(note, 7, 0, 1, 4)

        return box

    def _make_solo_group(self) -> QGroupBox:
        box = QGroupBox("Solo Mining / monerod")
        grid = QGridLayout(box)

        self.solo_wallet_edit = QLineEdit("")
        self.solo_rpc_edit = QLineEdit("http://127.0.0.1:18081")
        self.solo_zmq_edit = QLineEdit("tcp://127.0.0.1:18083")
        self.solo_use_zmq_check = QCheckBox("Use ZMQ pub notifications")
        self.solo_use_zmq_check.setChecked(True)

        self.solo_poll_spin = QDoubleSpinBox()
        self.solo_poll_spin.setDecimals(2)
        self.solo_poll_spin.setRange(0.25, 60.0)
        self.solo_poll_spin.setSingleStep(0.25)
        self.solo_poll_spin.setValue(2.0)

        self.solo_reserve_spin = QSpinBox()
        self.solo_reserve_spin.setRange(0, 255)
        self.solo_reserve_spin.setValue(60)

        grid.addWidget(QLabel("Wallet address"), 0, 0)
        grid.addWidget(self.solo_wallet_edit, 0, 1, 1, 3)

        grid.addWidget(QLabel("Daemon RPC URL"), 1, 0)
        grid.addWidget(self.solo_rpc_edit, 1, 1, 1, 3)

        grid.addWidget(QLabel("ZMQ pub URL"), 2, 0)
        grid.addWidget(self.solo_zmq_edit, 2, 1, 1, 3)

        grid.addWidget(self.solo_use_zmq_check, 3, 0, 1, 2)
        grid.addWidget(QLabel("Fallback poll (s)"), 3, 2)
        grid.addWidget(self.solo_poll_spin, 3, 3)

        grid.addWidget(QLabel("Reserve size"), 4, 0)
        grid.addWidget(self.solo_reserve_spin, 4, 1)

        return box

    def _make_opencl_group(self) -> QGroupBox:
        box = QGroupBox("OpenCL / RandomX")
        grid = QGridLayout(box)

        self.kernel_edit = QLineEdit("kernels/blocknet_randomx_vm_opencl.cl")
        self.loader_edit = QLineEdit("OpenCL.dll")

        self.verifier_dll_edit = QLineEdit("randomx_verify.dll")
        self.randomx_runtime_dll_edit = QLineEdit("randomx-dll.dll")
        self.preload_randomx_runtime_check = QCheckBox("Preload RandomX runtime before verifier")
        self.preload_randomx_runtime_check.setChecked(True)

        self.build_opts_edit = QLineEdit(
            "-cl-std=CL1.2 "
            "-DBN_PREFILTER_ROUNDS=96 "
            "-DBN_TAIL_PREDICT_ROUNDS=8 "
            "-DBN_TUNE_WORDS=256 "
            "-DBN_LOCAL_STAGE_SIZE=128 "
            "-DBN_LOCAL_TOPK=8"
        )

        self.platform_spin = QSpinBox()
        self.platform_spin.setRange(0, 64)

        self.device_spin = QSpinBox()
        self.device_spin.setRange(0, 128)

        self.gws_spin = QSpinBox()
        self.gws_spin.setRange(1, 100_000_000)
        self.gws_spin.setValue(2_097_152)

        self.lws_spin = QSpinBox()
        self.lws_spin.setRange(0, 4096)
        self.lws_spin.setValue(128)

        self.max_results_spin = QSpinBox()
        self.max_results_spin.setRange(1, 65535)
        self.max_results_spin.setValue(4096)

        self.nonce_offset_spin = QSpinBox()
        self.nonce_offset_spin.setRange(0, 255)
        self.nonce_offset_spin.setValue(39)

        browse_kernel_btn = QPushButton("Browse…")
        browse_kernel_btn.clicked.connect(self.browse_kernel)

        browse_loader_btn = QPushButton("Browse…")
        browse_loader_btn.clicked.connect(self.browse_loader)

        browse_verifier_btn = QPushButton("Browse…")
        browse_verifier_btn.clicked.connect(self.browse_verifier_dll)

        browse_runtime_btn = QPushButton("Browse…")
        browse_runtime_btn.clicked.connect(self.browse_randomx_runtime_dll)

        grid.addWidget(QLabel("Kernel path"), 0, 0)
        grid.addWidget(self.kernel_edit, 0, 1, 1, 2)
        grid.addWidget(browse_kernel_btn, 0, 3)

        grid.addWidget(QLabel("Verifier DLL"), 1, 0)
        grid.addWidget(self.verifier_dll_edit, 1, 1, 1, 2)
        grid.addWidget(browse_verifier_btn, 1, 3)

        grid.addWidget(QLabel("RandomX runtime DLL"), 2, 0)
        grid.addWidget(self.randomx_runtime_dll_edit, 2, 1, 1, 2)
        grid.addWidget(browse_runtime_btn, 2, 3)

        grid.addWidget(QLabel("OpenCL loader"), 3, 0)
        grid.addWidget(self.loader_edit, 3, 1, 1, 2)
        grid.addWidget(browse_loader_btn, 3, 3)

        grid.addWidget(self.preload_randomx_runtime_check, 4, 0, 1, 2)

        grid.addWidget(QLabel("Build options"), 5, 0)
        grid.addWidget(self.build_opts_edit, 5, 1, 1, 3)

        grid.addWidget(QLabel("Platform index"), 6, 0)
        grid.addWidget(self.platform_spin, 6, 1)
        grid.addWidget(QLabel("Device index"), 6, 2)
        grid.addWidget(self.device_spin, 6, 3)

        grid.addWidget(QLabel("Global work size (chunk mode)"), 7, 0)
        grid.addWidget(self.gws_spin, 7, 1)
        grid.addWidget(QLabel("Local work size (0 = auto)"), 7, 2)
        grid.addWidget(self.lws_spin, 7, 3)

        grid.addWidget(QLabel("Max GPU results"), 8, 0)
        grid.addWidget(self.max_results_spin, 8, 1)
        grid.addWidget(QLabel("Nonce offset"), 8, 2)
        grid.addWidget(self.nonce_offset_spin, 8, 3)

        return box

    def _make_pipeline_group(self) -> QGroupBox:
        box = QGroupBox("Pipeline / Modes / Throughput")
        grid = QGridLayout(box)

        self.scan_mode_combo = QComboBox()
        self.scan_mode_combo.addItems(["chunk", "hash_batch"])
        self.scan_mode_combo.currentTextChanged.connect(self._sync_scan_mode_controls)
        self.scan_mode_combo.currentTextChanged.connect(self.schedule_save)

        self.enable_cpu_verify_check = QCheckBox("Enable CPU verify")
        self.enable_cpu_verify_check.setChecked(True)
        self.enable_cpu_verify_check.toggled.connect(self._sync_verification_controls)

        self.submit_unverified_check = QCheckBox("Submit unverified shares when CPU verify is off")
        self.submit_unverified_check.setChecked(False)

        self.enable_cpu_verify_batch_check = QCheckBox("Use native CPU verify batch")
        self.enable_cpu_verify_batch_check.setChecked(True)
        self.enable_cpu_verify_batch_check.toggled.connect(self._sync_verification_controls)

        self.enable_cpu_hash_batch_check = QCheckBox("Use exact hash teacher batch")
        self.enable_cpu_hash_batch_check.setChecked(True)
        self.enable_cpu_hash_batch_check.toggled.connect(self._sync_verification_controls)

        self.enable_cpu_tail_batch_check = QCheckBox("Use tail-only prescreen batch")
        self.enable_cpu_tail_batch_check.setChecked(True)
        self.enable_cpu_tail_batch_check.toggled.connect(self._sync_verification_controls)

        self.enable_bucket_tuning_check = QCheckBox("Enable seed EMA tuning")
        self.enable_bucket_tuning_check.setChecked(True)
        self.enable_bucket_tuning_check.toggled.connect(self._sync_tuning_controls)

        self.enable_job_tuning_check = QCheckBox("Enable per-job tuning")
        self.enable_job_tuning_check.setChecked(True)
        self.enable_job_tuning_check.toggled.connect(self._sync_tuning_controls)

        self.adaptive_queue_throttle_check = QCheckBox("Adaptive queue / stale-risk throttling")
        self.adaptive_queue_throttle_check.setChecked(True)
        self.adaptive_queue_throttle_check.toggled.connect(self._sync_adaptive_controls)

        self.sort_candidates_check = QCheckBox("Sort candidates before processing")
        self.sort_candidates_check.setChecked(True)

        self.scan_chunk_spin = QSpinBox()
        self.scan_chunk_spin.setRange(1, 100_000_000)
        self.scan_chunk_spin.setValue(262_144)

        self.hash_batch_size_spin = QSpinBox()
        self.hash_batch_size_spin.setRange(1, 100_000_000)
        self.hash_batch_size_spin.setValue(262_144)

        self.scan_target_spin = QSpinBox()
        self.scan_target_spin.setRange(1, 65535)
        self.scan_target_spin.setValue(512)

        self.max_scan_ms_spin = QSpinBox()
        self.max_scan_ms_spin.setRange(1, 5000)
        self.max_scan_ms_spin.setValue(15)

        self.cpu_verify_limit_spin = QSpinBox()
        self.cpu_verify_limit_spin.setRange(1, 1_000_000)
        self.cpu_verify_limit_spin.setValue(1024)

        self.verify_threads_spin = QSpinBox()
        self.verify_threads_spin.setRange(1, 64)
        self.verify_threads_spin.setValue(1)

        self.cpu_verify_batch_size_spin = QSpinBox()
        self.cpu_verify_batch_size_spin.setRange(1, 4096)
        self.cpu_verify_batch_size_spin.setValue(64)

        self.cpu_verify_batch_wait_ms_spin = QSpinBox()
        self.cpu_verify_batch_wait_ms_spin.setRange(0, 100)
        self.cpu_verify_batch_wait_ms_spin.setValue(2)

        self.cpu_verify_native_threads_spin = QSpinBox()
        self.cpu_verify_native_threads_spin.setRange(0, 128)
        self.cpu_verify_native_threads_spin.setValue(0)

        self.cpu_hash_batch_min_size_spin = QSpinBox()
        self.cpu_hash_batch_min_size_spin.setRange(2, 4096)
        self.cpu_hash_batch_min_size_spin.setValue(8)

        self.cpu_hash_batch_threads_spin = QSpinBox()
        self.cpu_hash_batch_threads_spin.setRange(0, 128)
        self.cpu_hash_batch_threads_spin.setValue(0)

        self.cpu_tail_batch_min_size_spin = QSpinBox()
        self.cpu_tail_batch_min_size_spin.setRange(2, 4096)
        self.cpu_tail_batch_min_size_spin.setValue(32)

        self.cpu_tail_batch_threads_spin = QSpinBox()
        self.cpu_tail_batch_threads_spin.setRange(0, 128)
        self.cpu_tail_batch_threads_spin.setValue(0)

        self.verify_queue_limit_spin = QSpinBox()
        self.verify_queue_limit_spin.setRange(1, 100_000)
        self.verify_queue_limit_spin.setValue(2048)

        self.submit_queue_limit_spin = QSpinBox()
        self.submit_queue_limit_spin.setRange(1, 100_000)
        self.submit_queue_limit_spin.setValue(512)

        self.scan_pause_ms_spin = QSpinBox()
        self.scan_pause_ms_spin.setRange(0, 5000)
        self.scan_pause_ms_spin.setValue(0)

        self.stats_update_ms_spin = QSpinBox()
        self.stats_update_ms_spin.setRange(50, 5000)
        self.stats_update_ms_spin.setValue(500)

        self.verify_queue_soft_pct_spin = QDoubleSpinBox()
        self.verify_queue_soft_pct_spin.setDecimals(2)
        self.verify_queue_soft_pct_spin.setRange(0.10, 0.98)
        self.verify_queue_soft_pct_spin.setSingleStep(0.05)
        self.verify_queue_soft_pct_spin.setValue(0.70)

        self.submit_queue_soft_pct_spin = QDoubleSpinBox()
        self.submit_queue_soft_pct_spin.setDecimals(2)
        self.submit_queue_soft_pct_spin.setRange(0.10, 0.98)
        self.submit_queue_soft_pct_spin.setSingleStep(0.05)
        self.submit_queue_soft_pct_spin.setValue(0.70)

        self.min_candidate_target_spin = QSpinBox()
        self.min_candidate_target_spin.setRange(1, 65535)
        self.min_candidate_target_spin.setValue(64)

        self.min_dynamic_work_pct_spin = QDoubleSpinBox()
        self.min_dynamic_work_pct_spin.setDecimals(2)
        self.min_dynamic_work_pct_spin.setRange(0.05, 1.00)
        self.min_dynamic_work_pct_spin.setSingleStep(0.05)
        self.min_dynamic_work_pct_spin.setValue(0.25)

        self.job_age_soft_ms_spin = QSpinBox()
        self.job_age_soft_ms_spin.setRange(1, 10000)
        self.job_age_soft_ms_spin.setValue(800)

        self.job_age_hard_ms_spin = QSpinBox()
        self.job_age_hard_ms_spin.setRange(1, 30000)
        self.job_age_hard_ms_spin.setValue(2500)

        grid.addWidget(QLabel("GPU scan mode"), 0, 0)
        grid.addWidget(self.scan_mode_combo, 0, 1)
        grid.addWidget(QLabel("Candidate target / pass"), 0, 2)
        grid.addWidget(self.scan_target_spin, 0, 3)

        grid.addWidget(self.enable_cpu_verify_check, 1, 0, 1, 2)
        grid.addWidget(self.submit_unverified_check, 1, 2, 1, 2)

        grid.addWidget(self.enable_cpu_verify_batch_check, 2, 0, 1, 2)
        grid.addWidget(QLabel("Native verify batch threads (0 = auto)"), 2, 2)
        grid.addWidget(self.cpu_verify_native_threads_spin, 2, 3)

        grid.addWidget(QLabel("CPU verify batch size"), 3, 0)
        grid.addWidget(self.cpu_verify_batch_size_spin, 3, 1)
        grid.addWidget(QLabel("Verify batch wait (ms)"), 3, 2)
        grid.addWidget(self.cpu_verify_batch_wait_ms_spin, 3, 3)

        grid.addWidget(self.enable_cpu_hash_batch_check, 4, 0, 1, 2)
        grid.addWidget(QLabel("Teacher hash batch threads (0 = auto)"), 4, 2)
        grid.addWidget(self.cpu_hash_batch_threads_spin, 4, 3)

        grid.addWidget(QLabel("Teacher min batch size"), 5, 0)
        grid.addWidget(self.cpu_hash_batch_min_size_spin, 5, 1)
        grid.addWidget(self.enable_cpu_tail_batch_check, 5, 2, 1, 2)

        grid.addWidget(QLabel("Tail prescreen min batch size"), 6, 0)
        grid.addWidget(self.cpu_tail_batch_min_size_spin, 6, 1)
        grid.addWidget(QLabel("Tail prescreen threads (0 = auto)"), 6, 2)
        grid.addWidget(self.cpu_tail_batch_threads_spin, 6, 3)

        grid.addWidget(self.enable_bucket_tuning_check, 7, 0, 1, 2)
        grid.addWidget(self.enable_job_tuning_check, 7, 2, 1, 2)

        grid.addWidget(self.adaptive_queue_throttle_check, 8, 0, 1, 2)
        grid.addWidget(self.sort_candidates_check, 8, 2, 1, 2)

        grid.addWidget(QLabel("Chunk size"), 9, 0)
        grid.addWidget(self.scan_chunk_spin, 9, 1)
        grid.addWidget(QLabel("Hash batch size"), 9, 2)
        grid.addWidget(self.hash_batch_size_spin, 9, 3)

        grid.addWidget(QLabel("Max scan time (ms, chunk mode)"), 10, 0)
        grid.addWidget(self.max_scan_ms_spin, 10, 1)
        grid.addWidget(QLabel("Candidate process limit"), 10, 2)
        grid.addWidget(self.cpu_verify_limit_spin, 10, 3)

        grid.addWidget(QLabel("Python verify threads"), 11, 0)
        grid.addWidget(self.verify_threads_spin, 11, 1)
        grid.addWidget(QLabel("Verify queue limit"), 11, 2)
        grid.addWidget(self.verify_queue_limit_spin, 11, 3)

        grid.addWidget(QLabel("Scan pause (ms)"), 12, 0)
        grid.addWidget(self.scan_pause_ms_spin, 12, 1)
        grid.addWidget(QLabel("Submit queue limit"), 12, 2)
        grid.addWidget(self.submit_queue_limit_spin, 12, 3)

        grid.addWidget(QLabel("Verify queue soft pct"), 13, 0)
        grid.addWidget(self.verify_queue_soft_pct_spin, 13, 1)
        grid.addWidget(QLabel("Submit queue soft pct"), 13, 2)
        grid.addWidget(self.submit_queue_soft_pct_spin, 13, 3)

        grid.addWidget(QLabel("Min candidate target"), 14, 0)
        grid.addWidget(self.min_candidate_target_spin, 14, 1)
        grid.addWidget(QLabel("Min dynamic work pct"), 14, 2)
        grid.addWidget(self.min_dynamic_work_pct_spin, 14, 3)

        grid.addWidget(QLabel("Job age soft (ms)"), 15, 0)
        grid.addWidget(self.job_age_soft_ms_spin, 15, 1)
        grid.addWidget(QLabel("Job age hard (ms)"), 15, 2)
        grid.addWidget(self.job_age_hard_ms_spin, 15, 3)

        grid.addWidget(QLabel("Stats update (ms)"), 16, 0)
        grid.addWidget(self.stats_update_ms_spin, 16, 1)

        return box

    def _make_tuning_group(self) -> QGroupBox:
        box = QGroupBox("EMA Learning / Bucket Tuning")
        grid = QGridLayout(box)

        self.ema_alpha_spin = QDoubleSpinBox()
        self.ema_alpha_spin.setDecimals(3)
        self.ema_alpha_spin.setRange(0.010, 0.500)
        self.ema_alpha_spin.setSingleStep(0.005)
        self.ema_alpha_spin.setValue(0.080)

        self.tune_verified_spin = QDoubleSpinBox()
        self.tune_verified_spin.setDecimals(2)
        self.tune_verified_spin.setRange(0.0, 10.0)
        self.tune_verified_spin.setValue(1.0)

        self.tune_accepted_spin = QDoubleSpinBox()
        self.tune_accepted_spin.setDecimals(2)
        self.tune_accepted_spin.setRange(0.0, 10.0)
        self.tune_accepted_spin.setValue(2.0)

        self.tune_cpu_reject_spin = QDoubleSpinBox()
        self.tune_cpu_reject_spin.setDecimals(2)
        self.tune_cpu_reject_spin.setRange(0.0, 20.0)
        self.tune_cpu_reject_spin.setValue(1.5)

        self.tune_pool_reject_spin = QDoubleSpinBox()
        self.tune_pool_reject_spin.setDecimals(2)
        self.tune_pool_reject_spin.setRange(0.0, 20.0)
        self.tune_pool_reject_spin.setValue(2.0)

        self.tune_pressure_spin = QDoubleSpinBox()
        self.tune_pressure_spin.setDecimals(2)
        self.tune_pressure_spin.setRange(0.0, 50.0)
        self.tune_pressure_spin.setValue(8.0)

        self.tune_work_bonus_spin = QDoubleSpinBox()
        self.tune_work_bonus_spin.setDecimals(2)
        self.tune_work_bonus_spin.setRange(0.0, 20.0)
        self.tune_work_bonus_spin.setValue(2.0)

        self.tune_tail_bins_spin = QSpinBox()
        self.tune_tail_bins_spin.setRange(1, 64)
        self.tune_tail_bins_spin.setValue(16)

        self.tune_confidence_div_spin = QDoubleSpinBox()
        self.tune_confidence_div_spin.setDecimals(2)
        self.tune_confidence_div_spin.setRange(0.5, 64.0)
        self.tune_confidence_div_spin.setValue(8.0)

        self.tune_quality_reward_spin = QDoubleSpinBox()
        self.tune_quality_reward_spin.setDecimals(2)
        self.tune_quality_reward_spin.setRange(0.0, 20.0)
        self.tune_quality_reward_spin.setValue(1.0)

        self.tune_stale_penalty_spin = QDoubleSpinBox()
        self.tune_stale_penalty_spin.setDecimals(2)
        self.tune_stale_penalty_spin.setRange(0.0, 20.0)
        self.tune_stale_penalty_spin.setValue(1.0)

        self.tune_duplicate_penalty_spin = QDoubleSpinBox()
        self.tune_duplicate_penalty_spin.setDecimals(2)
        self.tune_duplicate_penalty_spin.setRange(0.0, 20.0)
        self.tune_duplicate_penalty_spin.setValue(0.25)

        self.tune_invalid_penalty_spin = QDoubleSpinBox()
        self.tune_invalid_penalty_spin.setDecimals(2)
        self.tune_invalid_penalty_spin.setRange(0.0, 50.0)
        self.tune_invalid_penalty_spin.setValue(2.5)

        self.tune_backend_error_penalty_spin = QDoubleSpinBox()
        self.tune_backend_error_penalty_spin.setDecimals(2)
        self.tune_backend_error_penalty_spin.setRange(0.0, 20.0)
        self.tune_backend_error_penalty_spin.setValue(0.20)

        grid.addWidget(QLabel("EMA alpha"), 0, 0)
        grid.addWidget(self.ema_alpha_spin, 0, 1)
        grid.addWidget(QLabel("Verify reward"), 0, 2)
        grid.addWidget(self.tune_verified_spin, 0, 3)

        grid.addWidget(QLabel("Accepted reward"), 1, 0)
        grid.addWidget(self.tune_accepted_spin, 1, 1)
        grid.addWidget(QLabel("CPU reject penalty"), 1, 2)
        grid.addWidget(self.tune_cpu_reject_spin, 1, 3)

        grid.addWidget(QLabel("Pool reject penalty"), 2, 0)
        grid.addWidget(self.tune_pool_reject_spin, 2, 1)
        grid.addWidget(QLabel("Pressure penalty"), 2, 2)
        grid.addWidget(self.tune_pressure_spin, 2, 3)

        grid.addWidget(QLabel("Work bonus scale"), 3, 0)
        grid.addWidget(self.tune_work_bonus_spin, 3, 1)
        grid.addWidget(QLabel("Tail bins"), 3, 2)
        grid.addWidget(self.tune_tail_bins_spin, 3, 3)

        grid.addWidget(QLabel("Confidence divisor"), 4, 0)
        grid.addWidget(self.tune_confidence_div_spin, 4, 1)
        grid.addWidget(QLabel("Quality reward"), 4, 2)
        grid.addWidget(self.tune_quality_reward_spin, 4, 3)

        grid.addWidget(QLabel("Stale penalty"), 5, 0)
        grid.addWidget(self.tune_stale_penalty_spin, 5, 1)
        grid.addWidget(QLabel("Duplicate penalty"), 5, 2)
        grid.addWidget(self.tune_duplicate_penalty_spin, 5, 3)

        grid.addWidget(QLabel("Invalid penalty"), 6, 0)
        grid.addWidget(self.tune_invalid_penalty_spin, 6, 1)
        grid.addWidget(QLabel("Backend error penalty"), 6, 2)
        grid.addWidget(self.tune_backend_error_penalty_spin, 6, 3)

        return box

    def _make_controls_group(self) -> QGroupBox:
        box = QGroupBox("Controls")
        layout = QVBoxLayout(box)

        row1 = QHBoxLayout()
        self.start_btn = QPushButton("Start Mining")
        self.stop_btn = QPushButton("Stop")
        row1.addWidget(self.start_btn)
        row1.addWidget(self.stop_btn)

        row2 = QHBoxLayout()
        self.save_btn = QPushButton("Save Config")
        self.reload_btn = QPushButton("Reload Config")
        self.clear_log_btn = QPushButton("Clear Log")
        row2.addWidget(self.save_btn)
        row2.addWidget(self.reload_btn)
        row2.addWidget(self.clear_log_btn)

        self.start_btn.clicked.connect(self.start_mining)
        self.stop_btn.clicked.connect(self.stop_mining)
        self.save_btn.clicked.connect(self.save_config)
        self.reload_btn.clicked.connect(self.load_config)
        self.clear_log_btn.clicked.connect(self.clear_logs)

        layout.addLayout(row1)
        layout.addLayout(row2)
        return box

    def _make_right_tabs(self) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        dashboard = QWidget()
        dash_layout = QVBoxLayout(dashboard)
        dash_layout.setContentsMargins(12, 12, 12, 12)
        dash_layout.setSpacing(10)

        self.summary_lbl = QLabel("Ready")
        big = QFont()
        big.setPointSize(22)
        big.setBold(True)
        self.summary_lbl.setFont(big)

        self.summary_sub_lbl = QLabel("Accepted: 0    Rejected: 0    Job: -")
        self.summary_sub_lbl.setObjectName("SubTitle")
        self.summary_sub_lbl.setWordWrap(True)

        self.detail_metrics_lbl = QLabel("VerifyYield: 0.000    AcceptYield: 0.000")
        self.detail_metrics_lbl.setObjectName("SubTitle")
        self.detail_metrics_lbl.setWordWrap(True)

        self.reject_breakdown_lbl = QLabel("Rejects[stale/dup/invalid/backend]=0/0/0/0")
        self.reject_breakdown_lbl.setObjectName("SubTitle")
        self.reject_breakdown_lbl.setWordWrap(True)

        self.pressure_lbl = QLabel("EffTarget: -    EffWork: -    AgeMs: 0    Q8[v/s/st]=0/0/0")
        self.pressure_lbl.setObjectName("SubTitle")
        self.pressure_lbl.setWordWrap(True)

        dash_layout.addWidget(self.summary_lbl)
        dash_layout.addWidget(self.summary_sub_lbl)
        dash_layout.addWidget(self.detail_metrics_lbl)
        dash_layout.addWidget(self.reject_breakdown_lbl)
        dash_layout.addWidget(self.pressure_lbl)
        dash_layout.addStretch(1)

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(0, 0, 0, 0)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(10)
        self.log_edit.setFont(mono)
        log_layout.addWidget(self.log_edit)

        raw_tab = QWidget()
        raw_layout = QVBoxLayout(raw_tab)
        raw_layout.setContentsMargins(12, 12, 12, 12)

        self.raw_stats_edit = QPlainTextEdit()
        self.raw_stats_edit.setReadOnly(True)
        self.raw_stats_edit.setFont(mono)
        raw_layout.addWidget(self.raw_stats_edit)

        self.tabs.addTab(dashboard, "Dashboard")
        self.tabs.addTab(log_tab, "Log")
        self.tabs.addTab(raw_tab, "Raw Stats")

        return wrap

    def _connect_autosave_signals(self) -> None:
        for edit in [
            self.host_edit,
            self.login_edit,
            self.pass_edit,
            self.agent_edit,
            self.kernel_edit,
            self.loader_edit,
            self.verifier_dll_edit,
            self.randomx_runtime_dll_edit,
            self.build_opts_edit,
            self.blocknet_relay_edit,
            self.blocknet_token_edit,
            self.blocknet_prefix_edit,
            self.monero_rpc_url_edit,
            self.monero_rpc_token_edit,
            self.monero_rpc_prefix_edit,
            self.monero_rpc_client_id_edit,
            self.solo_wallet_edit,
            self.solo_rpc_edit,
            self.solo_zmq_edit,
        ]:
            edit.textChanged.connect(self.schedule_save)

        for check in [
            self.tls_check,
            self.blocknet_verify_tls_check,
            self.monero_rpc_verify_tls_check,
            self.monero_rpc_require_leases_check,
            self.enable_cpu_verify_check,
            self.submit_unverified_check,
            self.enable_cpu_verify_batch_check,
            self.enable_cpu_hash_batch_check,
            self.enable_cpu_tail_batch_check,
            self.enable_bucket_tuning_check,
            self.enable_job_tuning_check,
            self.adaptive_queue_throttle_check,
            self.sort_candidates_check,
            self.solo_use_zmq_check,
            self.preload_randomx_runtime_check,
        ]:
            check.toggled.connect(self.schedule_save)

        for combo in [
            self.backend_combo,
            self.scan_mode_combo,
            self.blocknet_force_scheme_combo,
            self.monero_rpc_force_scheme_combo,
            self.monero_rpc_feeder_mode_combo,
        ]:
            combo.currentTextChanged.connect(self.schedule_save)

        self.tabs.currentChanged.connect(self.schedule_save)

        for spin in [
            self.port_spin,
            self.platform_spin,
            self.device_spin,
            self.gws_spin,
            self.lws_spin,
            self.max_results_spin,
            self.nonce_offset_spin,
            self.blocknet_timeout_spin,
            self.blocknet_poll_ms_spin,
            self.blocknet_max_msgs_spin,
            self.monero_rpc_timeout_spin,
            self.monero_rpc_poll_ms_spin,
            self.monero_rpc_lease_size_spin,
            self.monero_rpc_feeder_poll_ms_spin,
            self.scan_chunk_spin,
            self.hash_batch_size_spin,
            self.scan_target_spin,
            self.max_scan_ms_spin,
            self.cpu_verify_limit_spin,
            self.verify_threads_spin,
            self.cpu_verify_batch_size_spin,
            self.cpu_verify_batch_wait_ms_spin,
            self.cpu_verify_native_threads_spin,
            self.cpu_hash_batch_min_size_spin,
            self.cpu_hash_batch_threads_spin,
            self.cpu_tail_batch_min_size_spin,
            self.cpu_tail_batch_threads_spin,
            self.verify_queue_limit_spin,
            self.submit_queue_limit_spin,
            self.scan_pause_ms_spin,
            self.stats_update_ms_spin,
            self.verify_queue_soft_pct_spin,
            self.submit_queue_soft_pct_spin,
            self.min_candidate_target_spin,
            self.min_dynamic_work_pct_spin,
            self.job_age_soft_ms_spin,
            self.job_age_hard_ms_spin,
            self.ema_alpha_spin,
            self.tune_verified_spin,
            self.tune_accepted_spin,
            self.tune_cpu_reject_spin,
            self.tune_pool_reject_spin,
            self.tune_pressure_spin,
            self.tune_work_bonus_spin,
            self.tune_tail_bins_spin,
            self.tune_confidence_div_spin,
            self.tune_quality_reward_spin,
            self.tune_stale_penalty_spin,
            self.tune_duplicate_penalty_spin,
            self.tune_invalid_penalty_spin,
            self.tune_backend_error_penalty_spin,
            self.solo_reserve_spin,
            self.solo_poll_spin,
        ]:
            spin.valueChanged.connect(self.schedule_save)

    def _sync_backend_controls(self) -> None:
        backend = self.backend_combo.currentText().strip().lower()
        feeder_mode = self._monero_feeder_mode()

        using_stratum = backend == "stratum"
        using_blocknet = backend == "blocknet"
        using_solo = backend == "solo"
        using_monerorpc = backend == "monerorpc"

        self.host_edit.setEnabled(using_stratum)
        self.port_spin.setEnabled(using_stratum)
        self.login_edit.setEnabled(using_stratum)
        self.pass_edit.setEnabled(using_stratum)
        self.tls_check.setEnabled(using_stratum)

        enable_blocknet_group = using_blocknet or (using_monerorpc and feeder_mode == "blocknet")
        enable_solo_group = using_solo or (using_monerorpc and feeder_mode == "solo")

        for w in [
            self.blocknet_relay_edit,
            self.blocknet_token_edit,
            self.blocknet_prefix_edit,
            self.blocknet_force_scheme_combo,
            self.blocknet_verify_tls_check,
            self.blocknet_timeout_spin,
            self.blocknet_poll_ms_spin,
            self.blocknet_max_msgs_spin,
        ]:
            w.setEnabled(enable_blocknet_group)

        for w in [
            self.monero_rpc_url_edit,
            self.monero_rpc_token_edit,
            self.monero_rpc_prefix_edit,
            self.monero_rpc_force_scheme_combo,
            self.monero_rpc_verify_tls_check,
            self.monero_rpc_timeout_spin,
            self.monero_rpc_poll_ms_spin,
            self.monero_rpc_client_id_edit,
            self.monero_rpc_lease_size_spin,
            self.monero_rpc_require_leases_check,
            self.monero_rpc_feeder_mode_combo,
            self.monero_rpc_feeder_poll_ms_spin,
        ]:
            w.setEnabled(using_monerorpc)

        for w in [
            self.solo_wallet_edit,
            self.solo_rpc_edit,
            self.solo_zmq_edit,
            self.solo_use_zmq_check,
            self.solo_poll_spin,
            self.solo_reserve_spin,
        ]:
            w.setEnabled(enable_solo_group)

        if using_solo or using_monerorpc:
            self.submit_unverified_check.setChecked(False)

        self._sync_verification_controls()

    def _sync_scan_mode_controls(self) -> None:
        mode = self.scan_mode_combo.currentText().strip().lower()
        is_chunk = mode == "chunk"
        is_hash_batch = mode == "hash_batch"

        self.gws_spin.setEnabled(is_chunk)
        self.scan_chunk_spin.setEnabled(is_chunk)
        self.max_scan_ms_spin.setEnabled(is_chunk)
        self.hash_batch_size_spin.setEnabled(is_hash_batch)

    def _sync_verification_controls(self) -> None:
        verify_on = self.enable_cpu_verify_check.isChecked()
        verify_batch_on = verify_on and self.enable_cpu_verify_batch_check.isChecked()
        hash_teacher_on = verify_on and self.enable_cpu_hash_batch_check.isChecked()
        tail_batch_on = verify_on and self.enable_cpu_tail_batch_check.isChecked()
        backend = self.backend_combo.currentText().strip().lower()
        raw_forbidden = backend in {"solo", "monerorpc"}

        self.submit_unverified_check.setEnabled((not verify_on) and (not raw_forbidden))
        if raw_forbidden:
            self.submit_unverified_check.setChecked(False)

        if (verify_batch_on or hash_teacher_on or tail_batch_on) and self.verify_threads_spin.value() != 1:
            self.verify_threads_spin.blockSignals(True)
            self.verify_threads_spin.setValue(1)
            self.verify_threads_spin.blockSignals(False)

        self.verify_threads_spin.setEnabled(
            verify_on and (not verify_batch_on) and (not hash_teacher_on) and (not tail_batch_on)
        )
        self.verify_queue_limit_spin.setEnabled(verify_on)

        self.enable_cpu_verify_batch_check.setEnabled(verify_on)
        self.cpu_verify_batch_size_spin.setEnabled(verify_batch_on or hash_teacher_on or tail_batch_on)
        self.cpu_verify_batch_wait_ms_spin.setEnabled(verify_batch_on or hash_teacher_on or tail_batch_on)
        self.cpu_verify_native_threads_spin.setEnabled(verify_batch_on)

        self.enable_cpu_hash_batch_check.setEnabled(verify_on)
        self.cpu_hash_batch_min_size_spin.setEnabled(hash_teacher_on)
        self.cpu_hash_batch_threads_spin.setEnabled(hash_teacher_on)

        self.enable_cpu_tail_batch_check.setEnabled(verify_on)
        self.cpu_tail_batch_min_size_spin.setEnabled(tail_batch_on)
        self.cpu_tail_batch_threads_spin.setEnabled(tail_batch_on)

    def _sync_tuning_controls(self) -> None:
        enabled = self.enable_bucket_tuning_check.isChecked() or self.enable_job_tuning_check.isChecked()
        for w in [
            self.ema_alpha_spin,
            self.tune_verified_spin,
            self.tune_accepted_spin,
            self.tune_cpu_reject_spin,
            self.tune_pool_reject_spin,
            self.tune_pressure_spin,
            self.tune_work_bonus_spin,
            self.tune_tail_bins_spin,
            self.tune_confidence_div_spin,
            self.tune_quality_reward_spin,
            self.tune_stale_penalty_spin,
            self.tune_duplicate_penalty_spin,
            self.tune_invalid_penalty_spin,
            self.tune_backend_error_penalty_spin,
        ]:
            w.setEnabled(enabled)

    def _sync_adaptive_controls(self) -> None:
        enabled = self.adaptive_queue_throttle_check.isChecked()
        for w in [
            self.verify_queue_soft_pct_spin,
            self.submit_queue_soft_pct_spin,
            self.min_candidate_target_spin,
            self.min_dynamic_work_pct_spin,
            self.job_age_soft_ms_spin,
            self.job_age_hard_ms_spin,
        ]:
            w.setEnabled(enabled)

    def schedule_save(self) -> None:
        self._save_timer.start()

    def browse_verifier_dll(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select verifier DLL / shared library",
            self.verifier_dll_edit.text().strip() or "",
            "Libraries (*.dll *.so *.dylib);;All files (*)",
        )
        if path:
            self.verifier_dll_edit.setText(path)

    def browse_randomx_runtime_dll(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select RandomX runtime DLL / shared library",
            self.randomx_runtime_dll_edit.text().strip() or "",
            "Libraries (*.dll *.so *.dylib);;All files (*)",
        )
        if path:
            self.randomx_runtime_dll_edit.setText(path)

    def browse_kernel(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select OpenCL kernel",
            self.kernel_edit.text().strip() or "",
            "OpenCL (*.cl);;All files (*)",
        )
        if path:
            self.kernel_edit.setText(path)

    def browse_loader(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select OpenCL loader",
            self.loader_edit.text().strip() or "",
            "Libraries (*.dll *.so *.dylib);;All files (*)",
        )
        if path:
            self.loader_edit.setText(path)

    def clear_logs(self) -> None:
        self.log_edit.clear()
        self.raw_stats_edit.clear()

    def build_config(self) -> MinerConfig:
        lws = self.lws_spin.value() or None

        backend = self.backend_combo.currentText().strip().lower()
        feeder_mode = self._monero_feeder_mode()
        host = self.host_edit.text().strip()
        kernel_path = self.kernel_edit.text().strip()
        opencl_loader = self.loader_edit.text().strip()
        verifier_dll_path = self.verifier_dll_edit.text().strip()
        randomx_runtime_dll_path = self.randomx_runtime_dll_edit.text().strip()
        scan_mode = self.scan_mode_combo.currentText().strip().lower()
        agent = self.agent_edit.text().strip() or "GPUMiner-PyQt5/0.2"

        if not kernel_path:
            raise ValueError("Kernel path is required.")
        if not opencl_loader:
            raise ValueError("OpenCL loader is required.")
        if not verifier_dll_path:
            raise ValueError("Verifier DLL path is required.")
        if scan_mode not in {"chunk", "hash_batch"}:
            raise ValueError("GPU scan mode must be 'chunk' or 'hash_batch'.")

        if backend == "blocknet":
            if not self.blocknet_relay_edit.text().strip():
                raise ValueError("BlockNet relay is required when BlockNet backend is selected.")
        elif backend == "solo":
            if not self.solo_wallet_edit.text().strip():
                raise ValueError("Solo wallet address is required.")
            if not self.solo_rpc_edit.text().strip():
                raise ValueError("Solo daemon RPC URL is required.")
        elif backend == "monerorpc":
            if not self.monero_rpc_url_edit.text().strip():
                raise ValueError("MoneroRPC URL is required when MoneroRPC backend is selected.")
            if feeder_mode == "solo":
                if not self.solo_wallet_edit.text().strip():
                    raise ValueError("Solo wallet address is required for MoneroRPC feeder=solo.")
                if not self.solo_rpc_edit.text().strip():
                    raise ValueError("Solo daemon RPC URL is required for MoneroRPC feeder=solo.")
            if feeder_mode == "blocknet":
                if not self.blocknet_relay_edit.text().strip():
                    raise ValueError("BlockNet relay is required for MoneroRPC feeder=blocknet.")
        else:
            if not host:
                raise ValueError("Host is required.")
            if not self.login_edit.text().strip():
                raise ValueError("Login / wallet is required for direct stratum.")

        submit_unverified = self.submit_unverified_check.isChecked()
        if backend in {"solo", "monerorpc"}:
            submit_unverified = False

        return MinerConfig(
            mining_backend=backend,

            host=host,
            port=int(self.port_spin.value()),
            login=self.login_edit.text().strip(),
            password=self.pass_edit.text().strip(),
            agent=agent,
            use_tls=self.tls_check.isChecked(),

            blocknet_api_relay=self.blocknet_relay_edit.text().strip(),
            blocknet_api_token=self.blocknet_token_edit.text().strip(),
            blocknet_api_prefix=self.blocknet_prefix_edit.text().strip() or "/v1",
            blocknet_verify_tls=self.blocknet_verify_tls_check.isChecked(),
            blocknet_timeout_s=float(self.blocknet_timeout_spin.value()),
            blocknet_force_scheme=self._combo_value_or_none(self.blocknet_force_scheme_combo),
            blocknet_poll_interval_ms=int(self.blocknet_poll_ms_spin.value()),
            blocknet_poll_max_msgs=int(self.blocknet_max_msgs_spin.value()),

            monero_rpc_url=self.monero_rpc_url_edit.text().strip(),
            monero_rpc_token=self.monero_rpc_token_edit.text().strip(),
            monero_rpc_prefix=self.monero_rpc_prefix_edit.text().strip() or "/v1",
            monero_rpc_verify_tls=self.monero_rpc_verify_tls_check.isChecked(),
            monero_rpc_timeout_s=float(self.monero_rpc_timeout_spin.value()),
            monero_rpc_force_scheme=self._combo_value_or_none(self.monero_rpc_force_scheme_combo),
            monero_rpc_poll_interval_ms=int(self.monero_rpc_poll_ms_spin.value()),
            monero_rpc_client_id=self.monero_rpc_client_id_edit.text().strip(),
            monero_rpc_lease_size=int(self.monero_rpc_lease_size_spin.value()),
            monero_rpc_require_leases=self.monero_rpc_require_leases_check.isChecked(),
            monero_rpc_feeder_mode=feeder_mode,
            monero_rpc_feeder_poll_interval_ms=int(self.monero_rpc_feeder_poll_ms_spin.value()),

            solo_wallet_address=self.solo_wallet_edit.text().strip(),
            solo_daemon_rpc_url=self.solo_rpc_edit.text().strip(),
            solo_zmq_pub_url=self.solo_zmq_edit.text().strip(),
            solo_use_zmq=self.solo_use_zmq_check.isChecked(),
            solo_poll_fallback_s=float(self.solo_poll_spin.value()),
            solo_reserve_size=int(self.solo_reserve_spin.value()),

            kernel_path=kernel_path,
            opencl_loader=opencl_loader,
            build_options=self.build_opts_edit.text().strip(),
            platform_index=int(self.platform_spin.value()),
            device_index=int(self.device_spin.value()),
            global_work_size=int(self.gws_spin.value()),
            local_work_size=lws,
            max_results=int(self.max_results_spin.value()),
            nonce_offset=int(self.nonce_offset_spin.value()),
            randomx_dll_path=verifier_dll_path,
            randomx_runtime_dll_path=randomx_runtime_dll_path,
            preload_randomx_runtime=self.preload_randomx_runtime_check.isChecked(),
            require_dataset=True,

            enable_cpu_verify=self.enable_cpu_verify_check.isChecked(),
            submit_unverified_shares=submit_unverified,

            enable_cpu_verify_batch=self.enable_cpu_verify_batch_check.isChecked(),
            cpu_verify_batch_size=int(self.cpu_verify_batch_size_spin.value()),
            cpu_verify_batch_wait_ms=int(self.cpu_verify_batch_wait_ms_spin.value()),
            cpu_verify_native_threads=int(self.cpu_verify_native_threads_spin.value()),

            enable_cpu_hash_batch=self.enable_cpu_hash_batch_check.isChecked(),
            cpu_hash_batch_min_size=int(self.cpu_hash_batch_min_size_spin.value()),
            cpu_hash_batch_threads=int(self.cpu_hash_batch_threads_spin.value()),

            enable_cpu_tail_batch=self.enable_cpu_tail_batch_check.isChecked(),
            cpu_tail_batch_min_size=int(self.cpu_tail_batch_min_size_spin.value()),
            cpu_tail_batch_threads=int(self.cpu_tail_batch_threads_spin.value()),

            gpu_scan_mode=scan_mode,
            hash_batch_size=int(self.hash_batch_size_spin.value()),

            scan_chunk_size=int(self.scan_chunk_spin.value()),
            scan_candidate_target=int(self.scan_target_spin.value()),
            max_scan_time_ms=int(self.max_scan_ms_spin.value()),

            cpu_verify_limit=int(self.cpu_verify_limit_spin.value()),
            verify_threads=int(self.verify_threads_spin.value()),
            verify_queue_limit=int(self.verify_queue_limit_spin.value()),
            submit_queue_limit=int(self.submit_queue_limit_spin.value()),

            adaptive_queue_throttle=self.adaptive_queue_throttle_check.isChecked(),
            verify_queue_soft_pct=float(self.verify_queue_soft_pct_spin.value()),
            submit_queue_soft_pct=float(self.submit_queue_soft_pct_spin.value()),
            min_candidate_target=int(self.min_candidate_target_spin.value()),
            min_dynamic_work_pct=float(self.min_dynamic_work_pct_spin.value()),
            job_age_soft_ms=int(self.job_age_soft_ms_spin.value()),
            job_age_hard_ms=int(self.job_age_hard_ms_spin.value()),

            scan_pause_ms=int(self.scan_pause_ms_spin.value()),
            enable_bucket_tuning=self.enable_bucket_tuning_check.isChecked(),
            enable_job_tuning=self.enable_job_tuning_check.isChecked(),
            sort_candidates=self.sort_candidates_check.isChecked(),
            stats_update_ms=int(self.stats_update_ms_spin.value()),

            ema_alpha=float(self.ema_alpha_spin.value()),
            tune_verified_reward=float(self.tune_verified_spin.value()),
            tune_accepted_reward=float(self.tune_accepted_spin.value()),
            tune_cpu_reject_penalty=float(self.tune_cpu_reject_spin.value()),
            tune_pool_reject_penalty=float(self.tune_pool_reject_spin.value()),
            tune_pressure_penalty=float(self.tune_pressure_spin.value()),
            tune_work_bonus_scale=float(self.tune_work_bonus_spin.value()),

            tune_tail_bins=int(self.tune_tail_bins_spin.value()),
            tune_confidence_div=float(self.tune_confidence_div_spin.value()),
            tune_quality_reward=float(self.tune_quality_reward_spin.value()),
            tune_stale_penalty=float(self.tune_stale_penalty_spin.value()),
            tune_duplicate_penalty=float(self.tune_duplicate_penalty_spin.value()),
            tune_invalid_penalty=float(self.tune_invalid_penalty_spin.value()),
            tune_backend_error_penalty=float(self.tune_backend_error_penalty_spin.value()),
        )

    def append_log(self, text: str) -> None:
        text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            return

        self.log_edit.appendPlainText(text)

        max_blocks = 6000
        doc = self.log_edit.document()
        if doc.blockCount() > max_blocks:
            cursor = self.log_edit.textCursor()
            cursor.movePosition(cursor.Start)
            for _ in range(doc.blockCount() - max_blocks):
                cursor.select(cursor.LineUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()

    def update_stats(self, stats: dict) -> None:
        accepted = int(stats.get("accepted", 0))
        rejected = int(stats.get("rejected", 0))
        candidates = int(stats.get("candidates", 0))
        verified = int(stats.get("verified", 0))
        verify_rejected = int(stats.get("verify_rejected", 0))
        stale_rejects = int(stats.get("submit_stale", 0))
        dup_rejects = int(stats.get("submit_duplicate", 0))
        invalid_rejects = int(stats.get("submit_invalid", 0))
        backend_rejects = int(stats.get("submit_backend_error", 0))
        teacher_tail_accepted = int(stats.get("teacher_tail_accepted", 0))

        job_id = str(stats.get("job_id", "-"))
        height = str(stats.get("height", "-"))
        accepted_rate = int(stats.get("p2pool_rate_15m", stats.get("hashrate_est", 0)))
        scan_rate = int(stats.get("gpu_scan_rate_15m", 0))
        verified_rate = int(stats.get("verified_rate_15m", 0))

        cq = int(stats.get("candidate_queue_depth", 0))
        sq = int(stats.get("submit_queue_depth", 0))
        backend = str(stats.get("backend", "unknown"))
        scan_mode = str(stats.get("scan_mode", self.scan_mode_combo.currentText()))
        launches = int(stats.get("scan_launches_last", 0))

        verify_enabled = bool(stats.get("verification_enabled", self.enable_cpu_verify_check.isChecked()))
        verify_batch_enabled = bool(stats.get("verification_batch_enabled", self.enable_cpu_verify_batch_check.isChecked()))
        hash_batch_enabled = bool(stats.get("hash_batch_enabled", self.enable_cpu_hash_batch_check.isChecked()))
        tail_batch_enabled = bool(stats.get("tail_batch_enabled", self.enable_cpu_tail_batch_check.isChecked()))
        hash_batch_min_size = int(stats.get("cpu_hash_batch_min_size", self.cpu_hash_batch_min_size_spin.value()))
        tail_batch_min_size = int(stats.get("cpu_tail_batch_min_size", self.cpu_tail_batch_min_size_spin.value()))
        verify_batch_size = int(stats.get("cpu_verify_batch_size", self.cpu_verify_batch_size_spin.value()))
        job_tuning_enabled = bool(stats.get("job_tuning_enabled", self.enable_job_tuning_check.isChecked()))
        submitted_unverified = int(stats.get("submitted_unverified", 0))
        verify_yield = float(stats.get("verify_yield", 0.0))
        accept_yield = float(stats.get("accept_yield", 0.0))
        eff_target = int(stats.get("effective_candidate_target_last", self.scan_target_spin.value()))
        eff_work = int(stats.get("effective_work_items_last", 0))
        job_age_ms = int(stats.get("job_age_ms_last", 0))
        verify_q8 = int(stats.get("verify_pressure_q8_last", 0))
        submit_q8 = int(stats.get("submit_pressure_q8_last", 0))
        stale_q8 = int(stats.get("stale_risk_q8_last", 0))
        tail_bins = int(stats.get("tail_bins", self.tune_tail_bins_spin.value()))
        scan_window_source = str(stats.get("scan_window_source_last", "local"))
        scan_window_count = int(stats.get("scan_window_count_last", 0))

        self.card_hashrate.set_value(f"{accepted_rate:,} H/s", "accepted work")
        self.card_scan.set_value(f"{scan_rate:,} H/s", "raw scan")
        self.card_verified_rate.set_value(f"{verified_rate:,} H/s", "verified work")
        self.card_accepted.set_value(str(accepted), f"raw_submits={submitted_unverified}")
        self.card_rejected.set_value(
            str(rejected),
            f"stale={stale_rejects} dup={dup_rejects} invalid={invalid_rejects} backend={backend_rejects}",
        )
        self.card_candidates.set_value(
            str(candidates),
            f"verify_rejected={verify_rejected} tail_accepts={teacher_tail_accepted}",
        )
        self.card_verified.set_value(str(verified), f"launches={launches}")
        self.card_verify_yield.set_value(f"{verify_yield:.3f}", "verified / candidates")
        self.card_accept_yield.set_value(f"{accept_yield:.3f}", "accepted / verified")
        self.card_job.set_value(job_id, f"tail_bins={tail_bins}")
        self.card_height.set_value(height, "")
        self.card_queues.set_value(f"{cq} / {sq}", "verify / submit")
        self.card_backend.set_value(
            f"{backend} / {scan_mode}",
            f"verify={'on' if verify_enabled else 'off'} window={scan_window_source}/{scan_window_count}",
        )
        self.card_effective_target.set_value(str(eff_target), f"stale_q8={stale_q8}")
        self.card_effective_work.set_value(str(eff_work), f"age_ms={job_age_ms}")

        self.summary_lbl.setText(f"{accepted_rate:,} H/s")
        self.summary_sub_lbl.setText(
            f"Accepted: {accepted}    Rejected: {rejected}    Candidates: {candidates}    "
            f"Verified: {verified}    VerifyReject: {verify_rejected}    TailAccept: {teacher_tail_accepted}    "
            f"RawSubmits: {submitted_unverified}    Job: {job_id}    Height: {height}    Queues: {cq}/{sq}    "
            f"Mode: {scan_mode}    Backend: {backend}    Window: {scan_window_source}/{scan_window_count}    "
            f"Verify: {'on' if verify_enabled else 'off'}    "
            f"VerifyBatch: {'on' if verify_batch_enabled else 'off'}({verify_batch_size})    "
            f"HashTeacher: {'on' if hash_batch_enabled else 'off'}({hash_batch_min_size})    "
            f"TailBatch: {'on' if tail_batch_enabled else 'off'}({tail_batch_min_size})    "
            f"JobTune: {'on' if job_tuning_enabled else 'off'}    "
            f"SplitTune: {'on' if bool(stats.get('split_tuning', True)) else 'off'}    "
            f"Launches: {launches}"
        )

        self.detail_metrics_lbl.setText(
            f"VerifyYield: {verify_yield:.3f}    "
            f"AcceptYield: {accept_yield:.3f}    "
            f"Accepted15m: {accepted_rate:,} H/s    "
            f"Scan15m: {scan_rate:,} H/s    "
            f"Verified15m: {verified_rate:,} H/s"
        )

        self.reject_breakdown_lbl.setText(
            f"Rejects[stale/dup/invalid/backend]={stale_rejects}/{dup_rejects}/{invalid_rejects}/{backend_rejects}"
        )

        self.pressure_lbl.setText(
            f"EffTarget: {eff_target}    EffWork: {eff_work}    "
            f"AgeMs: {job_age_ms}    Q8[v/s/st]={verify_q8}/{submit_q8}/{stale_q8}    "
            f"TailBins: {tail_bins}    Window: {scan_window_source}/{scan_window_count}"
        )

        try:
            self.raw_stats_edit.setPlainText(json.dumps(stats, indent=2, sort_keys=True))
        except Exception:
            self.raw_stats_edit.setPlainText(str(stats))

    def set_status(self, status: str) -> None:
        s = (status or "idle").strip()
        upper = s.upper()
        self.status_pill.setText(upper)

        if upper in {"RUNNING", "CONNECTED", "MINING"}:
            self.status_pill.setStyleSheet("background:#153a29; color:#eaffea;")
        elif upper in {"STARTING", "RECONNECTING", "STOPPING", "CONNECTING", "WAITING_FOR_LEASE", "WAITING_FOR_FEEDER"}:
            self.status_pill.setStyleSheet("background:#3b3418; color:#fff7da;")
        else:
            self.status_pill.setStyleSheet("background:#4b2020; color:#ffecec;")

        self.start_btn.setEnabled(upper in {"IDLE", "STOPPED", "ERROR"})
        self.stop_btn.setEnabled(
            upper in {
                "RUNNING",
                "STARTING",
                "RECONNECTING",
                "STOPPING",
                "CONNECTED",
                "CONNECTING",
                "MINING",
                "WAITING_FOR_LEASE",
                "WAITING_FOR_FEEDER",
            }
        )

    def _update_uptime(self) -> None:
        if not self.started_at:
            self.uptime_lbl.setText("Uptime: 00:00:00")
            return
        dt = max(0, int(time.time() - self.started_at))
        h = dt // 3600
        m = (dt % 3600) // 60
        s = dt % 60
        self.uptime_lbl.setText(f"Uptime: {h:02d}:{m:02d}:{s:02d}")

    def _config_to_dict(self) -> dict[str, Any]:
        return {
            "mining_backend": self.backend_combo.currentText().strip(),
            "host": self.host_edit.text().strip(),
            "port": int(self.port_spin.value()),
            "login": self.login_edit.text().strip(),
            "password": self.pass_edit.text(),
            "agent": self.agent_edit.text().strip(),
            "use_tls": bool(self.tls_check.isChecked()),

            "blocknet_api_relay": self.blocknet_relay_edit.text().strip(),
            "blocknet_api_token": self.blocknet_token_edit.text().strip(),
            "blocknet_api_prefix": self.blocknet_prefix_edit.text().strip(),
            "blocknet_force_scheme": self.blocknet_force_scheme_combo.currentText().strip(),
            "blocknet_verify_tls": bool(self.blocknet_verify_tls_check.isChecked()),
            "blocknet_timeout_s": int(self.blocknet_timeout_spin.value()),
            "blocknet_poll_interval_ms": int(self.blocknet_poll_ms_spin.value()),
            "blocknet_poll_max_msgs": int(self.blocknet_max_msgs_spin.value()),

            "monero_rpc_url": self.monero_rpc_url_edit.text().strip(),
            "monero_rpc_token": self.monero_rpc_token_edit.text().strip(),
            "monero_rpc_prefix": self.monero_rpc_prefix_edit.text().strip(),
            "monero_rpc_force_scheme": self.monero_rpc_force_scheme_combo.currentText().strip(),
            "monero_rpc_verify_tls": bool(self.monero_rpc_verify_tls_check.isChecked()),
            "monero_rpc_timeout_s": int(self.monero_rpc_timeout_spin.value()),
            "monero_rpc_poll_interval_ms": int(self.monero_rpc_poll_ms_spin.value()),
            "monero_rpc_client_id": self.monero_rpc_client_id_edit.text().strip(),
            "monero_rpc_lease_size": int(self.monero_rpc_lease_size_spin.value()),
            "monero_rpc_require_leases": bool(self.monero_rpc_require_leases_check.isChecked()),
            "monero_rpc_feeder_mode": self.monero_rpc_feeder_mode_combo.currentText().strip(),
            "monero_rpc_feeder_poll_interval_ms": int(self.monero_rpc_feeder_poll_ms_spin.value()),

            "solo_wallet_address": self.solo_wallet_edit.text().strip(),
            "solo_daemon_rpc_url": self.solo_rpc_edit.text().strip(),
            "solo_zmq_pub_url": self.solo_zmq_edit.text().strip(),
            "solo_use_zmq": bool(self.solo_use_zmq_check.isChecked()),
            "solo_poll_fallback_s": float(self.solo_poll_spin.value()),
            "solo_reserve_size": int(self.solo_reserve_spin.value()),

            "kernel_path": self.kernel_edit.text().strip(),
            "opencl_loader": self.loader_edit.text().strip(),
            "build_options": self.build_opts_edit.text().strip(),
            "platform_index": int(self.platform_spin.value()),
            "device_index": int(self.device_spin.value()),
            "global_work_size": int(self.gws_spin.value()),
            "local_work_size": int(self.lws_spin.value()),
            "max_results": int(self.max_results_spin.value()),
            "nonce_offset": int(self.nonce_offset_spin.value()),
            "randomx_dll_path": self.verifier_dll_edit.text().strip(),
            "randomx_runtime_dll_path": self.randomx_runtime_dll_edit.text().strip(),
            "preload_randomx_runtime": bool(self.preload_randomx_runtime_check.isChecked()),

            "gpu_scan_mode": self.scan_mode_combo.currentText().strip(),
            "enable_cpu_verify": bool(self.enable_cpu_verify_check.isChecked()),
            "submit_unverified_shares": bool(self.submit_unverified_check.isChecked()),
            "enable_cpu_verify_batch": bool(self.enable_cpu_verify_batch_check.isChecked()),
            "cpu_verify_batch_size": int(self.cpu_verify_batch_size_spin.value()),
            "cpu_verify_batch_wait_ms": int(self.cpu_verify_batch_wait_ms_spin.value()),
            "cpu_verify_native_threads": int(self.cpu_verify_native_threads_spin.value()),
            "enable_cpu_hash_batch": bool(self.enable_cpu_hash_batch_check.isChecked()),
            "cpu_hash_batch_min_size": int(self.cpu_hash_batch_min_size_spin.value()),
            "cpu_hash_batch_threads": int(self.cpu_hash_batch_threads_spin.value()),
            "enable_cpu_tail_batch": bool(self.enable_cpu_tail_batch_check.isChecked()),
            "cpu_tail_batch_min_size": int(self.cpu_tail_batch_min_size_spin.value()),
            "cpu_tail_batch_threads": int(self.cpu_tail_batch_threads_spin.value()),

            "enable_bucket_tuning": bool(self.enable_bucket_tuning_check.isChecked()),
            "enable_job_tuning": bool(self.enable_job_tuning_check.isChecked()),
            "adaptive_queue_throttle": bool(self.adaptive_queue_throttle_check.isChecked()),
            "sort_candidates": bool(self.sort_candidates_check.isChecked()),

            "scan_chunk_size": int(self.scan_chunk_spin.value()),
            "hash_batch_size": int(self.hash_batch_size_spin.value()),
            "scan_candidate_target": int(self.scan_target_spin.value()),
            "max_scan_time_ms": int(self.max_scan_ms_spin.value()),
            "cpu_verify_limit": int(self.cpu_verify_limit_spin.value()),
            "verify_threads": int(self.verify_threads_spin.value()),
            "verify_queue_limit": int(self.verify_queue_limit_spin.value()),
            "submit_queue_limit": int(self.submit_queue_limit_spin.value()),
            "scan_pause_ms": int(self.scan_pause_ms_spin.value()),
            "stats_update_ms": int(self.stats_update_ms_spin.value()),

            "verify_queue_soft_pct": float(self.verify_queue_soft_pct_spin.value()),
            "submit_queue_soft_pct": float(self.submit_queue_soft_pct_spin.value()),
            "min_candidate_target": int(self.min_candidate_target_spin.value()),
            "min_dynamic_work_pct": float(self.min_dynamic_work_pct_spin.value()),
            "job_age_soft_ms": int(self.job_age_soft_ms_spin.value()),
            "job_age_hard_ms": int(self.job_age_hard_ms_spin.value()),

            "ema_alpha": float(self.ema_alpha_spin.value()),
            "tune_verified_reward": float(self.tune_verified_spin.value()),
            "tune_accepted_reward": float(self.tune_accepted_spin.value()),
            "tune_cpu_reject_penalty": float(self.tune_cpu_reject_spin.value()),
            "tune_pool_reject_penalty": float(self.tune_pool_reject_spin.value()),
            "tune_pressure_penalty": float(self.tune_pressure_spin.value()),
            "tune_work_bonus_scale": float(self.tune_work_bonus_spin.value()),
            "tune_tail_bins": int(self.tune_tail_bins_spin.value()),
            "tune_confidence_div": float(self.tune_confidence_div_spin.value()),
            "tune_quality_reward": float(self.tune_quality_reward_spin.value()),
            "tune_stale_penalty": float(self.tune_stale_penalty_spin.value()),
            "tune_duplicate_penalty": float(self.tune_duplicate_penalty_spin.value()),
            "tune_invalid_penalty": float(self.tune_invalid_penalty_spin.value()),
            "tune_backend_error_penalty": float(self.tune_backend_error_penalty_spin.value()),

            "geometry": {
                "width": int(self.width()),
                "height": int(self.height()),
            },
            "splitter_sizes": self.main_split.sizes(),
            "selected_tab": int(self.tabs.currentIndex()),
        }

    def _apply_config_dict(self, data: dict[str, Any]) -> None:
        backend = str(data.get("mining_backend", "")).strip().lower()
        if backend in {"monero_rpc", "monero-rpc"}:
            backend = "monerorpc"
        if backend not in {"stratum", "blocknet", "solo", "monerorpc"}:
            if bool(data.get("use_blocknet", False)):
                backend = "blocknet"
            else:
                backend = "stratum"
        self.backend_combo.setCurrentText(backend)

        self.host_edit.setText(str(data.get("host", self.host_edit.text())))
        self.port_spin.setValue(int(data.get("port", self.port_spin.value())))
        self.login_edit.setText(str(data.get("login", self.login_edit.text())))
        self.pass_edit.setText(str(data.get("password", self.pass_edit.text())))
        self.agent_edit.setText(str(data.get("agent", self.agent_edit.text())))
        self.tls_check.setChecked(bool(data.get("use_tls", self.tls_check.isChecked())))

        self.blocknet_relay_edit.setText(str(data.get("blocknet_api_relay", self.blocknet_relay_edit.text())))
        self.blocknet_token_edit.setText(str(data.get("blocknet_api_token", self.blocknet_token_edit.text())))
        self.blocknet_prefix_edit.setText(str(data.get("blocknet_api_prefix", self.blocknet_prefix_edit.text())))
        self._set_combo_from_optional(self.blocknet_force_scheme_combo, data.get("blocknet_force_scheme", ""))
        self.blocknet_verify_tls_check.setChecked(
            bool(data.get("blocknet_verify_tls", self.blocknet_verify_tls_check.isChecked()))
        )
        self.blocknet_timeout_spin.setValue(int(data.get("blocknet_timeout_s", self.blocknet_timeout_spin.value())))
        self.blocknet_poll_ms_spin.setValue(
            int(data.get("blocknet_poll_interval_ms", self.blocknet_poll_ms_spin.value()))
        )
        self.blocknet_max_msgs_spin.setValue(
            int(data.get("blocknet_poll_max_msgs", self.blocknet_max_msgs_spin.value()))
        )

        self.monero_rpc_url_edit.setText(str(data.get("monero_rpc_url", self.monero_rpc_url_edit.text())))
        self.monero_rpc_token_edit.setText(str(data.get("monero_rpc_token", self.monero_rpc_token_edit.text())))
        self.monero_rpc_prefix_edit.setText(str(data.get("monero_rpc_prefix", self.monero_rpc_prefix_edit.text())))
        self._set_combo_from_optional(self.monero_rpc_force_scheme_combo, data.get("monero_rpc_force_scheme", ""))
        self.monero_rpc_verify_tls_check.setChecked(
            bool(data.get("monero_rpc_verify_tls", self.monero_rpc_verify_tls_check.isChecked()))
        )
        self.monero_rpc_timeout_spin.setValue(
            int(data.get("monero_rpc_timeout_s", self.monero_rpc_timeout_spin.value()))
        )
        self.monero_rpc_poll_ms_spin.setValue(
            int(data.get("monero_rpc_poll_interval_ms", self.monero_rpc_poll_ms_spin.value()))
        )
        self.monero_rpc_client_id_edit.setText(
            str(data.get("monero_rpc_client_id", self.monero_rpc_client_id_edit.text()))
        )
        self.monero_rpc_lease_size_spin.setValue(
            int(data.get("monero_rpc_lease_size", self.monero_rpc_lease_size_spin.value()))
        )
        self.monero_rpc_require_leases_check.setChecked(
            bool(data.get("monero_rpc_require_leases", self.monero_rpc_require_leases_check.isChecked()))
        )
        self._set_combo_from_optional(
            self.monero_rpc_feeder_mode_combo,
            data.get("monero_rpc_feeder_mode", "none"),
        )
        self.monero_rpc_feeder_poll_ms_spin.setValue(
            int(data.get("monero_rpc_feeder_poll_interval_ms", self.monero_rpc_feeder_poll_ms_spin.value()))
        )

        self.solo_wallet_edit.setText(str(data.get("solo_wallet_address", self.solo_wallet_edit.text())))
        self.solo_rpc_edit.setText(str(data.get("solo_daemon_rpc_url", self.solo_rpc_edit.text())))
        self.solo_zmq_edit.setText(str(data.get("solo_zmq_pub_url", self.solo_zmq_edit.text())))
        self.solo_use_zmq_check.setChecked(bool(data.get("solo_use_zmq", self.solo_use_zmq_check.isChecked())))
        self.solo_poll_spin.setValue(float(data.get("solo_poll_fallback_s", self.solo_poll_spin.value())))
        self.solo_reserve_spin.setValue(int(data.get("solo_reserve_size", self.solo_reserve_spin.value())))

        self.kernel_edit.setText(str(data.get("kernel_path", self.kernel_edit.text())))
        self.loader_edit.setText(str(data.get("opencl_loader", self.loader_edit.text())))
        self.build_opts_edit.setText(str(data.get("build_options", self.build_opts_edit.text())))
        self.verifier_dll_edit.setText(str(data.get("randomx_dll_path", self.verifier_dll_edit.text())))
        self.randomx_runtime_dll_edit.setText(
            str(data.get("randomx_runtime_dll_path", self.randomx_runtime_dll_edit.text()))
        )
        self.preload_randomx_runtime_check.setChecked(
            bool(data.get("preload_randomx_runtime", self.preload_randomx_runtime_check.isChecked()))
        )

        scan_mode = str(data.get("gpu_scan_mode", self.scan_mode_combo.currentText())).strip().lower()
        if scan_mode not in {"chunk", "hash_batch"}:
            scan_mode = "chunk"
        self.scan_mode_combo.setCurrentText(scan_mode)

        self.enable_cpu_verify_check.setChecked(
            bool(data.get("enable_cpu_verify", self.enable_cpu_verify_check.isChecked()))
        )
        self.submit_unverified_check.setChecked(
            bool(data.get("submit_unverified_shares", self.submit_unverified_check.isChecked()))
        )
        self.enable_cpu_verify_batch_check.setChecked(
            bool(data.get("enable_cpu_verify_batch", self.enable_cpu_verify_batch_check.isChecked()))
        )
        self.enable_cpu_hash_batch_check.setChecked(
            bool(data.get("enable_cpu_hash_batch", self.enable_cpu_hash_batch_check.isChecked()))
        )
        self.enable_cpu_tail_batch_check.setChecked(
            bool(data.get("enable_cpu_tail_batch", self.enable_cpu_tail_batch_check.isChecked()))
        )
        self.enable_bucket_tuning_check.setChecked(
            bool(data.get("enable_bucket_tuning", self.enable_bucket_tuning_check.isChecked()))
        )
        self.enable_job_tuning_check.setChecked(
            bool(data.get("enable_job_tuning", self.enable_job_tuning_check.isChecked()))
        )
        self.adaptive_queue_throttle_check.setChecked(
            bool(data.get("adaptive_queue_throttle", self.adaptive_queue_throttle_check.isChecked()))
        )
        self.sort_candidates_check.setChecked(
            bool(data.get("sort_candidates", self.sort_candidates_check.isChecked()))
        )

        self.platform_spin.setValue(int(data.get("platform_index", self.platform_spin.value())))
        self.device_spin.setValue(int(data.get("device_index", self.device_spin.value())))
        self.gws_spin.setValue(int(data.get("global_work_size", self.gws_spin.value())))
        self.lws_spin.setValue(int(data.get("local_work_size", self.lws_spin.value()) or 0))
        self.max_results_spin.setValue(int(data.get("max_results", self.max_results_spin.value())))
        self.nonce_offset_spin.setValue(int(data.get("nonce_offset", self.nonce_offset_spin.value())))

        self.scan_chunk_spin.setValue(int(data.get("scan_chunk_size", self.scan_chunk_spin.value())))
        self.hash_batch_size_spin.setValue(int(data.get("hash_batch_size", self.hash_batch_size_spin.value())))
        self.scan_target_spin.setValue(int(data.get("scan_candidate_target", self.scan_target_spin.value())))
        self.max_scan_ms_spin.setValue(int(data.get("max_scan_time_ms", self.max_scan_ms_spin.value())))
        self.cpu_verify_limit_spin.setValue(int(data.get("cpu_verify_limit", self.cpu_verify_limit_spin.value())))
        self.verify_threads_spin.setValue(int(data.get("verify_threads", self.verify_threads_spin.value())))
        self.cpu_verify_batch_size_spin.setValue(
            int(data.get("cpu_verify_batch_size", self.cpu_verify_batch_size_spin.value()))
        )
        self.cpu_verify_batch_wait_ms_spin.setValue(
            int(data.get("cpu_verify_batch_wait_ms", self.cpu_verify_batch_wait_ms_spin.value()))
        )
        self.cpu_verify_native_threads_spin.setValue(
            int(data.get("cpu_verify_native_threads", self.cpu_verify_native_threads_spin.value()))
        )
        self.cpu_hash_batch_min_size_spin.setValue(
            int(data.get("cpu_hash_batch_min_size", self.cpu_hash_batch_min_size_spin.value()))
        )
        self.cpu_hash_batch_threads_spin.setValue(
            int(data.get("cpu_hash_batch_threads", self.cpu_hash_batch_threads_spin.value()))
        )
        self.cpu_tail_batch_min_size_spin.setValue(
            int(data.get("cpu_tail_batch_min_size", self.cpu_tail_batch_min_size_spin.value()))
        )
        self.cpu_tail_batch_threads_spin.setValue(
            int(data.get("cpu_tail_batch_threads", self.cpu_tail_batch_threads_spin.value()))
        )
        self.verify_queue_limit_spin.setValue(int(data.get("verify_queue_limit", self.verify_queue_limit_spin.value())))
        self.submit_queue_limit_spin.setValue(int(data.get("submit_queue_limit", self.submit_queue_limit_spin.value())))
        self.scan_pause_ms_spin.setValue(int(data.get("scan_pause_ms", self.scan_pause_ms_spin.value())))
        self.stats_update_ms_spin.setValue(int(data.get("stats_update_ms", self.stats_update_ms_spin.value())))

        self.verify_queue_soft_pct_spin.setValue(
            float(data.get("verify_queue_soft_pct", self.verify_queue_soft_pct_spin.value()))
        )
        self.submit_queue_soft_pct_spin.setValue(
            float(data.get("submit_queue_soft_pct", self.submit_queue_soft_pct_spin.value()))
        )
        self.min_candidate_target_spin.setValue(
            int(data.get("min_candidate_target", self.min_candidate_target_spin.value()))
        )
        self.min_dynamic_work_pct_spin.setValue(
            float(data.get("min_dynamic_work_pct", self.min_dynamic_work_pct_spin.value()))
        )
        self.job_age_soft_ms_spin.setValue(int(data.get("job_age_soft_ms", self.job_age_soft_ms_spin.value())))
        self.job_age_hard_ms_spin.setValue(int(data.get("job_age_hard_ms", self.job_age_hard_ms_spin.value())))

        self.ema_alpha_spin.setValue(float(data.get("ema_alpha", self.ema_alpha_spin.value())))
        self.tune_verified_spin.setValue(float(data.get("tune_verified_reward", self.tune_verified_spin.value())))
        self.tune_accepted_spin.setValue(float(data.get("tune_accepted_reward", self.tune_accepted_spin.value())))
        self.tune_cpu_reject_spin.setValue(
            float(data.get("tune_cpu_reject_penalty", self.tune_cpu_reject_spin.value()))
        )
        self.tune_pool_reject_spin.setValue(
            float(data.get("tune_pool_reject_penalty", self.tune_pool_reject_spin.value()))
        )
        self.tune_pressure_spin.setValue(float(data.get("tune_pressure_penalty", self.tune_pressure_spin.value())))
        self.tune_work_bonus_spin.setValue(
            float(data.get("tune_work_bonus_scale", self.tune_work_bonus_spin.value()))
        )
        self.tune_tail_bins_spin.setValue(int(data.get("tune_tail_bins", self.tune_tail_bins_spin.value())))
        self.tune_confidence_div_spin.setValue(
            float(data.get("tune_confidence_div", self.tune_confidence_div_spin.value()))
        )
        self.tune_quality_reward_spin.setValue(
            float(data.get("tune_quality_reward", self.tune_quality_reward_spin.value()))
        )
        self.tune_stale_penalty_spin.setValue(
            float(data.get("tune_stale_penalty", self.tune_stale_penalty_spin.value()))
        )
        self.tune_duplicate_penalty_spin.setValue(
            float(data.get("tune_duplicate_penalty", self.tune_duplicate_penalty_spin.value()))
        )
        self.tune_invalid_penalty_spin.setValue(
            float(data.get("tune_invalid_penalty", self.tune_invalid_penalty_spin.value()))
        )
        self.tune_backend_error_penalty_spin.setValue(
            float(data.get("tune_backend_error_penalty", self.tune_backend_error_penalty_spin.value()))
        )

        geom = data.get("geometry") or {}
        w = int(geom.get("width", self.width()))
        h = int(geom.get("height", self.height()))
        self.resize(max(1250, w), max(820, h))

        splitter_sizes = data.get("splitter_sizes")
        if isinstance(splitter_sizes, list) and len(splitter_sizes) == 2:
            self.main_split.setSizes([int(splitter_sizes[0]), int(splitter_sizes[1])])

        tab_index = int(data.get("selected_tab", 0))
        if 0 <= tab_index < self.tabs.count():
            self.tabs.setCurrentIndex(tab_index)

        self._sync_backend_controls()
        self._sync_scan_mode_controls()
        self._sync_verification_controls()
        self._sync_tuning_controls()
        self._sync_adaptive_controls()

    def save_config(self) -> None:
        try:
            CFG_PATH.write_text(json.dumps(self._config_to_dict(), indent=2), encoding="utf-8")
            self.statusBar().showMessage("Config saved", 1500)
        except Exception as exc:
            self.statusBar().showMessage(f"Config save failed: {exc}", 2500)

    def load_config(self) -> None:
        if not CFG_PATH.exists():
            return
        try:
            data = json.loads(CFG_PATH.read_text(encoding="utf-8"))
            self._apply_config_dict(data)
            self.statusBar().showMessage("Config loaded", 1500)
        except Exception as exc:
            self.statusBar().showMessage(f"Config load failed: {exc}", 2500)

    def start_mining(self) -> None:
        if self.worker_thread is not None or self.worker is not None:
            return

        try:
            cfg = self.build_config()
        except Exception as exc:
            QMessageBox.critical(self, "Config error", str(exc))
            return

        self.save_config()
        self.clear_logs()
        self.append_log("[gui] starting miner...")
        self.append_log(
            f"[gui] backend={cfg.mining_backend_name()} "
            f"mode={cfg.gpu_scan_mode} "
            f"verify={'on' if cfg.enable_cpu_verify else 'off'} "
            f"verify_batch={'on' if cfg.enable_cpu_verify_batch else 'off'} "
            f"verify_batch_size={cfg.cpu_verify_batch_size} "
            f"verify_batch_wait_ms={cfg.cpu_verify_batch_wait_ms} "
            f"verify_native_threads={cfg.cpu_verify_native_threads} "
            f"hash_teacher={'on' if cfg.enable_cpu_hash_batch else 'off'} "
            f"hash_teacher_min={cfg.cpu_hash_batch_min_size} "
            f"hash_teacher_threads={cfg.cpu_hash_batch_threads} "
            f"tail_batch={'on' if cfg.enable_cpu_tail_batch else 'off'} "
            f"tail_batch_min={cfg.cpu_tail_batch_min_size} "
            f"tail_batch_threads={cfg.cpu_tail_batch_threads} "
            f"job_tuning={'on' if cfg.enable_job_tuning else 'off'} "
            f"seed_tuning={'on' if cfg.enable_bucket_tuning else 'off'} "
            f"adaptive={'on' if cfg.adaptive_queue_throttle else 'off'} "
            f"tail_bins={cfg.normalized_tail_bins()} "
            f"split_tuning=rank+threshold+credit+confidence "
            f"submit_unverified={'on' if cfg.submit_unverified_shares else 'off'} "
            f"sort={'on' if cfg.sort_candidates else 'off'} "
            f"device=P{cfg.platform_index}/D{cfg.device_index}"
        )
        self.append_log(
            f"[gui] verifier_dll={cfg.randomx_dll_path or '-'} "
            f"randomx_runtime={cfg.randomx_runtime_dll_path or '-'} "
            f"preload_randomx_runtime={'on' if cfg.preload_randomx_runtime else 'off'}"
        )

        if cfg.use_blocknet:
            self.append_log(
                f"[gui] blocknet relay={cfg.blocknet_api_relay or '-'} "
                f"prefix={cfg.blocknet_api_prefix or '/v1'} "
                f"verify_tls={'on' if cfg.blocknet_verify_tls else 'off'} "
                f"poll_ms={cfg.blocknet_poll_interval_ms}"
            )

        if cfg.use_solo:
            self.append_log(
                f"[gui] solo rpc={cfg.solo_daemon_rpc_url} "
                f"zmq={'on' if cfg.solo_use_zmq else 'off'} "
                f"wallet={cfg.solo_wallet_address[:16]}..."
            )

        if cfg.use_monero_rpc:
            self.append_log(
                f"[gui] monerorpc url={cfg.monero_rpc_url or '-'} "
                f"prefix={cfg.monero_rpc_prefix or '/v1'} "
                f"verify_tls={'on' if cfg.monero_rpc_verify_tls else 'off'} "
                f"poll_ms={cfg.monero_rpc_poll_interval_ms} "
                f"lease_size={cfg.monero_rpc_lease_size} "
                f"require_leases={'on' if cfg.monero_rpc_require_leases else 'off'} "
                f"client_id={cfg.monero_rpc_client_id or '-'} "
                f"feeder={cfg.normalized_monero_rpc_feeder_mode()} "
                f"feeder_poll_ms={cfg.monero_rpc_feeder_poll_interval_ms}"
            )

        self.worker_thread = QThread(self)
        self.worker = MinerWorker(cfg)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.stats.connect(self.update_stats)
        self.worker.status.connect(self.set_status)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._on_worker_stopped)

        self.started_at = time.time()
        self._uptime_timer.start()
        self.set_status("starting")
        self.worker_thread.start()

    def stop_mining(self) -> None:
        if self.worker is None:
            return

        self.append_log("[gui] stopping miner...")
        self.set_status("stopping")

        try:
            self.worker.stop()
        except Exception as exc:
            self.append_log(f"[gui] stop warning: {exc}")

    def _on_worker_stopped(self) -> None:
        self.append_log("[gui] miner stopped")
        self.started_at = 0.0
        self._uptime_timer.stop()
        self._update_uptime()
        self.set_status("stopped")
        self.worker = None
        self.worker_thread = None

    def closeEvent(self, event) -> None:
        self.save_config()

        if self.worker is not None:
            try:
                self.worker.stop()
            except Exception:
                pass

        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(3000)

        super().closeEvent(event)


def run() -> None:
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()