import socket
from collections import namedtuple

import pytest

import run

Addr = namedtuple("Addr", "family address")
Stat = namedtuple("Stat", "isup")
INET = socket.AF_INET


@pytest.mark.parametrize(
    "name",
    [
        "lo",
        "lo0",
        "Loopback Pseudo-Interface 1",
        "docker0",
        "br-1a2b3c",
        "veth12ab",
        "virbr0",
        "vEthernet (WSL)",
        "vEthernet (Default Switch)",
        "VirtualBox Host-Only Network",
        "VMware Network Adapter VMnet8",
        "tailscale0",
        "tun0",
        "utun3",
        "wg0",
    ],
)
def test_virtual_interfaces_are_detected(name):
    assert run.is_virtual_interface(name)


@pytest.mark.parametrize(
    "name", ["eth0", "enp3s0", "wlan0", "Wi-Fi", "Ethernet", "Local Area Connection", "en0"]
)
def test_real_interfaces_are_kept(name):
    """'Local Area Connection' empieza por 'lo' y antes se descartaba por error."""
    assert not run.is_virtual_interface(name)


def test_lan_addresses_filtering_and_order():
    addrs = {
        "lo": [Addr(INET, "127.0.0.1")],
        "Local Area Connection": [Addr(INET, "192.168.1.42"), Addr(socket.AF_INET6, "fe80::1")],
        "vEthernet (WSL)": [Addr(INET, "172.20.0.1")],
        "Ethernet 2": [Addr(INET, "169.254.10.3")],  # APIPA
        "Wi-Fi": [Addr(INET, "10.0.0.7")],
        "Wi-Fi apagado": [Addr(INET, "10.9.9.9")],
        "Public": [Addr(INET, "8.8.4.4")],  # no privada
        "Sin estado": [Addr(INET, "192.168.50.5")],
    }
    stats = {name: Stat(True) for name in addrs}
    stats["Wi-Fi apagado"] = Stat(False)
    del stats["Sin estado"]  # psutil a veces no devuelve estado: se asume activa

    result = run.get_lan_addresses(addrs, stats, primary="10.0.0.7")
    assert result == [
        ("10.0.0.7", "Wi-Fi"),  # la principal va primero
        ("192.168.1.42", "Local Area Connection"),
        ("192.168.50.5", "Sin estado"),
    ]


def test_lan_addresses_without_primary_still_works():
    addrs = {"eth0": [Addr(INET, "192.168.0.10")]}
    assert run.get_lan_addresses(addrs, {"eth0": Stat(True)}, primary="") == [
        ("192.168.0.10", "eth0")
    ]


def test_primary_ip_ignores_loopback(monkeypatch):
    class FakeSocket:
        def __init__(self, *a): ...
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def connect(self, addr): ...
        def getsockname(self):
            return ("127.0.0.1", 0)

    monkeypatch.setattr(run.socket, "socket", FakeSocket)
    assert run.get_primary_ip() is None


def test_primary_ip_handles_unreachable_network(monkeypatch):
    class FailingSocket:
        def __init__(self, *a): ...
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def connect(self, addr):
            raise OSError("Network is unreachable")

    monkeypatch.setattr(run.socket, "socket", FailingSocket)
    assert run.get_primary_ip() is None
