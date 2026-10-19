"""Arranque del servidor con detección de IPs de LAN (funciona sin salida a Internet)."""
import ipaddress
import socket

import psutil
import uvicorn

HOST = "0.0.0.0"
PORT = 8000


def get_lan_ips(include_loopback: bool = False) -> list[str]:
    """Devuelve las IPs IPv4 privadas de las interfaces del equipo."""
    lan_ips = set()
    for iface_name, addrs in psutil.net_if_addrs().items():
        if iface_name.lower().startswith(("lo", "docker", "veth", "virbr")):
            continue
        for addr in addrs:
            if addr.family != socket.AF_INET:
                continue
            ip = addr.address
            if ip.startswith("127."):
                if include_loopback:
                    lan_ips.add(ip)
                continue
            try:
                if ipaddress.ip_address(ip).is_private:
                    lan_ips.add(ip)
            except ValueError:
                continue
    return sorted(lan_ips)


if __name__ == "__main__":
    lan_ips = get_lan_ips()

    print("=" * 60)
    print("🚀 Servidor de traducción iniciado")
    print(f"   📍 Localhost:  http://127.0.0.1:{PORT}")

    if lan_ips:
        for ip in lan_ips:
            print(f"   🌐 Red local:  http://{ip}:{PORT}")
        print(f"   📖 Docs:       http://{lan_ips[0]}:{PORT}/docs")
    else:
        print("   ⚠️  No se detectó ninguna IP de LAN.")

    print("=" * 60)
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)