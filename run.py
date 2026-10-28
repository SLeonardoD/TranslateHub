"""Arranque del servidor con detección de IPs de LAN (funciona sin salida a Internet)."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse  # noqa: E402
import ipaddress  # noqa: E402
import re  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402

import psutil  # noqa: E402
import uvicorn  # noqa: E402

try:
    import app.config as config  # noqa: E402
    from app.config import ENV_FILE_LOADED, Settings, load_catalog  # noqa: E402
    from app.core.exceptions import CatalogError  # noqa: E402
    from app.logging_setup import build_log_config  # noqa: E402
except ValueError as exc:  # variable de entorno TRANSLATOR_* inválida
    sys.exit(f"❌ Configuración inválida: {exc}")

# Adaptadores que existen en el equipo pero NO son alcanzables desde otros equipos de la LAN.
_VIRTUAL_NAME_PARTS = (
    "docker",
    "veth",
    "virbr",
    "vmnet",
    "vboxnet",
    "virtualbox",
    "vmware",
    "hyper-v",
    "wsl",
    "loopback",
    "pseudo",
    "zerotier",
    "tailscale",
    "podman",
    "cni",
    "flannel",
    "calico",
)
_VIRTUAL_NAME_PREFIXES = ("br-", "bridge", "tun", "tap", "wg", "utun", "zt")
_LOOPBACK_NAME = re.compile(r"lo\d*")


def is_virtual_interface(name: str) -> bool:
    lowered = name.lower()
    return (
        bool(_LOOPBACK_NAME.fullmatch(lowered))
        or lowered.startswith(_VIRTUAL_NAME_PREFIXES)
        or any(part in lowered for part in _VIRTUAL_NAME_PARTS)
    )


def get_primary_ip() -> str | None:
    """IP de la interfaz que el sistema usaría para salir a la red (no envía ningún paquete).

    Funciona sin Internet siempre que haya una ruta por defecto (el router de la LAN).
    """
    for probe in ("10.254.254.254", "192.168.254.254", "172.16.254.254"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect((probe, 1))
                ip = sock.getsockname()[0]
        except OSError:
            continue
        if ip and not ip.startswith(("0.", "127.")):
            return ip
    return None


def get_lan_addresses(
    addrs: dict | None = None,
    stats: dict | None = None,
    primary: str | None = None,
) -> list[tuple[str, str]]:
    """Devuelve [(ip, interfaz)] de las IPs IPv4 de LAN, con la principal primero.

    Descarta interfaces apagadas, virtuales (Docker, WSL, VirtualBox, VPN...), loopback y
    direcciones APIPA (169.254.x.x). Los argumentos permiten probar la función sin psutil.
    """
    addrs = psutil.net_if_addrs() if addrs is None else addrs
    stats = psutil.net_if_stats() if stats is None else stats
    primary = get_primary_ip() if primary is None else primary

    found: list[tuple[str, str]] = []
    for iface, entries in addrs.items():
        status = stats.get(iface)
        if (status is not None and not status.isup) or is_virtual_interface(iface):
            continue
        for entry in entries:
            if entry.family != socket.AF_INET:
                continue
            try:
                ip = ipaddress.ip_address(entry.address)
            except ValueError:
                continue
            if ip.is_private and not ip.is_loopback and not ip.is_link_local:
                found.append((str(ip), iface))
    return sorted(found, key=lambda t: (t[0] != primary, t[1], ipaddress.ip_address(t[0])))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Servidor de traducción local. Por defecto solo escucha en este equipo."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--local",
        action="store_true",
        help="solo este equipo (127.0.0.1). Es el modo por defecto.",
    )
    mode.add_argument(
        "--lan",
        action="store_true",
        help="visible para otros equipos de la red local (0.0.0.0).",
    )
    return parser.parse_args(argv)


def apply_mode_flag(args: argparse.Namespace) -> None:
    """El flag manda sobre TRANSLATOR_HOST (entorno / .env). Debe ejecutarse antes de que
    uvicorn importe ``app.main``, que es quien lee los ajustes."""
    mode = "lan" if args.lan else "local" if args.local else None
    if mode is not None:
        os.environ["TRANSLATOR_HOST"] = mode
        config.settings = Settings.from_env()


def access_lines(host: str, port: int, lan: list[tuple[str, str]] | None = None) -> list[str]:
    """Líneas del banner con las URLs de acceso según cómo escuche el servidor."""
    lowered = host.lower()
    if lowered in ("localhost", "::1") or host.startswith("127."):
        return [
            f"   📍 Localhost:  http://127.0.0.1:{port}",
            "   🔒 Modo local: solo este equipo. Para abrirlo a la red: python run.py --lan",
        ]
    if host in ("0.0.0.0", "::"):
        lines = [f"   📍 Localhost:  http://127.0.0.1:{port}"]
        for i, (ip, iface) in enumerate(lan or []):
            label = "Red local:" if i == 0 else "Otra red: "
            lines.append(f"   🌐 {label} http://{ip}:{port}   ({iface})")
        if lan:
            lines.append(f"   📖 Docs:       http://{lan[0][0]}:{port}/docs")
        else:
            lines.append("   ⚠️  No se detectó ninguna IP de LAN.")
        lines.append("   🌐 Modo LAN: visible para toda la red local (sin API key).")
        return lines
    # IP concreta (p. ej. la de la tarjeta de red de la LAN): solo se llega por esa IP.
    return [
        f"   🌐 Red local:  http://{host}:{port}",
        f"   📖 Docs:       http://{host}:{port}/docs",
        f"   🌐 Modo LAN: escuchando solo en {host}.",
    ]


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")  # consolas Windows sin UTF-8

    apply_mode_flag(parse_args(argv))
    settings = config.settings

    try:
        catalog = load_catalog()  # falla aquí, con un mensaje claro, si model_info.json es inválido
    except CatalogError as exc:
        sys.exit(f"❌ {exc}")

    lan = get_lan_addresses() if settings.host in ("0.0.0.0", "::") else None
    print("=" * 60)
    print("🚀 Servidor de traducción iniciado")
    for line in access_lines(settings.host, settings.port, lan):
        print(line)
    print(f"   ⚙️  Config: {ENV_FILE_LOADED or 'sin .env (valores por defecto / entorno)'}")
    if settings.max_batch_items == 0:
        print("   ♾️  Lotes sin límite de textos (TRANSLATOR_MAX_BATCH_ITEMS=0).")
    print(f"   📦 Modelos en el catálogo: {len(catalog)}")
    if settings.api_key:
        print("   🔑 API key activada (cabecera X-API-Key).")
    log_file = settings.resolve_log_file()
    print(f"   📝 Log: {log_file if log_file else 'desactivado'}")
    print("=" * 60)

    # Consola resumida + archivo con todo (se escribe en vivo). El access log de uvicorn se
    # sustituye por el de la aplicación (translator.access), que añade quién llama y para qué.
    # workers=1: la cola y el modelo viven en un solo proceso; con varios se romperían.
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        workers=1,
        access_log=False,
        log_config=build_log_config(
            settings.log_level,
            log_file,
            settings.log_max_mb,
            settings.log_backups,
            settings.console_log,
        ),
    )


if __name__ == "__main__":
    main()
