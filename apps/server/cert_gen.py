"""Self-signed cert generation with proper SAN for LAN access.

Why this exists
---------------
Browsers (especially Chrome on Android) refuse to expose getUserMedia
(camera / microphone) on pages whose TLS cert is invalid. The previously
shipped cert had `CN=minicpmo45-dev` and *no* Subject Alternative Names,
which means:

  * Chrome 58+ rejects it as "common name only" (NET::ERR_CERT_COMMON_NAME_INVALID)
  * Even after the user clicks through the warning, mediaDevices may not
    be exposed because the page isn't a "true" secure context

The fix is to issue a cert that lists every name the user is likely to
type into the address bar:

  * localhost
  * 127.0.0.1
  * the host's current LAN IPv4 address (e.g. 192.168.1.42)

We re-issue the cert whenever the LAN IP changes, so phones keep working
when the host moves between Wi-Fi networks.

Notes
-----
* This is still a self-signed cert; the user must accept the browser
  warning the first time. After that, getUserMedia works.
* For zero-warning UX we'd need a real CA (mkcert, ACME, etc.). That's
  way too much friction for a desktop app.
"""
from __future__ import annotations

import datetime
import ipaddress
import logging
import os
import socket
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger("comni.cert_gen")

CERT_VALIDITY_DAYS = 825  # Apple's max; Chrome accepts up to this
CERT_RSA_BITS = 2048

# Filenames inside the certs dir.
_CERT_NAME = "cert.pem"
_KEY_NAME = "key.pem"


# ── LAN IP discovery ─────────────────────────────────────────────────────

def _is_real_lan_ipv4(ip: str) -> bool:
    """True if `ip` is a "useful for the user's phone" address.

    Filters out:
      * loopback (127/8)
      * link-local APIPA (169.254/16)
      * IETF benchmark / fake-IP ranges that Clash/V2Ray love (198.18/15)
      * multicast / reserved
      * non-private addresses (we don't want to bake a public IP into a
        self-signed cert by accident)

    Keeps RFC1918 (10/8, 172.16/12, 192.168/16) and CGNAT (100.64/10)
    which is what real Wi-Fi networks use.
    """
    try:
        ipa = ipaddress.IPv4Address(ip)
    except (ValueError, ipaddress.AddressValueError):
        return False
    if ipa.is_loopback or ipa.is_link_local or ipa.is_multicast:
        return False
    if ipa.is_reserved or ipa.is_unspecified:
        return False
    if not (ipa.is_private):
        return False
    # 198.18.0.0/15 is technically "is_private==False" per stdlib (it's a
    # benchmark range, not RFC1918). Belt-and-suspenders explicit reject:
    if ipaddress.IPv4Network("198.18.0.0/15").supernet_of(
            ipaddress.IPv4Network(f"{ip}/32")):
        return False
    return True


def get_lan_ipv4_addresses() -> List[str]:
    """Return non-loopback IPv4 addresses likely useful for LAN access.

    Order: the address used to reach the public internet first (most
    likely the Wi-Fi NIC), then the rest. Virtual / fake-IP interfaces
    are filtered out — see `_is_real_lan_ipv4`.
    """
    addrs: list[str] = []

    # Open a UDP socket to a public IP; the kernel picks the interface
    # that would route to it without actually sending anything.
    primary: Optional[str] = None
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 53))
            primary = s.getsockname()[0]
    except OSError:
        primary = None

    try:
        hostname = socket.gethostname()
        infos = socket.getaddrinfo(hostname, None, socket.AF_INET)
        for info in infos:
            ip = info[4][0]
            if ip and ip not in addrs and _is_real_lan_ipv4(ip):
                addrs.append(ip)
    except OSError:
        pass

    if primary and _is_real_lan_ipv4(primary):
        if primary in addrs:
            addrs.remove(primary)
        addrs.insert(0, primary)

    return addrs


# ── Cert inspection ──────────────────────────────────────────────────────

def cert_san_entries(cert_path: Path) -> Tuple[List[str], List[str]]:
    """Return (dns_names, ip_addresses) listed in cert's SAN, ([], []) if
    the cert is missing/unreadable/has no SAN extension.
    """
    if not cert_path.exists():
        return ([], [])
    try:
        from cryptography import x509
        from cryptography.hazmat.backends import default_backend
        with open(cert_path, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read(), default_backend())
        try:
            ext = cert.extensions.get_extension_for_oid(
                x509.ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
            san = ext.value
        except x509.ExtensionNotFound:
            return ([], [])
        # cryptography ≥42 returns unwrapped values (str / IPv4Address);
        # earlier versions returned wrapped objects with `.value`. Handle both.
        def _unwrap(v):
            return v.value if hasattr(v, "value") else v
        dns = [_unwrap(n) for n in san.get_values_for_type(x509.DNSName)]
        ips = [str(_unwrap(n)) for n in san.get_values_for_type(x509.IPAddress)]
        return (dns, ips)
    except Exception as e:
        logger.warning("cannot read cert %s: %s", cert_path, e)
        return ([], [])


def cert_is_expired_or_soon(cert_path: Path, soon_days: int = 30) -> bool:
    """True if the cert is missing, expired, or expires within `soon_days`."""
    if not cert_path.exists():
        return True
    try:
        from cryptography import x509
        from cryptography.hazmat.backends import default_backend
        with open(cert_path, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read(), default_backend())
        # Use timezone-aware UTC; cryptography 42+ deprecates naive variants.
        try:
            not_after = cert.not_valid_after_utc
            now = datetime.datetime.now(datetime.timezone.utc)
        except AttributeError:
            not_after = cert.not_valid_after
            now = datetime.datetime.utcnow()
        return (not_after - now).days <= soon_days
    except Exception:
        return True


# ── Cert generation ──────────────────────────────────────────────────────

def _build_san(dns_names: Iterable[str], ip_addrs: Iterable[str]):
    from cryptography import x509
    entries: list = []
    for n in dns_names:
        if n:
            entries.append(x509.DNSName(n))
    for ip in ip_addrs:
        if not ip:
            continue
        try:
            entries.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            logger.warning("skipping invalid IP in SAN: %s", ip)
    return x509.SubjectAlternativeName(entries)


def generate_self_signed(
    cert_path: Path,
    key_path: Path,
    dns_names: Iterable[str],
    ip_addrs: Iterable[str],
    cn: str = "Comni Local",
) -> None:
    """Write a fresh RSA self-signed cert + key with the given SAN."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=CERT_RSA_BITS)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, cn),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Comni"),
    ])

    now = datetime.datetime.now(datetime.timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=CERT_VALIDITY_DAYS))
        .add_extension(_build_san(dns_names, ip_addrs), critical=False)
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False)
    )
    cert = builder.sign(private_key=key, algorithm=hashes.SHA256())

    cert_path.parent.mkdir(parents=True, exist_ok=True)

    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )

    # Atomic write so a crash mid-generation doesn't leave a half-written
    # cert that breaks the next launch.
    tmp_cert = cert_path.with_suffix(cert_path.suffix + ".tmp")
    tmp_key = key_path.with_suffix(key_path.suffix + ".tmp")
    tmp_cert.write_bytes(cert_pem)
    tmp_key.write_bytes(key_pem)
    tmp_cert.replace(cert_path)
    tmp_key.replace(key_path)

    try:
        os.chmod(key_path, 0o600)  # no-op on Windows but cheap
    except OSError:
        pass

    logger.info(
        "generated cert %s (DNS=%s, IP=%s, valid %d days)",
        cert_path, list(dns_names), list(ip_addrs), CERT_VALIDITY_DAYS,
    )


# ── Public entry point ───────────────────────────────────────────────────

def ensure_cert_for_lan(certs_dir: Path) -> Tuple[Path, Path, List[str]]:
    """Make sure cert.pem / key.pem cover the current LAN IP(s).

    Regenerates the pair if any of the following are true:
      * cert or key file missing
      * cert expired or expires within 30 days
      * a current LAN IP is not present in the cert SAN

    Returns (cert_path, key_path, lan_ips_used).

    Raises ImportError if the `cryptography` package isn't installed —
    callers should handle that and fall back to whatever shipped on disk.
    """
    cert_path = certs_dir / _CERT_NAME
    key_path = certs_dir / _KEY_NAME

    lan_ips = get_lan_ipv4_addresses()
    desired_ips = ["127.0.0.1"] + lan_ips
    desired_dns = ["localhost"]

    needs_regen = False
    if not cert_path.exists() or not key_path.exists():
        needs_regen = True
        logger.info("cert/key missing, will generate")
    elif cert_is_expired_or_soon(cert_path):
        needs_regen = True
        logger.info("cert expired or near expiry, will regenerate")
    else:
        san_dns, san_ips = cert_san_entries(cert_path)
        missing_ips = [ip for ip in desired_ips if ip not in san_ips]
        missing_dns = [d for d in desired_dns if d not in san_dns]
        if missing_ips or missing_dns:
            needs_regen = True
            logger.info(
                "cert SAN does not cover current LAN: missing DNS=%s IP=%s "
                "(have DNS=%s IP=%s); regenerating",
                missing_dns, missing_ips, san_dns, san_ips,
            )

    if needs_regen:
        generate_self_signed(
            cert_path, key_path,
            dns_names=desired_dns,
            ip_addrs=desired_ips,
        )
    else:
        logger.info("cert already valid for %s; reusing", desired_ips)

    return (cert_path, key_path, lan_ips)
