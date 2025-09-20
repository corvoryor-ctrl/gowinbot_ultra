# python_machineid/__init__.py
# Minimal drop-in replacement for macOS (works in CI/build too)
import hashlib, os, platform, subprocess, uuid

def _mac_platform_uuid():
    """Return IOPlatformUUID on macOS, or None."""
    try:
        out = subprocess.check_output(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            text=True
        )
        for line in out.splitlines():
            if "IOPlatformUUID" in line:
                # line like: "IOPlatformUUID" = "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"
                parts = line.split('"')
                if len(parts) >= 4:
                    return parts[3]
    except Exception:
        pass
    return None

def get_machine_id():
    """
    Stable machine id for licensing/analytics.
    - Use env MACHINE_ID if provided
    - Else macOS IOPlatformUUID
    - Else fallback to hostname / MAC address hash
    Returns a hex sha256 string.
    """
    mid = os.environ.get("MACHINE_ID")
    if mid:
        return mid

    u = _mac_platform_uuid()
    if not u:
        # Fallback: hostname or mac address (non-PII hash)
        u = platform.node() or hex(uuid.getnode())

    return hashlib.sha256(str(u).encode("utf-8")).hexdigest()

# Backward-compat aliases some libs expect
machine_id = get_machine_id
id = get_machine_id
