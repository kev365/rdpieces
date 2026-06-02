"""Offline parsing of external host artifacts that constrain reconstruction.

Sources (client-side): ``.rdp`` connection files (resolution/monitor), the
Terminal Server Client registry MRU (hosts/timeline), and the RdpCoreTS
Operational ``.evtx`` (Event 168 = negotiated resolution, 169 = OS, 104 = tz).

The extraction logic is separated from binary I/O so it is unit-testable with
injected records; the real hive/evtx readers (python-registry / python-evtx) are
thin wrappers used by the CLI.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# Event XML comes from untrusted forensic .evtx; defusedxml blocks XXE/entity-expansion.
import defusedxml.ElementTree as ET
from xml.etree.ElementTree import ParseError


@dataclass(slots=True)
class RdpFileInfo:
    width: int | None = None
    height: int | None = None
    fullscreen: bool | None = None
    multimon: bool | None = None
    full_address: str | None = None


def parse_rdp_file(path: str) -> RdpFileInfo:
    """Parse a .rdp connection file's display + host settings."""
    fields: dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            # lines look like "key:type:value"
            parts = line.rstrip("\r\n").split(":", 2)
            if len(parts) == 3:
                key, _type, value = parts
                fields[key.strip().lower()] = value

    def as_int(key):
        try:
            return int(fields[key])
        except (KeyError, ValueError):
            return None

    info = RdpFileInfo(
        width=as_int("desktopwidth"),
        height=as_int("desktopheight"),
        full_address=fields.get("full address"),
    )
    mode = as_int("screen mode id")
    if mode is not None:
        info.fullscreen = mode == 2  # 1 = windowed, 2 = fullscreen
    multimon = as_int("use multimon")
    if multimon is not None:
        info.multimon = multimon == 1
    return info


def registry_mru_from_values(values: list[tuple[str, str]]) -> list[str]:
    """Given ``Terminal Server Client\\Default`` (name, data) value pairs, return the
    connected hosts ordered MRU0..MRUn (most-recent first)."""
    mru = []
    for name, data in values:
        m = re.fullmatch(r"MRU(\d+)", name)
        if m and data:
            mru.append((int(m.group(1)), data))
    return [data for _, data in sorted(mru)]


def rdpcorets_events_from_xml(xml_records) -> list[dict]:
    """Extract RdpCoreTS events of interest from rendered event XML strings."""
    events = []
    for record in xml_records:
        try:
            root = ET.fromstring(record)
        except (ParseError, ValueError):
            continue
        # Strip namespaces so tag lookups are simple.
        for el in root.iter():
            el.tag = el.tag.rsplit("}", 1)[-1]

        eid_el = root.find(".//EventID")
        if eid_el is None or not (eid_el.text or "").strip().isdigit():
            continue
        event_id = int(eid_el.text)
        data = {}
        for d in root.findall(".//EventData/Data"):
            name = d.get("Name")
            if name:
                data[name] = (d.text or "").strip()
        time_el = root.find(".//TimeCreated")
        time = time_el.get("SystemTime") if time_el is not None else None

        record_out = {"event_id": event_id, "time": time}
        if "DesktopWidth" in data:
            record_out["desktop_width"] = int(data["DesktopWidth"])
        if "DesktopHeight" in data:
            record_out["desktop_height"] = int(data["DesktopHeight"])
        if "OSMajorType" in data:
            record_out["os_major"] = data["OSMajorType"]
        events.append(record_out)
    return events


# --- thin binary readers (python-registry / python-evtx); logic above is what's tested ---

def read_registry_default_mru(hive_path: str) -> list[str]:
    """Read MRU hosts from a Terminal Server Client key in an NTUSER.DAT hive."""
    from Registry import Registry

    reg = Registry.Registry(hive_path)
    for path in (
        r"Software\Microsoft\Terminal Server Client\Default",
        r"Microsoft\Terminal Server Client\Default",
    ):
        try:
            key = reg.open(path)
        except Registry.RegistryKeyNotFoundException:
            continue
        return registry_mru_from_values([(v.name(), v.value()) for v in key.values()])
    return []


def read_rdpcorets_evtx(path: str) -> list[dict]:
    """Read RdpCoreTS events of interest from an .evtx file."""
    from Evtx.Evtx import Evtx

    xmls = []
    with Evtx(path) as log:
        for record in log.records():
            xmls.append(record.xml())
    return rdpcorets_events_from_xml(xmls)


@dataclass(slots=True)
class ArtifactBundle:
    width: int | None = None
    height: int | None = None
    resolution_source: str | None = None
    multimon: bool | None = None
    os_major: str | None = None
    hosts: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


def collect_artifacts(path: str) -> ArtifactBundle:
    """Scan a file or directory for .rdp / NTUSER.DAT / .evtx and aggregate signals.

    Resolution preference: RdpCoreTS Event 168 (authoritative) over a .rdp file.
    """
    files = [path] if os.path.isfile(path) else [os.path.join(path, f) for f in sorted(os.listdir(path))]
    bundle = ArtifactBundle()
    rdp_res = evtx_res = None
    for f in files:
        low = f.lower()
        base = os.path.basename(f).upper()
        if low.endswith(".rdp"):
            info = parse_rdp_file(f)
            bundle.sources.append(f)
            if info.width and info.height:
                rdp_res = (info.width, info.height)
            if info.multimon is not None:
                bundle.multimon = info.multimon
            if info.full_address:
                bundle.hosts.append(info.full_address)
        elif low.endswith(".evtx"):
            try:
                events = read_rdpcorets_evtx(f)
            except Exception:
                continue
            bundle.sources.append(f)
            bundle.events.extend(events)
            for e in events:
                if e.get("desktop_width") and e.get("desktop_height"):
                    evtx_res = (e["desktop_width"], e["desktop_height"])
                if e.get("os_major"):
                    bundle.os_major = e["os_major"]
        elif base == "NTUSER.DAT":
            try:
                bundle.hosts.extend(read_registry_default_mru(f))
                bundle.sources.append(f)
            except Exception:
                continue

    if evtx_res:
        bundle.width, bundle.height = evtx_res
        bundle.resolution_source = "evtx_event168"
    elif rdp_res:
        bundle.width, bundle.height = rdp_res
        bundle.resolution_source = "rdp_file"
    return bundle
