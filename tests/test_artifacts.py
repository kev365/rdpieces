"""Tests for external host-artifact parsing (resolution / monitor / OS / timezone).

Registry- and evtx-extraction logic is tested via injected records, so it doesn't
need sample hives or .evtx binaries.
"""

from __future__ import annotations

from rdpieces.constraints.artifacts import (
    collect_artifacts,
    parse_rdp_file,
    registry_mru_from_values,
    rdpcorets_events_from_xml,
)


def test_parse_rdp_file_extracts_resolution_and_multimon(tmp_path):
    rdp = tmp_path / "server.rdp"
    rdp.write_text(
        "screen mode id:i:2\n"
        "use multimon:i:1\n"
        "desktopwidth:i:1920\n"
        "desktopheight:i:1080\n"
        "session bpp:i:32\n"
        "full address:s:10.0.0.5\n"
    )

    info = parse_rdp_file(str(rdp))

    assert (info.width, info.height) == (1920, 1080)
    assert info.fullscreen is True  # screen mode id 2
    assert info.multimon is True
    assert info.full_address == "10.0.0.5"


def test_parse_rdp_file_missing_fields_are_none(tmp_path):
    rdp = tmp_path / "min.rdp"
    rdp.write_text("full address:s:host1\n")
    info = parse_rdp_file(str(rdp))
    assert info.width is None and info.height is None
    assert info.full_address == "host1"


def test_registry_mru_from_values_orders_and_extracts_hosts():
    # Simulated 'Terminal Server Client\\Default' values (name, data).
    values = [("MRU1", "10.0.0.9"), ("MRU0", "fileserver"), ("MRU2", "10.0.0.5")]
    hosts = registry_mru_from_values(values)
    assert hosts == ["fileserver", "10.0.0.9", "10.0.0.5"]  # MRU0 first


def test_collect_artifacts_from_directory_uses_rdp_resolution(tmp_path):
    (tmp_path / "conn.rdp").write_text("desktopwidth:i:1366\ndesktopheight:i:768\nfull address:s:dc01\n")
    bundle = collect_artifacts(str(tmp_path))
    assert (bundle.width, bundle.height) == (1366, 768)
    assert bundle.resolution_source == "rdp_file"
    assert "dc01" in bundle.hosts


def test_rdpcorets_events_from_xml_pulls_resolution_os_tz():
    xml_records = [
        # Event 168: negotiated desktop size
        '<Event><System><EventID>168</EventID>'
        '<TimeCreated SystemTime="2020-05-28T12:00:00Z"/></System>'
        '<EventData><Data Name="DesktopWidth">1920</Data>'
        '<Data Name="DesktopHeight">1080</Data></EventData></Event>',
        # Event 169: OS type
        '<Event><System><EventID>169</EventID>'
        '<TimeCreated SystemTime="2020-05-28T12:00:01Z"/></System>'
        '<EventData><Data Name="OSMajorType">6</Data></EventData></Event>',
    ]
    events = rdpcorets_events_from_xml(xml_records)
    by_id = {e["event_id"]: e for e in events}
    assert by_id[168]["desktop_width"] == 1920
    assert by_id[168]["desktop_height"] == 1080
    assert by_id[168]["time"] == "2020-05-28T12:00:00Z"
    assert by_id[169]["event_id"] == 169
