"""rdpieces — automated RDP bitmap-cache reconstruction + OCR.

A clean-room Python rewrite of Brian Moran's rdpieces.pl. Cache-format knowledge
is derived from the public [MS-RDPEGDI]/[MS-RDPBCGR] specifications and direct
byte inspection of sample caches — it does not copy ANSSI bmc-tools code.
"""

__version__ = "2.0.0.dev0"
