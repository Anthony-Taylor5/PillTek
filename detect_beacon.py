import asyncio
from bleak import BleakScanner

BEACON_MAC = "DD:34:02:0A:2D:F1".upper()

#BEACON_MAC2 = "DD:34:02:0A:2D:DA".upper()
SCAN_INTERVAL = 2.0  # seconds between scans

async def scan_for_beacon():
    print(f"Scanning for KBeacon {BEACON_MAC} (Ctrl+C to stop)\n") #and {BEACON_MAC2}... (Ctrl+C to stop)\n")
    
    last_seen = None
    scan_count = 0
    
    while True:
        scan_count += 1
        try:
            # Use return_adv=True to get advertisement data
            devices = await BleakScanner.discover(timeout=3.0, return_adv=True)
            
            # Search for your beacon
            found = False
            for address, (device, adv_data) in devices.items():
                if device.address.upper() == BEACON_MAC: # or device.address.upper() == BEACON_MAC2:
                    print(f"[Scan #{scan_count}] ✓ DETECTED | RSSI: {adv_data.rssi} dBm | {device.address.upper() or 'Unknown'}")
                    found = True
                    last_seen = scan_count
                    break

            
            if not found:
                if last_seen is None:
                    print(f"[Scan #{scan_count}] ✗ Not detected (never seen)")
                else:
                    print(f"[Scan #{scan_count}] ✗ Not detected (last seen: scan #{last_seen})")
            
            await asyncio.sleep(SCAN_INTERVAL)
                    
        except Exception as e:
            print(f"[Scan #{scan_count}] Error: {e}")
            await asyncio.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    try:
        asyncio.run(scan_for_beacon())
    except KeyboardInterrupt:
        print("\n\nScan stopped.")