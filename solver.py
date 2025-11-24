#!/usr/bin/env python3
"""
BTC Puzzle #71 Solver - GCE VM Edition (No Metrics Server)
Optimized for c4-standard-2 (2 vCPUs, 7GB RAM)
All metrics sent via Discord webhooks with progress reports
"""

import os
import sys
import time
import json
import signal
import hashlib
import argparse
import logging
import threading
import urllib.request
import urllib.error
from multiprocessing import Pool, cpu_count
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any

# Dependencies check
try:
    import coincurve
    BACKEND = "coincurve"
except ImportError:
    print("Installing required dependencies...")
    os.system("pip install coincurve psutil")
    try:
        import coincurve
        BACKEND = "coincurve"
    except:
        print("Failed to install coincurve. Please run manually:")
        print("pip install coincurve psutil")
        sys.exit(1)

try:
    import psutil
except ImportError:
    os.system("pip install psutil")
    import psutil

# ============== Constants ==============

TARGET_ADDRESS = '1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU'  # Puzzle #71
START_RANGE = 0x40000000000000000  # 2^70
END_RANGE = 0x80000000000000000    # 2^71

# Discord Webhooks
PROGRESS_WEBHOOKS = [
    "https://discord.com/api/webhooks"
]
HIT_WEBHOOKS = [
    "https://discord.com/api/webhooks/"
]

# Optimized for 2 vCPUs
DEFAULT_PROCESSES = 2  # Match your vCPU count
DEFAULT_BATCH_SIZE = 50000  # Smaller batches for 2 cores
DISPLAY_INTERVAL_SEC = 10  # Update display every 10 seconds
CHECKPOINT_INTERVAL = 10000000  # Save every 10M keys
WEBHOOK_INTERVAL_SEC = 600  # Send webhook every 10 minutes

# ============== Bitcoin Functions ==============

def hash160(data: bytes) -> bytes:
    """SHA256 + RIPEMD160"""
    return hashlib.new('ripemd160', hashlib.sha256(data).digest()).digest()

BASE58_ALPHABET = b'123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
BASE58_LOOKUP = {c: i for i, c in enumerate(BASE58_ALPHABET)}

def base58_decode(s: str) -> bytes:
    """Decode Base58 address"""
    if not s or len(s) > 50:
        raise ValueError(f"Invalid Base58 length: {len(s)}")
    
    num_leading_zeros = len(s) - len(s.lstrip('1'))
    num = 0
    
    try:
        for ch in s:
            num = num * 58 + BASE58_LOOKUP[ord(ch)]
    except KeyError:
        raise ValueError("Invalid Base58 characters")
    
    # Convert to bytes
    if num == 0:
        payload = b'\x00' * 25
    else:
        hex_str = hex(num)[2:]
        if len(hex_str) % 2:
            hex_str = '0' + hex_str
        payload = b'\x00' * num_leading_zeros + bytes.fromhex(hex_str)
    
    if len(payload) != 25:
        raise ValueError(f"Decoded length {len(payload)}, expected 25")
    
    # Verify checksum
    checksum = hashlib.sha256(hashlib.sha256(payload[:-4]).digest()).digest()[:4]
    if checksum != payload[-4:]:
        raise ValueError("Checksum mismatch")
    
    return payload

def extract_hash160(address: str) -> bytes:
    """Extract hash160 from Bitcoin address"""
    decoded = base58_decode(address)
    return decoded[1:21]  # Skip version byte, return 20-byte hash160

# Pre-compute target hash160
TARGET_HASH160 = extract_hash160(TARGET_ADDRESS)

def generate_public_key(private_key_int: int) -> bytes:
    """Generate uncompressed public key from private key integer"""
    private_key_bytes = private_key_int.to_bytes(32, 'big')
    privkey = coincurve.PrivateKey(private_key_bytes)
    return privkey.public_key.format(compressed=False)

# ============== System Stats ==============

def get_system_stats() -> Dict[str, Any]:
    """Get current system statistics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network stats if available
        try:
            net = psutil.net_io_counters()
            net_stats = {
                'bytes_sent': net.bytes_sent,
                'bytes_recv': net.bytes_recv
            }
        except:
            net_stats = {'bytes_sent': 0, 'bytes_recv': 0}
        
        return {
            'cpu_percent': cpu_percent,
            'cpu_count': psutil.cpu_count(),
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'memory_percent': memory.percent,
            'disk_used_gb': disk.used / (1024**3),
            'disk_total_gb': disk.total / (1024**3),
            'disk_percent': disk.percent,
            'network': net_stats
        }
    except Exception as e:
        return {
            'cpu_percent': 0,
            'memory_percent': 0,
            'error': str(e)
        }

# ============== Discord Webhook Handler ==============

class WebhookNotifier:
    """Send Discord webhook notifications with system stats"""
    
    def __init__(self, progress_urls: List[str], hit_urls: List[str]):
        self.progress_urls = progress_urls or []
        self.hit_urls = hit_urls or []
        self.last_progress_time = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def _send_webhook(self, url: str, payload: Dict[str, Any], retries: int = 3) -> bool:
        """Send webhook with retry logic"""
        data = json.dumps(payload).encode('utf-8')
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Puzzle71Solver/1.0'
        }
        
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method='POST')
                with urllib.request.urlopen(req, timeout=10) as response:
                    if 200 <= response.status < 300:
                        return True
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
        return False
    
    def send_progress(self, stats: Dict[str, Any]):
        """Send progress update with system stats to Discord"""
        now = time.time()
        with self.lock:
            if now - self.last_progress_time < WEBHOOK_INTERVAL_SEC:
                return
            self.last_progress_time = now
        
        # Get system stats
        sys_stats = get_system_stats()
        uptime = now - self.start_time
        
        # Format ETA
        eta_sec = stats.get('eta_seconds', float('inf'))
        if eta_sec == float('inf'):
            eta_str = "Unknown"
        elif eta_sec > 86400 * 365:
            eta_str = f"{eta_sec/(86400*365):.1f} years"
        elif eta_sec > 86400:
            eta_str = f"{eta_sec/86400:.1f} days"
        elif eta_sec > 3600:
            eta_str = f"{eta_sec/3600:.1f} hours"
        else:
            eta_str = f"{eta_sec/60:.1f} minutes"
        
        # Format uptime
        if uptime > 86400:
            uptime_str = f"{uptime/86400:.1f} days"
        elif uptime > 3600:
            uptime_str = f"{uptime/3600:.1f} hours"
        else:
            uptime_str = f"{uptime/60:.1f} minutes"
        
        embed = {
            "title": "Puzzle #71 Progress Report",
            "color": 0x00FF00,
            "fields": [
                {"name": "Progress", "value": f"{stats['progress_pct']:.10f}%", "inline": True},
                {"name": "Speed", "value": f"{stats['rate']:,.0f} keys/sec", "inline": True},
                {"name": "ETA", "value": eta_str, "inline": True},
                {"name": "Keys Checked", "value": f"{stats['keys_checked']:,}", "inline": True},
                {"name": "Session Keys", "value": f"{stats.get('session_keys', 0):,}", "inline": True},
                {"name": "Uptime", "value": uptime_str, "inline": True},
                {"name": "Current Position", "value": f"`{stats['current_hex'][:16]}...`", "inline": False},
                {"name": "CPU Usage", "value": f"{sys_stats['cpu_percent']:.1f}%", "inline": True},
                {"name": "Memory", "value": f"{sys_stats.get('memory_used_gb', 0):.1f}/{sys_stats.get('memory_total_gb', 0):.1f} GB ({sys_stats.get('memory_percent', 0):.1f}%)", "inline": True},
                {"name": "Disk", "value": f"{sys_stats.get('disk_percent', 0):.1f}%", "inline": True},
                {"name": "Instance", "value": "GCE c4-standard-2", "inline": True},
                {"name": "Workers", "value": str(stats.get('processes', 2)), "inline": True},
                {"name": "Backend", "value": BACKEND, "inline": True}
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": f"BTC Puzzle #71 Solver | Batch Size: {stats.get('batch_size', DEFAULT_BATCH_SIZE):,}"}
        }
        
        payload = {
            "embeds": [embed],
            "username": "Puzzle71 Bot"
        }
        
        for url in self.progress_urls:
            self._send_webhook(url, payload)
    
    def send_hit(self, private_key_int: int):
        """Send KEY FOUND notification"""
        hex_key = hex(private_key_int)[2:].upper()
        
        # Get final system stats
        sys_stats = get_system_stats()
        
        embed = {
            "title": "🎉 PUZZLE #71 SOLVED - KEY FOUND! 🎉",
            "color": 0xFF0000,
            "fields": [
                {"name": "Target Address", "value": TARGET_ADDRESS, "inline": False},
                {"name": "Private Key (Hex)", "value": f"```{hex_key}```", "inline": False},
                {"name": "Private Key (Int)", "value": str(private_key_int), "inline": False},
                {"name": "Import Command", "value": f"```importprivkey {hex_key}```", "inline": False},
                {"name": "Found By", "value": f"GCE Instance (CPU: {sys_stats['cpu_percent']:.1f}%, Mem: {sys_stats.get('memory_percent', 0):.1f}%)", "inline": False}
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        payload = {
            "content": "@everyone PUZZLE #71 HAS BEEN SOLVED!",
            "embeds": [embed],
            "username": "Puzzle71 Bot"
        }
        
        for url in self.hit_urls:
            self._send_webhook(url, payload, retries=5)

# ============== Worker Function ==============

def check_range(args: Tuple[int, int, bytes]) -> Optional[int]:
    """Check a range of private keys (for multiprocessing)"""
    start, count, target_hash = args
    
    for i in range(count):
        private_key = start + i
        try:
            public_key = generate_public_key(private_key)
            if hash160(public_key) == target_hash:
                return private_key
        except Exception:
            continue
    
    return None

# ============== Main Solver ==============

class Puzzle71Solver:
    """Main solver class - no metrics server"""
    
    def __init__(self, processes: int = DEFAULT_PROCESSES, batch_size: int = DEFAULT_BATCH_SIZE):
        self.processes = processes
        self.batch_size = batch_size
        self.progress_file = Path('puzzle71_progress.json')
        self.key_found_file = Path('PUZZLE71_KEY_FOUND.txt')
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('Puzzle71')
        
        # Components
        self.webhooks = WebhookNotifier(PROGRESS_WEBHOOKS, HIT_WEBHOOKS)
        self.pool = None
        self.shutdown_flag = False
        self.start_time = time.time()
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("Shutdown signal received, saving progress...")
        self.shutdown_flag = True
        if self.pool:
            self.pool.terminate()
    
    def load_progress(self) -> Tuple[int, int]:
        """Load saved progress"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    return data.get('current_value', START_RANGE), data.get('total_keys_checked', 0)
            except Exception as e:
                self.logger.warning(f"Failed to load progress: {e}")
        return START_RANGE, 0
    
    def save_progress(self, current: int, total_checked: int, session_keys: int = 0):
        """Save current progress"""
        try:
            data = {
                'current_value': current,
                'total_keys_checked': total_checked,
                'session_keys': session_keys,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'backend': BACKEND,
                'instance': 'GCE c4-standard-2'
            }
            
            temp_file = self.progress_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.progress_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")
    
    def save_found_key(self, private_key_int: int):
        """Save found private key"""
        hex_key = hex(private_key_int)[2:].upper()
        
        info = f"""
BITCOIN PUZZLE #71 - PRIVATE KEY FOUND!
{'='*60}
Timestamp: {datetime.now(timezone.utc).isoformat()}
Private Key (Hex): {hex_key}
Private Key (Int): {private_key_int}
Target Address: {TARGET_ADDRESS}
Backend: {BACKEND}
Instance: GCE c4-standard-2
{'='*60}

Import in Bitcoin Core:
importprivkey {hex_key}

Import in Electrum:
Create new wallet -> Import Bitcoin addresses or private keys
Enter: {hex_key}
"""
        
        with open(self.key_found_file, 'w') as f:
            f.write(info)
        
        print("\n" + "!"*60)
        print(info)
        print("!"*60)
        
        self.logger.info(f"KEY FOUND: {hex_key}")
    
    def run(self):
        """Main solving loop"""
        print("="*70)
        print(f"Bitcoin Puzzle #71 Solver - GCE Optimized")
        print(f"Instance: c4-standard-2 (2 vCPUs, 7GB RAM)")
        print(f"Backend: {BACKEND}")
        print(f"Target: {TARGET_ADDRESS}")
        print(f"Target Hash160: {TARGET_HASH160.hex()}")
        print(f"Range: {hex(START_RANGE)} to {hex(END_RANGE)}")
        print(f"Total Keys: {END_RANGE - START_RANGE:,}")
        print(f"Processes: {self.processes} | Batch Size: {self.batch_size:,}")
        print(f"Discord Webhooks: Enabled (every {WEBHOOK_INTERVAL_SEC}s)")
        print("="*70)
        
        # Load progress
        current, total_checked_all_time = self.load_progress()
        if current > START_RANGE:
            print(f"Resuming from: {hex(current)}")
            print(f"Total keys checked (all-time): {total_checked_all_time:,}")
        
        # Initialize pool
        self.pool = Pool(processes=self.processes)
        
        # Tracking variables
        session_start = time.time()
        session_keys = 0
        last_display = time.time()
        last_checkpoint = 0
        total_range = END_RANGE - START_RANGE
        
        print("\nStarting search...\n")
        
        try:
            while current < END_RANGE and not self.shutdown_flag:
                # Prepare batch
                batch_remaining = min(self.batch_size, END_RANGE - current)
                
                # Split work among processes
                work_per_process = batch_remaining // self.processes
                work_items = []
                
                for i in range(self.processes):
                    start = current + i * work_per_process
                    count = work_per_process if i < self.processes - 1 else batch_remaining - i * work_per_process
                    if count > 0:
                        work_items.append((start, count, TARGET_HASH160))
                
                # Execute parallel search
                results = self.pool.map(check_range, work_items)
                
                # Check for hit
                for result in results:
                    if result is not None:
                        # Verify the key
                        if hash160(generate_public_key(result)) == TARGET_HASH160:
                            self.save_found_key(result)
                            self.webhooks.send_hit(result)
                            self.pool.terminate()
                            return True
                
                # Update progress
                current += batch_remaining
                session_keys += batch_remaining
                total_checked = total_checked_all_time + session_keys
                
                # Calculate stats
                elapsed = time.time() - session_start
                rate = session_keys / elapsed if elapsed > 0 else 0
                progress_pct = ((current - START_RANGE) / total_range) * 100
                remaining = END_RANGE - current
                eta_seconds = remaining / rate if rate > 0 else float('inf')
                
                # Display progress
                now = time.time()
                if now - last_display >= DISPLAY_INTERVAL_SEC:
                    print(f"\rProgress: {progress_pct:.10f}% | "
                          f"Speed: {rate:,.0f} keys/s | "
                          f"Checked: {total_checked:,} | "
                          f"Current: {hex(current)[:16]}...", end='', flush=True)
                    
                    # Send webhook with all stats
                    self.webhooks.send_progress({
                        'progress_pct': progress_pct,
                        'rate': rate,
                        'eta_seconds': eta_seconds,
                        'keys_checked': total_checked,
                        'session_keys': session_keys,
                        'current_hex': hex(current),
                        'processes': self.processes,
                        'batch_size': self.batch_size
                    })
                    
                    last_display = now
                
                # Checkpoint
                if session_keys - last_checkpoint >= CHECKPOINT_INTERVAL:
                    self.save_progress(current, total_checked, session_keys)
                    self.logger.info(f"Checkpoint saved: {total_checked:,} keys total")
                    last_checkpoint = session_keys
            
            # Save final progress
            self.save_progress(current, total_checked_all_time + session_keys, session_keys)
            
            if current >= END_RANGE:
                print(f"\n\nSearch range exhausted. Checked {total_checked_all_time + session_keys:,} keys total.")
            else:
                print(f"\n\nStopped at {hex(current)}. Progress saved.")
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            self.save_progress(current, total_checked_all_time + session_keys, session_keys)
            return False
        
        finally:
            if self.pool:
                self.pool.close()
                self.pool.join()

# ============== Entry Point ==============

def main():
    parser = argparse.ArgumentParser(description='Bitcoin Puzzle #71 Solver - GCE Edition (No Metrics Server)')
    parser.add_argument('--processes', type=int, default=DEFAULT_PROCESSES,
                        help=f'Number of worker processes (default: {DEFAULT_PROCESSES})')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Keys per batch (default: {DEFAULT_BATCH_SIZE})')
    
    args = parser.parse_args()
    
    solver = Puzzle71Solver(
        processes=max(1, args.processes),
        batch_size=max(10000, args.batch_size)
    )
    
    solver.run()

if __name__ == '__main__':
    main()
