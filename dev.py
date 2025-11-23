#!/usr/bin/env python3

import sys
import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


def terminate_process(process, timeout=2):
    """Terminate a subprocess gracefully or forcefully if needed."""
    if process:
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()


class ReloadHandler(FileSystemEventHandler):
    def __init__(self, script_path):
        self.script_path = script_path
        self.process = None
        self.restart()
    
    def restart(self):
        if self.process:
            print("\n🔄 Reloading...")
            terminate_process(self.process)
        
        print(f"▶️  Starting {self.script_path.name}")
        self.process = subprocess.Popen([sys.executable, str(self.script_path)])
    
    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            print(f"\n📝 Detected change in {Path(event.src_path).name}")
            self.restart()


def main():
    script_path = Path(__file__).parent / "main.py"
    
    if not script_path.exists():
        print(f"Error: {script_path} not found")
        return
    
    print(f"👀 Watching {script_path.name} for changes...")
    print("Press Ctrl+C to stop\n")
    
    handler = ReloadHandler(script_path)
    observer = Observer()
    observer.schedule(handler, str(script_path.parent), recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping...")
        observer.stop()
        terminate_process(handler.process)
    
    observer.join()
    print("✅ Stopped")


if __name__ == "__main__":
    main()
