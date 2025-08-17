#!/usr/bin/env python
"""
Startup script for the Steam Game Recommender system.
This script starts both the backend API and frontend development server.
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path

def start_backend():
    """Start the FastAPI backend server."""
    print("🚀 Starting FastAPI backend server...")
    
    # Change to project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Start backend server
    backend_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "src.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000",
        "--reload"
    ])
    
    print("✅ Backend server started on http://localhost:8000")
    return backend_process

def start_frontend():
    """Start the React frontend development server."""
    print("🎨 Starting React frontend server...")
    
    # Change to frontend directory
    frontend_dir = Path(__file__).parent / "frontend"
    os.chdir(frontend_dir)
    
    # Check if node_modules exists, if not install dependencies
    if not (frontend_dir / "node_modules").exists():
        print("📦 Installing frontend dependencies...")
        subprocess.run(["npm", "install"], check=True)
    
    # Start frontend server
    frontend_process = subprocess.Popen(["npm", "start"])
    
    print("✅ Frontend server started on http://localhost:3000")
    return frontend_process

def main():
    """Main startup function."""
    print("🎮 Steam Game Recommender System")
    print("=" * 40)
    
    processes = []
    
    try:
        # Start backend
        backend_process = start_backend()
        processes.append(backend_process)
        
        # Wait a bit for backend to start
        time.sleep(3)
        
        # Start frontend
        frontend_process = start_frontend()
        processes.append(frontend_process)
        
        print("\n🎉 System started successfully!")
        print("📱 Frontend: http://localhost:3000")
        print("🔧 Backend API: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop all services...")
        
        # Wait for processes
        for process in processes:
            process.wait()
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down services...")
        
        # Stop all processes
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        
        print("✅ All services stopped")
        
    except Exception as e:
        print(f"\n💥 Error starting system: {e}")
        
        # Clean up processes
        for process in processes:
            try:
                process.terminate()
            except:
                pass
        
        sys.exit(1)

if __name__ == "__main__":
    main() 