#!/usr/bin/env python
"""
Main startup script for the Steam Game Recommender system.
This script processes the user's data files and starts the complete system.
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_data_files():
    """Check if required data files exist."""
    logger.info("Checking data files...")
    
    required_files = [
        "data/items_v2.json.gz",
        "data/reviews_v2.json.gz", 
        "data/bundles.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing required data files: {missing_files}")
        logger.error("Please ensure all data files are present in the data/ directory")
        return False
    
    logger.info("✅ All required data files found")
    return True

def process_data():
    """Process the Steam data files."""
    logger.info("Processing Steam data files...")
    
    try:
        # Run data processing script
        result = subprocess.run([
            sys.executable, "scripts/process_steam_data.py"
        ], capture_output=True, text=True, check=True)
        
        logger.info("✅ Data processing completed successfully")
        logger.info(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Data processing failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def build_models():
    """Build recommendation models from processed data."""
    logger.info("Building recommendation models...")
    
    try:
        # Run model building script
        result = subprocess.run([
            sys.executable, "scripts/build_recommendation_models.py"
        ], capture_output=True, text=True, check=True)
        
        logger.info("✅ Model building completed successfully")
        logger.info(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Model building failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def start_backend():
    """Start the FastAPI backend server."""
    logger.info("🚀 Starting FastAPI backend server...")
    
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
    
    logger.info("✅ Backend server started on http://localhost:8000")
    return backend_process

def start_frontend():
    """Start the React frontend development server."""
    logger.info("🎨 Starting React frontend server...")
    
    # Change to frontend directory
    frontend_dir = Path(__file__).parent / "frontend"
    os.chdir(frontend_dir)
    
    # Check if node_modules exists, if not install dependencies
    if not (frontend_dir / "node_modules").exists():
        logger.info("📦 Installing frontend dependencies...")
        subprocess.run(["npm", "install"], check=True)
    
    # Start frontend server
    frontend_process = subprocess.Popen(["npm", "start"])
    
    logger.info("✅ Frontend server started on http://localhost:3000")
    return frontend_process

def main():
    """Main startup function."""
    print("🎮 Steam Game Recommender System")
    print("=" * 50)
    print("Using data files:")
    print("- data/items_v2.json.gz")
    print("- data/reviews_v2.json.gz")
    print("- data/bundles.json")
    print("=" * 50)
    
    # Check data files
    if not check_data_files():
        sys.exit(1)
    
    # Process data
    if not process_data():
        logger.error("Failed to process data. Exiting.")
        sys.exit(1)
    
    # Build models
    if not build_models():
        logger.error("Failed to build models. Exiting.")
        sys.exit(1)
    
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
        
        print("\n🎉 Steam Game Recommender System Started Successfully!")
        print("=" * 50)
        print("📱 Frontend: http://localhost:3000")
        print("🔧 Backend API: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("🔍 Health Check: http://localhost:8000/health")
        print("\n💡 Your data has been processed and models are ready!")
        print("🎮 Start exploring game recommendations!")
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
        logger.error(f"💥 Error starting system: {e}")
        
        # Clean up processes
        for process in processes:
            try:
                process.terminate()
            except:
                pass
        
        sys.exit(1)

if __name__ == "__main__":
    main() 