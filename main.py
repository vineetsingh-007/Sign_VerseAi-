import argparse
import sys
import subprocess
import os

def launch_dashboard():
    """Launches the Streamlit dashboard app."""
    dashboard_path = os.path.join("dashboard", "streamlit_dashboard.py")
    if not os.path.exists(dashboard_path):
        print(f"Error: Streamlit dashboard file not found at {dashboard_path}")
        return
        
    print("\nStarting SignVerse AI Streamlit Dashboard...")
    try:
        # Launch Streamlit server with file watcher disabled to avoid PyTorch conflicts
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path, "--server.fileWatcherType=none"], check=True)
    except KeyboardInterrupt:
        print("\nDashboard server stopped.")
    except Exception as e:
        print(f"Error launching Streamlit dashboard: {e}")
        print("Please check that streamlit is installed: pip install streamlit")

def main():
    parser = argparse.ArgumentParser(
        description="SIGNVERSE AI - Real-Time Sign Language Recognition & Assistant Orchestrator"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # Subcommand: generate-data
    subparsers.add_parser("generate-data", help="Generates synthetic landmark training datasets")
    
    # Subcommand: train
    train_parser = subparsers.add_parser("train", help="Trains the PyTorch 1D CNN model")
    train_parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    # Subcommand: evaluate
    subparsers.add_parser("evaluate", help="Evaluates the trained model on test split & plots Confusion Matrix")
    
    # Subcommand: webcam
    subparsers.add_parser("webcam", help="Launches the standalone OpenCV webcam reader client")
    
    # Subcommand: dashboard
    subparsers.add_parser("dashboard", help="Launches the Streamlit dark dashboard panel")
    
    # Subcommand: report
    subparsers.add_parser("report", help="Generates database communication summary reports")
    
    args = parser.parse_args()
    
    if args.command == "generate-data":
        from training.dataset_loader import generate_synthetic_dataset
        print("Initializing data generation...")
        generate_synthetic_dataset(samples_per_class=150)
        
    elif args.command == "train":
        from training.train import train_model
        print(f"Initializing PyTorch training pipeline (epochs={args.epochs}, lr={args.lr})...")
        train_model(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        
    elif args.command == "evaluate":
        from training.evaluate import evaluate_model
        print("Initializing test evaluation...")
        evaluate_model()
        
    elif args.command == "webcam":
        from realtime.webcam_app import run_webcam_application
        print("Initializing desktop webcam client...")
        run_webcam_application()
        
    elif args.command == "dashboard":
        launch_dashboard()
        
    elif args.command == "report":
        from analytics.reports import AnalyticsReporter
        print("Compiling database analytical summary...")
        reporter = AnalyticsReporter()
        reporter.write_report_to_file()
        
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
