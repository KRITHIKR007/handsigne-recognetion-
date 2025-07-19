#!/usr/bin/env python3
"""
Complete Hand Sign Recognition System - All-in-One GUI
This is the final, complete version with all features integrated.
"""

import sys
import os

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def main():
    """Launch the complete hand sign recognition system."""
    try:
        from complete_gui import CompleteHandSignGUI
        print("✅ Successfully imported CompleteHandSignGUI")
        
        # Create and run the application
        app = CompleteHandSignGUI()
        print("✅ Complete GUI initialized successfully")
        print("📹 Camera ready")
        print("🖱️  Use mouse wheel to scroll through all features")
        print("🎯 All-in-one solution: Collection + Training + Recognition")
        print("🚀 Starting complete system...")
        
        app.run()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure all required packages are installed:")
        print("pip install opencv-python mediapipe tensorflow pillow numpy scikit-learn")
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
