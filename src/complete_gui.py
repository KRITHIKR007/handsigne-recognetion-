import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import cv2
from PIL import Image, ImageTk
import threading
import time
import numpy as np
from typing import Optional
import os

# Import our custom modules
from data_collection import HandSignDataCollector
from model import HandSignModel

class CompleteHandSignGUI:
    """
    Complete Hand Sign Recognition GUI with all features in one place.
    Features: Scrollable interface, prominent controls, all aspects visible.
    """
    
    def __init__(self):
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("🤲 Hand Sign Recognition System - Complete Edition")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#0f0f0f')
        self.root.minsize(1400, 900)
        
        # Configure better styling with safer font handling
        try:
            # Try to set default font family
            default_font = self.root.tk.call("font", "families")
            if "Segoe UI" in default_font:
                self.root.option_add('*Font', ('Segoe UI', 10))
            else:
                self.root.option_add('*Font', ('Arial', 10))
        except:
            # Fallback to system default
            pass
        
        # Set window icon (if available)
        try:
            self.root.iconbitmap(default='icon.ico')
        except:
            pass
        
        # Initialize components
        self.data_collector = HandSignDataCollector()
        self.ml_model = HandSignModel()
        
        # GUI state variables
        self.camera_active = False
        self.current_mode = "data_collection"
        self.video_label = None
        self.stop_update = False
        self.collection_thread = None
        self.training_thread = None
        
        # Create scrollable main frame
        self.create_scrollable_frame()
        
        # Create GUI elements
        self.create_widgets()
        self.setup_layout()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Auto-update data info
        self.update_data_info()
    
    def get_safe_font(self, family="Segoe UI", size=10, weight="normal"):
        """Get a safe font that works on the system."""
        try:
            # Try the requested font
            test_font = (family, size, weight)
            # Test if font works by creating a temporary label
            temp_label = tk.Label(self.root, font=test_font)
            temp_label.destroy()
            return test_font
        except:
            # Fallback to Arial
            try:
                fallback_font = ("Arial", size, weight)
                temp_label = tk.Label(self.root, font=fallback_font)
                temp_label.destroy()
                return fallback_font
            except:
                # Final fallback to default
                return ("TkDefaultFont", size, weight)
    
    def create_scrollable_frame(self):
        """Create a scrollable main frame with proper mouse wheel support."""
        # Create main container
        self.main_container = tk.Frame(self.root, bg='#0f0f0f')
        self.main_container.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Create canvas and scrollbar
        self.main_canvas = tk.Canvas(
            self.main_container,
            bg='#0f0f0f',
            highlightthickness=0,
            relief='flat'
        )
        
        self.scrollbar = ttk.Scrollbar(
            self.main_container,
            orient="vertical",
            command=self.main_canvas.yview
        )
        
        # Create scrollable frame inside canvas
        self.scrollable_frame = tk.Frame(self.main_canvas, bg='#0f0f0f')
        
        # Configure scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        # Create window in canvas
        self.canvas_window = self.main_canvas.create_window(
            (0, 0), 
            window=self.scrollable_frame, 
            anchor="nw"
        )
        
        # Configure canvas scroll
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack canvas and scrollbar
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel and resize events
        self.bind_scroll_events()
    
    def bind_scroll_events(self):
        """Bind mouse wheel and resize events for smooth scrolling."""
        def _on_mousewheel(event):
            self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            self.main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            self.main_canvas.unbind_all("<MouseWheel>")
        
        def _on_canvas_configure(event):
            # Update scrollable frame width to fill canvas
            canvas_width = event.width
            self.main_canvas.itemconfig(self.canvas_window, width=canvas_width)
        
        # Bind events
        self.main_canvas.bind('<Enter>', _bind_to_mousewheel)
        self.main_canvas.bind('<Leave>', _unbind_from_mousewheel)
        self.main_canvas.bind('<Configure>', _on_canvas_configure)
    
    def create_widgets(self):
        """Create all GUI widgets with modern styling."""
        
        # Main title with emoji and modern look
        self.title_frame = tk.Frame(self.scrollable_frame, bg='#0f0f0f')
        
        title_font = self.get_safe_font("Segoe UI", 32, "bold")
        self.title_label = tk.Label(
            self.title_frame,
            text="🤲 Hand Sign Recognition System",
            font=title_font,
            bg='#0f0f0f',
            fg='#ffffff'
        )
        
        subtitle_font = self.get_safe_font("Segoe UI", 14, "normal")
        self.subtitle_label = tk.Label(
            self.title_frame,
            text="Real-time ML-powered hand gesture recognition with complete control",
            font=subtitle_font,
            bg='#0f0f0f',
            fg='#b0b0b0'
        )
        
        # Mode selection with modern radio buttons
        self.create_mode_selection()
        
        # Camera section with prominent controls
        self.create_camera_section()
        
        # Main control panels
        self.create_control_panels()
        
        # Status and info section
        self.create_status_section()
    
    def create_mode_selection(self):
        """Create modern mode selection interface."""
        self.mode_frame = tk.Frame(self.scrollable_frame, bg='#1e1e1e', relief=tk.RAISED, bd=3)
        
        mode_title = tk.Label(
            self.mode_frame,
            text="🎛️ Select Operation Mode",
            font=self.get_safe_font("Segoe UI", 20, "bold"),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        mode_title.pack(pady=20)
        
        # Mode selection with colored frames
        self.mode_var = tk.StringVar(value="data_collection")
        
        modes_container = tk.Frame(self.mode_frame, bg='#1e1e1e')
        modes_container.pack(pady=15)
        
        # Data Collection Mode
        data_mode_frame = tk.Frame(modes_container, bg='#3d5afe', relief=tk.RAISED, bd=3)
        data_mode_frame.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.data_collection_radio = tk.Radiobutton(
            data_mode_frame,
            text="📊 Data Collection\nCapture hand signs",
            variable=self.mode_var,
            value="data_collection",
            command=self.change_mode,
            bg='#3d5afe',
            fg='white',
            selectcolor='#1e88e5',
            font=self.get_safe_font("Segoe UI", 14, "bold"),
            justify=tk.CENTER,
            padx=30,
            pady=20
        )
        self.data_collection_radio.pack()
        
        # Training Mode
        train_mode_frame = tk.Frame(modes_container, bg='#4caf50', relief=tk.RAISED, bd=3)
        train_mode_frame.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.training_radio = tk.Radiobutton(
            train_mode_frame,
            text="🧠 Model Training\nTrain neural network",
            variable=self.mode_var,
            value="training",
            command=self.change_mode,
            bg='#4caf50',
            fg='white',
            selectcolor='#388e3c',
            font=("Segoe UI", 14, "bold"),
            justify=tk.CENTER,
            padx=30,
            pady=20
        )
        self.training_radio.pack()
        
        # Recognition Mode
        recog_mode_frame = tk.Frame(modes_container, bg='#ff5722', relief=tk.RAISED, bd=3)
        recog_mode_frame.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.recognition_radio = tk.Radiobutton(
            recog_mode_frame,
            text="🎯 Recognition\nReal-time detection",
            variable=self.mode_var,
            value="recognition",
            command=self.change_mode,
            bg='#ff5722',
            fg='white',
            selectcolor='#d84315',
            font=("Segoe UI", 14, "bold"),
            justify=tk.CENTER,
            padx=30,
            pady=20
        )
        self.recognition_radio.pack()
    
    def create_camera_section(self):
        """Create camera section with prominent controls."""
        self.camera_frame = tk.Frame(self.scrollable_frame, bg='#1e1e1e', relief=tk.RAISED, bd=4)
        
        # Camera title
        camera_title = tk.Label(
            self.camera_frame,
            text="📹 Camera Control Center",
            font=("Segoe UI", 22, "bold"),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        camera_title.pack(pady=20)
        
        # Video display area with modern styling
        self.video_container = tk.Frame(self.camera_frame, bg='#000000', relief=tk.SUNKEN, bd=4)
        self.video_container.pack(pady=15, padx=30)
        
        self.video_label = tk.Label(
            self.video_container,
            text="📹 Camera Ready\n\n🟢 Click the GREEN button below to start camera\n\nCamera will show live hand tracking with landmarks",
            width=75,
            height=28,
            bg='#000000',
            fg='#00ff00',
            font=("Consolas", 14),
            relief=tk.FLAT,
            bd=0,
            justify=tk.CENTER
        )
        self.video_label.pack(padx=15, pady=15)
        
        # Camera controls with better-sized button
        self.camera_controls_frame = tk.Frame(self.camera_frame, bg='#1e1e1e')
        self.camera_controls_frame.pack(pady=25)
        
        self.start_camera_btn = tk.Button(
            self.camera_controls_frame,
            text="🎥 START CAMERA",
            command=self.toggle_camera,
            width=20,
            height=2,
            bg='#4CAF50',
            fg='white',
            font=("Segoe UI", 16, "bold"),
            relief=tk.RAISED,
            bd=6,
            activebackground='#45a049',
            cursor='hand2'
        )
        self.start_camera_btn.pack()
        
        # Camera status
        self.camera_status_label = tk.Label(
            self.camera_controls_frame,
            text="Status: Camera Ready",
            font=("Segoe UI", 14),
            bg='#1e1e1e',
            fg='#4CAF50'
        )
        self.camera_status_label.pack(pady=15)
    
    def create_control_panels(self):
        """Create all control panels in organized sections."""
        self.control_container = tk.Frame(self.scrollable_frame, bg='#0f0f0f')
        
        # Data collection controls
        self.create_data_collection_panel()
        
        # Training controls
        self.create_training_panel()
        
        # Recognition controls
        self.create_recognition_panel()
    
    def create_data_collection_panel(self):
        """Create comprehensive data collection panel."""
        self.data_collection_frame = tk.Frame(self.control_container, bg='#1e3a8a', relief=tk.RAISED, bd=3)
        
        # Panel title
        title = tk.Label(
            self.data_collection_frame,
            text="📊 Data Collection Panel",
            font=("Segoe UI", 18, "bold"),
            bg='#1e3a8a',
            fg='white'
        )
        title.pack(pady=15)
        
        # Input section
        input_frame = tk.Frame(self.data_collection_frame, bg='#1e3a8a')
        input_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Label entry with modern styling
        tk.Label(
            input_frame,
            text="✋ Hand Sign Label:",
            bg='#1e3a8a',
            fg='white',
            font=("Segoe UI", 12, "bold")
        ).pack(anchor='w', pady=5)
        
        self.label_entry = tk.Entry(
            input_frame,
            width=50,
            font=("Segoe UI", 13),
            relief=tk.FLAT,
            bd=8,
            bg='#ffffff',
            fg='#000000'
        )
        self.label_entry.pack(pady=8, fill=tk.X)
        self.label_entry.insert(0, "Enter sign name (e.g., hello, peace, thumbs_up)")
        
        # Duration and sample controls
        controls_frame = tk.Frame(self.data_collection_frame, bg='#1e3a8a')
        controls_frame.pack(pady=15, padx=20, fill=tk.X)
        
        # Duration
        duration_frame = tk.Frame(controls_frame, bg='#1e3a8a')
        duration_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(
            duration_frame,
            text="⏱️ Duration (seconds):",
            bg='#1e3a8a',
            fg='white',
            font=("Segoe UI", 11, "bold")
        ).pack()
        
        self.duration_var = tk.StringVar(value="10")
        self.duration_entry = tk.Entry(
            duration_frame,
            textvariable=self.duration_var,
            width=10,
            font=("Segoe UI", 13),
            justify=tk.CENTER,
            bg='#ffffff',
            fg='#000000',
            relief=tk.FLAT,
            bd=5
        )
        self.duration_entry.pack(pady=8)
        
        # Sample count
        samples_frame = tk.Frame(controls_frame, bg='#1e3a8a')
        samples_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(
            samples_frame,
            text="📈 Target Samples:",
            bg='#1e3a8a',
            fg='white',
            font=("Segoe UI", 11, "bold")
        ).pack()
        
        self.samples_var = tk.StringVar(value="100")
        self.samples_entry = tk.Entry(
            samples_frame,
            textvariable=self.samples_var,
            width=10,
            font=("Segoe UI", 13),
            justify=tk.CENTER,
            bg='#ffffff',
            fg='#000000',
            relief=tk.FLAT,
            bd=5
        )
        self.samples_entry.pack(pady=8)
        
        # Action buttons
        button_frame = tk.Frame(self.data_collection_frame, bg='#1e3a8a')
        button_frame.pack(pady=25)
        
        self.collect_btn = tk.Button(
            button_frame,
            text="🎥 START COLLECTION",
            command=self.start_data_collection,
            width=18,
            height=2,
            bg='#FF5722',
            fg='white',
            font=("Segoe UI", 13, "bold"),
            relief=tk.RAISED,
            bd=4
        )
        self.collect_btn.pack(side=tk.LEFT, padx=15)
        
        self.save_data_btn = tk.Button(
            button_frame,
            text="💾 SAVE DATA",
            command=self.save_collected_data,
            width=15,
            height=2,
            bg='#2196F3',
            fg='white',
            font=("Segoe UI", 12, "bold"),
            relief=tk.RAISED,
            bd=4
        )
        self.save_data_btn.pack(side=tk.LEFT, padx=15)
        
        self.load_data_btn = tk.Button(
            button_frame,
            text="📂 LOAD DATA",
            command=self.load_data,
            width=15,
            height=2,
            bg='#9C27B0',
            fg='white',
            font=("Segoe UI", 12, "bold"),
            relief=tk.RAISED,
            bd=4
        )
        self.load_data_btn.pack(side=tk.LEFT, padx=15)
        
        # Collection progress
        self.collection_progress_frame = tk.Frame(self.data_collection_frame, bg='#1e3a8a')
        self.collection_progress_frame.pack(pady=15, padx=20, fill=tk.X)
        
        self.collection_status_label = tk.Label(
            self.collection_progress_frame,
            text="Ready to collect data",
            bg='#1e3a8a',
            fg='white',
            font=("Segoe UI", 12)
        )
        self.collection_status_label.pack()
        
        self.collection_progress_var = tk.DoubleVar()
        self.collection_progress_bar = ttk.Progressbar(
            self.collection_progress_frame,
            variable=self.collection_progress_var,
            maximum=100,
            length=500,
            style='TProgressbar'
        )
        self.collection_progress_bar.pack(pady=15)
    
    def create_training_panel(self):
        """Create comprehensive training panel."""
        self.training_frame = tk.Frame(self.control_container, bg='#2e7d32', relief=tk.RAISED, bd=3)
        
        # Panel title
        title = tk.Label(
            self.training_frame,
            text="🧠 Neural Network Training Panel",
            font=("Segoe UI", 18, "bold"),
            bg='#2e7d32',
            fg='white'
        )
        title.pack(pady=15)
        
        # Training parameters
        params_frame = tk.Frame(self.training_frame, bg='#2e7d32')
        params_frame.pack(pady=15, padx=20)
        
        # Parameters grid
        param_grid = tk.Frame(params_frame, bg='#2e7d32')
        param_grid.pack()
        
        # Epochs
        tk.Label(
            param_grid,
            text="🔄 Epochs:",
            bg='#2e7d32',
            fg='white',
            font=("Segoe UI", 12, "bold")
        ).grid(row=0, column=0, padx=15, pady=10, sticky='e')
        
        self.epochs_var = tk.StringVar(value="100")
        self.epochs_entry = tk.Entry(
            param_grid,
            textvariable=self.epochs_var,
            width=10,
            font=("Segoe UI", 12),
            justify=tk.CENTER
        )
        self.epochs_entry.grid(row=0, column=1, padx=15, pady=10)
        
        # Batch Size
        tk.Label(
            param_grid,
            text="📦 Batch Size:",
            bg='#2e7d32',
            fg='white',
            font=("Segoe UI", 12, "bold")
        ).grid(row=0, column=2, padx=15, pady=10, sticky='e')
        
        self.batch_size_var = tk.StringVar(value="32")
        self.batch_size_entry = tk.Entry(
            param_grid,
            textvariable=self.batch_size_var,
            width=10,
            font=("Segoe UI", 12),
            justify=tk.CENTER
        )
        self.batch_size_entry.grid(row=0, column=3, padx=15, pady=10)
        
        # Learning Rate
        tk.Label(
            param_grid,
            text="📊 Learning Rate:",
            bg='#2e7d32',
            fg='white',
            font=("Segoe UI", 12, "bold")
        ).grid(row=1, column=0, padx=15, pady=10, sticky='e')
        
        self.learning_rate_var = tk.StringVar(value="0.001")
        self.learning_rate_entry = tk.Entry(
            param_grid,
            textvariable=self.learning_rate_var,
            width=10,
            font=("Segoe UI", 12),
            justify=tk.CENTER
        )
        self.learning_rate_entry.grid(row=1, column=1, padx=15, pady=10)
        
        # Validation Split
        tk.Label(
            param_grid,
            text="✅ Validation Split:",
            bg='#2e7d32',
            fg='white',
            font=("Segoe UI", 12, "bold")
        ).grid(row=1, column=2, padx=15, pady=10, sticky='e')
        
        self.validation_split_var = tk.StringVar(value="0.2")
        self.validation_split_entry = tk.Entry(
            param_grid,
            textvariable=self.validation_split_var,
            width=10,
            font=("Segoe UI", 12),
            justify=tk.CENTER
        )
        self.validation_split_entry.grid(row=1, column=3, padx=15, pady=10)
        
        # Training buttons
        button_frame = tk.Frame(self.training_frame, bg='#2e7d32')
        button_frame.pack(pady=25)
        
        self.train_btn = tk.Button(
            button_frame,
            text="🚀 START TRAINING",
            command=self.start_training,
            width=18,
            height=2,
            bg='#FF9800',
            fg='white',
            font=("Segoe UI", 13, "bold"),
            relief=tk.RAISED,
            bd=4
        )
        self.train_btn.pack(side=tk.LEFT, padx=15)
        
        self.save_model_btn = tk.Button(
            button_frame,
            text="💾 SAVE MODEL",
            command=self.save_model,
            width=15,
            height=2,
            bg='#795548',
            fg='white',
            font=("Segoe UI", 12, "bold"),
            relief=tk.RAISED,
            bd=4
        )
        self.save_model_btn.pack(side=tk.LEFT, padx=15)
        
        self.load_model_btn = tk.Button(
            button_frame,
            text="📂 LOAD MODEL",
            command=self.load_model,
            width=15,
            height=2,
            bg='#607D8B',
            fg='white',
            font=("Segoe UI", 12, "bold"),
            relief=tk.RAISED,
            bd=4
        )
        self.load_model_btn.pack(side=tk.LEFT, padx=15)
        
        # Training progress
        self.training_progress_frame = tk.Frame(self.training_frame, bg='#2e7d32')
        self.training_progress_frame.pack(pady=15, padx=20, fill=tk.X)
        
        self.training_status_label = tk.Label(
            self.training_progress_frame,
            text="Ready to train model",
            bg='#2e7d32',
            fg='white',
            font=("Segoe UI", 12)
        )
        self.training_status_label.pack()
        
        self.training_progress_var = tk.DoubleVar()
        self.training_progress_bar = ttk.Progressbar(
            self.training_progress_frame,
            variable=self.training_progress_var,
            maximum=100,
            length=400
        )
        self.training_progress_bar.pack(pady=10)
        
        # Model info
        self.model_info_label = tk.Label(
            self.training_progress_frame,
            text="Model: Not loaded",
            bg='#2e7d32',
            fg='white',
            font=("Segoe UI", 10)
        )
        self.model_info_label.pack(pady=5)
    
    def create_recognition_panel(self):
        """Create comprehensive recognition panel."""
        self.recognition_frame = tk.Frame(self.control_container, bg='#c62828', relief=tk.RAISED, bd=3)
        
        # Panel title
        title = tk.Label(
            self.recognition_frame,
            text="🎯 Real-time Recognition Panel",
            font=("Segoe UI", 18, "bold"),
            bg='#c62828',
            fg='white'
        )
        title.pack(pady=15)
        
        # Recognition display
        display_frame = tk.Frame(self.recognition_frame, bg='#c62828')
        display_frame.pack(pady=15)
        
        # Current prediction
        self.prediction_display = tk.Frame(display_frame, bg='#000000', relief=tk.SUNKEN, bd=5)
        self.prediction_display.pack(pady=10)
        
        self.prediction_label = tk.Label(
            self.prediction_display,
            text="Prediction: None",
            font=("Segoe UI", 20, "bold"),
            bg='#000000',
            fg='#00ff00',
            width=30,
            height=3
        )
        self.prediction_label.pack(padx=20, pady=20)
        
        # Confidence meter
        confidence_frame = tk.Frame(display_frame, bg='#c62828')
        confidence_frame.pack(pady=10)
        
        tk.Label(
            confidence_frame,
            text="🎯 Confidence Level:",
            bg='#c62828',
            fg='white',
            font=("Segoe UI", 14, "bold")
        ).pack()
        
        self.confidence_var = tk.DoubleVar()
        self.confidence_meter = ttk.Progressbar(
            confidence_frame,
            variable=self.confidence_var,
            maximum=100,
            length=300,
            mode='determinate'
        )
        self.confidence_meter.pack(pady=10)
        
        self.confidence_label = tk.Label(
            confidence_frame,
            text="Confidence: 0%",
            font=("Segoe UI", 12, "bold"),
            bg='#c62828',
            fg='white'
        )
        self.confidence_label.pack(pady=5)
        
        # Recognition controls
        button_frame = tk.Frame(self.recognition_frame, bg='#c62828')
        button_frame.pack(pady=25)
        
        self.start_recognition_btn = tk.Button(
            button_frame,
            text="🎯 START RECOGNITION",
            command=self.start_recognition,
            width=18,
            height=2,
            bg='#E91E63',
            fg='white',
            font=("Segoe UI", 13, "bold"),
            relief=tk.RAISED,
            bd=4
        )
        self.start_recognition_btn.pack(side=tk.LEFT, padx=15)
        
        self.test_model_btn = tk.Button(
            button_frame,
            text="🧪 TEST MODEL",
            command=self.test_model,
            width=15,
            height=2,
            bg='#9C27B0',
            fg='white',
            font=("Segoe UI", 12, "bold"),
            relief=tk.RAISED,
            bd=4
        )
        self.test_model_btn.pack(side=tk.LEFT, padx=15)
        
        # Recognition stats
        stats_frame = tk.Frame(self.recognition_frame, bg='#c62828')
        stats_frame.pack(pady=15, padx=20, fill=tk.X)
        
        self.recognition_stats_label = tk.Label(
            stats_frame,
            text="Recognition Stats: Ready",
            bg='#c62828',
            fg='white',
            font=("Segoe UI", 12)
        )
        self.recognition_stats_label.pack()
    
    def create_status_section(self):
        """Create comprehensive status section."""
        self.status_frame = tk.Frame(self.scrollable_frame, bg='#424242', relief=tk.RAISED, bd=3)
        
        # Status title
        status_title = tk.Label(
            self.status_frame,
            text="📊 System Status & Information",
            font=("Segoe UI", 16, "bold"),
            bg='#424242',
            fg='white'
        )
        status_title.pack(pady=15)
        
        # Status grid
        status_grid = tk.Frame(self.status_frame, bg='#424242')
        status_grid.pack(pady=10, padx=20, fill=tk.X)
        
        # System status
        self.system_status_label = tk.Label(
            status_grid,
            text="🟢 System: Ready",
            font=("Segoe UI", 12, "bold"),
            bg='#424242',
            fg='#4CAF50'
        )
        self.system_status_label.pack(anchor='w', pady=2)
        
        # Data info
        self.data_info_label = tk.Label(
            status_grid,
            text="📊 Data: No data loaded",
            font=("Segoe UI", 12),
            bg='#424242',
            fg='#FFC107'
        )
        self.data_info_label.pack(anchor='w', pady=2)
        
        # Model info
        self.model_status_label = tk.Label(
            status_grid,
            text="🧠 Model: Not trained",
            font=("Segoe UI", 12),
            bg='#424242',
            fg='#FF5722'
        )
        self.model_status_label.pack(anchor='w', pady=2)
        
        # Camera info
        self.camera_info_label = tk.Label(
            status_grid,
            text="📹 Camera: Inactive",
            font=("Segoe UI", 12),
            bg='#424242',
            fg='#9E9E9E'
        )
        self.camera_info_label.pack(anchor='w', pady=2)
        
        # Global progress bar
        self.global_progress_var = tk.DoubleVar()
        self.global_progress_bar = ttk.Progressbar(
            status_grid,
            variable=self.global_progress_var,
            maximum=100,
            length=500
        )
        self.global_progress_bar.pack(pady=15, fill=tk.X)
        
        # Quick actions
        quick_actions_frame = tk.Frame(self.status_frame, bg='#424242')
        quick_actions_frame.pack(pady=15)
        
        tk.Label(
            quick_actions_frame,
            text="⚡ Quick Actions:",
            font=("Segoe UI", 12, "bold"),
            bg='#424242',
            fg='white'
        ).pack()
        
        quick_buttons_frame = tk.Frame(quick_actions_frame, bg='#424242')
        quick_buttons_frame.pack(pady=15)
        
        self.clear_cache_btn = tk.Button(
            quick_buttons_frame,
            text="🗑️ Clear Cache",
            command=self.clear_cache,
            width=14,
            height=1,
            bg='#F44336',
            fg='white',
            font=("Segoe UI", 11, "bold")
        )
        self.clear_cache_btn.pack(side=tk.LEFT, padx=8)
        
        self.refresh_btn = tk.Button(
            quick_buttons_frame,
            text="🔄 Refresh",
            command=self.refresh_interface,
            width=14,
            height=1,
            bg='#2196F3',
            fg='white',
            font=("Segoe UI", 11, "bold")
        )
        self.refresh_btn.pack(side=tk.LEFT, padx=8)
        
        self.help_btn = tk.Button(
            quick_buttons_frame,
            text="❓ Help",
            command=self.show_help,
            width=14,
            height=1,
            bg='#9C27B0',
            fg='white',
            font=("Segoe UI", 11, "bold")
        )
        self.help_btn.pack(side=tk.LEFT, padx=8)
    
    def setup_layout(self):
        """Setup the layout of all widgets with proper spacing."""
        
        # Title section
        self.title_frame.pack(pady=25, fill=tk.X)
        self.title_label.pack()
        self.subtitle_label.pack(pady=8)
        
        # Mode selection
        self.mode_frame.pack(pady=25, padx=25, fill=tk.X)
        
        # Camera section
        self.camera_frame.pack(pady=25, padx=25, fill=tk.X)
        
        # Control panels container
        self.control_container.pack(pady=25, padx=25, fill=tk.X)
        
        # Status section
        self.status_frame.pack(pady=25, padx=25, fill=tk.X)
        
        # Initially show data collection mode
        self.change_mode()
    
    def change_mode(self):
        """Change the application mode and show appropriate panel."""
        mode = self.mode_var.get()
        self.current_mode = mode
        
        # Hide all panels first
        self.data_collection_frame.pack_forget()
        self.training_frame.pack_forget()
        self.recognition_frame.pack_forget()
        
        # Show appropriate panel
        if mode == "data_collection":
            self.data_collection_frame.pack(pady=25, fill=tk.X)
            self.update_system_status("Mode: Data Collection - Ready to capture hand signs")
        elif mode == "training":
            self.training_frame.pack(pady=25, fill=tk.X)
            self.update_system_status("Mode: Training - Ready to train neural network")
        elif mode == "recognition":
            self.recognition_frame.pack(pady=25, fill=tk.X)
            self.update_system_status("Mode: Recognition - Ready for real-time detection")
        
        # Update scroll region
        self.root.after(100, lambda: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all")))
    
    def toggle_camera(self):
        """Toggle camera on/off with enhanced feedback."""
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera with enhanced error handling and feedback."""
        try:
            self.update_system_status("Starting camera...")
            
            if self.data_collector.start_camera():
                self.camera_active = True
                
                # Update button appearance
                self.start_camera_btn.configure(
                    text="⏹️ STOP CAMERA",
                    bg='#F44336',
                    activebackground='#d32f2f'
                )
                
                # Update status labels
                self.camera_status_label.configure(
                    text="Status: Camera Active ✅",
                    fg='#4CAF50'
                )
                self.camera_info_label.configure(
                    text="📹 Camera: Active - Live feed running",
                    fg='#4CAF50'
                )
                
                self.update_system_status("Camera started successfully - Live hand tracking active")
                self.start_video_update()
                
            else:
                messagebox.showerror("Camera Error", "Failed to start camera. Please check if camera is available and not being used by another application.")
                self.update_system_status("❌ Camera failed to start")
                
        except Exception as e:
            messagebox.showerror("Camera Error", f"Error starting camera: {str(e)}")
            self.update_system_status(f"❌ Camera error: {str(e)}")
    
    def stop_camera(self):
        """Stop camera with proper cleanup."""
        self.camera_active = False
        self.stop_update = True
        self.data_collector.stop_camera()
        
        # Update button appearance
        self.start_camera_btn.configure(
            text="🎥 START CAMERA",
            bg='#4CAF50',
            activebackground='#45a049'
        )
        
        # Update status labels
        self.camera_status_label.configure(
            text="Status: Camera Stopped",
            fg='#FF5722'
        )
        self.camera_info_label.configure(
            text="📹 Camera: Inactive",
            fg='#9E9E9E'
        )
        
        # Reset video display
        self.video_label.configure(
            image="",
            text="📹 Camera Ready\n\n🟢 Click the GREEN button below to start camera\n\nCamera will show live hand tracking with landmarks"
        )
        self.video_label.image = None
        
        self.update_system_status("Camera stopped")
    
    def start_video_update(self):
        """Start updating video feed with enhanced display."""
        self.stop_update = False
        self.update_video_feed()
    
    def update_video_feed(self):
        """Update video feed with hand landmarks and modern styling."""
        if not self.camera_active or self.stop_update:
            return
        
        try:
            # Get frame and landmarks from data collector
            frame_data = self.data_collector.get_frame()
            
            # Handle the tuple return from get_frame()
            if isinstance(frame_data, tuple):
                frame, landmarks = frame_data
            else:
                frame = frame_data
                landmarks = None
            
            if frame is not None:
                # Convert and resize frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize for display
                height, width = frame_rgb.shape[:2]
                max_width = 600
                max_height = 450
                
                if width > max_width or height > max_height:
                    scale = min(max_width/width, max_height/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
                
                # Convert to PhotoImage
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image)
                
                # Update display
                self.video_label.configure(image=photo, text="")
                self.video_label.image = photo
                
                # Update recognition if in recognition mode
                if self.current_mode == "recognition" and hasattr(self.ml_model, 'model') and self.ml_model.model:
                    self.update_recognition()
        
        except Exception as e:
            print(f"Error updating video: {e}")
        
        # Schedule next update
        if self.camera_active and not self.stop_update:
            self.root.after(30, self.update_video_feed)
    
    def start_data_collection(self):
        """Enhanced data collection with progress tracking."""
        if not self.camera_active:
            messagebox.showwarning("Camera Required", "Please start the camera first!")
            return
        
        label = self.label_entry.get().strip()
        if not label or "Enter sign name" in label:
            messagebox.showwarning("Label Required", "Please enter a valid hand sign label!")
            return
        
        try:
            duration = int(self.duration_var.get())
            target_samples = int(self.samples_var.get())
            
            if duration <= 0 or target_samples <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showwarning("Invalid Parameters", "Please enter valid duration and sample count!")
            return
        
        # Start collection in thread
        def collect_data():
            try:
                self.collect_btn.configure(state='disabled')
                self.collection_status_label.configure(text=f"Preparing to collect '{label}'...")
                
                # Countdown
                for i in range(3, 0, -1):
                    self.collection_status_label.configure(text=f"Starting in {i}...")
                    self.collection_progress_var.set((4-i) * 25)
                    time.sleep(1)
                
                self.collection_status_label.configure(text=f"Collecting '{label}' data...")
                self.collection_progress_var.set(0)
                
                # Collect data with progress updates
                collected_count = 0
                start_time = time.time()
                
                while time.time() - start_time < duration:
                    landmarks = self.data_collector.get_current_landmarks()
                    if landmarks is not None:
                        collected_count += 1
                        self.data_collector.add_sample(label, landmarks)
                    
                    # Update progress
                    elapsed = time.time() - start_time
                    progress = (elapsed / duration) * 100
                    self.collection_progress_var.set(progress)
                    
                    self.collection_status_label.configure(
                        text=f"Collecting '{label}': {collected_count} samples ({progress:.1f}%)"
                    )
                    
                    time.sleep(0.1)
                
                self.collection_progress_var.set(100)
                self.collection_status_label.configure(
                    text=f"✅ Collection complete! {collected_count} samples for '{label}'"
                )
                
                self.update_data_info()
                self.update_system_status(f"Data collection complete: {collected_count} samples for '{label}'")
                
            except Exception as e:
                self.collection_status_label.configure(text=f"❌ Collection error: {str(e)}")
                messagebox.showerror("Collection Error", f"Data collection failed: {str(e)}")
            finally:
                self.collect_btn.configure(state='normal')
        
        # Start collection thread
        if self.collection_thread and self.collection_thread.is_alive():
            return
        
        self.collection_thread = threading.Thread(target=collect_data)
        self.collection_thread.daemon = True
        self.collection_thread.start()
    
    def save_collected_data(self):
        """Save collected data with enhanced file dialog."""
        try:
            if not self.data_collector.has_data():
                messagebox.showwarning("No Data", "No data to save! Please collect some data first.")
                return
            
            filename = filedialog.asksaveasfilename(
                title="Save Hand Sign Data",
                defaultextension=".npz",
                filetypes=[
                    ("NumPy Compressed", "*.npz"),
                    ("NumPy Array", "*.npy"),
                    ("All files", "*.*")
                ],
                initialdir="./data"
            )
            
            if filename:
                self.data_collector.save_data(filename)
                self.update_system_status(f"✅ Data saved to {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Data saved successfully to:\n{filename}")
        
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save data: {str(e)}")
    
    def load_data(self):
        """Load data with enhanced file dialog."""
        try:
            filename = filedialog.askopenfilename(
                title="Load Hand Sign Data",
                filetypes=[
                    ("NumPy Compressed", "*.npz"),
                    ("NumPy Array", "*.npy"),
                    ("All files", "*.*")
                ],
                initialdir="./data"
            )
            
            if filename:
                self.data_collector.load_data(filename)
                self.update_data_info()
                self.update_system_status(f"✅ Data loaded from {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Data loaded successfully from:\n{filename}")
        
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load data: {str(e)}")
    
    def start_training(self):
        """Enhanced model training with progress tracking."""
        if not self.data_collector.has_data():
            messagebox.showwarning("No Data", "No training data available! Please collect or load data first.")
            return
        
        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            learning_rate = float(self.learning_rate_var.get())
            validation_split = float(self.validation_split_var.get())
            
            if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or validation_split <= 0:
                raise ValueError()
        
        except ValueError:
            messagebox.showwarning("Invalid Parameters", "Please enter valid training parameters!")
            return
        
        # Start training in thread
        def train_model():
            try:
                self.train_btn.configure(state='disabled')
                self.training_status_label.configure(text="Preparing training data...")
                self.training_progress_var.set(10)
                
                # Get training data
                X, y, labels = self.data_collector.get_training_data()
                
                self.training_status_label.configure(text=f"Training with {len(X)} samples...")
                self.training_progress_var.set(20)
                
                # Train model with progress updates
                self.training_status_label.configure(text="Training neural network...")
                
                # Simulate training progress (in real implementation, you'd get this from the model)
                for epoch in range(epochs):
                    progress = 20 + (epoch / epochs) * 70
                    self.training_progress_var.set(progress)
                    self.training_status_label.configure(
                        text=f"Training: Epoch {epoch+1}/{epochs} ({progress:.1f}%)"
                    )
                    
                    if epoch == 0:  # Actually train on first iteration
                        history = self.ml_model.train(X, y, labels, epochs=epochs, batch_size=batch_size)
                    
                    time.sleep(0.1)  # Visual feedback
                
                self.training_progress_var.set(100)
                self.training_status_label.configure(text="✅ Training completed successfully!")
                
                # Update model info
                self.model_info_label.configure(text=f"Model: Trained on {len(set(y))} classes")
                self.model_status_label.configure(
                    text="🧠 Model: Trained and ready",
                    fg='#4CAF50'
                )
                
                self.update_system_status("✅ Model training completed successfully")
                messagebox.showinfo("Training Complete", "Model training completed successfully!")
                
            except Exception as e:
                self.training_status_label.configure(text=f"❌ Training error: {str(e)}")
                messagebox.showerror("Training Error", f"Training failed: {str(e)}")
            finally:
                self.train_btn.configure(state='normal')
        
        # Start training thread
        if self.training_thread and self.training_thread.is_alive():
            return
        
        self.training_thread = threading.Thread(target=train_model)
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def save_model(self):
        """Save trained model with enhanced dialog."""
        if not hasattr(self.ml_model, 'model') or self.ml_model.model is None:
            messagebox.showwarning("No Model", "No trained model to save!")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                title="Save Trained Model",
                defaultextension=".h5",
                filetypes=[
                    ("Keras Model", "*.h5"),
                    ("TensorFlow SavedModel", "*.pb"),
                    ("All files", "*.*")
                ],
                initialdir="./models"
            )
            
            if filename:
                self.ml_model.save_model(filename)
                self.update_system_status(f"✅ Model saved to {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Model saved successfully to:\n{filename}")
        
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        """Load trained model with enhanced dialog."""
        try:
            filename = filedialog.askopenfilename(
                title="Load Trained Model",
                filetypes=[
                    ("Keras Model", "*.h5"),
                    ("TensorFlow SavedModel", "*.pb"),
                    ("All files", "*.*")
                ],
                initialdir="./models"
            )
            
            if filename:
                self.ml_model.load_model(filename)
                self.model_status_label.configure(
                    text="🧠 Model: Loaded and ready",
                    fg='#4CAF50'
                )
                self.update_system_status(f"✅ Model loaded from {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Model loaded successfully from:\n{filename}")
        
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load model: {str(e)}")
    
    def start_recognition(self):
        """Start real-time recognition with enhanced feedback."""
        if not self.camera_active:
            messagebox.showwarning("Camera Required", "Please start the camera first!")
            return
        
        if not hasattr(self.ml_model, 'model') or self.ml_model.model is None:
            messagebox.showwarning("No Model", "No trained model available! Please train or load a model first.")
            return
        
        self.recognition_stats_label.configure(text="🎯 Recognition: Active - Analyzing hand signs")
        self.update_system_status("Real-time recognition started!")
    
    def update_recognition(self):
        """Update recognition results with enhanced display."""
        try:
            landmarks = self.data_collector.get_current_landmarks()
            
            if landmarks is not None:
                prediction, confidence = self.ml_model.predict(landmarks)
                
                # Update prediction display
                self.prediction_label.configure(text=f"✋ {prediction}")
                
                # Update confidence
                self.confidence_var.set(confidence)
                self.confidence_label.configure(text=f"Confidence: {confidence:.1f}%")
                
                # Color coding
                if confidence > 80:
                    color = '#00ff00'  # Green
                    self.prediction_label.configure(fg=color)
                elif confidence > 60:
                    color = '#ffff00'  # Yellow
                    self.prediction_label.configure(fg=color)
                else:
                    color = '#ff6600'  # Orange
                    self.prediction_label.configure(fg=color)
                
                # Update stats
                self.recognition_stats_label.configure(
                    text=f"🎯 Recognition: {prediction} ({confidence:.1f}% confidence)"
                )
        
        except Exception as e:
            print(f"Recognition error: {e}")
    
    def test_model(self):
        """Test model performance with detailed results."""
        if not hasattr(self.ml_model, 'model') or self.ml_model.model is None:
            messagebox.showwarning("No Model", "No trained model available!")
            return
        
        if not self.data_collector.has_data():
            messagebox.showwarning("No Data", "No test data available!")
            return
        
        try:
            self.update_system_status("Testing model performance...")
            
            X, y, labels = self.data_collector.get_training_data()
            accuracy = self.ml_model.evaluate(X, y)
            
            result_text = f"Model Performance Test Results:\n\n"
            result_text += f"Overall Accuracy: {accuracy:.2f}%\n"
            result_text += f"Test Samples: {len(X)}\n"
            result_text += f"Classes: {len(set(y))}\n"
            result_text += f"Unique Signs: {', '.join(labels)}"
            
            self.update_system_status(f"✅ Model test complete - Accuracy: {accuracy:.2f}%")
            messagebox.showinfo("Model Test Results", result_text)
        
        except Exception as e:
            messagebox.showerror("Test Error", f"Model testing failed: {str(e)}")
    
    def clear_cache(self):
        """Clear all cached data with confirmation."""
        result = messagebox.askyesno(
            "Clear Cache", 
            "This will clear all collected data and reset the system.\n\nAre you sure you want to continue?"
        )
        
        if result:
            self.data_collector.clear_data()
            self.update_data_info()
            self.update_system_status("✅ Cache cleared - System reset")
            messagebox.showinfo("Success", "Cache cleared successfully!")
    
    def refresh_interface(self):
        """Refresh the entire interface."""
        self.update_data_info()
        self.update_system_status("🔄 Interface refreshed")
        
        # Reset progress bars
        self.collection_progress_var.set(0)
        self.training_progress_var.set(0)
        self.confidence_var.set(0)
        self.global_progress_var.set(0)
        
        messagebox.showinfo("Refresh", "Interface refreshed successfully!")
    
    def show_help(self):
        """Show comprehensive help dialog."""
        help_text = """
🤲 Hand Sign Recognition System - Help

🎯 GETTING STARTED:
1. Start the camera using the green button
2. Switch to Data Collection mode
3. Collect hand sign data for different gestures
4. Train the neural network model
5. Use real-time recognition

📊 DATA COLLECTION:
• Enter a descriptive label for your hand sign
• Set duration (recommended: 10+ seconds)
• Keep your hand steady and visible
• Collect multiple samples for better accuracy

🧠 TRAINING:
• Use 50+ epochs for better results
• Batch size 32 works well for most cases
• Higher learning rates train faster but may be unstable
• 20% validation split is recommended

🎯 RECOGNITION:
• Ensure good lighting
• Keep hand in camera view
• Higher confidence = more accurate prediction
• Green = High confidence, Yellow = Medium, Orange = Low

⚙️ TIPS:
• Collect data in good lighting
• Use consistent hand positions
• Train with diverse samples
• Test your model before use

For technical support, check the README.md file.
        """
        
        # Create help window
        help_window = tk.Toplevel(self.root)
        help_window.title("Help - Hand Sign Recognition")
        help_window.geometry("600x500")
        help_window.configure(bg='#2d2d2d')
        
        # Add scrollable text
        text_frame = tk.Frame(help_window, bg='#2d2d2d')
        text_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            bg='#1a1a1a',
            fg='white',
            relief=tk.FLAT,
            padx=20,
            pady=20
        )
        
        scrollbar_help = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar_help.set)
        
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar_help.pack(side="right", fill="y")
        
        text_widget.insert(tk.END, help_text)
        text_widget.configure(state='disabled')
    
    def update_system_status(self, message):
        """Update system status with timestamp."""
        if hasattr(self, 'system_status_label'):
            self.system_status_label.configure(text=f"🟢 System: {message}")
            self.root.update_idletasks()
    
    def update_data_info(self):
        """Update data information display."""
        if hasattr(self, 'data_info_label'):
            if self.data_collector.has_data():
                total_samples = len(self.data_collector.landmarks_data)
                unique_labels = len(set(self.data_collector.labels_data))
                self.data_info_label.configure(
                    text=f"📊 Data: {total_samples} samples, {unique_labels} unique signs",
                    fg='#4CAF50'
                )
            else:
                self.data_info_label.configure(
                    text="📊 Data: No data loaded",
                    fg='#FFC107'
                )
    
    def on_closing(self):
        """Handle application closing with proper cleanup."""
        if self.camera_active:
            self.stop_camera()
        
        # Stop any running threads
        if self.collection_thread and self.collection_thread.is_alive():
            self.stop_update = True
        
        if self.training_thread and self.training_thread.is_alive():
            self.stop_update = True
        
        self.root.destroy()
    
    def run(self):
        """Start the complete GUI application."""
        print("🚀 Starting Complete Hand Sign Recognition System...")
        print("✅ All features integrated: Data Collection, Training, Recognition")
        print("🖱️  Use mouse wheel to scroll through all panels")
        print("📱 Responsive design with modern styling")
        
        self.update_system_status("Application started - All systems ready")
        self.root.mainloop()

def main():
    """Main function to run the complete application."""
    app = CompleteHandSignGUI()
    app.run()

if __name__ == "__main__":
    main()
