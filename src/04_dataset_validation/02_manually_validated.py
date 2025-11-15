import argparse
import os
import sys
import pandas as pd
import re
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import copy


class ManualValidationApp:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.year = self.extract_year_from_filename(csv_path)
        
        if not self.year:
            raise ValueError(f"Could not extract year from filename: {csv_path}")
        
        # Setup directories
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(self.script_dir, '../../'))
        self.png_dir = os.path.join(
            self.project_root,
            f'data/01_dataset_construction/csvs/Patentamt_{self.year}/page_by_page/PNG'
        )
        self.output_csv_dir = os.path.join(self.project_root, 'data/04_dataset_validation/02_manually_validated/csv')
        self.output_xlsx_dir = os.path.join(self.project_root, 'data/04_dataset_validation/02_manually_validated/xlsx')
        self.output_log_dir = os.path.join(self.project_root, 'data/04_dataset_validation/02_manually_validated/logs')
        
        # Create output directories
        os.makedirs(self.output_csv_dir, exist_ok=True)
        os.makedirs(self.output_xlsx_dir, exist_ok=True)
        os.makedirs(self.output_log_dir, exist_ok=True)
        
        # Setup logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.output_log_dir, f'manually_validated_log_{self.year}_{timestamp}.txt')
        
        # Load data
        self.df = None
        self.tasks = []
        self.current_task_idx = 0
        self.undo_stack = []
        self.zoom_level = 1.0  # Will be calculated to fit screen width
        self.original_image = None
        self.photo_image = None
        
        # Session tracking
        self.session_start_time = datetime.now()
        self.current_task_category = None  # Track category changes
        self.tasks_since_last_save = 0  # Track tasks for batch saving
        
        # Track original values for logging (to avoid keystroke-by-keystroke logs)
        self.original_entry = None
        self.original_patent_id = None
        
        # Load and prepare data
        self.load_data()
        self.parse_tasks()
        
        # Log session start
        self.log(f"SESSION START - File: Patentamt_{self.year}")
        self.log(f"Total tasks to review: {len(self.tasks)}")
        
        # Setup GUI
        self.root = tk.Tk()
        self.root.title(f"Manual Validation - Patentamt {self.year}")
        self.root.state('zoomed')  # Maximize window
        
        self.setup_gui()
        self.setup_keyboard_shortcuts()
        
        # Display first task
        if self.tasks:
            self.display_task()
        else:
            messagebox.showinfo("Complete", "No validation issues found in this file!")
            self.root.destroy()
    
    def extract_year_from_filename(self, filename):
        """Extract year from filename."""
        basename = os.path.basename(filename)
        match = re.search(r'Patentamt_(\d{4})_', basename)
        return match.group(1) if match else None
    
    def load_data(self):
        """Load CSV and add manually_validated column. Check for existing work-in-progress file."""
        # Check if manually_validated CSV already exists (resuming)
        output_csv = os.path.join(self.output_csv_dir, f"Patentamt_{self.year}_manually_validated.csv")
        
        if os.path.exists(output_csv):
            self.log("Found existing manually_validated file - RESUMING previous session")
            print(f"✓ Resuming from previous session: {output_csv}")
            self.df = pd.read_csv(output_csv, dtype=str)
            
            # Count remaining tasks
            remaining = (self.df['manually_validated'] == '0').sum()
            self.log(f"Resuming: {remaining} tasks remaining")
            print(f"✓ {remaining} validation tasks remaining")
        else:
            self.log("Starting fresh - loading from validated CSV")
            print(f"✓ Starting new validation session")
            self.df = pd.read_csv(self.csv_path, dtype=str)
            
            # Add manually_validated column if it doesn't exist
            if 'manually_validated' not in self.df.columns:
                self.df['manually_validated'] = self.df['validation_notes'].apply(
                    lambda x: '1' if str(x).strip() == 'Valid' else '0'
                )
                self.log(f"Added manually_validated column: {(self.df['manually_validated'] == '1').sum()} valid entries")
        
        # Add deletion flag column if it doesn't exist
        if 'marked_for_deletion' not in self.df.columns:
            self.df['marked_for_deletion'] = '0'
        
        # Add manually_edited column if it doesn't exist (tracks actual changes)
        if 'manually_edited' not in self.df.columns:
            self.df['manually_edited'] = '0'
            self.log("Added manually_edited column (initialized to 0)")
        
        # Add old_patent_id column if it doesn't exist (stores original patent_id for tracking)
        if 'old_patent_id' not in self.df.columns:
            self.df['old_patent_id'] = self.df['patent_id'].copy()
            self.log("Added old_patent_id column (copied from patent_id)")
    
    def parse_tasks(self):
        """Parse validation_notes to create task queue. Only include unvalidated rows."""
        self.log("Parsing validation issues...")
        
        # Define patterns for the three issue types
        duplicate_pattern = r'Duplicate patent_id (\d+)'
        less_than_pattern = r'patent_id \d+ < start'
        greater_than_pattern = r'patent_id \d+ > end'
        
        # Group duplicates by patent_id number
        duplicate_groups = {}
        less_than = []
        greater_than = []
        
        for idx, row in self.df.iterrows():
            # Only process rows that haven't been manually validated yet
            if str(row['manually_validated']).strip() != '0':
                continue
                
            validation_notes = str(row['validation_notes']).strip()
            
            # Check for duplicates and group by patent_id
            dup_match = re.search(duplicate_pattern, validation_notes)
            if dup_match:
                patent_id = int(dup_match.group(1))
                if patent_id not in duplicate_groups:
                    duplicate_groups[patent_id] = []
                duplicate_groups[patent_id].append(idx)
            elif re.search(less_than_pattern, validation_notes):
                less_than.append(idx)
            elif re.search(greater_than_pattern, validation_notes):
                greater_than.append(idx)
        
        # Sort duplicates by patent_id, then flatten
        duplicates = []
        for patent_id in sorted(duplicate_groups.keys()):
            duplicates.extend(sorted(duplicate_groups[patent_id]))
        
        # Sort other categories by row index
        less_than.sort()
        greater_than.sort()
        
        # Combine in order: duplicates, < start, > end
        self.tasks = duplicates + less_than + greater_than
        
        self.log(f"Found {len(duplicates)} duplicate issues to validate (grouped by patent_id)")
        self.log(f"Found {len(less_than)} '< start' issues to validate")
        self.log(f"Found {len(greater_than)} '> end' issues to validate")
    
    def setup_gui(self):
        """Build the tkinter interface."""
        # Top frame - Progress info and progress bar
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        # Progress info (task, percentage, time) - bigger and bolder
        progress_info_frame = ttk.Frame(top_frame)
        progress_info_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.progress_label = ttk.Label(progress_info_frame, text="", font=('Arial', 13, 'bold'))
        self.progress_label.pack(side=tk.LEFT)
        
        self.session_timer_label = ttk.Label(progress_info_frame, text="Session: 00:00", font=('Arial', 13, 'bold'))
        self.session_timer_label.pack(side=tk.RIGHT, padx=10)
        
        self.progress_percentage_label = ttk.Label(progress_info_frame, text="0%", font=('Arial', 13, 'bold'))
        self.progress_percentage_label.pack(side=tk.RIGHT)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(top_frame, mode='determinate', length=400)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Start timer update
        self.update_session_timer()
        
        # Image frame with zoom controls
        image_frame = ttk.Frame(self.root, padding="10")
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Zoom controls
        zoom_control_frame = ttk.Frame(image_frame)
        zoom_control_frame.pack()
        
        ttk.Button(zoom_control_frame, text="-", command=self.zoom_out, width=3).pack(side=tk.LEFT, padx=5)
        self.zoom_label = ttk.Label(zoom_control_frame, text="100%", width=8)
        self.zoom_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_control_frame, text="+", command=self.zoom_in, width=3).pack(side=tk.LEFT, padx=5)
        
        self.page_label = ttk.Label(zoom_control_frame, text="", font=('Arial', 10))
        self.page_label.pack(side=tk.LEFT, padx=20)
        
        # Scrollable canvas for image (with both scrollbars)
        canvas_frame = ttk.Frame(image_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.image_canvas = tk.Canvas(canvas_frame, bg='gray')
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        
        self.image_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mouse wheel for scrolling
        self.image_canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows/MacOS
        self.image_canvas.bind("<Button-4>", self.on_mouse_wheel)  # Linux scroll up
        self.image_canvas.bind("<Button-5>", self.on_mouse_wheel)  # Linux scroll down
        
        # Entry fields frame
        fields_frame = ttk.Frame(self.root, padding="10")
        fields_frame.pack(fill=tk.X)
        
        # Entry position on page (bold and prominent) - moved to left
        position_frame = ttk.Frame(fields_frame)
        position_frame.pack(fill=tk.X, pady=5)
        
        self.entry_position_label = tk.Label(
            position_frame, 
            text="", 
            font=('Arial', 14, 'bold'),
            fg='#0066cc',  # Blue color for visibility
            bg='#f0f0f0',  # Light gray background
            padx=10,
            pady=2,
            relief=tk.RAISED,
            borderwidth=2
        )
        self.entry_position_label.pack(side=tk.LEFT)
        
        # Entry text area (no label)
        self.entry_text = scrolledtext.ScrolledText(fields_frame, height=4, wrap=tk.WORD, font=('Arial', 10))
        self.entry_text.pack(fill=tk.X, pady=5)
        self.entry_text.bind('<Tab>', self.on_entry_tab)
        self.entry_text.bind('<KeyRelease>', self.on_field_change)
        
        # Patent ID with deletion indicator
        patent_id_frame = ttk.Frame(fields_frame)
        patent_id_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(patent_id_frame, text="Patent ID:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        self.patent_id_entry = ttk.Entry(patent_id_frame, font=('Arial', 10), width=20)
        self.patent_id_entry.pack(side=tk.LEFT, padx=5)
        self.patent_id_entry.bind('<Tab>', self.on_patent_id_tab)
        self.patent_id_entry.bind('<KeyRelease>', self.on_field_change)
        
        # Deletion indicator (shown when row marked for deletion)
        self.deletion_indicator = tk.Label(
            patent_id_frame,
            text="🗑️ MARKED FOR DELETION",
            font=('Arial', 10, 'bold'),
            fg='red',
            bg='#ffe6e6',
            padx=10,
            pady=3,
            relief=tk.RAISED,
            borderwidth=2
        )
        # Initially hidden
        # self.deletion_indicator.pack(side=tk.LEFT, padx=10)
        
        # Bottom buttons
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.pack(fill=tk.X)
        
        button_style_frame = ttk.Frame(button_frame)
        button_style_frame.pack()
        
        ttk.Button(button_style_frame, text="Back (⌘A)", command=self.go_back, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_style_frame, text="Next (⌘S)", command=self.next_task, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_style_frame, text="Delete (⌘D)", command=self.toggle_delete, width=15).pack(side=tk.LEFT, padx=5)
        
        # Status bar for subtle feedback
        status_frame = ttk.Frame(self.root, relief=tk.SUNKEN, padding="2")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(status_frame, text="Ready", font=('Arial', 9))
        self.status_label.pack(side=tk.LEFT)
    
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts: Cmd+A=Back, Cmd+S=Next, Cmd+D=Delete."""
        self.root.bind('<Command-a>', lambda e: self.go_back())
        self.root.bind('<Command-A>', lambda e: self.go_back())
        self.root.bind('<Command-s>', lambda e: self.next_task())
        self.root.bind('<Command-S>', lambda e: self.next_task())
        self.root.bind('<Command-d>', lambda e: self.toggle_delete())
        self.root.bind('<Command-D>', lambda e: self.toggle_delete())
    
    def display_task(self):
        """Display current task in GUI (optimized for smooth loading)."""
        if self.current_task_idx >= len(self.tasks):
            # All tasks completed - show message and save final files
            self.complete_validation()
            return
        
        # Update UI asynchronously for smoother feel
        self.root.update_idletasks()
        
        # Get current row
        row_idx = self.tasks[self.current_task_idx]
        row = self.df.iloc[row_idx]
        
        # Check for task category change and show warning
        current_category = self.get_task_category(row['validation_notes'])
        if self.current_task_category is not None and self.current_task_category != current_category:
            category_name = {
                'duplicate': 'Duplicate patent IDs',
                'less_than': 'Patent IDs below valid range (< start)',
                'greater_than': 'Patent IDs above valid range (> end)'
            }
            messagebox.showinfo(
                "Task Category Changed",
                f"Now starting: {category_name.get(current_category, 'New category')}"
            )
            self.log(f"Task category changed from '{self.current_task_category}' to '{current_category}'")
        self.current_task_category = current_category
        
        # Update progress bar and percentage
        progress_percentage = int((self.current_task_idx / len(self.tasks)) * 100)
        self.progress_bar['maximum'] = len(self.tasks)
        self.progress_bar['value'] = self.current_task_idx
        self.progress_percentage_label.config(text=f"{progress_percentage}%")
        
        # Update progress text
        progress_text = f"Task {self.current_task_idx + 1} / {len(self.tasks)} | {row['validation_notes']}"
        self.progress_label.config(text=progress_text)
        
        # Calculate entry position on this page
        current_page = str(row['page'])
        entries_on_page = self.df[self.df['page'] == current_page]
        # Get indices in their actual DataFrame order (top to bottom) - DO NOT SORT
        indices_in_order = entries_on_page.index.tolist()
        entry_position = indices_in_order.index(row_idx) + 1  # +1 for 1-based indexing
        total_on_page = len(entries_on_page)
        
        # Update entry position label (bold and prominent)
        self.entry_position_label.config(text=f"{entry_position}/{total_on_page}")
        
        # Load and display image
        page_num = str(row['page']).zfill(4)
        self.page_label.config(text=f"Page: page_{page_num}.png")
        image_path = os.path.join(self.png_dir, f"page_{page_num}.png")
        
        if os.path.exists(image_path):
            # Show loading feedback
            self.show_status("📄 Loading image...", 500)
            self.original_image = Image.open(image_path)
            # Calculate zoom to fit screen width
            self.calculate_fit_zoom()
            self.update_image_display()
        else:
            self.log(f"WARNING: Image not found: {image_path}")
            self.show_status(f"⚠ Image not found: page_{page_num}.png", 3000)
        
        # Populate fields
        self.entry_text.delete('1.0', tk.END)
        self.entry_text.insert('1.0', str(row['entry']))
        
        self.patent_id_entry.delete(0, tk.END)
        self.patent_id_entry.insert(0, str(row['patent_id']))
        
        # Store original values for change tracking (to avoid logging every keystroke)
        self.original_entry = str(row['entry'])
        self.original_patent_id = str(row['patent_id'])
        
        # Show/hide deletion indicator based on marked_for_deletion status
        if str(row['marked_for_deletion']) == '1':
            self.deletion_indicator.pack(side=tk.LEFT, padx=10)
        else:
            self.deletion_indicator.pack_forget()
        
        # Focus on entry text
        self.entry_text.focus_set()
    
    def get_task_category(self, validation_notes):
        """Determine task category from validation notes."""
        if 'Duplicate patent_id' in validation_notes:
            return 'duplicate'
        elif '< start' in validation_notes:
            return 'less_than'
        elif '> end' in validation_notes:
            return 'greater_than'
        return 'other'
    
    def calculate_fit_zoom(self):
        """Calculate zoom level to fit image width to canvas width (reduced size)."""
        if self.original_image is None:
            self.zoom_level = 0.5
            return
        
        # Get canvas width (use update to ensure proper dimensions)
        self.image_canvas.update_idletasks()
        canvas_width = self.image_canvas.winfo_width()
        
        # If canvas not yet rendered, use default
        if canvas_width < 100:
            canvas_width = 1200  # Reasonable default
        
        # Calculate zoom to fit width with some margin (reduced by 30%)
        image_width = self.original_image.width
        margin = 40  # Leave some margin
        fit_zoom = (canvas_width - margin) / image_width
        
        # Reduce the fit zoom by 30% to make images narrower
        fit_zoom = fit_zoom * 0.7
        
        # Cap zoom levels for usability
        self.zoom_level = max(0.25, min(fit_zoom, 1.5))
    
    def update_image_display(self):
        """Update image display with current zoom level and center it (optimized)."""
        if self.original_image is None:
            return
        
        # Calculate new size
        new_width = int(self.original_image.width * self.zoom_level)
        new_height = int(self.original_image.height * self.zoom_level)
        
        # Use BILINEAR for faster rendering (LANCZOS is higher quality but slower)
        resized_image = self.original_image.resize((new_width, new_height), Image.BILINEAR)
        self.photo_image = ImageTk.PhotoImage(resized_image)
        
        # Update canvas idletasks for accurate dimensions
        self.image_canvas.update_idletasks()
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        # Calculate centered position
        x_center = max(0, (canvas_width - new_width) // 2)
        y_center = max(0, (canvas_height - new_height) // 2)
        
        # Update canvas efficiently
        self.image_canvas.delete("all")
        self.image_canvas.create_image(x_center, y_center, anchor=tk.NW, image=self.photo_image)
        
        # Set scroll region (both horizontal and vertical)
        scroll_width = max(new_width + x_center * 2, canvas_width)
        scroll_height = max(new_height + y_center * 2, canvas_height)
        self.image_canvas.config(scrollregion=(0, 0, scroll_width, scroll_height))
        
        # Update zoom label
        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
    
    def zoom_in(self):
        """Zoom in the image."""
        self.zoom_level = min(self.zoom_level + 0.25, 5.0)
        self.update_image_display()
    
    def zoom_out(self):
        """Zoom out the image."""
        self.zoom_level = max(self.zoom_level - 0.25, 0.25)
        self.update_image_display()
    
    def on_entry_tab(self, event):
        """Handle Tab key in entry text - move to patent_id field."""
        self.patent_id_entry.focus_set()
        return 'break'
    
    def on_patent_id_tab(self, event):
        """Handle Tab key in patent_id field - move to next task."""
        self.next_task()
        return 'break'
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel scrolling with smoother increments."""
        # Use smaller scroll units for smoother scrolling
        scroll_amount = 3  # Increased from 1 for smoother feel
        
        if event.num == 5 or event.delta < 0:
            # Scroll down
            self.image_canvas.yview_scroll(scroll_amount, "units")
        elif event.num == 4 or event.delta > 0:
            # Scroll up
            self.image_canvas.yview_scroll(-scroll_amount, "units")
        return 'break'
    
    def update_session_timer(self):
        """Update the session timer display."""
        elapsed = datetime.now() - self.session_start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            time_str = f"Session: {hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            time_str = f"Session: {minutes:02d}:{seconds:02d}"
        
        self.session_timer_label.config(text=time_str)
        
        # Schedule next update in 1 second
        self.root.after(1000, self.update_session_timer)
    
    def on_field_change(self, event):
        """Mark entry as modified (save happens every 10 tasks now)."""
        # Just save to DataFrame, don't write to file yet
        self.save_current_entry()
    
    def save_current_entry(self, log_changes=False):
        """Save current entry data to DataFrame (always reads from UI fields).
        
        Args:
            log_changes: If True, log changes to the log file (only when moving to next/back task)
        """
        if self.current_task_idx >= len(self.tasks):
            return
        
        row_idx = self.tasks[self.current_task_idx]
        old_row = self.df.iloc[row_idx].copy()
        
        # Always get current values from UI fields (not from stored state)
        # This ensures we capture whatever the user typed, even without pressing Enter
        new_entry = self.entry_text.get('1.0', tk.END).strip()
        new_patent_id = self.patent_id_entry.get().strip()
        
        # Check if values changed compared to ORIGINAL values (not intermediate states)
        entry_changed = self.original_entry != new_entry
        patent_id_changed = self.original_patent_id != new_patent_id
        
        # Update entry if changed
        if new_entry != str(old_row['entry']):
            self.df.at[row_idx, 'entry'] = new_entry
        
        # Update patent_id AND patent_id_cleaned if changed
        if new_patent_id != str(old_row['patent_id']):
            self.df.at[row_idx, 'patent_id'] = new_patent_id
            # Also update patent_id_cleaned to match
            self.df.at[row_idx, 'patent_id_cleaned'] = new_patent_id
        
        # Log changes only when requested (e.g., when moving to next task)
        if log_changes:
            if entry_changed:
                self.log(f"Row {row_idx} (id={old_row['id']}): entry changed from '{self.original_entry}' to '{new_entry}'")
            
            if patent_id_changed:
                self.log(f"Row {row_idx} (id={old_row['id']}): patent_id changed from '{self.original_patent_id}' to '{new_patent_id}'")
        
        # Mark as manually validated if any field was changed
        if (entry_changed or patent_id_changed) and str(old_row['manually_validated']) == '0':
            self.df.at[row_idx, 'manually_validated'] = '1'
            if log_changes:
                self.log(f"Row {row_idx} (id={old_row['id']}): manually_validated set to 1")
        
        # Mark as manually edited if any field was changed (tracks actual edits)
        if entry_changed or patent_id_changed:
            self.df.at[row_idx, 'manually_edited'] = '1'
            if log_changes:
                self.log(f"Row {row_idx} (id={old_row['id']}): manually_edited set to 1")
    
    def save_to_files(self):
        """Save DataFrame to CSV only (XLSX created only at completion) - optimized."""
        output_csv = os.path.join(self.output_csv_dir, f"Patentamt_{self.year}_manually_validated.csv")
        
        # Save with all columns including internal tracking columns (async-like, non-blocking)
        try:
            # Use a background thread-like approach by deferring heavy I/O
            self.df.to_csv(output_csv, index=False)
        except Exception as e:
            self.log(f"ERROR saving CSV: {e}")
            # Don't show error dialog during auto-save to avoid interrupting workflow
            if not hasattr(self, '_save_error_shown'):
                self._save_error_shown = True
                messagebox.showerror("Save Error", f"Could not save CSV: {e}")
    
    def show_status(self, message, duration=2000):
        """Show a status message briefly in the status bar."""
        self.status_label.config(text=message)
        # Clear after duration
        self.root.after(duration, lambda: self.status_label.config(text="Ready"))
    
    def toggle_delete(self):
        """Toggle deletion marking via keyboard (D key) with visual indicator only."""
        if self.current_task_idx >= len(self.tasks):
            return
        
        row_idx = self.tasks[self.current_task_idx]
        current_status = str(self.df.at[row_idx, 'marked_for_deletion'])
        
        # Toggle the deletion status
        new_status = '0' if current_status == '1' else '1'
        self.df.at[row_idx, 'marked_for_deletion'] = new_status
        
        # Update visual indicator and status bar (no popup)
        if new_status == '1':
            self.deletion_indicator.pack(side=tk.LEFT, padx=10)
            self.log(f"Row {row_idx} (id={self.df.at[row_idx, 'id']}): marked for deletion")
            self.show_status(f"✓ Row {self.df.at[row_idx, 'id']} marked for deletion")
        else:
            self.deletion_indicator.pack_forget()
            self.log(f"Row {row_idx} (id={self.df.at[row_idx, 'id']}): unmarked for deletion")
            self.show_status(f"✓ Row {self.df.at[row_idx, 'id']} unmarked")
    
    def next_task(self):
        """Move to next task (always captures current field values, saves every 10 tasks)."""
        # IMPORTANT: Always save current field values first (whatever is in the UI)
        # This captures edits even if Enter wasn't pressed
        # Pass log_changes=True to log the final changes
        self.save_current_entry(log_changes=True)
        
        # Show immediate feedback
        self.show_status("→ Moving to next task...", 1000)
        
        # Save current state to undo stack (after capturing field values)
        self.undo_stack.append({
            'df': self.df.copy(),
            'task_idx': self.current_task_idx
        })
        
        # Increment task counter
        self.tasks_since_last_save += 1
        
        # Save to file every 10 tasks or at completion
        if self.tasks_since_last_save >= 10 or self.current_task_idx + 1 >= len(self.tasks):
            self.save_to_files()
            self.tasks_since_last_save = 0
            self.log(f"Auto-saved at task {self.current_task_idx + 1}")
            self.show_status("💾 Saved to disk", 2000)
        
        # Move to next
        self.current_task_idx += 1
        
        # Display next task (or complete if done)
        self.display_task()
    
    def go_back(self):
        """Go back to previous task (undo) with status feedback."""
        if not self.undo_stack:
            self.show_status("⚠ Already at the first task", 2000)
            return
        
        # Save current field values before going back (in case user made edits)
        # This ensures we don't lose any work in progress
        # Pass log_changes=True to log the final changes
        self.save_current_entry(log_changes=True)
        
        # Show immediate feedback
        self.show_status("← Going back...", 1000)
        
        # Restore previous state
        prev_state = self.undo_stack.pop()
        self.df = prev_state['df']
        self.current_task_idx = prev_state['task_idx']
        
        # Reset task counter when going back
        if self.tasks_since_last_save > 0:
            self.tasks_since_last_save -= 1
        
        self.log(f"UNDO: Back to task {self.current_task_idx + 1}")
        self.display_task()
    
    def complete_validation(self):
        """Complete validation: keep marked rows (don't delete), save final CSV + XLSX, and exit."""
        # Calculate session duration
        elapsed = datetime.now() - self.session_start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        session_duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours > 0 else f"{minutes:02d}:{seconds:02d}"
        
        self.log("All validation tasks completed!")
        self.log(f"Session duration: {session_duration}")
        
        # Count rows marked for deletion (but don't delete them)
        rows_marked_for_deletion = (self.df['marked_for_deletion'] == '1').sum()
        if rows_marked_for_deletion > 0:
            marked_ids = self.df[self.df['marked_for_deletion'] == '1']['id'].tolist()
            self.log(f"Rows marked for deletion (kept in dataset): {rows_marked_for_deletion} rows (IDs: {marked_ids})")
        
        # Keep marked_for_deletion column in the final output (don't remove it)
        # No deletion, no reindexing - keep everything as is
        
        # Save final CSV and XLSX
        output_csv = os.path.join(self.output_csv_dir, f"Patentamt_{self.year}_manually_validated.csv")
        output_xlsx = os.path.join(self.output_xlsx_dir, f"Patentamt_{self.year}_manually_validated.xlsx")
        
        try:
            self.df.to_csv(output_csv, index=False)
            self.df.to_excel(output_xlsx, index=False, engine='openpyxl')
            self.log(f"Final files saved: CSV and XLSX")
        except Exception as e:
            self.log(f"ERROR saving final files: {e}")
            messagebox.showerror("Save Error", f"Could not save final files: {e}")
            return
        
        self.log(f"SESSION END - {len(self.tasks)} tasks completed, {rows_marked_for_deletion} rows marked for deletion (kept in dataset), Duration: {session_duration}")
        
        # Show completion message
        completion_message = (
            f"✓ Validation Complete!\n\n"
            f"Total tasks processed: {len(self.tasks)}\n"
            f"Session duration: {session_duration}\n"
            f"Rows marked for deletion (kept): {rows_marked_for_deletion}\n"
            f"Total row count: {len(self.df)}\n\n"
            f"Files saved:\n"
            f"• CSV: {output_csv}\n"
            f"• XLSX: {output_xlsx}\n\n"
            f"Log file: {self.log_file}"
        )
        
        messagebox.showinfo("🎉 Validation Complete", completion_message)
        self.root.destroy()
    
    def log(self, message):
        """Write to log file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Manual validation GUI for patent data")
    parser.add_argument('--csv', required=True, help='CSV filename from 01_validated/csv folder (e.g., Patentamt_1878_validated.csv)')
    args = parser.parse_args()
    
    # Locate the CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../../'))
    
    csv_file = args.csv
    
    # Check if absolute path
    if os.path.isabs(csv_file):
        csv_path = csv_file
    else:
        # Try in 01_validated/csv folder
        csv_path = os.path.join(project_root, 'data/04_dataset_validation/01_validated/csv', csv_file)
        
        if not os.path.exists(csv_path):
            # Try in current directory
            if os.path.exists(csv_file):
                csv_path = csv_file
            else:
                print(f"Error: Could not find CSV file: {csv_file}")
                print(f"Looked in: {os.path.join(project_root, 'data/04_dataset_validation/01_validated/csv')}")
                sys.exit(1)
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file does not exist: {csv_path}")
        sys.exit(1)
    
    try:
        app = ManualValidationApp(csv_path)
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

