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
        self.output_csv_dir = os.path.join(self.project_root, 'data/04_dataset_validation/manually_validated_csv')
        self.output_xlsx_dir = os.path.join(self.project_root, 'data/04_dataset_validation/manually_validated_xlsx')
        
        # Create output directories
        os.makedirs(self.output_csv_dir, exist_ok=True)
        os.makedirs(self.output_xlsx_dir, exist_ok=True)
        
        # Setup logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.project_root, f'manually_validated_log_{self.year}_{timestamp}.txt')
        
        # Load data
        self.df = None
        self.tasks = []
        self.current_task_idx = 0
        self.undo_stack = []
        self.zoom_level = 1.0
        self.original_image = None
        self.photo_image = None
        
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
        """Load CSV and add manually_validated column."""
        self.log("Loading data...")
        self.df = pd.read_csv(self.csv_path, dtype=str)
        
        # Add manually_validated column if it doesn't exist
        if 'manually_validated' not in self.df.columns:
            self.df['manually_validated'] = self.df['validation_notes'].apply(
                lambda x: '1' if str(x).strip() == 'Valid' else '0'
            )
            self.log(f"Added manually_validated column: {(self.df['manually_validated'] == '1').sum()} valid entries")
        
        # Add deletion flag column
        if 'marked_for_deletion' not in self.df.columns:
            self.df['marked_for_deletion'] = '0'
    
    def parse_tasks(self):
        """Parse validation_notes to create task queue."""
        self.log("Parsing validation issues...")
        
        # Define patterns for the three issue types
        duplicate_pattern = r'Duplicate patent_id \d+'
        less_than_pattern = r'patent_id \d+ < start'
        greater_than_pattern = r'patent_id \d+ > end'
        
        duplicates = []
        less_than = []
        greater_than = []
        
        for idx, row in self.df.iterrows():
            validation_notes = str(row['validation_notes']).strip()
            
            if re.search(duplicate_pattern, validation_notes):
                duplicates.append(idx)
            elif re.search(less_than_pattern, validation_notes):
                less_than.append(idx)
            elif re.search(greater_than_pattern, validation_notes):
                greater_than.append(idx)
        
        # Combine in order: duplicates, < start, > end
        self.tasks = duplicates + less_than + greater_than
        
        self.log(f"Found {len(duplicates)} duplicate issues")
        self.log(f"Found {len(less_than)} '< start' issues")
        self.log(f"Found {len(greater_than)} '> end' issues")
    
    def setup_gui(self):
        """Build the tkinter interface."""
        # Top frame - Progress bar
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        self.progress_label = ttk.Label(top_frame, text="", font=('Arial', 12, 'bold'))
        self.progress_label.pack()
        
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
        
        # Scrollable canvas for image
        canvas_frame = ttk.Frame(image_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.image_canvas = tk.Canvas(canvas_frame, bg='gray')
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        
        self.image_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Entry fields frame
        fields_frame = ttk.Frame(self.root, padding="10")
        fields_frame.pack(fill=tk.X)
        
        # Entry text
        entry_label_frame = ttk.Frame(fields_frame)
        entry_label_frame.pack(fill=tk.X, pady=5)
        
        self.entry_info_label = ttk.Label(entry_label_frame, text="Entry:", font=('Arial', 10, 'bold'))
        self.entry_info_label.pack(side=tk.LEFT)
        
        self.entry_text = scrolledtext.ScrolledText(fields_frame, height=4, wrap=tk.WORD, font=('Arial', 10))
        self.entry_text.pack(fill=tk.X, pady=5)
        self.entry_text.bind('<Return>', self.on_entry_return)
        self.entry_text.bind('<KeyRelease>', self.on_field_change)
        
        # Patent ID
        patent_id_frame = ttk.Frame(fields_frame)
        patent_id_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(patent_id_frame, text="Patent ID:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        self.patent_id_entry = ttk.Entry(patent_id_frame, font=('Arial', 10), width=20)
        self.patent_id_entry.pack(side=tk.LEFT, padx=5)
        self.patent_id_entry.bind('<Return>', self.on_patent_id_return)
        self.patent_id_entry.bind('<KeyRelease>', self.on_field_change)
        
        # Delete checkbox
        self.delete_var = tk.BooleanVar()
        self.delete_checkbox = ttk.Checkbutton(
            patent_id_frame,
            text="Mark for deletion",
            variable=self.delete_var,
            command=self.on_delete_toggle
        )
        self.delete_checkbox.pack(side=tk.LEFT, padx=20)
        
        # Bottom buttons
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.pack(fill=tk.X)
        
        button_style_frame = ttk.Frame(button_frame)
        button_style_frame.pack()
        
        ttk.Button(button_style_frame, text="Manual Save (S)", command=self.manual_save, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_style_frame, text="Back (A)", command=self.go_back, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_style_frame, text="Next Task (D)", command=self.next_task, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_style_frame, text="Finish & Exit (Q)", command=self.finish_and_exit, width=15).pack(side=tk.LEFT, padx=5)
        
        # Shortcuts help
        help_frame = ttk.Frame(self.root, padding="5")
        help_frame.pack(fill=tk.X)
        help_text = "Shortcuts: A=Back | D=Next | S=Save | E=Toggle Delete | Q=Finish | Space=Next | Enter=Next Field"
        ttk.Label(help_frame, text=help_text, font=('Arial', 9), foreground='gray').pack()
    
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts."""
        self.root.bind('a', lambda e: self.go_back())
        self.root.bind('A', lambda e: self.go_back())
        self.root.bind('d', lambda e: self.next_task())
        self.root.bind('D', lambda e: self.next_task())
        self.root.bind('s', lambda e: self.manual_save())
        self.root.bind('S', lambda e: self.manual_save())
        self.root.bind('e', lambda e: self.toggle_delete())
        self.root.bind('E', lambda e: self.toggle_delete())
        self.root.bind('q', lambda e: self.finish_and_exit())
        self.root.bind('Q', lambda e: self.finish_and_exit())
        self.root.bind('<space>', lambda e: self.next_task())
    
    def display_task(self):
        """Display current task in GUI."""
        if self.current_task_idx >= len(self.tasks):
            messagebox.showinfo("Complete", "All tasks completed!")
            self.finish_and_exit()
            return
        
        # Get current row
        row_idx = self.tasks[self.current_task_idx]
        row = self.df.iloc[row_idx]
        
        # Update progress
        progress_text = f"Task {self.current_task_idx + 1} / {len(self.tasks)} | {row['validation_notes']}"
        self.progress_label.config(text=progress_text)
        
        # Update entry info label
        self.entry_info_label.config(text=f"Entry (from page_{row['page'].zfill(4)}.png):")
        
        # Load and display image
        page_num = str(row['page']).zfill(4)
        self.page_label.config(text=f"Page: page_{page_num}.png")
        image_path = os.path.join(self.png_dir, f"page_{page_num}.png")
        
        if os.path.exists(image_path):
            self.original_image = Image.open(image_path)
            self.zoom_level = 1.0
            self.update_image_display()
        else:
            self.log(f"WARNING: Image not found: {image_path}")
            messagebox.showwarning("Image Not Found", f"Could not find image: page_{page_num}.png")
        
        # Populate fields
        self.entry_text.delete('1.0', tk.END)
        self.entry_text.insert('1.0', str(row['entry']))
        
        self.patent_id_entry.delete(0, tk.END)
        self.patent_id_entry.insert(0, str(row['patent_id']))
        
        self.delete_var.set(str(row['marked_for_deletion']) == '1')
        
        # Focus on entry text
        self.entry_text.focus_set()
    
    def update_image_display(self):
        """Update image display with current zoom level."""
        if self.original_image is None:
            return
        
        # Calculate new size
        new_width = int(self.original_image.width * self.zoom_level)
        new_height = int(self.original_image.height * self.zoom_level)
        
        # Resize image
        resized_image = self.original_image.resize((new_width, new_height), Image.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(resized_image)
        
        # Update canvas
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        self.image_canvas.config(scrollregion=(0, 0, new_width, new_height))
        
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
    
    def on_entry_return(self, event):
        """Handle Enter key in entry text."""
        # Prevent default newline behavior
        self.patent_id_entry.focus_set()
        return 'break'
    
    def on_patent_id_return(self, event):
        """Handle Enter key in patent_id field."""
        self.next_task()
        return 'break'
    
    def on_field_change(self, event):
        """Auto-save when field changes."""
        # Debounce: only save after a brief pause
        if hasattr(self, '_save_timer'):
            self.root.after_cancel(self._save_timer)
        self._save_timer = self.root.after(500, self.auto_save)
    
    def auto_save(self):
        """Auto-save current changes."""
        self.save_current_entry()
        self.save_to_files()
    
    def manual_save(self):
        """Manual save triggered by user."""
        self.save_current_entry()
        self.save_to_files()
        messagebox.showinfo("Saved", "File saved successfully!")
    
    def save_current_entry(self):
        """Save current entry data to DataFrame."""
        if self.current_task_idx >= len(self.tasks):
            return
        
        row_idx = self.tasks[self.current_task_idx]
        old_row = self.df.iloc[row_idx].copy()
        
        # Get new values
        new_entry = self.entry_text.get('1.0', tk.END).strip()
        new_patent_id = self.patent_id_entry.get().strip()
        
        # Update DataFrame
        if str(old_row['entry']) != new_entry:
            self.log(f"Row {row_idx} (id={old_row['id']}): entry changed")
            self.df.at[row_idx, 'entry'] = new_entry
        
        if str(old_row['patent_id']) != new_patent_id:
            self.log(f"Row {row_idx} (id={old_row['id']}): patent_id changed from {old_row['patent_id']} to {new_patent_id}")
            self.df.at[row_idx, 'patent_id'] = new_patent_id
        
        # Mark as manually validated if it was changed
        if str(old_row['manually_validated']) == '0':
            self.df.at[row_idx, 'manually_validated'] = '1'
            self.log(f"Row {row_idx} (id={old_row['id']}): manually_validated changed to 1")
    
    def save_to_files(self):
        """Save DataFrame to CSV and XLSX."""
        output_csv = os.path.join(self.output_csv_dir, f"Patentamt_{self.year}_manually_validated.csv")
        output_xlsx = os.path.join(self.output_xlsx_dir, f"Patentamt_{self.year}_manually_validated.xlsx")
        
        # Remove marked_for_deletion column for output (internal use only)
        output_df = self.df.copy()
        
        try:
            output_df.to_csv(output_csv, index=False)
            output_df.to_excel(output_xlsx, index=False, engine='openpyxl')
        except Exception as e:
            self.log(f"ERROR saving files: {e}")
            messagebox.showerror("Save Error", f"Could not save files: {e}")
    
    def on_delete_toggle(self):
        """Handle delete checkbox toggle."""
        if self.current_task_idx >= len(self.tasks):
            return
        
        row_idx = self.tasks[self.current_task_idx]
        is_marked = '1' if self.delete_var.get() else '0'
        self.df.at[row_idx, 'marked_for_deletion'] = is_marked
        
        if is_marked == '1':
            self.log(f"Row {row_idx} (id={self.df.at[row_idx, 'id']}): marked for deletion")
        else:
            self.log(f"Row {row_idx} (id={self.df.at[row_idx, 'id']}): unmarked for deletion")
        
        self.auto_save()
    
    def toggle_delete(self):
        """Toggle delete checkbox via keyboard."""
        self.delete_var.set(not self.delete_var.get())
        self.on_delete_toggle()
    
    def next_task(self):
        """Move to next task."""
        # Save current state to undo stack
        self.undo_stack.append({
            'df': self.df.copy(),
            'task_idx': self.current_task_idx
        })
        
        # Save current entry
        self.save_current_entry()
        self.save_to_files()
        
        # Move to next
        self.current_task_idx += 1
        
        if self.current_task_idx < len(self.tasks):
            self.display_task()
        else:
            messagebox.showinfo("Complete", "All tasks completed!")
            self.finish_and_exit()
    
    def go_back(self):
        """Go back to previous task (undo)."""
        if not self.undo_stack:
            messagebox.showinfo("Info", "Already at the first task!")
            return
        
        # Restore previous state
        prev_state = self.undo_stack.pop()
        self.df = prev_state['df']
        self.current_task_idx = prev_state['task_idx']
        
        self.log(f"UNDO: Back to task {self.current_task_idx + 1}")
        self.display_task()
    
    def finish_and_exit(self):
        """Finish validation, delete marked rows, reindex, and exit."""
        # Confirm with user
        response = messagebox.askyesno(
            "Finish & Exit",
            f"Are you sure you want to finish?\n\n"
            f"This will:\n"
            f"- Delete {(self.df['marked_for_deletion'] == '1').sum()} marked rows\n"
            f"- Reindex the 'id' column\n"
            f"- Save final files\n"
            f"- Exit the application"
        )
        
        if not response:
            return
        
        # Save current entry first
        if self.current_task_idx < len(self.tasks):
            self.save_current_entry()
        
        # Delete marked rows
        rows_to_delete = self.df[self.df['marked_for_deletion'] == '1'].index.tolist()
        if rows_to_delete:
            self.log(f"Deleting {len(rows_to_delete)} marked rows: {rows_to_delete}")
            self.df = self.df[self.df['marked_for_deletion'] != '1'].reset_index(drop=True)
        
        # Remove marked_for_deletion column
        if 'marked_for_deletion' in self.df.columns:
            self.df = self.df.drop(columns=['marked_for_deletion'])
        
        # Reindex id column from 1 to N
        self.df['id'] = range(1, len(self.df) + 1)
        self.log(f"Reindexed 'id' column: 1 to {len(self.df)}")
        
        # Final save
        self.save_to_files()
        
        self.log(f"SESSION END - {self.current_task_idx} tasks completed, {len(rows_to_delete)} rows deleted")
        
        messagebox.showinfo("Complete", f"Validation complete!\n\nFiles saved to:\n{self.output_csv_dir}")
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
    parser.add_argument('--csv', required=True, help='CSV filename from validated_csv folder (e.g., Patentamt_1878_cleaned_with_variables_validated.csv)')
    args = parser.parse_args()
    
    # Locate the CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../../'))
    
    csv_file = args.csv
    
    # Check if absolute path
    if os.path.isabs(csv_file):
        csv_path = csv_file
    else:
        # Try in validated_csv folder
        csv_path = os.path.join(project_root, 'data/04_dataset_validation/validated_csv', csv_file)
        
        if not os.path.exists(csv_path):
            # Try in current directory
            if os.path.exists(csv_file):
                csv_path = csv_file
            else:
                print(f"Error: Could not find CSV file: {csv_file}")
                print(f"Looked in: {os.path.join(project_root, 'data/04_dataset_validation/validated_csv')}")
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

