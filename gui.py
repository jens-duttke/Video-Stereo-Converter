import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import subprocess
import threading
import os
import sys
import queue

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class VideoStereoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Stereo Converter GUI")
        self.root.geometry("700x500")
        
        self.video_path = tk.StringVar()
        self.process = None
        self.queue = queue.Queue()
        
        self.create_widgets()
        self.root.after(100, self.process_queue)
        
    def create_widgets(self):
        # Frame for top controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # Video Selection
        ttk.Label(control_frame, text="Input Video:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.entry_video = ttk.Entry(control_frame, textvariable=self.video_path, width=60)
        self.entry_video.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="Browse...", command=self.browse_video).grid(row=0, column=2, pady=5)
        
        # Actions
        actions_frame = ttk.LabelFrame(self.root, text="Actions", padding="10")
        actions_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.btn_init = ttk.Button(actions_frame, text="1. Initialize Workflow", command=self.run_init)
        self.btn_init.grid(row=0, column=0, padx=5, pady=5)
        
        self.btn_extract = ttk.Button(actions_frame, text="2. Extract Frames", command=self.run_extract)
        self.btn_extract.grid(row=0, column=1, padx=5, pady=5)
        
        self.btn_depth = ttk.Button(actions_frame, text="3. Generate Depth Maps", command=self.run_depth)
        self.btn_depth.grid(row=0, column=2, padx=5, pady=5)
        
        self.btn_test = ttk.Button(actions_frame, text="4. Test 3D Settings", command=self.run_test)
        self.btn_test.grid(row=1, column=0, padx=5, pady=5)
        
        self.btn_start = ttk.Button(actions_frame, text="5. Run Full Conversion / Resume", command=self.run_conversion)
        self.btn_start.grid(row=1, column=1, padx=5, pady=5)
        
        self.btn_cleanup = ttk.Button(actions_frame, text="6. Cleanup Temporary Files", command=self.run_cleanup)
        self.btn_cleanup.grid(row=1, column=2, padx=5, pady=5)
        
        # Console Output
        console_frame = ttk.LabelFrame(self.root, text="Console Output", padding="10")
        console_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.console = ScrolledText(console_frame, wrap=tk.WORD, state='disabled', bg='black', fg='white')
        self.console.pack(fill=tk.BOTH, expand=True)
        
    def browse_video(self):
        filename = filedialog.askopenfilename(
            title="Select Video",
            filetypes=(("Video files", "*.mp4 *.mkv *.avi *.mov"), ("All files", "*.*"))
        )
        if filename:
            self.video_path.set(filename)
            
    def log(self, message):
        self.queue.put(message)
        
    def process_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                self.console.config(state='normal')
                self.console.insert(tk.END, msg)
                self.console.see(tk.END)
                self.console.config(state='disabled')
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)
        
    def run_command(self, cmd, success_msg=None):
        if self.process is not None:
            messagebox.showwarning("Warning", "A process is already running!")
            return
        self.process = True  # reserve the slot synchronously before the thread starts

        def target():
            self.log(f"> {' '.join(cmd)}\n")
            try:
                # Use subprocess.PIPE to capture output
                self.process = subprocess.Popen(
                    cmd,
                    cwd=SCRIPT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )

                for line in self.process.stdout:
                    self.log(line)

                self.process.wait()

                if self.process.returncode == 0 and success_msg:
                    self.log(f"\n[SUCCESS] {success_msg}\n")
                elif self.process.returncode != 0:
                    self.log(f"\n[ERROR] Command finished with exit code {self.process.returncode}\n")

            except Exception as e:
                self.log(f"\n[EXCEPTION] {str(e)}\n")
            finally:
                self.process = None

        thread = threading.Thread(target=target, daemon=True)
        thread.start()

    def get_workflow_dir(self):
        video = self.video_path.get()
        if not video:
            messagebox.showerror("Error", "Please select a video file first.")
            return None
        
        video_dir = os.path.dirname(video)
        video_stem = os.path.splitext(os.path.basename(video))[0]
        workflow_dir = os.path.join(video_dir, video_stem, "workflow")
        return workflow_dir

    def run_init(self):
        video = self.video_path.get()
        if not video:
            messagebox.showerror("Error", "Please select a video file first.")
            return
            
        workflow_dir = self.get_workflow_dir()
        cmd = [sys.executable, "workflow_init.py", "--input-video", video, "--workflow-dir", workflow_dir]
        self.run_command(cmd, "Workflow initialized successfully.")
        
    def run_extract(self):
        workflow_dir = self.get_workflow_dir()
        if not workflow_dir or not os.path.exists(workflow_dir):
            messagebox.showerror("Error", "Workflow directory not found. Please initialize first.")
            return
            
        cmd = [sys.executable, "frame_extractor.py", workflow_dir]
        self.run_command(cmd, "Frames extracted successfully.")
        
    def run_depth(self):
        workflow_dir = self.get_workflow_dir()
        if not workflow_dir or not os.path.exists(workflow_dir):
            messagebox.showerror("Error", "Workflow directory not found. Please initialize first.")
            return
            
        cmd = [sys.executable, "depth_map_generator.py", workflow_dir]
        self.run_command(cmd, "Depth maps generated successfully.")
        
    def run_test(self):
        workflow_dir = self.get_workflow_dir()
        if not workflow_dir or not os.path.exists(workflow_dir):
            messagebox.showerror("Error", "Workflow directory not found. Please initialize first.")
            return
            
        cmd = [sys.executable, "sbs_tester.py", workflow_dir]
        self.run_command(cmd, "Tester closed.")
        
    def run_conversion(self):
        workflow_dir = self.get_workflow_dir()
        if not workflow_dir or not os.path.exists(workflow_dir):
            messagebox.showerror("Error", "Workflow directory not found. Please initialize first.")
            return

        if self.process is not None:
            messagebox.showwarning("Warning", "A process is already running!")
            return

        yaml_path = os.path.join(SCRIPT_DIR, "workflows_gui.yaml")
        try:
            with open(yaml_path, "w") as f:
                f.write(f"{workflow_dir}:\n")
            self.log(f"Created {yaml_path}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create yaml: {e}")
            return

        cmd = [sys.executable, "workflow_orchestrator.py", yaml_path]
        self.log("Opening Workflow Orchestrator in a new terminal window...\n")
        self.process = True  # reserve the slot until the detached console process exits

        def target():
            try:
                self.process = subprocess.Popen(cmd, cwd=SCRIPT_DIR, creationflags=subprocess.CREATE_NEW_CONSOLE)
                self.process.wait()
            except Exception as e:
                self.log(f"\n[EXCEPTION] {str(e)}\n")
            finally:
                self.process = None

        threading.Thread(target=target, daemon=True).start()

    def run_cleanup(self):
        workflow_dir = self.get_workflow_dir()
        if not workflow_dir or not os.path.exists(workflow_dir):
            messagebox.showerror("Error", "Workflow directory not found.")
            return
            
        import shutil
        folders_to_delete = ["frames", "depth_maps", "sbs", "chunks"]
        deleted = []
        for folder in folders_to_delete:
            path = os.path.join(workflow_dir, folder)
            if os.path.exists(path):
                try:
                    shutil.rmtree(path)
                    deleted.append(folder)
                except Exception as e:
                    self.log(f"Error deleting {folder}: {e}\n")
        
        if deleted:
            messagebox.showinfo("Success", f"Deleted temporary folders: {', '.join(deleted)}")
            self.log("Cleanup completed successfully.\n")
        else:
            messagebox.showinfo("Info", "No temporary folders found to delete.")

if __name__ == "__main__":
    root = tk.Tk()
    
    style = ttk.Style()
    if 'clam' in style.theme_names():
        style.theme_use('clam')
        
    app = VideoStereoGUI(root)
    root.mainloop()
