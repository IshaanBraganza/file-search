# allows for a user to drag and drop files onto a given space.

import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
import subprocess
import platform
import os
from urllib.parse import urljoin
from urllib.request import pathname2url

def handle_drop(event):
    ########
    # Gets path of the file dropped
    ########

    # Safe path parsing
    file_paths = root.splitlist(event.data)
    if not file_paths:
        return
        
    file_path = file_paths[0]
    absolute_path = os.path.abspath(file_path)

    # Update UI
    label.config(text=f"Dropped file:\n{absolute_path}")
    print(f"File absolute path: {absolute_path}")


    ########
    # Opens the file
    ########

    # Determine OS and open with native system tool
    current_os = platform.system()
    
    try:
        if current_os == "Windows":
            os.startfile(absolute_path)
        elif current_os == "Darwin":  # macOS
            subprocess.run(["open", absolute_path], check=True)
        else:  # Linux
            subprocess.run(["xdg-open", absolute_path], check=True)
    except Exception as e:
        print(f"Native open failed: {e}. Falling back to browser URL format...")
        # Emergency backup: Convert path to strict browser URL structure (file:///...)
        file_url = urljoin('file:', pathname2url(absolute_path))
        import webbrowser
        webbrowser.open(file_url)



# Initialize window
root = TkinterDnD.Tk()
root.title("Drag and Drop Files")
root.geometry("400x200")

label = tk.Label(root, text="Drag and drop a file anywhere here", bg="lightgray", width=40, height=8)
label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
label.drop_target_register(DND_FILES)
label.dnd_bind("<<Drop>>", handle_drop)

root.mainloop()