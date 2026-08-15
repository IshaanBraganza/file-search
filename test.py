import threading
import tkinter as tk
from tkinter import scrolledtext
import ollama

# Configuration
MODEL_NAME = "llama3"  # Change to your local model


def stream_ai_response():
    """Fetches the stream from Ollama and updates the UI safely."""
    user_input = entry_box.get().strip()
    if not user_input:
        return

    # Clear previous output and disable input during generation
    output_box.config(state=tk.NORMAL)
    output_box.delete("1.0", tk.END)
    send_button.config(state=tk.DISABLED)

    try:
        # Call the Ollama stream API
        stream = ollama.generate(
            model=MODEL_NAME, prompt=user_input, stream=True
        )

        for chunk in stream:
            text_chunk = chunk["response"]
            # Insert text at the end of the text box
            output_box.insert(tk.END, text_chunk)
            # Auto-scroll to the bottom
            output_box.see(tk.END)
            # Force Tkinter to update the visual display immediately
            window.update_idletasks()

    except Exception as e:
        output_box.insert(tk.END, f"\nError: {str(e)}")

    finally:
        # Re-enable UI elements
        output_box.config(state=tk.DISABLED)
        send_button.config(state=tk.NORMAL)


def start_generation_thread():
    """Starts the AI generation in a background thread to prevent UI freezing."""
    threading.Thread(target=stream_ai_response, daemon=True).start()


# --- GUI Setup ---
window = tk.Tk()
window.title("Ollama Stream Chat")
window.geometry("600x500")

# Input field
entry_box = tk.Entry(window, font=("Arial", 12))
entry_box.pack(fill=tk.X, padx=10, pady=5)

# Send button
send_button = tk.Button(
    window, text="Send", command=start_generation_thread, bg="blue", fg="white"
)
send_button.pack(pady=5)

# Scrollable output text area
output_box = scrolledtext.ScrolledText(
    window, wrap=tk.WORD, font=("Arial", 12), state=tk.DISABLED
)
output_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Run the application
window.mainloop()
