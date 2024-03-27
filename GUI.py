import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

def select_points(image_path):
    root = tk.Tk()
    root.title("Click Four Points")

    # Initialize variables
    clicked_points = []
    canvas = tk.Canvas(root, cursor="cross")
    canvas.pack(fill="both", expand=True)

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(image=Image.fromarray(image))
    canvas.create_image(0, 0, image=photo, anchor="nw")
    root.geometry(f"{image.shape[1]}x{image.shape[0]}")

    def on_click(event):
        # Calculate clicked position in the image
        clicked_x = canvas.canvasx(event.x)
        clicked_y = canvas.canvasy(event.y)
        clicked_points.append((clicked_x, clicked_y))

        # Mark the clicked position on the canvas
        radius = 5
        if len(clicked_points) <= 2:  # First and second click, draw horizontal lines
            canvas.create_line(0, clicked_y, image.shape[1], clicked_y, fill='red')
        elif len(clicked_points) <= 4:  # Third and fourth click, draw vertical lines
            canvas.create_line(clicked_x, 0, clicked_x, image.shape[0], fill='red')

        if len(clicked_points) == 4:
            root.quit()

    # Bind click event
    canvas.bind("<Button-1>", on_click)

    # Start the GUI event loop
    root.mainloop()
    root.destroy()

    clicked_points = [clicked_points[0][1], clicked_points[1][1], clicked_points[2][0], clicked_points[3][0]]
    
    return clicked_points

def select_first_frame():
    
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select the first frame",
        filetypes=[("Image files", "*.png *.jpg *.jpeg")])  # Filter to show only images
    root.destroy()

    return file_path

def select_masks_folder():
     # Initialize the Tkinter window
    root = tk.Tk()
    # Hide the main window
    root.withdraw()

    # Open the folder selection dialog and get the selected folder path
    folder_selected = filedialog.askdirectory()

    # Close the Tkinter window
    root.destroy()

    # Return the selected folder path
    return folder_selected

def select_video():
    
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select the video frame",
        filetypes=[("Image files", "*.mp4")])  # Filter to show only video
    root.destroy()

    return file_path

# Example usage
if __name__ == "__main__":
    
    dir = os.getcwd()
    dir = os.path.join(dir, "mask")
    docs = os.listdir(dir)

    for docs_now in docs:
        image_path = os.path.join(dir, docs_now)# Update this to your image path
        points = select_points(image_path)
        print("Selected Points:", points)
