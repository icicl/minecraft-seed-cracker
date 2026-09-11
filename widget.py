import tkinter as tk
from PIL import Image, ImageTk

GRID_ROWS = 3
GRID_COLS = 9

GRID_X = 50     # top-left X of grid (relative to first image)
GRID_Y = 50     # top-left Y of grid
GRID_WIDTH = 450
GRID_HEIGHT = 150

def launch(detected_img, actual_img, scale=4):
    root = tk.Tk()
    root.title("Cracker")

    tk_img1 = ImageTk.PhotoImage(detected_img)
    tk_img2 = ImageTk.PhotoImage(actual_img)

    
    canvas = tk.Canvas(root, width=max(detected_img.width, actual_img.width), height=detected_img.height + actual_img.height)
    canvas.pack()

    detected_x0 = (max(detected_img.width, actual_img.width) - detected_img.width)//2
    canvas.create_image(detected_x0, 0, anchor="nw", image=tk_img1)
    canvas.create_image((max(detected_img.width, actual_img.width) - actual_img.width)//2, detected_img.height, anchor="nw", image=tk_img2)

    # Create Grid (Initially Hidden)
    grid_cells = []
    active_cell = [None]

    for row in range(3):
        for col in range(9):
            x1 = (detected_x0 + scale) + col * (16 + 1) * scale
            y1 = scale + row * (16 + 1) * scale
            x2 = x1 + 16*scale
            y2 = y1 + 16*scale

            rect = canvas.create_rectangle(
                x1, y1, x2, y2,
                outline="red",
                width=2,
                fill="yellow",
                stipple="gray25",  # semi-transparent effect
                state="hidden"
            )
            grid_cells.append(rect)

    def on_click(event):
        # Only react if click is inside first image
        if event.y > detected_img.height:
            return

        g_x = (event.x - detected_x0 - scale) // (scale*(16 + 1))
        g_y = (event.y - scale) // (scale*(16 + 1))

        if 0 <= g_x < 9 and 0 <= g_y < 3:
            index = 9*g_y + g_x
            if active_cell[0] == index:
                canvas.itemconfigure(grid_cells[index], state="hidden")
                active_cell[0] = None
            else:
                if active_cell[0] is not None: canvas.itemconfigure(grid_cells[active_cell[0]], state="hidden")
                canvas.itemconfigure(grid_cells[index], state="normal")
                active_cell[0] = index


    canvas.bind("<Button-1>", on_click)

    root.mainloop()