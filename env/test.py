import tkinter as tk
from tkinter import messagebox
import game
import time

class GameUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('2048 Game')
        self.geometry('400x400')
        
        self.game = game.Game()
        
        self.canvas = tk.Canvas(self, bg='gray', width=380, height=380)
        self.canvas.pack(pady=10)
        
        self.bind_all('<Key>', self.on_keypress)

        self.update_ui()

    def update_ui(self, intermediate_state=None):
        self.canvas.delete('all')
        state = intermediate_state or self.game.get_state()
        for i, row in enumerate(state):
            for j, value in enumerate(row):
                if value:
                    self.canvas.create_rectangle(20 + j*90, 20 + i*90, 110 + j*90, 110 + i*90, fill=self.get_color(value))
                    self.canvas.create_text(65 + j*90, 65 + i*90, text=str(value), font=('Arial', 24, 'bold'))

    def on_keypress(self, event):
        moves = {'Up': 1, 'Right': 2, 'Down': 0, 'Left': 3}
        key = event.keysym
        if key in moves:
            old_state = [row.copy() for row in self.game.get_state()]
            _, reward, done = self.game.action(moves[key])
            new_state = self.game.get_state()
            self.animate_movement(old_state, new_state)
            if done:
                messagebox.showinfo('Game Over', f'Final Score: {self.game.get_current_score()}')
                self.game = game.Game()
                self.update_ui()

    def animate_movement(self, old_state, new_state):
        # For simplicity, we're just going to create intermediary steps without smooth sliding
        steps = 10
        for step in range(1, steps+1):
            intermediate_state = [[0]*4 for _ in range(4)]
            for i in range(4):
                for j in range(4):
                    if old_state[i][j] == new_state[i][j]:
                        intermediate_state[i][j] = old_state[i][j]
                    elif old_state[i][j] == 0:
                        intermediate_state[i][j] = new_state[i][j] if step == steps else 0
                    elif new_state[i][j] == 0:
                        intermediate_state[i][j] = old_state[i][j] if step == 1 else 0
                    else:
                        intermediate_state[i][j] = old_state[i][j] if step < steps/2 else new_state[i][j]

            self.update_ui(intermediate_state)
            self.canvas.update()
            time.sleep(0.02)

    def get_color(self, value):
        colors = {
            2: '#eee4da',
            4: '#ede0c8',
            8: '#f2b179',
            16: '#f59563',
            32: '#f67c5f',
            64: '#f65e3b',
            128: '#edcf72',
            256: '#edcc61',
            512: '#edc850',
            1024: '#edc53f',
            2048: '#edc22e'
        }
        return colors.get(value, '#cdc1b4')

if __name__ == '__main__':
    app = GameUI()
    app.mainloop()
