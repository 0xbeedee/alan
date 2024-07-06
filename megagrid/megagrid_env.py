class GridManager:
    def __init__(self):
        self.current_grid = None

    def load_grid(self, grid):
        self.current_grid = grid

    def transition_to_next_grid(self, next_grid):
        self.load_grid(next_grid)
        # Add logic for smooth transition

    def get_current_grid(self):
        return self.current_grid
