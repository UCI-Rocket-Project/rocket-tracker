import numpy as np
class Rocket:
    def __init__(self):
        self.pos = np.array([0,0,0], dtype=float)
        self.vel = np.array([0,0,0], dtype=float)
        self.acc = np.array([0,0,0], dtype=float)

    def step(self, delta_t):
        self.pos += self.vel * delta_t
        self.vel += self.acc * delta_t