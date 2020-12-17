import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import copy
import os

class Bomb:

    def __init__(self, col):
        self.col = col
        self.row = 0

    def update(self):
        self.row += 1

    def isDroped(self, n_rows):
        return True if self.row >= n_rows else False


class Invador:

    def __init__(self, time_limit=True, simple=False, plot=False):
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.IMG_SIZE = 8 if simple else 16
        self.IMG_SIZE = 8 if simple else 16
        self.player_length = 3
        self.enable_actions = (0, 1, 2)
        self.frame_rate = 5
        self.bomb_post_interval = 4
        self.bomb_past_time = 0
        self.past_time = 0
        self.bombs = []
        self.time_limit = time_limit
        self.simple = simple

        self.nb_actions = len(self.enable_actions)
        self.timestep_limit = 500

        # variables
        self.reset()

        # animation
        if plot:
            plt.ion()
            self.fig = plt.figure()

    def update(self, action):
        """
        action:
            0: do nothing
            1: move left
            2: move right
        """
        # update player position
        if action == self.enable_actions[1]:
            # move left
            self.player_col = max(0, self.player_col - 1)
        elif action == self.enable_actions[2]:
            # move right
            self.player_col = min(self.player_col + 1, self.IMG_SIZE - self.player_length)
        else:
            # do nothing
            pass

        # update bomb position
        for b in self.bombs:
            b.row += 1

        if self.bomb_past_time == self.bomb_post_interval:
            self.bomb_past_time = 0
            new_pos = np.random.randint(self.IMG_SIZE)
            if not self.simple:
                while len(self.bombs) > 0 and (abs(new_pos - self.bombs[-1].col) > self.bomb_post_interval + self.player_length - 1 or abs(new_pos - self.bombs[-1].col) < self.player_length):
                    new_pos = np.random.randint(self.IMG_SIZE)
            else:
                while len(self.bombs) > 0 and abs(new_pos - self.bombs[-1].col) < self.player_length:
                    new_pos = np.random.randint(self.IMG_SIZE)
            self.bombs.append(Bomb(new_pos))
        else:
            self.bomb_past_time += 1

        # collision detection
        # self.reward = 0
        self.terminal = False

        self.past_time += 1
        if self.time_limit and self.past_time > self.timestep_limit:
            self.terminal = True

        if self.bombs[0].row == self.IMG_SIZE - 1:
            if self.player_col <= self.bombs[0].col < self.player_col + self.player_length:
                # catch
                self.reward += 1
            else:
                # drop
                self.reward -= 1
                self.terminal = True

        new_bombs = []
        for b in self.bombs:
            if not b.isDroped(self.IMG_SIZE):
                new_bombs.append(b)
        self.bombs = copy.copy(new_bombs)

    def draw(self):
        # reset screen
        self.screen = np.zeros((self.IMG_SIZE, self.IMG_SIZE))

        # draw player
        self.screen[self.player_row, self.player_col:self.player_col + self.player_length] = 1

        # draw bomb
        for b in self.bombs:
            self.screen[b.row, b.col] = 0.5

    def observe(self):
        self.draw()
        return self.screen, self.reward, self.terminal

    def step(self, action):
        self.update(action)
        return self.screen, self.reward, self.terminal

    def reset(self):
        # reset player position
        self.player_row = self.IMG_SIZE - 1
        self.player_col = np.random.randint(self.IMG_SIZE - self.player_length)

        # reset bomb position
        self.bombs = []
        self.bombs.append(Bomb(np.random.randint(self.IMG_SIZE)))

        # reset other variables
        self.reward = 0
        self.terminal = False
        self.past_time = 0
        self.bomb_past_time = 0

        self.draw()
        return self.screen
    
    def update_plot(self):
        self.draw()
        self.fig.clear()
        plt.imshow(self.screen)
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        plt.pause(0.2)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


