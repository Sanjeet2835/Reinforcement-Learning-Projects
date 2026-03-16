import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()

font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# colors
WHITE = (255,255,255)
RED = (200,0,0)
BLUE1 = (0,0,255)
BLUE2 = (0,100,255)
BLACK = (0,0,0)

BLOCK_SIZE = 20 #Each block takes 20 pixels.
SPEED = 15


class SnakeGame:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake wala game")
        self.clock = pygame.time.Clock()

        self.reset()


    def reset(self):
        """Reset game state"""
        self.direction = Direction.RIGHT
        self.head = Point(self.w//2, self.h//2)
        self.snake = [
            self.head,
            Point(self.head.x-BLOCK_SIZE, self.head.y),
            Point(self.head.x-(2*BLOCK_SIZE), self.head.y)
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0 # to track how long it's taking


    def _place_food(self):

        x = random.randint(0,(self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0,(self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x,y)

        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        self.frame_iteration += 1 # <-- Increment the frame counter

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action[0] == 1:
            new_dir = clock_wise[idx] # no change
        elif action[1] == 1:
            new_dir = clock_wise[(idx + 1) % 4] # right turn
        elif action[2] == 1:
            new_dir = clock_wise[(idx - 1) % 4] # left turn

        self.direction = new_dir

        self._move(self.direction) 
        self.snake.insert(0,self.head)

        # --- REWARD LOGIC STARTS HERE ---
        reward = 0
        game_over = False

        # Check collision OR starvation (taking too long without eating)
        # We give it 100 frames * its length to find food
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10 # <-- PENALTY!
            return reward, game_over, self.score

        # Check if eating food
        if self.head == self.food:
            self.score += 1
            reward = 10 # <-- TREAT!
            self._place_food()
            self.frame_iteration = 0 # Reset timer when it eats
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score 

    def _is_collision(self, pt=None): 
            if pt is None:
                pt = self.head

            # Check if hitting boundaries
            if pt.x > self.w-BLOCK_SIZE or pt.x < 0:
                return True
            if pt.y > self.h-BLOCK_SIZE or pt.y < 0:
                return True
            
            # Check if hitting itself
            if pt in self.snake[1:]:
                return True

            return False

    def get_state(self):
        head = self.snake[0]
        
        # Calculate the points directly around the snake's head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        # Check current absolute direction
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        # Food relative location logic
        food_straight = (dir_u and self.food.y < head.y) or \
                        (dir_d and self.food.y > head.y) or \
                        (dir_l and self.food.x < head.x) or \
                        (dir_r and self.food.x > head.x)

        food_behind = (dir_u and self.food.y > head.y) or \
                      (dir_d and self.food.y < head.y) or \
                      (dir_l and self.food.x > head.x) or \
                      (dir_r and self.food.x < head.x)

        food_left = (dir_u and self.food.x < head.x) or \
                    (dir_d and self.food.x > head.x) or \
                    (dir_l and self.food.y > head.y) or \
                    (dir_r and self.food.y < head.y)

        food_right = (dir_u and self.food.x > head.x) or \
                     (dir_d and self.food.x < head.x) or \
                     (dir_l and self.food.y < head.y) or \
                     (dir_r and self.food.y > head.y)

        state = [
            # Danger straight
            (dir_r and self._is_collision(point_r)) or 
            (dir_l and self._is_collision(point_l)) or 
            (dir_u and self._is_collision(point_u)) or 
            (dir_d and self._is_collision(point_d)),

            # Danger right
            (dir_u and self._is_collision(point_r)) or 
            (dir_d and self._is_collision(point_l)) or 
            (dir_l and self._is_collision(point_u)) or 
            (dir_r and self._is_collision(point_d)),

            # Danger left
            (dir_d and self._is_collision(point_r)) or 
            (dir_u and self._is_collision(point_l)) or 
            (dir_r and self._is_collision(point_u)) or 
            (dir_l and self._is_collision(point_d)),
            
            # Relative Food Location
            food_straight,
            food_behind,
            food_left,
            food_right
        ]

        # Convert True/False booleans to 1s and 0s
        return np.array(state, dtype=int)    
    
    def _update_ui(self):

        self.display.fill(BLACK)

        for pt in self.snake:

            pygame.draw.rect(self.display, BLUE1,
                             pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

            pygame.draw.rect(self.display, BLUE2,
                             pygame.Rect(pt.x+4, pt.y+4, 12, 12))


        pygame.draw.rect(self.display, RED,
                         pygame.Rect(self.food.x,self.food.y,BLOCK_SIZE,BLOCK_SIZE))


        text = font.render("Score: "+str(self.score),True,WHITE)
        self.display.blit(text,[0,0])

        pygame.display.flip()


    def _move(self,direction):

        x = self.head.x
        y = self.head.y

        if direction == Direction.RIGHT:
            x += BLOCK_SIZE

        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE

        elif direction == Direction.DOWN:
            y += BLOCK_SIZE

        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x,y)


    def _game_over_screen(self):

        text = font.render("Game Over!", True, RED)

        self.display.blit(text,[self.w/2-60,self.h/2])

        pygame.display.flip()

        pygame.time.delay(1000)



