#ref https://www.youtube.com/watch?v=L8ypSXwyBds&t=4861s

import numpy as np
import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font(None, 50)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
# green
FOOD = (0, 255, 0)

# snake head
# red
HEAD = (255, 0, 0)
# snake body
# grey
BODY = (200, 200, 200)

# background color
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 60

# reset
# reward
# play(action) -> direction
# game_iteration
# is_collision


class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('SnakeAI')
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        
        self.frame_iteration += 1
        
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        reward = 0
        
        # 3. check if game over
        game_over = False
        
        # check frame_iteration
        if self.is_collision() or self.frame_iteration > 100*len(self.  snake):
            reward = -10
            game_over = True
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            reward = 10
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, head=None):
        if head is None:
            head = self.head
        # hits boundary
        if head.x > self.w - BLOCK_SIZE or head.x < 0 or head.y > self.h - BLOCK_SIZE or head.y < 0:
            return True
        # hits itself
        if head in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            if pt == self.head:
                pygame.draw.rect(self.display, HEAD, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            else:
                pygame.draw.rect(self.display, HEAD, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BODY, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
            
        pygame.draw.rect(self.display, FOOD, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        # action: [straight, right, left]
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        
        # get current direction index in clock_wise
        current_direction = clock_wise.index(self.direction)

        # compare direction with action
        if np.array_equal(action, [1, 0, 0]): # move straight
            new_direction = current_direction
        elif np.array_equal(action, [0, 1, 0]): # move right
            new_direction = (current_direction + 1) % 4
        else: # move left
            new_direction = (current_direction - 1) % 4
        
        self.direction = clock_wise[new_direction]
            
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)