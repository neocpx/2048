import sys, pygame
import random
import constants as consts

state=[[0,0,0,0],
       [0,0,0,0],
       [0,0,0,0],
       [0,0,0,0]]

pygame.init()
screen=pygame.display.set_mode((consts.width,consts.height))
pygame.display.set_caption('2048')
font = pygame.font.SysFont("Comic Sans MS", 30)


def setNumber(n=1):
       for _ in range(n):
              emptyPos = [(i,j) for i in range(4) for j in range(4) if state[i][j]==0]
              if not emptyPos: return False
              randPos = random.choice(emptyPos)
              state[randPos[0]][randPos[1]] = random.choice((2,4))
       return True

def display():
    screen.fill(consts.defaultBgColor)
    for r in range(4):
           for c in range(4):
                  rect_x = c * consts.width // 4 + 10
                  rect_y = r * consts.height // 4 + 10
                  rect_w = consts.width // 4 - 2 * 10
                  rect_h = consts.height // 4 - 2 * 10 
                  pygame.draw.rect(
                                screen,
                                consts.backgroundColor[state[r][c]],
                                pygame.Rect(rect_x, rect_y, rect_w, rect_h),
                                border_radius = 8,
                                )
                  if state[r][c]==0: continue
                  text = font.render(f'{state[r][c]}', True, consts.cellColor[state[r][c]])
                  score = text.get_rect(
                    center=(rect_x + rect_w / 2, rect_y + rect_h / 2))
                  screen.blit(text, score)

def getEvent():
       while True:
             for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                           return 'quit'
                    elif event.type == pygame.KEYDOWN:
                           keys = pygame.key.get_pressed()
                           if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                                  return 'left'
                           elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                                  return 'right'
                           elif keys[pygame.K_UP] or keys[pygame.K_w]:
                                  return 'up'
                           elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                                  return 'down' 

def compact(line):
    """Compact the line by removing zeros."""
    return [tile for tile in line if tile != 0] + [0] * line.count(0)


def merge(line):
    """Merge adjacent equal tiles."""
    for i in range(len(line) - 1):
        if line[i] == line[i + 1]:
            line[i] *= 2
            line[i + 1] = 0
    return line



def move(command):
       global state
       newState = [r[:] for r in state]
       if command == 'left':
              for row in newState:
                     row[:] = compact(row)  # Compact the row
                     row[:] = merge(row)    # Merge the tiles
                     row[:] = compact(row)  # Compact again

       elif command == 'right':
              for row in newState:
                     row[:] = compact(row[::-1])  # Compact the reversed row
                     row[:] = merge(row[::-1])    # Merge the reversed row
                     row[:] = compact(row[::-1])  # Compact again

       elif command == 'up':
              for col in range(len(newState[0])):
                     column = [newState[row][col] for row in range(len(newState))]
                     column = compact(column)  # Compact the column
                     column = merge(column)    # Merge the tiles
                     column = compact(column)  # Compact again
                     for row in range(len(newState)):
                            newState[row][col] = column[row]

       elif command == 'down':
              for col in range(len(newState[0])):
                     column = [newState[row][col] for row in range(len(newState))]
                     column = compact(column[::-1])  # Compact the reversed column
                     column = merge(column[::-1])    # Merge the reversed column
                     column = compact(column[::-1])  # Compact again
                     for row in range(len(newState)):
                            newState[row][col] = column[row]
       state = newState

def play():
       setNumber(2)
       while True:
              display()
              pygame.display.flip()
              event = getEvent()
              if event == 'quit':
                     break
              move(event)
              if not setNumber():
                     display()
                     break

              
play()
