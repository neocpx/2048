import pygame
import random
import constants as consts

state=[[0,0,0,0],
       [0,0,0,0],
       [0,0,0,0],
       [0,0,0,0]]

pygame.init()
screen=pygame.display.set_mode((consts.width,consts.height))
pygame.display.set_caption('2048')
icon = pygame.image.load('./imgs/logo.svg')
pygame.display.set_icon(icon)
font = pygame.font.SysFont('./fonts/openSans.ttf', 60)


def setNumber(n=1):
       for _ in range(n):
              emptyPos = [(i,j) for i in range(4) for j in range(4) if state[i][j]==0]
              if not emptyPos: return False
              randPos = random.choice(emptyPos)
              state[randPos[0]][randPos[1]] = random.choice((2,4))
       return True

def getScore():
    global state
    return sum(tile_value for row in state for tile_value in row if tile_value not in (0, 2))


def display(score):
    instructions = 'Use arrow keys (or WASD) to move tiles. Combine matching tiles to reach 2048!. use TAB for AI mode and N for new game'
    screen.fill(consts.defaultBgColor)

    logo_text = font.render(f'2048', True, (0, 0, 0))
    logo_rect = logo_text.get_rect(topright=(consts.height+(consts.width - consts.height - 20), 20))
    screen.blit(logo_text, logo_rect)

    # Render score text
    score_text = font.render(f'Score: {score}', True, (0, 0, 0))
    score_rect = score_text.get_rect(topright=(consts.width - 20, 65))
    screen.blit(score_text, score_rect)

    instructions_width = consts.width - consts.height - 40
    words = instructions.split(' ')
    lines = ['']
    current_line = 0

    for word in words:
        test_line = lines[current_line] + word + ' '
        test_width, _ = font.size(test_line)

        if test_width <= instructions_width:
            lines[current_line] = test_line
        else:
            lines.append(word + ' ')
            current_line += 1

    for i, line in enumerate(lines):
        text_surface = font.render(line, True, (0, 0, 0))
        text_rect = text_surface.get_rect(topleft=(consts.height + 20, consts.height + 5 + 20))
        screen.blit(text_surface, text_rect)

    # Render game tiles
    for r in range(4):
        for c in range(4):
            rect_x = c * consts.height // 4 + 10
            rect_y = r * consts.height // 4 + 10
            rect_w = consts.height // 4 - 2 * 10
            rect_h = consts.height // 4 - 2 * 10
            pygame.draw.rect(
                screen,
                consts.backgroundColor[state[r][c]],
                pygame.Rect(rect_x, rect_y, rect_w, rect_h),
                border_radius=8,
            )
            if state[r][c] == 0:
                continue
            text = font.render(f'{state[r][c]}', True, consts.cellColor[state[r][c]])
            num_rect = text.get_rect(center=(rect_x + rect_w / 2, rect_y + rect_h / 2))
            screen.blit(text, num_rect)
       

    pygame.display.flip()


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

def compact_and_merge(line):
    """Compact the line by removing zeros and merge adjacent equal tiles."""
    compacted_line = [tile for tile in line if tile != 0]
    merged_line = []
    i = 0
    while i < len(compacted_line) - 1:
        if compacted_line[i] == compacted_line[i + 1]:
            merged_line.append(compacted_line[i] * 2)
            i += 1
        else:
            merged_line.append(compacted_line[i])
        i += 1
    if i == len(compacted_line) - 1:
        merged_line.append(compacted_line[i])

    # Fill the remaining spaces with zeros
    merged_line += [0] * (len(line) - len(merged_line))

    return merged_line


def move(command):
    global state
    newState = [r[:] for r in state]
    for idx, row_or_column in enumerate(newState):
        if command == 'left':
            newState[idx] = compact_and_merge(row_or_column)
        elif command == 'right':
            newState[idx] = compact_and_merge(row_or_column[::-1])[::-1]
        elif command == 'up':
            column = compact_and_merge([newState[row][idx] for row in range(len(newState))])
            for row in range(len(newState)):
                newState[row][idx] = column[row]
        elif command == 'down':
            column = compact_and_merge([newState[row][idx] for row in range(len(newState))][::-1])[::-1]
            for row in range(len(newState)):
                newState[row][idx] = column[row]
    state = newState

def play():
   score=0 
   setNumber(2)
   while True:
          display(score)
          print(f'\rscore: {score}',end=' ', flush=True)
          pygame.display.flip()
          event = getEvent()
          if event == 'quit':
                 break
          move(event)
          score+=getScore()
          if not setNumber():
                 display(score)
                 break

              
play()
