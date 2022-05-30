import sys
import pygame
import random
import os
import neat

pygame.init()
width, height = 800, 800
GEN = 0
screen = pygame.display.set_mode((width, height))
background_img = pygame.image.load('background.jpg')
obstacle_img = pygame.transform.scale(pygame.image.load('obstacle.png'), (100, 50))
car_img = pygame.transform.scale(pygame.image.load('car.png'), (100, 200))


class Obstacle:
    img = obstacle_img
    velocity = 20

    def __init__(self):
        self.x = random.randrange(0, screen.get_width() - 100)
        self.y = -100

    def move(self):
        self.y += self.velocity

    def draw(self, win):
        win.blit(self.img, (self.x, self.y))

    def collide(self, car):
        car_mask = car.get_mask()
        block_mask = pygame.mask.from_surface(self.img)
        offset = (car.x + car.width/2 - (self.x + self.img.get_width()/2),
                  car.y - self.y)
        point = block_mask.overlap(car_mask, offset)
        if point:
            return True
        return False


class Road:
    img = background_img
    velocity = 20

    def __init__(self, y):
        self.y = y

    def move(self):
        self.y += self.velocity
        if self.y > self.img.get_height():
            self.y = -self.img.get_height()

    def draw(self, win):
        win.blit(self.img, (0, self.y))


class Car:
    img = car_img
    velocity = 20
    rotation_speed = 10

    def __init__(self, x):
        self.x = x
        self.y = 570
        self.rotation_angle = 0
        self.width = self.img.get_width()
        self.height = self.img.get_height()

    def move(self, direction):
        if direction == 'right' and self.x < width - self.img.get_width():
            self.x += self.velocity
            if self.rotation_angle > -20:
                self.rotation_angle -= self.rotation_speed + 10

        elif direction == 'left' and self.x > 0:
            self.x -= self.velocity
            if self.rotation_angle < 20:
                self.rotation_angle += self.rotation_speed + 10

        else:
            if self.rotation_angle < 0:
                self.rotation_angle += self.rotation_speed / 5
                if 5 < self.rotation_angle < -5:
                    self.rotation_angle = 0
            elif self.rotation_angle > 0:
                self.rotation_angle += self.rotation_speed / 5
                if 5 < self.rotation_angle < -5:
                    self.rotation_angle = 0

    def draw(self, win):
        rotated_img = pygame.transform.rotate(self.img, self.rotation_angle)
        new_rect = rotated_img.get_rect(center=(self.x, self.y))
        win.blit(rotated_img, new_rect)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


def draw_window(win, roads, obstacles, cars, score, gen, pop):
    for road in roads:
        road.draw(win)
    for obstacle in obstacles:
        obstacle.draw(win)
    for car in cars:
        car.draw(win)
    text1 = pygame.font.SysFont("sheriff", 50).render("Score: " + str(score), True, (255, 255, 255))
    text2 = pygame.font.SysFont("sheriff", 50).render("Generation: " + str(gen - 1), True, (255, 255, 255))
    text3 = pygame.font.SysFont("sheriff", 50).render("Population: " + str(pop), 1, (255, 255, 255))
    win.blit(text1, (30, 10))
    win.blit(text2, (30, 10 + text1.get_height()))
    win.blit(text3, (30, 10 + text2.get_height() + text1.get_height()))
    pygame.display.update()


def main(genomes, config):
    global GEN
    cars = []
    roads = [Road(0), Road(-800)]
    obstacles = []
    nets = []
    ge = []

    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        nets.append(net)
        cars.append(Car(width / 2 - car_img.get_width() / 2))  # add 100 cars to the middle of the screen
        ge.append(genome)

    GEN += 1
    score = 0
    clock = pygame.time.Clock()

    while True:
        clock.tick(30)
        score += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # elif event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_RIGHT:
            #         cars[0].move("right")
            #     elif event.key == pygame.K_LEFT:
            #         cars[0].move("left")

        if len(cars) > 0:
            if len(obstacles) > 0:
                for i, car in enumerate(cars):
                    ge[i].fitness += 0.1
                    obs = obstacles[0]
                    # middle of the car
                    output = nets[i].activate((abs(car.x + car.width / 2),
                                               # distance from car to bottom of obstacle (Y-AXIS)
                                               abs(car.y - (obs.y + obs.img.get_height())),
                                               # distance from car to bottom of obstacle (X-AXIS)
                                               abs(car.x + car.width - (obs.x + obs.img.get_width())),
                                               abs(obs.x),
                                               abs(obs.x - car.x),
                                               abs(width - (obs.x + obs.img.get_width()))))
                    best = output[0]
                    ind_best = 0
                    for index, o in enumerate(output):
                        if o > best:
                            best = o
                            ind_best = index

                    if ind_best == 0:
                        car.move('left')
                    elif ind_best == 1:
                        car.move('right')
                    elif ind_best == 2:
                        car.move('')
        else:
            break

        if len(obstacles) == 0:
            obstacles.append(Obstacle())
        for i, obstacle in enumerate(obstacles):
            if len(obstacles) <= 2:  # to limit of obstacles that can spawn
                if obstacle.y > 150:
                    obstacles.append(Obstacle())
                if obstacle.y > 350:
                    obstacles.append(Obstacle())

            obstacle.move()

            if obstacle.y > screen.get_height():
                obstacles.pop(i)

            for x, car in enumerate(cars):
                if obstacle.y > car.y + car.height:
                    ge[x].fitness += 10
                if car.x < 0 or car.x + car.width > width or obstacle.collide(car) or abs(car.rotation_angle) > 45:
                    ge[x].fitness -= 3
                    cars.pop(x)
                    ge.pop(x)
                    nets.pop(x)

        for road in roads:
            road.move()

        draw_window(screen, roads, obstacles, cars, score, GEN, len(cars))


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.run(main, 300)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)
