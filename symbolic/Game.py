
from enum import Enum
import pygame as pg
from pygame.locals import *
vec = pg.math.Vector2


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Collider(pg.sprite.Sprite):
    def __init__(self, rect) -> None:
        super().__init__()
        self.rect = rect


class Player(pg.sprite.Sprite):
    def __init__(self, position):
        super().__init__()
        tileset = pg.image.load("res/tiles.png")
        self.surf = pg.Surface((16, 16))
        self.surf.fill((0, 0, 0))
        self.surf.set_colorkey((0, 0, 0))
        self.surf.blit(tileset, dest=pg.Rect(
            0, 0, 16, 16), area=pg.Rect(6*16, 15*16, 16, 16))
        self.rect: Rect = self.surf.get_rect()
        self.pos = vec((position[0]*16+16, position[1]*16+16))
        self._update()
        self.walls = None

    def collide(self, dir: Direction) -> bool:
        if dir is Direction.LEFT:
            self.rect.centerx = self.pos.x - 1
        elif dir is Direction.RIGHT:
            self.rect.centerx = self.pos.x + 1
        elif dir is Direction.UP:
            self.rect.centery = self.pos.y + 1
        elif dir is Direction.DOWN:
            self.rect.centery = self.pos.y - 1

        collided = pg.sprite.spritecollide(
            Collider(self.rect), self.walls, False)
        self.rect.center = self.pos
        return len(collided) > 0

    def move(self, dir: Direction) -> bool:
        if not self.collide(dir):
            if dir is Direction.LEFT:
                self.pos.x -= 1
            elif dir is Direction.RIGHT:
                self.pos.x += 1
            elif dir is Direction.UP:
                self.pos.y += 1
            elif dir is Direction.DOWN:
                self.pos.y -= 1

            self._update()
            return True
        return False

    def _update(self):
        self.rect.center = self.pos


class Wall(pg.sprite.Sprite):
    def __init__(self, position):
        super().__init__()
        self.surf = pg.Surface((16, 16))
        tileset = pg.image.load("res/tiles.png")
        self.surf = pg.Surface((16, 16))
        # self.surf.fill((128, 255, 40))
        self.surf.blit(tileset, dest=pg.Rect(
            0, 0, 16, 16), area=pg.Rect(1*16, 1*16, 16, 16))
        self.rect = self.surf.get_rect(
            center=vec((position[0]*16+8, position[1]*16+8)))


class Ground(pg.sprite.Sprite):
    def __init__(self, position):
        super().__init__()
        self.surf = pg.Surface((16, 16))
        tileset = pg.image.load("res/tiles.png")
        self.surf = pg.Surface((16, 16))
        # self.surf.fill((128, 255, 40))
        self.surf.blit(tileset, dest=pg.Rect(
            0, 0, 16, 16), area=pg.Rect(2*16, 3*16, 16, 16))
        self.rect = self.surf.get_rect(
            center=vec((position[0]*16+8, position[1]*16+8)))


class MiniGame:
    def __init__(self,) -> None:
        pg.init()
        self.fps = pg.time.Clock()
        self.WIDTH, self.HEIGHT = 720, 720
        self.screen = pg.display.set_mode((self.WIDTH, self.HEIGHT))
        self.tmp_screen = pg.Surface((320, 320))
        # creating objects
        self.player = Player((2, 2))
        # Adding sprites to game
        self.sprites = pg.sprite.Group()
        self.walls = pg.sprite.Group()

        map_array = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

        for y in range(len(map_array)):
            for x in range(len(map_array[y])):
                sprite: Wall | Ground | None = None
                if map_array[y][x] == 0:
                    sprite = Ground((x, y))

                if map_array[y][x] == 1:
                    sprite = Wall((x, y))
                    self.walls.add(sprite)

                if sprite is not None:
                    self.sprites.add(sprite)

        self.player.walls = self.walls

        self.sprites.add(self.player)

    def draw(self):
        self.tmp_screen.fill((0, 0, 0))

        for entity in self.sprites:
            self.tmp_screen.blit(entity.surf, entity.rect)
        pg.transform.scale(
            self.tmp_screen, (self.WIDTH, self.HEIGHT), self.screen)
        pg.display.update()
        self.fps.tick()
