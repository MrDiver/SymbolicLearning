
from enum import Enum
import pygame as pg
from pygame.locals import *
import random as rng
import numpy as np
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
        self.surf = self._create_surface()
        self.rect: Rect = self.surf.get_rect()
        self.pos = vec((position[0]*16+16, position[1]*16+16))
        self._update()
        self.walls = None
        self.step_size = rng.randint(1, 3)

    def _create_surface(self) -> pg.SurfaceType:
        tileset = pg.image.load("res/tiles.png")
        surf: pg.SurfaceType = pg.Surface((16, 16))
        surf = pg.Surface.convert_alpha(surf)
        surf.set_colorkey((0, 0, 0))
        surf.blit(tileset, dest=pg.Rect(
            0, 0, 16, 16), area=pg.Rect(6*16, 15*16, 16, 16))
        return surf

    def collide(self, dir: Direction) -> bool:
        if dir is Direction.LEFT:
            self.rect.centerx = self.pos.x - self.step_size
        elif dir is Direction.RIGHT:
            self.rect.centerx = self.pos.x + self.step_size
        elif dir is Direction.UP:
            self.rect.centery = self.pos.y + self.step_size
        elif dir is Direction.DOWN:
            self.rect.centery = self.pos.y - self.step_size

        collided = pg.sprite.spritecollide(
            Collider(self.rect), self.walls, False)
        self.rect.center = self.pos
        return len(collided) > 0

    def move(self, dir: Direction) -> bool:
        if not self.collide(dir):
            if dir is Direction.LEFT:
                self.pos.x -= self.step_size
            elif dir is Direction.RIGHT:
                self.pos.x += self.step_size
            elif dir is Direction.UP:
                self.pos.y += self.step_size
            elif dir is Direction.DOWN:
                self.pos.y -= self.step_size

            self._update()
            self._update_step()
            return True
        return False

    def _update_step(self):
        self.step_size = rng.randint(1, 3)

    def set_state(self, x, y):
        self.pos.x = x
        self.pos.y = y
        self.rect.center = self.pos

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
        self.WIDTH, self.HEIGHT = 720*2, 720*2
        self.screen = pg.display.set_mode((self.WIDTH, self.HEIGHT), 0, 32)
        self.tmp_screen = pg.Surface((320, 320))
        self.tmp_full_screen = pg.Surface((self.WIDTH, self.HEIGHT), 0, 32)
        # creating objects
        self.player = Player((2, 2))
        # Adding sprites to game
        self.sprites = pg.sprite.Group()
        self.walls = pg.sprite.Group()

        map_array = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
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
        # print("Drawing")
        for entity in self.sprites:
            self.tmp_screen.blit(entity.surf, entity.rect)
        pg.transform.scale(
            self.tmp_screen, (self.WIDTH, self.HEIGHT), self.screen)
        pg.display.update()
        self.fps.tick()

    def draw_background(self):
        self.tmp_screen.fill((0, 0, 0))
        for entity in self.sprites:
            if entity is not self.player:
                self.tmp_screen.blit(entity.surf, entity.rect)
        pg.transform.scale(
            self.tmp_screen, (self.WIDTH, self.HEIGHT), self.screen)
        pg.display.update()
        self.fps.tick()

    def overlay(self, color, alpha):
        self.tmp_screen.fill((0, 0, 0, 0))
        self.tmp_full_screen.fill((0, 0, 0, 0))
        self.tmp_full_screen.set_colorkey((0, 0, 0))
        surf: pg.Surface = self.player._create_surface()
        surfcolored = surf.copy()
        surfcolored.fill(color)
        surfcolored.set_alpha(128)
        surf.blit(surfcolored, (0, 0))
        self.tmp_screen.blit(surf, self.player.rect)
        pg.transform.scale(
            self.tmp_screen, (self.WIDTH, self.HEIGHT), self.tmp_full_screen)
        self.tmp_full_screen.set_alpha(alpha)
        self.screen.blit(self.tmp_full_screen, (0, 0))
        pg.display.update()

    def overlay_transition(self, start, end, start_color, end_color, alpha):
        self.tmp_screen.fill((0, 0, 0, 0))
        self.tmp_full_screen.fill((0, 0, 0, 0))
        self.tmp_full_screen.set_colorkey((0, 0, 0))

        # _xs_min = min(start[0], end[0])
        # _xs_max = max(start[0], end[0])
        # xs = np.arange(_xs_min, _xs_max+1, 1)
        # _ys_min = min(start[1], end[1])
        # _ys_max = max(start[1], end[1])
        # ys = np.arange(_ys_min, _ys_max+1, 1)

        # # print(_ys)
        # if len(xs) > len(ys):
        #     ys = np.interp(xs, [_xs_min, _xs_max], [_ys_min, _ys_max])
        # else:
        #     xs = np.interp(ys, [_ys_min, _ys_max], [_xs_min, _xs_max])
        #     # ys = np.interp(_ys, start[1], end[1])
        # points = np.transpose([xs, ys])
        points = np.linspace(start, end, 50)

        colors = np.linspace(start_color, end_color, len(points))

        for i in range(len(points)-1):
            pg.draw.line(self.tmp_screen, colors[i], points[i], points[i+1])
        pg.transform.scale(
            self.tmp_screen, (self.WIDTH, self.HEIGHT), self.tmp_full_screen)
        self.tmp_full_screen.set_alpha(alpha)
        self.screen.blit(self.tmp_full_screen, (0, 0))
        pg.display.update()

    def destroy(self):
        pg.display.quit()
