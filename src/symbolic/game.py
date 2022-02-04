import random as rng
from enum import Enum
from typing import List, Tuple

import numpy as np
import pygame as pg
from pygame.locals import Rect

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
        self.pos = vec((position[0] * 16 + 8, position[1] * 16 + 8))
        self._update()
        self.wall_sprites = None
        self.key_sprites = None
        self.step_size = rng.randint(1, 3)

    def _create_surface(self) -> pg.SurfaceType:
        tileset = pg.image.load("res/tiles.png")
        surf: pg.SurfaceType = pg.Surface((16, 16))
        surf = pg.Surface.convert_alpha(surf)
        surf.set_colorkey((0, 0, 0))
        surf.blit(
            tileset, dest=pg.Rect(0, 0, 16, 16), area=pg.Rect(6 * 16, 15 * 16, 16, 16)
        )
        return surf

    def collide(self, dir: Direction) -> bool:
        if dir is Direction.LEFT:
            self.rect.centerx = self.pos.x - self.step_size
            # self.rect.centery = self.pos.y - self.step_size
        elif dir is Direction.RIGHT:
            self.rect.centerx = self.pos.x + self.step_size
        elif dir is Direction.UP:
            self.rect.centery = self.pos.y - self.step_size
        elif dir is Direction.DOWN:
            self.rect.centery = self.pos.y + self.step_size

        collided = pg.sprite.spritecollide(
            Collider(self.rect), self.wall_sprites, False
        )
        self.rect.center = self.pos
        return len(collided) > 0

    def collide_key(self) -> bool:
        collided = pg.sprite.spritecollide(Collider(self.rect), self.key_sprites, False)
        return len(collided) > 0

    def pick_key(self) -> bool:
        collided = pg.sprite.spritecollide(Collider(self.rect), self.key_sprites, True)
        if len(collided) > 0:
            return True
        return False

    def move(self, dir: Direction) -> bool:
        if not self.collide(dir):
            if dir is Direction.LEFT:
                self.pos.x -= self.step_size
                # self.pos.y -= self.step_size
            elif dir is Direction.RIGHT:
                self.pos.x += self.step_size
            elif dir is Direction.UP:
                self.pos.y -= self.step_size
            elif dir is Direction.DOWN:
                self.pos.y += self.step_size

            self._update()
            self._update_step()
            return True
        return False

    def _update_step(self):
        self.step_size = rng.randint(1, 10)

    def set_position(self, x, y):
        self.pos.x = x
        self.pos.y = y
        self.rect.center = self.pos

    def get_position(self):
        return [self.pos.x, self.pos.y]

    def _update(self):
        self.rect.center = self.pos


class Key(pg.sprite.Sprite):
    key_id = 0

    def __init__(self, position):
        super().__init__()
        self.id = Key.key_id
        Key.key_id += 1
        self.surf = pg.Surface((16, 16))
        tileset = pg.image.load("res/tiles.png")
        self.surf = pg.Surface((16, 16))
        # self.surf.fill((128, 255, 40))
        self.surf.blit(
            tileset, dest=pg.Rect(0, 0, 16, 16), area=pg.Rect(7 * 16, 13 * 16, 16, 16)
        )
        self.rect = self.surf.get_rect(
            center=vec((position[0] * 16 + 8, position[1] * 16 + 8))
        )
        self.surf.set_colorkey((0, 0, 0))


class Wall(pg.sprite.Sprite):
    def __init__(self, position):
        super().__init__()
        self.surf = pg.Surface((16, 16))
        tileset = pg.image.load("res/tiles.png")
        self.surf = pg.Surface((16, 16))
        # self.surf.fill((128, 255, 40))
        self.surf.blit(
            tileset, dest=pg.Rect(0, 0, 16, 16), area=pg.Rect(1 * 16, 1 * 16, 16, 16)
        )
        self.rect = self.surf.get_rect(
            center=vec((position[0] * 16 + 8, position[1] * 16 + 8))
        )


class Ground(pg.sprite.Sprite):
    def __init__(self, position):
        super().__init__()
        self.surf = pg.Surface((16, 16))
        tileset = pg.image.load("res/tiles.png")
        self.surf = pg.Surface((16, 16))
        # self.surf.fill((128, 255, 40))
        self.surf.blit(
            tileset, dest=pg.Rect(0, 0, 16, 16), area=pg.Rect(2 * 16, 3 * 16, 16, 16)
        )
        self.rect = self.surf.get_rect(
            center=vec((position[0] * 16 + 8, position[1] * 16 + 8))
        )


class MiniGame:
    def __init__(
        self,
    ) -> None:
        pg.init()
        pg.font.init()
        self.fps = pg.time.Clock()
        self.WIDTH, self.HEIGHT = 720, 720
        self.screen: pg.Surface = pg.display.set_mode((self.WIDTH, self.HEIGHT), 0, 32)
        self.tmp_screen: pg.Surface = pg.Surface((320, 320))
        self.tmp_full_screen: pg.Surface = pg.Surface((self.WIDTH, self.HEIGHT), 0, 32)
        # creating objects
        self.player: Player = Player((1, 1))
        # Adding sprites to game
        self.sprites: pg.sprite.Group = pg.sprite.Group()
        self.walls: pg.sprite.Group = pg.sprite.Group()
        self.keys: pg.sprite.Group = pg.sprite.Group()
        self.key_list: List[Key] = []
        self.interactibles: pg.sprite.Group = pg.sprite.Group()

        # map_array = [
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        # ]

        map_array = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1],
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
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]

        self.map_array = map_array

        for y in range(len(map_array)):
            for x in range(len(map_array[y])):
                sprite = None
                if map_array[y][x] == 0:
                    sprite = Ground((x, y))

                if map_array[y][x] == 1:
                    sprite = Wall((x, y))
                    self.walls.add(sprite)

                if map_array[y][x] == 2:
                    sprite = Key((x, y))
                    self.sprites.add(Ground((x, y)))
                    self.keys.add(sprite)
                    self.key_list.append(sprite)
                    self.interactibles.add(sprite)

                if sprite is not None:
                    self.sprites.add(sprite)

        self.player.wall_sprites = self.walls
        self.player.key_sprites = self.keys

        self.sprites.add(self.player)
        self.interactibles.add(self.player)

    def get_key_states(self) -> List[int]:
        """Returns a list with 0 for non picked keys and 1 for picked keys

        Returns:
            List[int]: state list of keys
        """
        states = []
        for key in self.key_list:
            if key.alive():
                states.append(0)
            else:
                states.append(1)

        return states

    def set_key_states(self, states: List[int]):
        for i, key in enumerate(self.key_list):
            key.kill()
            if states[i] == 0:
                self.keys.add(key)
                self.interactibles.add(key)
                self.sprites.add(key)

    def reset(self):
        self.set_key_states([0] * len(self.key_list))

    def draw(self):
        self.tmp_screen.fill((0, 0, 0))
        # print("Drawing")
        for entity in self.sprites:
            self.tmp_screen.blit(entity.surf, entity.rect)
        pg.transform.scale(self.tmp_screen, (self.WIDTH, self.HEIGHT), self.screen)
        pg.display.update()
        self.fps.tick()

    def overlay_text(
        self,
        text: str,
        pos: Tuple[int, int],
        size: int = 12,
        color=(255, 255, 255),
        alpha=255,
        bg_color=(0, 0, 0),
        bg_alpha=0,
    ):
        font = pg.font.SysFont("monospace", size)
        textsurf = font.render(text, True, color)
        textsurf.set_alpha(alpha)
        bg_surf = textsurf.copy()
        bg_surf.fill(bg_color)
        bg_surf.set_alpha(bg_alpha)
        self.screen.blit(bg_surf, pos)
        self.screen.blit(textsurf, pos)

    def overlay_background(self):
        self.tmp_screen.fill((0, 0, 0))
        for entity in self.sprites:
            if not self.interactibles.has(entity):
                self.tmp_screen.blit(entity.surf, entity.rect)
        pg.transform.scale(self.tmp_screen, (self.WIDTH, self.HEIGHT), self.screen)
        # pg.display.update()
        # self.fps.tick()

    def overlay(self, color, alpha, ui_offset=0):
        self.tmp_screen.fill((0, 0, 0, 0))
        self.tmp_full_screen.fill((0, 0, 0, 0))
        self.tmp_full_screen.set_colorkey((0, 0, 0))

        for sprite in self.interactibles.copy():
            surf: pg.Surface = sprite.surf.convert_alpha()
            surfcolored = surf.copy()
            surfcolored.fill(color)
            surfcolored.set_alpha(128)
            surf.blit(surfcolored, (0, 0))
            self.tmp_screen.blit(surf, sprite.rect)
            pg.transform.scale(
                self.tmp_screen, (self.WIDTH, self.HEIGHT), self.tmp_full_screen
            )
            self.tmp_full_screen.set_alpha(alpha)
            self.screen.blit(self.tmp_full_screen, (0, 0))
        self.overlay_text(
            "Keys: {}".format(int(np.array(self.get_key_states()).sum())),
            (self.WIDTH - 100 + ui_offset, 10),
            size=15,
            color=color,
            alpha=alpha,
            bg_color=(255, 255, 255),
            bg_alpha=10,
        )
        # pg.display.update()

    def overlay_transition(self, start, end, start_color, end_color, alpha):
        self.tmp_screen.fill((0, 0, 0, 0))
        self.tmp_full_screen.fill((0, 0, 0, 0))
        self.tmp_full_screen.set_colorkey((0, 0, 0))

        points = np.linspace(start, end, 30)

        colors = np.linspace(start_color, end_color, len(points))

        for i in range(len(points) - 1):
            pg.draw.line(self.tmp_screen, colors[i], points[i], points[i + 1])
        pg.transform.scale(
            self.tmp_screen, (self.WIDTH, self.HEIGHT), self.tmp_full_screen
        )
        self.tmp_full_screen.set_alpha(alpha)
        self.screen.blit(self.tmp_full_screen, (0, 0))
        # pg.display.update()

    def update_screen(self):
        pg.display.update()
        self.fps.tick()

    def destroy(self):
        pg.display.quit()

    def screenshot(self, name: str):
        print("Writing", name, "-", self.WIDTH, self.HEIGHT)
        pg.image.save_extended(self.screen, name)
