#!/usr/bin/env python3
"""Модуль Б. Геометрия сетки ArUco-маркеров, граф и разбор лидара."""

from __future__ import annotations

import heapq
import math
from collections import deque


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_angle(angle):
    """Приводит угол к диапазону (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_from_odom(msg):
    return yaw_from_quaternion(msg.pose.pose.orientation)


def rotate_vector(x, y, angle):
    ca, sa = math.cos(angle), math.sin(angle)
    return x * ca - y * sa, x * sa + y * ca


CELL_COST = 1000
TURN_COST = 1


def step_direction(cell_a, cell_b):
    """
    Направление шага между соседними ячейками: (drow, dcol) из {-1, 0, 1}.
    """
    drow = cell_b[0] - cell_a[0]
    dcol = cell_b[1] - cell_a[1]
    if drow:
        drow = 1 if drow > 0 else -1
    if dcol:
        dcol = 1 if dcol > 0 else -1
    return drow, dcol


def turn_cost(previous, current):
    """Поворотов на 90 градусов между двумя направлениями. Разворот — два."""
    if previous is None or previous == current:
        return 0
    if previous[0] == -current[0] and previous[1] == -current[1]:
        return 2
    return 1


def count_turns(cells, start_direction=None):
    turns = 0
    direction = start_direction
    for index in range(1, len(cells)):
        step = step_direction(cells[index - 1], cells[index])
        turns += turn_cost(direction, step)
        direction = step
    return turns


def route_with_turns(graph, start, goal, cell_of, start_direction=None):
    """Кратчайший маршрут: минимум ячеек, при равенстве — минимум поворотов."""
    if start == goal:
        return [start]
    if start not in graph or goal not in graph:
        return None
    counter = 0
    best = {(start, start_direction): 0}
    parent = {(start, start_direction): None}
    heap = [(0, counter, start, start_direction)]
    while heap:
        cost, _, node, direction = heapq.heappop(heap)
        state = (node, direction)
        if cost > best.get(state, float("inf")):
            continue
        if node == goal:
            cells = []
            while state is not None:
                cells.append(state[0])
                state = parent[state]
            return list(reversed(cells))
        for neighbour in graph.get(node, ()):
            step = step_direction(cell_of(node), cell_of(neighbour))
            if step == (0, 0):
                continue
            new_cost = cost + CELL_COST + TURN_COST * turn_cost(direction, step)
            new_state = (neighbour, step)
            if new_cost < best.get(new_state, float("inf")):
                best[new_state] = new_cost
                parent[new_state] = (node, direction)
                counter += 1
                heapq.heappush(heap, (new_cost, counter, neighbour, step))
    return None


def shortest_path(graph, start, goal):
    """
    BFS по невзвешенному графу. Список ID маркеров или None, если пути нет.
    """
    if start == goal:
        return [start]
    queue = deque([start])
    parent = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for nbr in graph.get(current, []):
            if nbr not in parent:
                parent[nbr] = current
                queue.append(nbr)
    if goal not in parent:
        return None
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return list(reversed(path))


class MarkerGrid:
    """Регулярная сетка ArUco-маркеров."""

    def __init__(self, rows, cols, spacing, angle_offset=0.0, order="row_major"):
        self.rows = int(rows)
        self.cols = int(cols)
        # Нумерация маркеров. "row_major" - симулятор: соседний номер лежит
        # вбок, id = row * cols + col. "col_major" - поле финала ЧВТ: соседний
        # номер лежит вверх по столбцу, id = col * rows + row. Размер сетки и
        # правило нумерации - две независимые вещи, менять одно без другого
        # нельзя, иначе робот поедет не в ту клетку и никакой ошибки не будет.
        self.order = str(order)
        if self.order not in ("row_major", "col_major"):
            raise ValueError(f"Неизвестная нумерация поля: {self.order}")
        self.spacing = float(spacing)
        self.angle_offset = float(angle_offset)
        self._anchor = None
        self._theta = 0.0

    @property
    def size(self):
        return self.rows * self.cols

    def valid(self, marker_id):
        return 0 <= marker_id < self.size

    def coords(self, marker_id):
        """Координаты центра маркера в системе поля."""
        row, col = self.cell(marker_id)
        return -row * self.spacing, col * self.spacing

    def cell(self, marker_id):
        """Ячейка маркера как (строка, столбец)."""
        value = int(marker_id)
        if self.order == "row_major":
            return divmod(value, self.cols)
        column, row = divmod(value, self.rows)
        return row, column

    def marker(self, row, col):
        """Номер маркера по ячейке. None, если ячейка вне поля."""
        row = int(row)
        col = int(col)
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return None
        if self.order == "row_major":
            return row * self.cols + col
        return col * self.rows + row

    def direction_from_field_yaw(self, field_yaw):
        """Ближайшее из четырёх направлений сетки к текущему курсу в поле."""
        best_direction = None
        best_error = None
        for direction in ((-1, 0), (1, 0), (0, 1), (0, -1)):
            heading = math.atan2(direction[1], -direction[0])
            error = abs(normalize_angle(heading - field_yaw))
            if best_error is None or error < best_error:
                best_error = error
                best_direction = direction
        return best_direction

    def turns_in_route(self, route, start_direction=None):
        return count_turns([self.cell(mid) for mid in route], start_direction)

    def nearest(self, gx, gy):
        """
        (id, расстояние) ближайшего маркера к точке поля. (None, inf) вне
        сетки.
        """
        row = int(round(-gx / self.spacing))
        col = int(round(gy / self.spacing))
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return None, float("inf")
        mid = self.marker(row, col)
        if mid is None:
            return None, float("inf")
        mx, my = self.coords(mid)
        return mid, math.hypot(gx - mx, gy - my)

    def is_border(self, marker_id):
        row, col = self.cell(marker_id)
        return row in (0, self.rows - 1) or col in (0, self.cols - 1)

    def build_graph(self, exclude=()):
        exclude = set(exclude)
        graph = {}
        for mid in range(self.size):
            if mid in exclude:
                continue
            row, col = self.cell(mid)
            # Порядок обхода - это поведение, а не деталь: BFS разрывает им
            # ничью между равными маршрутами. Сохранён прежний.
            neighbors = [
                self.marker(row, col + 1),
                self.marker(row, col - 1),
                self.marker(row + 1, col),
                self.marker(row - 1, col),
            ]
            graph[mid] = [
                n for n in neighbors if n is not None and n not in exclude
            ]
        return graph

    def route(self, start, goal, exclude=(), start_direction=None):
        """
        Кратчайший маршрут: минимум ячеек, при равенстве — минимум поворотов.
        """
        return route_with_turns(
            self.build_graph(exclude), start, goal, self.cell, start_direction
        )


    def anchor(self, marker_id, odom_x, odom_y, odom_yaw):
        """
        Первичная привязка: робот стоит над marker_id в данной позе одометрии.
        """
        mx, my = self.coords(marker_id)
        self._anchor = (odom_x, odom_y, mx, my)
        self._theta = normalize_angle(odom_yaw + math.pi + self.angle_offset)

    def reanchor(self, marker_id, odom_x, odom_y):
        """
        Обнуление накопленного сдвига: робот подтверждённо стоит над marker_id,
        значит его текущая поза одометрии и есть точка этого маркера. Угол не
        трогаем — его уточняет refine_theta().
        """
        if self._anchor is None:
            return
        mx, my = self.coords(marker_id)
        self._anchor = (odom_x, odom_y, mx, my)

    def refine_theta(self, from_marker, from_odom, to_marker, to_odom,
                     gain=0.4, min_leg=0.5):
        """Уточнение угла поворота поля по пройденному отрезку."""
        if self._anchor is None:
            return None
        fx1, fy1 = self.coords(from_marker)
        fx2, fy2 = self.coords(to_marker)
        fdx, fdy = fx2 - fx1, fy2 - fy1
        odx, ody = to_odom[0] - from_odom[0], to_odom[1] - from_odom[1]
        if math.hypot(fdx, fdy) < min_leg or math.hypot(odx, ody) < min_leg:
            return None
        measured = normalize_angle(math.atan2(ody, odx) - math.atan2(fdy, fdx))
        error = normalize_angle(measured - self._theta)
        correction = gain * error
        self._theta = normalize_angle(self._theta + correction)
        return correction

    @property
    def anchored(self):
        return self._anchor is not None

    @property
    def theta(self):
        return self._theta

    def to_odom(self, gx, gy):
        ax, ay, amx, amy = self._anchor
        dx, dy = rotate_vector(gx - amx, gy - amy, self._theta)
        return ax + dx, ay + dy

    def to_grid(self, ox, oy):
        ax, ay, amx, amy = self._anchor
        dx, dy = rotate_vector(ox - ax, oy - ay, -self._theta)
        return amx + dx, amy + dy

    def marker_to_odom(self, marker_id):
        return self.to_odom(*self.coords(marker_id))


    def field_yaw(self, odom_yaw):
        return normalize_angle(odom_yaw - self._theta)

    def odom_yaw(self, field_yaw):
        return normalize_angle(field_yaw + self._theta)


def summarize_scan(scan, robot_radius, max_range, front_half_angle,
                   offset=(0.0, 0.0, 0.0)):
    """Один проход по объединённому скану."""
    tx, ty, tyaw = offset
    front = float("inf")
    points = []
    if not scan.ranges:
        return front, points
    for i, r in enumerate(scan.ranges):
        if not math.isfinite(r):
            continue
        if r < robot_radius or r > max_range:
            continue
        angle = scan.angle_min + i * scan.angle_increment
        lx = r * math.cos(angle)
        ly = r * math.sin(angle)
        bx, by = rotate_vector(lx, ly, tyaw)
        bx += tx
        by += ty
        if abs(normalize_angle(math.atan2(by, bx))) <= front_half_angle:
            front = min(front, math.hypot(bx, by))
        points.append((bx, by))
    return front, points


def points_to_markers(points, grid, robot_x, robot_y, robot_yaw, radius,
                      max_point_range=None):
    """
    Переводит точки скана (base_link) в занятые маркеры сетки. Возвращает
    {marker_id: число попавших точек}.
    """
    hits = {}
    if not grid.anchored:
        return hits
    for bx, by in points:
        if max_point_range is not None and math.hypot(bx, by) > max_point_range:
            continue
        dx, dy = rotate_vector(bx, by, robot_yaw)
        gx, gy = grid.to_grid(robot_x + dx, robot_y + dy)
        mid, dist = grid.nearest(gx, gy)
        if mid is not None and dist <= radius:
            hits[mid] = hits.get(mid, 0) + 1
    return hits


class OccupancyTracker:
    """
    Гистерезис занятости маркеров: маркер считается занятым после hits_to_block
    подтверждений подряд и освобождается, если подтверждений не было
    forget_time секунд. Защищает от одиночных шумовых лучей.
    """

    def __init__(self, hits_to_block=3, min_points=2, forget_time=6.0):
        self.hits_to_block = int(hits_to_block)
        self.min_points = int(min_points)
        self.forget_time = float(forget_time)
        self.blocked = {}
        self._counter = {}

    def update(self, hits, now, protected=()):
        """Возвращает (ставшие занятыми, ставшие свободными)."""
        protected = set(protected)
        newly_blocked, newly_free = [], []

        for mid in list(self._counter):
            if mid not in hits:
                self._counter[mid] = 0

        for mid, count in hits.items():
            if mid in protected or count < self.min_points:
                continue
            self._counter[mid] = self._counter.get(mid, 0) + 1
            if self._counter[mid] >= self.hits_to_block:
                if mid not in self.blocked:
                    newly_blocked.append(mid)
                self.blocked[mid] = now

        for mid in list(self.blocked):
            if mid in protected:
                del self.blocked[mid]
                self._counter[mid] = 0
                newly_free.append(mid)
            elif now - self.blocked[mid] > self.forget_time:
                del self.blocked[mid]
                self._counter[mid] = 0
                newly_free.append(mid)

        return newly_blocked, newly_free

    def ids(self):
        return sorted(self.blocked)
