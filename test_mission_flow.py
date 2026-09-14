#!/usr/bin/env python3
"""Модуль Б. Проверка порядка зачётной попытки без симулятора."""

import sys
import types
import unittest
from pathlib import Path


def install_ros_stubs():
    def module(name):
        sys.modules[name] = types.ModuleType(name)
        return sys.modules[name]

    class Parameter:
        def __init__(self, value):
            self.value = value

    class Logger:
        def __init__(self):
            self.lines = []

        def info(self, text):
            self.lines.append(text)

        warning = error = info

    class Clock:
        def __init__(self):
            self.seconds = 0.0

        def now(self):
            return types.SimpleNamespace(nanoseconds=self.seconds * 1e9)

    class Node:
        def __init__(self, name):
            self._params = {}
            self._logger = Logger()
            self._clock = Clock()

        def declare_parameter(self, name, value):
            override = getattr(self, "_overrides", {}).get(name, value)
            self._params[name] = Parameter(override)

        def get_parameter(self, name):
            return self._params[name]

        def get_logger(self):
            return self._logger

        def get_clock(self):
            return self._clock

        def create_publisher(self, *a, **k):
            return types.SimpleNamespace(publish=lambda msg: None)

        def create_subscription(self, *a, **k):
            return None

        def create_timer(self, *a, **k):
            return None

    rclpy = module("rclpy")
    rclpy.ok = lambda: True
    rclpy.node = module("rclpy.node")
    rclpy.node.Node = Node
    rclpy.qos = module("rclpy.qos")
    rclpy.qos.qos_profile_sensor_data = object()

    geometry = module("geometry_msgs")
    geometry.msg = module("geometry_msgs.msg")
    geometry.msg.Twist = lambda: types.SimpleNamespace(
        linear=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
        angular=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
    )

    for name, fields in (("nav_msgs", ("Odometry",)),
                         ("sensor_msgs", ("LaserScan",)),
                         ("std_msgs", ("Bool", "String"))):
        pkg = module(name)
        pkg.msg = module(name + ".msg")
        for field in fields:
            setattr(pkg.msg, field, object)

    tf2 = module("tf2_ros")
    tf2.Buffer = object
    for name in ("LookupException", "ConnectivityException",
                 "ExtrapolationException"):
        setattr(tf2, name, type(name, (Exception,), {}))
    tf2.transform_listener = module("tf2_ros.transform_listener")
    tf2.transform_listener.TransformListener = lambda buffer, node: None


install_ros_stubs()
sys.path.insert(0, str(Path(__file__).resolve().parent))

import main as mission_main  # noqa: E402

TOKENS = ("МАРШРУТ к цели", "МАРШРУТ на старт",
          "movement_start", "movement_stop", "МИССИЯ ЗАВЕРШЕНА")


class Harness:
    """Миссия с подменёнными приводом и камерой."""

    # Тесты этого файла проверяют конечный автомат миссии (порядок фаз,
    # паузу на фиксацию маршрута, учёт времени), а не раскладку поля - им
    # нужна ЛЮБАЯ фиксированная сетка, а не то, что сейчас стоит в DEFAULTS.
    # Раскладка задаётся здесь явно и не зависит от профиля площадки.
    GRID_DEFAULTS = {
        "grid_rows": 6,
        "grid_cols": 6,
        "grid_order": "row_major",
        "aruco_topic": "/RMC2/aruco_id",
        "scan_topic": "/RMC2/scan",
    }

    def __init__(self, **params):
        self.mission = mission_main.Mission.__new__(mission_main.Mission)
        self.mission._overrides = {**self.GRID_DEFAULTS, **params}
        mission_main.Mission.__init__(self.mission)
        self.mission.publish = lambda linear, angular: None
        self.mission.stop = lambda: None
        self.mission.diagnostics = lambda: None
        self.mission.pose = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.mission.yaw = 0.0
        self.mission.aruco_offset = lambda marker_id: (0.0, 0.0)

    @property
    def log(self):
        return self.mission.get_logger().lines

    def tokens(self):
        return [t for line in self.log for t in TOKENS if t in line]

    def advance(self, seconds):
        self.mission._clock.seconds += seconds

    def tick(self):
        self.mission.tick()

    def start(self, marker=0):
        self.mission.current_aruco = marker
        self.mission.aruco_stamp = self.mission.now()
        self.tick()

    def go(self):
        self.mission.request_go("тест")
        self.tick()

    def finish_leg(self):
        """Конечный маркер под камерой, доводка отработала, этап закрыт."""
        self.mission.arrive()
        for _ in range(50):
            if self.mission.phase != "ALIGN":
                return
            self.tick()
        raise AssertionError("доводка не завершилась")


class TestOrder(unittest.TestCase):
    def setUp(self):
        self.h = Harness(start_id=0, target_id=5)

    def test_route_printed_before_movement(self):
        self.h.start()
        self.assertEqual(self.h.mission.phase, "ARMED")
        self.assertIn("МАРШРУТ к цели", self.h.tokens())
        self.assertNotIn("movement_start", self.h.tokens())

    def test_route_is_shortest_with_fewest_turns(self):
        self.h.start()
        self.assertEqual(self.h.mission.route, [0, 1, 2, 3, 4, 5])

    def test_robot_waits_for_the_command(self):
        self.h.start()
        for _ in range(20):
            self.h.tick()
        self.assertEqual(self.h.mission.phase, "ARMED")
        self.assertNotIn("movement_start", self.h.tokens())

    def test_command_outside_armed_phase_is_refused(self):
        self.h.start()
        self.h.go()
        self.assertFalse(self.h.mission.request_go("тест"))

    def test_full_sequence(self):
        self.h.start()
        self.h.go()
        self.h.advance(25.0)
        self.h.finish_leg()
        self.h.mission.phase_deadline = 0.0
        self.h.tick()
        self.assertEqual(self.h.mission.phase, "ARMED")
        self.h.go()
        self.h.advance(20.0)
        self.h.finish_leg()
        self.assertEqual(self.h.tokens(), [
            "МАРШРУТ к цели", "movement_start", "movement_stop",
            "МАРШРУТ на старт", "movement_start", "movement_stop",
            "МИССИЯ ЗАВЕРШЕНА",
        ])

    def test_return_route_avoids_a_cell_found_occupied(self):
        self.h.start()
        self.h.go()
        self.h.finish_leg()
        self.h.mission.occupancy.blocked = {4: 0.0}
        self.h.mission.phase_deadline = 0.0
        self.h.tick()
        self.assertNotIn(4, self.h.mission.route)
        self.assertEqual(self.h.mission.route[-1], 0)


class TestTiming(unittest.TestCase):
    def test_pause_for_route_fixing_is_not_counted(self):
        h = Harness(start_id=0, target_id=2)
        h.start()
        h.advance(200.0)
        h.go()
        h.advance(28.0)
        h.finish_leg()
        h.advance(200.0)
        h.mission.phase_deadline = 0.0
        h.tick()
        h.go()
        h.advance(31.0)
        h.finish_leg()
        self.assertAlmostEqual(sum(h.mission.leg_times), 59.0, places=3)
        self.assertTrue(any("59.0 с" in line for line in h.log))


class TestNoRoute(unittest.TestCase):
    def test_waits_instead_of_arming(self):
        h = Harness(start_id=0, target_id=5, blocked_markers="1,6,7")
        h.start()
        self.assertEqual(h.mission.phase, "WAIT_CLEAR")
        self.assertNotIn("movement_start", h.tokens())


if __name__ == "__main__":
    unittest.main()
