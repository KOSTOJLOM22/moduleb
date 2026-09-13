#!/usr/bin/env python3
"""Тесты геометрии и планировщика модуля Б. ROS не требуется."""

import math
import unittest

from navigation_core import (
    MarkerGrid,
    OccupancyTracker,
    count_turns,
    normalize_angle,
    points_to_markers,
    route_with_turns,
    shortest_path,
    step_direction,
    turn_cost,
)


class FakeScan:
    """Имитация sensor_msgs/LaserScan объединённого лидара."""

    def __init__(self, ranges, angle_min=-math.pi, angle_max=math.pi):
        self.ranges = ranges
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.angle_increment = (angle_max - angle_min) / max(len(ranges) - 1, 1)


class TestAngles(unittest.TestCase):
    def test_normalize(self):
        self.assertAlmostEqual(abs(normalize_angle(3 * math.pi)), math.pi, places=6)
        self.assertAlmostEqual(abs(normalize_angle(-3 * math.pi)), math.pi, places=6)
        self.assertAlmostEqual(normalize_angle(0.5), 0.5, places=6)
        self.assertAlmostEqual(normalize_angle(2 * math.pi + 0.3), 0.3, places=6)
        self.assertTrue(-math.pi - 1e-9 <= normalize_angle(100.0) <= math.pi + 1e-9)


class TestGrid(unittest.TestCase):
    def setUp(self):
        self.grid = MarkerGrid(6, 6, 1.0)

    def test_coords(self):
        self.assertEqual(self.grid.coords(0), (0.0, 0.0))
        self.assertEqual(self.grid.coords(5), (0.0, 5.0))
        self.assertEqual(self.grid.coords(6), (-1.0, 0.0))
        self.assertEqual(self.grid.coords(35), (-5.0, 5.0))

    def test_nearest_roundtrip(self):
        for mid in range(self.grid.size):
            mx, my = self.grid.coords(mid)
            found, dist = self.grid.nearest(mx, my)
            self.assertEqual(found, mid)
            self.assertAlmostEqual(dist, 0.0, places=9)

    def test_nearest_outside(self):
        mid, dist = self.grid.nearest(2.0, 0.0)
        self.assertIsNone(mid)
        self.assertEqual(dist, float("inf"))

    def test_nearest_offset(self):
        mid, dist = self.grid.nearest(-1.0 + 0.2, 2.0 - 0.1)
        self.assertEqual(mid, 8)
        self.assertAlmostEqual(dist, math.hypot(0.2, 0.1), places=9)

    def test_border(self):
        self.assertTrue(self.grid.is_border(0))
        self.assertTrue(self.grid.is_border(35))
        self.assertFalse(self.grid.is_border(7))

    def test_anchor_roundtrip(self):
        """Поле -> одометрия -> поле должно возвращать исходную точку."""
        for yaw in (0.0, 0.7, -2.5, math.pi):
            grid = MarkerGrid(6, 6, 1.0, angle_offset=0.009)
            grid.anchor(0, 3.21, -1.75, yaw)
            for mid in (0, 7, 21, 35):
                gx, gy = grid.coords(mid)
                ox, oy = grid.to_odom(gx, gy)
                bx, by = grid.to_grid(ox, oy)
                self.assertAlmostEqual(gx, bx, places=9)
                self.assertAlmostEqual(gy, by, places=9)

    def test_anchor_preserves_distance(self):
        """Преобразование — движение без масштабирования."""
        grid = MarkerGrid(6, 6, 1.0)
        grid.anchor(0, 10.0, -4.0, 1.1)
        a = grid.marker_to_odom(0)
        b = grid.marker_to_odom(35)
        expected = math.hypot(5.0, 5.0)
        self.assertAlmostEqual(math.dist(a, b), expected, places=9)


class TestDriftCorrection(unittest.TestCase):
    """
    Одометрия копит угловую ошибку. Раньше угол поля фиксировался один раз на
    старте, из-за чего к концу маршрута робот приходил на маркеры со смещением
    и цеплял ножку стеллажа. refine_theta/reanchor убирают накопление.
    """

    def setUp(self):
        self.grid = MarkerGrid(6, 6, 1.0)
        self.grid.anchor(0, 0.0, 0.0, 0.0)

    def test_theta_starts_at_pi(self):
        self.assertAlmostEqual(abs(self.grid.theta), math.pi, places=6)

    def test_refine_pulls_theta_toward_measurement(self):
        expected = self.grid.marker_to_odom(1)
        drift = math.radians(6.0)
        actual = (expected[0] * math.cos(drift) - expected[1] * math.sin(drift),
                  expected[0] * math.sin(drift) + expected[1] * math.cos(drift))
        before = self.grid.theta
        corr = self.grid.refine_theta(0, (0.0, 0.0), 1, actual, gain=0.4)
        self.assertIsNotNone(corr)
        self.assertAlmostEqual(math.degrees(corr), 6.0 * 0.4, places=3)
        self.assertGreater(abs(normalize_angle(self.grid.theta - before)), 0)

    def test_refine_converges_over_several_legs(self):
        true_theta = normalize_angle(math.pi + math.radians(6.0))
        fdx, fdy = 0.0, 1.0
        actual = (fdx * math.cos(true_theta) - fdy * math.sin(true_theta),
                  fdx * math.sin(true_theta) + fdy * math.cos(true_theta))
        for _ in range(8):
            self.grid.refine_theta(0, (0.0, 0.0), 1, actual, gain=0.4)
        self.assertLess(abs(math.degrees(normalize_angle(
            self.grid.theta - true_theta))), 0.2)

    def test_refine_ignores_short_leg(self):
        self.assertIsNone(self.grid.refine_theta(0, (0.0, 0.0), 0, (0.1, 0.0)))

    def test_reanchor_zeroes_translation_drift(self):
        true_odom = self.grid.marker_to_odom(7)
        drifted = (true_odom[0] + 0.12, true_odom[1] - 0.07)
        self.grid.reanchor(7, drifted[0], drifted[1])
        self.assertAlmostEqual(math.dist(self.grid.marker_to_odom(7), drifted),
                               0.0, places=9)
        self.assertAlmostEqual(
            math.dist(self.grid.marker_to_odom(7), self.grid.marker_to_odom(8)),
            1.0, places=9)

    def test_reanchor_keeps_theta(self):
        before = self.grid.theta
        self.grid.reanchor(7, 5.0, 5.0)
        self.assertAlmostEqual(self.grid.theta, before, places=12)


class TestFinalOrientation(unittest.TestCase):
    """
    Робот должен закончить миссию в той же ориентации, в которой начал. Курс
    хранится в системе ПОЛЯ: theta уточняется по ходу маршрута, поэтому
    запоминать угол одометрии нельзя — вернувшись в него, робот встал бы криво
    ровно на величину накопленной поправки.
    """

    def setUp(self):
        self.grid = MarkerGrid(6, 6, 1.0)
        self.grid.anchor(0, 0.0, 0.0, 0.0)

    def test_field_yaw_roundtrip(self):
        for odom in (0.0, 1.3, -2.7, math.pi):
            back = self.grid.odom_yaw(self.grid.field_yaw(odom))
            self.assertAlmostEqual(math.cos(back), math.cos(odom), places=9)
            self.assertAlmostEqual(math.sin(back), math.sin(odom), places=9)

    def test_target_odom_yaw_follows_theta_correction(self):
        start_field = self.grid.field_yaw(0.0)
        before = self.grid.odom_yaw(start_field)
        true_theta = normalize_angle(self.grid.theta + math.radians(2.0))
        fdx, fdy = 0.0, 1.0
        actual = (fdx * math.cos(true_theta) - fdy * math.sin(true_theta),
                  fdx * math.sin(true_theta) + fdy * math.cos(true_theta))
        for _ in range(10):
            self.grid.refine_theta(0, (0.0, 0.0), 1, actual, gain=0.4)
        after = self.grid.odom_yaw(start_field)
        self.assertAlmostEqual(
            math.degrees(normalize_angle(after - before)), 2.0, places=1)

    def test_start_field_yaw_is_stable_under_reanchor(self):
        start_field = self.grid.field_yaw(0.4)
        self.grid.reanchor(7, 3.0, -2.0)
        self.assertAlmostEqual(self.grid.field_yaw(0.4), start_field, places=12)


class TestPlanner(unittest.TestCase):
    def setUp(self):
        self.grid = MarkerGrid(6, 6, 1.0)

    def test_shortest_is_manhattan(self):
        route = self.grid.route(0, 35)
        self.assertEqual(route[0], 0)
        self.assertEqual(route[-1], 35)
        self.assertEqual(len(route), 11)

    def test_same_marker(self):
        self.assertEqual(self.grid.route(7, 7), [7])

    def test_avoids_blocked(self):
        route = self.grid.route(0, 33, exclude={21, 27})
        self.assertNotIn(21, route)
        self.assertNotIn(27, route)
        self.assertEqual(route[-1], 33)

    def test_no_path_when_walled_in(self):
        route = self.grid.route(0, 7, exclude={1, 6, 8, 13})
        self.assertIsNone(route)

    def test_detour_is_longer(self):
        direct = self.grid.route(0, 2)
        detour = self.grid.route(0, 2, exclude={1})
        self.assertEqual(len(direct), 3)
        self.assertGreater(len(detour), len(direct))

    def test_shortest_path_missing_node(self):
        self.assertIsNone(shortest_path({0: []}, 0, 99))


class TestLidarMapping(unittest.TestCase):
    def setUp(self):
        self.grid = MarkerGrid(6, 6, 1.0)
        self.grid.anchor(0, 0.0, 0.0, 0.0)

    def test_point_maps_to_expected_marker(self):
        """
        Робот на маркере 0. Из-за поворота поля (yaw+pi) точка, лежащая на 1 м
        впереди робота, попадает на маркер 6 (row1,col0).
        """
        gx, gy = self.grid.to_grid(1.0, 0.0)
        mid, dist = self.grid.nearest(gx, gy)
        self.assertEqual(mid, 6)
        self.assertLess(dist, 0.05)

    def test_points_to_markers_counts(self):
        pts = [(1.0, 0.0), (1.02, 0.01), (0.98, -0.02)]
        hits = points_to_markers(pts, self.grid, 0.0, 0.0, 0.0, 0.35)
        self.assertEqual(hits.get(6), 3)

    def test_points_outside_field_ignored(self):
        pts = [(-2.0, 0.0)]
        hits = points_to_markers(pts, self.grid, 0.0, 0.0, 0.0, 0.35)
        self.assertEqual(hits, {})

    def test_far_from_marker_centre_ignored(self):
        gx, gy = -1.0, 0.5
        ox, oy = self.grid.to_odom(gx, gy)
        pts = [(-ox, -oy)]
        hits = points_to_markers(pts, self.grid, 0.0, 0.0, math.pi, 0.35)
        self.assertEqual(hits, {})


class TestAttributionRange(unittest.TestCase):
    """
    Дальняя привязка точек к маркерам заставляла робота разворачиваться, ещё не
    доехав до ближайшего маркера. Ограничение дальности это лечит.
    """

    def setUp(self):
        self.grid = MarkerGrid(6, 6, 1.0)
        self.grid.anchor(0, 0.0, 0.0, 0.0)

    def test_far_point_ignored_with_limit(self):
        far = [(3.0, 0.0), (3.02, 0.01), (2.98, -0.01)]
        without = points_to_markers(far, self.grid, 0.0, 0.0, 0.0, 0.35)
        with_limit = points_to_markers(far, self.grid, 0.0, 0.0, 0.0, 0.35,
                                       max_point_range=1.6)
        self.assertTrue(without)
        self.assertEqual(with_limit, {})

    def test_near_point_still_counted(self):
        near = [(1.0, 0.0), (1.01, 0.0)]
        hits = points_to_markers(near, self.grid, 0.0, 0.0, 0.0, 0.35,
                                 max_point_range=1.6)
        self.assertEqual(hits.get(6), 2)


class TestLegGeometry(unittest.TestCase):
    """Курс отрезка всегда кратен 90°, снос считается поперёк отрезка."""

    def setUp(self):
        self.grid = MarkerGrid(6, 6, 1.0)
        self.grid.anchor(0, 0.0, 0.0, 0.0)

    def leg(self, origin, target, point):
        ax, ay = self.grid.coords(origin)
        bx, by = self.grid.coords(target)
        heading = math.atan2(by - ay, bx - ax)
        dx, dy = point[0] - ax, point[1] - ay
        along = dx * math.cos(heading) + dy * math.sin(heading)
        cross = -dx * math.sin(heading) + dy * math.cos(heading)
        return heading, along, cross

    def test_headings_are_multiples_of_90(self):
        for origin, target in ((0, 1), (1, 0), (0, 6), (6, 0), (7, 8), (8, 14)):
            heading, _, _ = self.leg(origin, target, self.grid.coords(origin))
            self.assertAlmostEqual(math.degrees(heading) % 90.0, 0.0, places=6)

    def test_along_and_cross(self):
        heading, along, cross = self.leg(0, 1, (0.05, 0.5))
        self.assertAlmostEqual(along, 0.5, places=6)
        self.assertAlmostEqual(cross, -0.05, places=6)

    def test_cross_sign_flips_with_side(self):
        _, _, left = self.leg(0, 1, (-0.05, 0.5))
        _, _, right = self.leg(0, 1, (0.05, 0.5))
        self.assertGreater(left, 0)
        self.assertLess(right, 0)


class TestShortestRouteWithTurns(unittest.TestCase):
    """Определение кратчайшего маршрута из задания финала ЧВТ 2026."""

    def setUp(self):
        self.grid = MarkerGrid(6, 6, 1.0)

    def turns(self, route, start_direction=None):
        return self.grid.turns_in_route(route, start_direction)

    def test_same_row_is_straight_and_has_no_turns(self):
        route = self.grid.route(0, 5)
        self.assertEqual(route, [0, 1, 2, 3, 4, 5])
        self.assertEqual(self.turns(route), 0)

    def test_diagonal_target_uses_single_turn(self):
        route = self.grid.route(0, 35)
        self.assertEqual(len(route), 11)
        self.assertEqual(self.turns(route), 1)

    def test_plain_bfs_zigzags_when_obstacles_force_a_detour(self):
        """
        На пустой сетке BFS случайно даёт тот же маршрут, на занятой — нет.
        """
        blocked = {15, 16, 20, 26, 34}
        graph = self.grid.build_graph(blocked)
        bfs = shortest_path(graph, 30, 4)
        route = self.grid.route(30, 4, exclude=blocked)
        self.assertEqual(len(bfs), len(route))
        self.assertEqual(self.turns(bfs), 5)
        self.assertEqual(self.turns(route), 1)

    def test_empty_grid_bfs_is_already_turn_optimal(self):
        """Честная граница: на чистой сетке выигрыша по поворотам нет."""
        graph = self.grid.build_graph()
        for goal in range(36):
            with self.subTest(goal=goal):
                bfs = shortest_path(graph, 0, goal)
                route = self.grid.route(0, goal)
                self.assertEqual(self.turns(bfs), self.turns(route))

    def test_length_is_never_traded_for_fewer_turns(self):
        for goal in range(36):
            with self.subTest(goal=goal):
                graph = self.grid.build_graph()
                self.assertEqual(
                    len(self.grid.route(0, goal)), len(shortest_path(graph, 0, goal))
                )

    def test_initial_heading_is_counted_as_a_turn(self):
        facing_along_row = self.grid.direction_from_field_yaw(math.pi / 2.0)
        self.assertEqual(facing_along_row, (0, 1))
        route = self.grid.route(0, 35, start_direction=facing_along_row)
        self.assertEqual(self.turns(route, facing_along_row), 1)
        self.assertEqual(route[1], 1)

    def test_route_around_obstacle_keeps_only_free_cells(self):
        route = self.grid.route(0, 35, exclude={7, 8, 14})
        self.assertFalse({7, 8, 14} & set(route))
        self.assertEqual(route[0], 0)
        self.assertEqual(route[-1], 35)

    def test_no_route_returns_none(self):
        walled_off = {1, 6}
        self.assertIsNone(self.grid.route(0, 35, exclude=walled_off))

    def test_turn_cost_counts_reversal_as_two(self):
        self.assertEqual(turn_cost((0, 1), (0, 1)), 0)
        self.assertEqual(turn_cost((0, 1), (1, 0)), 1)
        self.assertEqual(turn_cost((0, 1), (0, -1)), 2)
        self.assertEqual(turn_cost(None, (0, 1)), 0)

    def test_step_direction_is_normalized(self):
        self.assertEqual(step_direction((0, 0), (0, 1)), (0, 1))
        self.assertEqual(step_direction((3, 2), (2, 2)), (-1, 0))

    def test_count_turns_on_explicit_cells(self):
        self.assertEqual(count_turns([(0, 0), (0, 1), (0, 2)]), 0)
        self.assertEqual(count_turns([(0, 0), (0, 1), (1, 1)]), 1)

    def test_route_with_turns_handles_trivial_and_missing(self):
        graph = self.grid.build_graph()
        self.assertEqual(route_with_turns(graph, 4, 4, self.grid.cell), [4])
        self.assertIsNone(route_with_turns(graph, 4, 99, self.grid.cell))


class TestDeclaredOccupancy(unittest.TestCase):
    """Ячейки, объявленные занятыми, не попадают в маршрут."""

    def setUp(self):
        self.grid = MarkerGrid(6, 6, 1.0)
        self.occupied = {7, 8, 20, 26}

    def test_route_never_contains_an_occupied_cell(self):
        route = self.grid.route(33, 0, exclude=self.occupied)
        self.assertFalse(self.occupied & set(route))
        self.assertEqual(route[0], 33)
        self.assertEqual(route[-1], 0)

    def test_detour_costs_little(self):
        direct = self.grid.route(33, 0)
        detour = self.grid.route(33, 0, exclude=self.occupied)
        self.assertEqual(len(detour), len(direct))

    def test_target_reachable_with_cells_occupied(self):
        self.assertIsNotNone(self.grid.route(0, 33, exclude=self.occupied))

    def test_occupied_endpoint_makes_the_route_unbuildable(self):
        """
        Отсечение занятых концов маршрута — забота plan(), не планировщика.
        """
        self.assertIsNone(self.grid.route(7, 35, exclude=self.occupied))
        self.assertIsNotNone(self.grid.route(7, 35, exclude=self.occupied - {7}))


class TestOccupancy(unittest.TestCase):
    def test_hysteresis_blocks_after_three_frames(self):
        tracker = OccupancyTracker(hits_to_block=3, min_points=2, forget_time=5.0)
        for t in range(2):
            newly, _ = tracker.update({12: 4}, now=float(t))
            self.assertEqual(newly, [])
        newly, _ = tracker.update({12: 4}, now=2.0)
        self.assertEqual(newly, [12])
        self.assertEqual(tracker.ids(), [12])

    def test_single_noisy_point_ignored(self):
        tracker = OccupancyTracker(hits_to_block=3, min_points=2, forget_time=5.0)
        for t in range(10):
            tracker.update({12: 1}, now=float(t))
        self.assertEqual(tracker.ids(), [])

    def test_interrupted_streak_resets(self):
        tracker = OccupancyTracker(hits_to_block=3, min_points=2, forget_time=5.0)
        tracker.update({12: 3}, now=0.0)
        tracker.update({12: 3}, now=1.0)
        tracker.update({}, now=2.0)
        tracker.update({12: 3}, now=3.0)
        self.assertEqual(tracker.ids(), [])

    def test_forgets_after_timeout(self):
        tracker = OccupancyTracker(hits_to_block=1, min_points=1, forget_time=5.0)
        tracker.update({12: 3}, now=0.0)
        self.assertEqual(tracker.ids(), [12])
        _, freed = tracker.update({}, now=10.0)
        self.assertEqual(freed, [12])
        self.assertEqual(tracker.ids(), [])

    def test_protected_marker_never_blocked(self):
        tracker = OccupancyTracker(hits_to_block=1, min_points=1, forget_time=5.0)
        for t in range(5):
            tracker.update({0: 9}, now=float(t), protected={0})
        self.assertEqual(tracker.ids(), [])


class TestScanFrameOffset(unittest.TestCase):
    """
    Скан приходит во фрейме laser_merged, повёрнутом относительно base_link.
    Без поправки точка, лежащая слева от робота, выглядит как точка спереди —
    ровно та ошибка, из-за которой робот тормозил перед пустым полем.
    """

    def _scan_with_point(self, angle, dist, n=361):
        ranges = [float("inf")] * n
        increment = 2 * math.pi / (n - 1)
        idx = int(round((angle + math.pi) / increment))
        ranges[idx] = dist
        return FakeScan(ranges)

    def test_without_offset_point_is_misplaced(self):
        from navigation_core import summarize_scan
        scan = self._scan_with_point(0.0, 0.95)
        front, points = summarize_scan(scan, 0.28, 3.0, math.radians(25))
        self.assertAlmostEqual(front, 0.95, places=2)

    def test_offset_moves_point_out_of_front_sector(self):
        from navigation_core import summarize_scan
        scan = self._scan_with_point(0.0, 0.95)
        front, points = summarize_scan(scan, 0.28, 3.0, math.radians(25),
                                       offset=(0.0, 0.0, math.radians(-60)))
        self.assertEqual(front, float("inf"))
        self.assertEqual(len(points), 1)
        bx, by = points[0]
        self.assertAlmostEqual(math.degrees(math.atan2(by, bx)), -60.0, places=1)
        self.assertAlmostEqual(math.hypot(bx, by), 0.95, places=3)

    def test_offset_translation_applied(self):
        from navigation_core import summarize_scan
        scan = self._scan_with_point(0.0, 1.0)
        _, points = summarize_scan(scan, 0.28, 3.0, math.radians(25),
                                   offset=(0.5, -0.2, 0.0))
        self.assertAlmostEqual(points[0][0], 1.5, places=3)
        self.assertAlmostEqual(points[0][1], -0.2, places=3)


class TestScanSummary(unittest.TestCase):
    def test_front_distance_and_filtering(self):
        from navigation_core import summarize_scan
        n = 361
        ranges = [float("inf")] * n
        ranges[n // 2] = 0.5
        ranges[0] = 0.4
        ranges[10] = 0.1
        scan = FakeScan(ranges)
        front, points = summarize_scan(scan, robot_radius=0.28,
                                       max_range=3.0, front_half_angle=math.radians(25))
        self.assertAlmostEqual(front, 0.5, places=6)
        self.assertEqual(len(points), 2)

    def test_empty_scan(self):
        from navigation_core import summarize_scan
        front, points = summarize_scan(FakeScan([]), 0.28, 3.0, 0.4)
        self.assertEqual(front, float("inf"))
        self.assertEqual(points, [])


class TestFieldLayout(unittest.TestCase):
    """Поле финала ЧВТ: 5x5 с нумерацией по столбцам.

    Раскладка записана так, как она нарисована в документе: строки сверху
    вниз, столбцы слева направо. Это единственная запись, которую можно
    сверить глазами с картинкой.
    """

    DOCUMENT_GRID = [
        [24, 19, 14, 9, 4],
        [23, 18, 13, 8, 3],
        [22, 17, 12, 7, 2],
        [21, 16, 11, 6, 1],
        [20, 15, 10, 5, 0],
    ]

    def field_grid(self):
        from navigation_core import MarkerGrid
        return MarkerGrid(rows=5, cols=5, spacing=1.0, order="col_major")

    def test_layout_matches_the_document(self):
        grid = self.field_grid()
        rendered = [
            [grid.marker(row, col) for col in range(grid.cols - 1, -1, -1)]
            for row in range(grid.rows - 1, -1, -1)
        ]
        self.assertEqual(rendered, self.DOCUMENT_GRID)

    def test_cell_and_marker_are_inverse(self):
        grid = self.field_grid()
        for marker_id in range(25):
            row, col = grid.cell(marker_id)
            self.assertEqual(grid.marker(row, col), marker_id)

    def test_neighbours_do_not_wrap_around_the_column_edge(self):
        """4 - верх правого столбца, 5 - низ следующего. Они не соседи."""
        graph = self.field_grid().build_graph()
        self.assertEqual(len(graph), 25)
        self.assertNotIn(5, graph[4])
        self.assertEqual(sorted(graph[4]), [3, 9])
        self.assertEqual(sorted(graph[0]), [1, 5])
        self.assertEqual(sorted(graph[12]), [7, 11, 13, 17])

    def test_simulator_layout_is_unchanged(self):
        from navigation_core import MarkerGrid
        grid = MarkerGrid(rows=6, cols=6, spacing=1.0, order="row_major")
        self.assertEqual(grid.cell(31), (5, 1))
        self.assertEqual(grid.marker(5, 1), 31)
        self.assertEqual(len(grid.build_graph()), 36)

    def test_unknown_order_is_refused(self):
        from navigation_core import MarkerGrid
        with self.assertRaises(ValueError):
            MarkerGrid(rows=5, cols=5, spacing=1.0, order="diagonal")


if __name__ == "__main__":
    unittest.main(verbosity=2)
