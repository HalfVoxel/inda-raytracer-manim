from manim import *
from manim.mobject.svg.svg_mobject import SVGMobject
from typing import Optional, Tuple
from numpy.linalg import norm
import math

# Calculates the intersection between an infinite line and a circle.
#
# Returns:
#   `None` if there is no intersection.
#   `(t1,t2)` where `t1 <= t2` if the line overlaps with the circle between the two points `line_origin + t1*line_direction` and `line_origin + t2*line_direction`.


def line_circle_intersection(
    line_origin: np.array,  # 2D
    line_direction: np.array,  # 2D
    circle_center: np.array,  # 2D
    circle_radius: float,
) -> Optional[Tuple[float, float]]:
    # Extract XY components
    line_origin = line_origin[0:2]
    line_direction = line_direction[0:2]
    circle_center = circle_center[0:2]

    circle_dir = circle_center - line_origin
    line_dir_sqr = (line_direction**2).sum()

    # Distance between the line and circle center divided by | line_direction | .
    # We store it divided by | line_direction | to avoid unnecessary square roots in the calculations.
    distance_to_line = np.cross(circle_dir, line_direction) / line_dir_sqr

    # Squared distance from closest approach between the line and circle and the point that the line hits on the circle
    # Also divided by | line_direction | ^2 to get a squared t value
    t_offset_from_closest_approach_sqr = circle_radius**2 / line_dir_sqr - distance_to_line**2

    if t_offset_from_closest_approach_sqr > 0.0:
        # t value for the closest point on the line from the circle center
        t_closest_approach = np.dot(circle_dir, line_direction) / line_dir_sqr
        t_offset_from_closest_approach = math.sqrt(t_offset_from_closest_approach_sqr)

        return (
            t_closest_approach - t_offset_from_closest_approach,
            t_closest_approach + t_offset_from_closest_approach,
        )
    else:
        # No intersection
        return None


class RaycastScene:
    def __init__(self):
        self.shapes = []

    def linecast(self, start, end):
        dir = end - start
        t_max = 1
        t_min = 0.01
        hit_shape = None
        for shape in self.shapes:
            hit = line_circle_intersection(start, dir, shape.get_center(), shape.radius)
            if hit is not None:
                (t1, t2) = hit
                if t2 >= t_min and t1 <= t_max:
                    t = max(t1, t_min)
                    t_max = t
                    hit_shape = shape

        return (hit_shape, start + dir * t_max)


def basic_scene():
    camera = SVGMobject("source/video_camera_icon.svg")
    camera.set_stroke(WHITE)
    camera.move_to([-5.0, 0.0, 0])
    camera.scale(0.4)

    cat = SVGMobject("source/cat.svg")
    cat.set_fill(ORANGE, opacity=0.0)
    cat.set_stroke(WHITE)
    cat.scale(1.0)
    cat.move_to([5.0, 0.0, 0.0])

    circle = Circle(radius=1.0)
    circle.set_fill(ORANGE, opacity=1.0)
    circle.set_stroke(WHITE)
    circle.move_to([4.0, 0.0, 0.0])

    return (camera, cat, circle)


class BasicRayTracing(Scene):
    def construct(self):

        (camera, cat, circle) = basic_scene()

        raycastScene = RaycastScene()
        raycastScene.shapes = [circle]

        anims = []
        hits = []
        white_pixels = []
        pixels = []
        n_rays = 19
        for i in range(0, n_rays):
            ray_start = camera.get_center() + [0.5, 0.0, 0.0]
            view_height = 8
            ray_end = circle.get_center() + [2.0, (i/(n_rays-1) - 0.5) * view_height, 0.0]
            far_dist = ray_end[0] - ray_start[0]
            pixel_dist = 4.0

            (hit, point) = raycastScene.linecast(ray_start, ray_end)
            line = Line(ray_start, point).set_color(BLUE)
            anims.append(line)

            hit_color = BLACK
            if hit is not None:
                hit_color = ORANGE

            hits.append(Circle(radius=0.2).shift(point).set_fill(hit_color, opacity=1.0).set_stroke(WHITE))

            pixel_size = pixel_dist / far_dist * view_height / n_rays
            pixel_pos = ray_start + (ray_end - ray_start) * pixel_dist/(ray_end[0] - ray_start[0])
            pixels.append(Square(side_length=pixel_size).move_to(pixel_pos).set_fill(hit_color, opacity=1.0).flip())
            white_pixels.append(Square(side_length=pixel_size).move_to(pixel_pos).flip())

        self.wait()

        self.play(FadeIn(camera))
        self.play(FadeIn(cat))
        self.wait()
        self.play(Transform(cat, circle))
        self.wait()
        self.play(ShowCreation(VGroup(*white_pixels)), lag_ratio=0.3)
        self.wait()

        self.play(ShowCreation(VGroup(*anims)), lag_ratio=0.3, run_time=2.0)
        self.play(FadeOut(VGroup(*white_pixels)), lag_ratio=0.3)

        self.play(*[GrowFromCenter(h) for h in hits], run_time=2.0)
        self.play(*[Transform(h, p) for (h, p) in zip(hits, pixels)], run_time=2.0)
        self.wait()
        self.wait(5.0)

        # self.play(FadeOut(square))


class Shadows(Scene):
    def construct(self):
        (camera, cat, circle) = basic_scene()

        circle = Circle(radius=12.0)
        circle.set_fill(ORANGE, opacity=1.0)
        circle.set_stroke(WHITE)
        circle.move_to([16.0, 0.0, 0.0])
        camera.set_y(-2)

        raycastScene = RaycastScene()
        raycastScene.shapes = [circle]

        self.play(FadeIn(camera), FadeIn(circle))
        self.wait()

        sun = SVGMobject("source/sun.svg").move_to([-2.0, 2.0, 0.0])

        blocker = Circle(radius=0.6).move_to([1.5, 0.5, 0.0]).set_fill(ORANGE, opacity=1.0).set_stroke(WHITE)

        self.play(FadeIn(sun), FadeIn(blocker))

        raycastScene.shapes.append(blocker)

        anims = []
        hits = []
        white_pixels = []
        pixels = []
        shadow_rays = []
        shadow_rays_dash = []

        n_rays = 19
        for i in range(0, n_rays):
            ray_start = camera.get_center() + [0.5, 0.0, 0.0]
            view_height = 10
            far_dist = circle.get_center()[0] - ray_start[0]
            ray_end = circle.get_center() + [0.0, (i/(n_rays-1) - 0.5) * view_height, 0.0]
            pixel_dist = 4.0

            (hit, point) = raycastScene.linecast(ray_start, ray_end)
            line = Line(ray_start, point).set_color(BLUE)
            anims.append(line)

            hit_color = BLACK
            if hit is not None:
                hit_color = ORANGE

            sun_pos = sun.get_center()
            (shadow_hit, shadow_hit_point) = raycastScene.linecast(point, sun_pos)
            shadow_line = Line(point, shadow_hit_point).set_color(RED)

            shadow_line_dash = DashedVMobject(Line(shadow_hit_point, sun_pos).set_color(RED).set_opacity(0.5))

            shadow_rays.append(shadow_line if hit is not None else None)
            shadow_rays_dash.append(shadow_line_dash)

            if shadow_hit is not None:
                hit_color = BLACK

            hits.append(Circle(radius=0.2).shift(point).set_fill(hit_color, opacity=1.0).set_stroke(WHITE))

            pixel_size = pixel_dist / far_dist * view_height / n_rays
            pixel_pos = ray_start + (ray_end - ray_start) * pixel_dist/(ray_end[0] - ray_start[0])
            pixels.append(Square(side_length=pixel_size).move_to(pixel_pos).set_fill(
                hit_color, opacity=1.0).set_stroke(None, width=0).flip())
            white_pixels.append(Square(side_length=pixel_size).move_to(pixel_pos).flip())

        self.play(ShowCreation(anims[11]))
        self.play(ShowCreation(VGroup(shadow_rays[11], shadow_rays_dash[11])))
        self.wait()
        hits[11].set_fill(ORANGE)
        self.play(GrowFromCenter(hits[11]))
        self.play(hits[11].set_fill, BLACK)
        self.wait()
        self.play(FadeOut(hits[11]), FadeOut(anims[11]), FadeOut(shadow_rays[11]), FadeOut(shadow_rays_dash[11]))
        self.wait()

        self.play(ShowCreation(VGroup(*[VGroup(*[x for x in [a, b] if x is not None])
                                        for (a, b) in zip(anims, shadow_rays)])), lag_ratio=1.0, run_time=5.0)
        self.play(*[GrowFromCenter(h) for h in hits], run_time=2.0)
        self.wait()
        self.play(*[Transform(h, p) for (h, p) in zip(hits, pixels)], run_time=2.0)
        self.wait()
        self.wait(5.0)


def random_on_circle():
    length = np.sqrt(np.random.uniform(0, 1))
    angle = np.pi * np.random.uniform(0, 2)

    x = length * np.cos(angle)
    y = length * np.sin(angle)
    return np.array([x,y, 0])

class SoftShadows(Scene):
    def construct(self):
        (camera, cat, circle) = basic_scene()

        circle = Circle(radius=12.0)
        circle.set_fill(ORANGE, opacity=1.0)
        circle.set_stroke(WHITE)
        circle.move_to([16.0, 0.0, 0.0])
        camera.set_y(-2)

        raycastScene = RaycastScene()
        raycastScene.shapes = [circle]

        self.play(FadeIn(camera), FadeIn(circle))
        self.wait()

        sun = SVGMobject("source/sun.svg").move_to([-2.0, 2.0, 0.0])

        blocker = Circle(radius=0.6).move_to([1.5, 0.5, 0.0]).set_fill(ORANGE, opacity=1.0).set_stroke(WHITE)

        self.play(FadeIn(sun), FadeIn(blocker), camera.set_y, -2)

        raycastScene.shapes.append(blocker)

        anims = []
        hits = []
        pixels = []
        shadow_rays = []
        shadow_rays_dash = []

        n_rays = 19
        for i in range(0, n_rays):
            ray_start = camera.get_center() + [0.5, 0.0, 0.0]
            view_height = 10
            far_dist = circle.get_center()[0] - ray_start[0]
            ray_end = circle.get_center() + [0.0, (i/(n_rays-1) - 0.5) * view_height, 0.0]
            pixel_dist = 4.0

            (hit, point) = raycastScene.linecast(ray_start, ray_end)
            line = Line(ray_start, point).set_color(BLUE)
            anims.append(line)

            hit_color = BLACK
            if hit is not None:
                hit_color = ORANGE

            sun_pos = sun.get_center()

            shadow_hits = 0
            n_shadow_rays = 20
            local_shadow_rays = []
            local_shadow_rays_dash = []
            shadow_rays.append(local_shadow_rays)
            shadow_rays_dash.append(local_shadow_rays_dash)

            for j in range(0, n_shadow_rays):
                sun_pos_jitter = sun_pos + random_on_circle()
                (shadow_hit, shadow_hit_point) = raycastScene.linecast(point, sun_pos_jitter)
                shadow_line = Line(point, shadow_hit_point).set_color(ORANGE)

                shadow_line_dash = Line(shadow_hit_point, sun_pos_jitter).set_color(RED).set_opacity(0.2)

                local_shadow_rays.append(shadow_line)
                local_shadow_rays_dash.append(shadow_line_dash)

                if shadow_hit is not None:
                    shadow_hits += 1

            hit_color = interpolate_color(hit_color, BLACK, shadow_hits/n_shadow_rays)

            hits.append(Circle(radius=0.2).shift(point).set_fill(hit_color, opacity=1.0).set_stroke(WHITE))

            pixel_size = pixel_dist / far_dist * view_height / n_rays
            pixel_pos = ray_start + (ray_end - ray_start) * pixel_dist/(ray_end[0] - ray_start[0])
            pixels.append(Square(side_length=pixel_size).move_to(pixel_pos).set_fill(
                hit_color, opacity=1.0).set_stroke(None, width=0).flip())

        self.play(ShowCreation(anims[7]))
        self.play(ShowCreation(VGroup(*[VGroup(a, b) for (a, b) in zip(shadow_rays[7], shadow_rays_dash[7])])))
        self.wait()
        # hits[11].set_fill(ORANGE)
        self.play(GrowFromCenter(hits[7]))
        # self.play(hits[11].set_fill, BLACK)
        # self.wait()
        # self.play(FadeOut(hits[11]), FadeOut(anims[11]), FadeOut(shadow_rays[11]))
        # self.wait()

        # self.play(
        #     ShowCreation(
        #         VGroup(*[VGroup(*[x for x in [a, b] if x is not None]) for (a, b) in zip(anims[11], shadow_rays[11])])
        #     ), lag_ratio=1.0, run_time=5.0)
        self.play(*[GrowFromCenter(h) for h in hits if h != hits[7]], run_time=2.0)
        self.wait()
        self.play(*[Transform(h, p) for (h, p) in zip(hits, pixels)], run_time=2.0)
        self.wait(5.0)


class Reflections(Scene):
    def construct(self):
        (camera, cat, circle) = basic_scene()

        circle = Circle(radius=12.0)
        circle.set_fill(ORANGE, opacity=1.0)
        circle.set_stroke(WHITE)
        circle.move_to([16.0, 0.0, 0.0])
        camera.set_y(-2)

        raycastScene = RaycastScene()
        raycastScene.shapes = [circle]

        self.play(FadeIn(camera), FadeIn(circle))
        self.wait()

        sun = SVGMobject("source/sun.svg").move_to([-2.0, 2.0, 0.0])

        blocker = Circle(radius=0.6).move_to([1.5, 0.5, 0.0]).set_fill(BLUE, opacity=1.0).set_stroke(WHITE)

        self.play(FadeIn(sun), FadeIn(blocker), camera.set_y, -2)

        raycastScene.shapes.append(blocker)


        anims = []
        hits = []
        pixels = []
        shadow_rays = []
        shadow_rays_dash = []

        n_rays = 19
        for i in range(0, n_rays):
            scene = self
            ray_start = camera.get_center() + [0.5, 0.0, 0.0]
            view_height = 10
            far_dist = circle.get_center()[0] - ray_start[0]
            ray_end = circle.get_center() + [0.0, (i/(n_rays-1) - 0.5) * view_height, 0.0]
            pixel_dist = 4.0

            def check_line(ray_start, ray_end):
                (hit, point) = raycastScene.linecast(ray_start, ray_end)
                line = Line(ray_start, point).set_color(BLUE)

                scene.play(ShowCreation(line), run_time=0.2)

                if hit is not None:
                    normal = (point - hit.get_center())
                    normal /= norm(normal)

                    prev_dir_norm = (ray_start - point) / norm(ray_start - point)
                    tangent = (ray_start - point) - np.dot(ray_start - point, normal) * normal
                    out_dir = (ray_start - point) - tangent*2
                    out_dir /= norm(out_dir)
                    out_dir *= 8

                    check_line(point, point + out_dir)
                
                scene.play(FadeOut(line), run_time=0.2)

            check_line(ray_start, ray_end)

        self.wait(5.0)

# BasicRayTracing()
# Shadows()
# SoftShadows()
Reflections()