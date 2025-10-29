import math
from typing import List

from epiclibcpp.epiclib import geometric_utilities as gu
from plum import dispatch

"""
This set of simple classes is used to compute positions and directions in the plane, 
in both cartesian and polar coordinates, and also with respect to line segments
and polygon regions.

Most classes are defined with "struct" to make all members public by default.

These classes make no assumptions about units of measurement of distance. Angles
can be specified in radians, but radians can be converted to and from 
trigonometry degrees in which 0 degrees corresponds to (x > 0, y = 0), 
and 90 degrees corresponds to (x = 0, y > 0).

Point is a set of (x, y) coordinates. 

A Cartesian_vector is (delta_x, delta_y) - a displacement in Cartesian coordinates.

A Polar_vector is (r, theta) - a displacement in polar coordinates using radians.

Various overloaded operators support computations of positions and directions.

A Line_segment represents a line that passes through two Points in 
both parametric and general form. Functions are provided (mostly inline for speed)
for computing distances of a Point from the line, and intersections between lines.
There are two definitions supported for distance of a point from the line:
    1. The distance from the line extended infinitely past the endpoints of the segment.
    2. The distance from the line segment only, which is either:
        a. the length of the line that both passes through the point and is perpendicular
        to the line segment and intersects the line segment between the endpoints. 
        b. the distance from the closest endpoint if the intersecting perpendicular does
        not intersect between the endpoints.

The Polygon class represents a polygon as a series of Points for vertices that make up
the endpoints of a set of line segments. The last Point wraps around to the first point
to define the last line segment making up the polygon. A distance_inside function
is used to compute whether a Point is inside the polygon and the distance of the Point
from the nearest line segment.
"""

GU_pi = math.pi


class Point:
    def __init__(self, x: float = 0.0, y: float = 0.0):
        ...

    def __new__(cls, *args, **kwargs):
        return gu.Point(*args, **kwargs)


class Size:
    def __init__(self, h: float = 0.0, v: float = 0.0):
        ...

    def __new__(cls, *args, **kwargs):
        return gu.Size(*args, **kwargs)


class Cartesian_vector:
    """A Cartesian_vector contains an x, y displacement"""

    @dispatch
    def __init__(self, delta_x: float = 0.0, delta_y: float = 0.0):
        ...

    @dispatch
    def __init__(self, p1: Point, p2: Point):
        """
        construct a Cartesian_vector from two Points,
        showing the vector from p1 to p2
        that is, p1 + cv => p2
        """
        ...

    @dispatch
    def __init__(self, pv: "Polar_vector"):
        """construct a Cartesian_vector from a Polar_vector"""
        ...

    def __new__(cls, *args, **kwargs):
        return gu.Cartesian_vector(*args, **kwargs)


class Polar_vector:
    """
    Polar_vector describes a displacement in terms of polar coordinates
    with angle in radians
    """

    @dispatch
    def __init__(self, r: float = 0.0, theta: float = 0.0):
        ...

    @dispatch
    def __init__(self, p1: Point, p2: Point):
        """
        construct a Polar_vector from two Points,
        showing the vector from p1 to p2
        that is, p1 + pv => p2
        """
        ...

    @dispatch
    def __init__(self, cv: Cartesian_vector):
        """construct a Polar_vector from a Cartesian_vector"""
        ...

    def __new__(cls, *args, **kwargs):
        return gu.Polar_vector(*args, **kwargs)


class Line_segment:
    """
    The Line_segment class stores two endpoints and the terms of the general form,
    and precomputes for speed some values used in various calculations.

    In the parametric form, line is specified in terms of its endpoints, p1 and p2,
    the differences dx = p2.x - p1.x, and dy = p2.y - p1.y, and a parameter t
    that describes where on the line a particular point is, starting from p1.

    A point is on the line if its (x, y) coordinates can be specified as
        x = p1.x + t * dx
        y = p1.y + t * dy
        for some value of t.
    Note that p1 corresponds to t = 0, p2 to t = 1. So for a given (x, y), a value
    of t between 0 and 1 means that the point lies between the endpoints. A value
    of t < 0 means that t comes "before" p1


    In the general form, a point (x, y) is on the line if it satisfies the equation
        A * x + B * y + C == 0, where

        len = distance between p1 and p2
        A = -dy/len;
        B = dx/len;
        C = -(-dy*x1 + dx*y1)/len

    For speed, most functions are inline, and dx, dy, len, and c (numerator of C)
    are computed and saved when Line_segment is initialized.

    Note that x and y values are doubles, and if they have fractional values, some
    functions and tests that assume exact values will not be accurate.
    But if all coordinates have integral values, then functions such as is_horizontal
    should be correct.
    """

    @dispatch
    def __init__(self):
        ...

    @dispatch
    def __init__(self, p1: Point, p2: Point):
        ...

    def __new__(cls, *args, **kwargs):
        return gu.Line_segment(*args, **kwargs)

    def get_p1(self) -> Point:
        ...

    def get_p2(self) -> Point:
        ...

    # return the center of the bounding box
    def get_center(self) -> Point:
        ...

    # return the Size of the bounding box
    def get_size(self) -> Size:
        ...

    def get_dx(self) -> float:
        ...

    def get_dy(self) -> float:
        ...

    def get_a(self) -> float:
        ...

    def get_b(self) -> float:
        ...

    def get_c(self) -> float:
        ...

    def get_length(self) -> float:
        ...

    def is_horizontal(self) -> bool:
        ...

    def is_vertical(self) -> bool:
        ...

    def is_on_infinite_line(self, p: Point) -> bool:
        """
        These functions treat the segment as an infinite line
        Return True if p is on the infinite line described by the segment
        """
        ...

    def closest_point_on_infinite_line(self, p: Point) -> Point:
        """Compute the closest point on the infinite line"""
        ...

    def distance_from_infinite_line(self, p: Point) -> float:
        """Return the distance from the line to a point"""
        ...

    def parameter(self, p: Point) -> float:
        """
        Return the parameter value for the point.
        The value will be between 0 and 1 if the point is between the endpoints.
        Otherwise, t corresponds to the position of the point along the line.
        t < 0 means p is "before" p1, t > 1, "after" p2.
        """
        ...

    def parameter_given_x(self, x: float) -> float:
        """Compute t given a value for x"""
        ...

    def x_given_parameter(self, t: float) -> float:
        """Return x of point on the line specified by t"""
        ...

    def parameter_given_y(self, y: float) -> float:
        """Compute t given a value for y"""
        ...

    def y_given_parameter(self, t: float) -> float:
        """Return y of point on the line specified by t"""
        ...

    def point_on_line(self, t: float) -> Point:
        """Return the point on the line specified by t"""
        ...

    def closest_point_on_segment(self, p: Point) -> Point:
        """
        Return the closest point on the line segment to p
        This is either a point on the line, or the closest endpoint
        """
        ...

    def distance_from_segment(self, p: Point) -> float:
        """Return the distance from p to the closest point on the line segment"""
        ...


class Polygon:
    """
    The polygon class stores a vector of points corresponding to adjacent vertices
    of the polygon - the last point "wraps" to the first point to close the polygon.
    """

    @dispatch
    def __init__(self):
        ...

    @dispatch
    def __init__(self, vertices: List[Point]):
        ...

    def __new__(cls, *args, **kwargs):
        return gu.Polygon(*args, **kwargs)

    def add_vertex(self, p: Point):
        """create the polygon by adding points in order"""
        ...

    def clear(self):
        """empty the polygon"""
        ...

    def get_vertices(self) -> List[Point]:
        """supply a reference to the vertices"""
        ...

    def get_center(self) -> Point:
        """return the center of the bounding box"""
        ...

    def get_size(self) -> Size:
        """return the Size of the bounding box"""
        ...

    def distance_inside(self, p: Point) -> float:
        """
        compute the distance of p from the edge of the polygon.
        A negative value means that p is outside the polygon, positive means p is inside
        """
        ...


def is_point_inside_rectangle(p: Point, rect_loc: Point, rect_size: Size) -> bool:
    """
    # Given a point and a rectangle, return True if the point is inside the rectangle,
    # False if not.
    """
    return gu.is_point_inside_rectangle(p, rect_loc, rect_size)


def clip_line_to_rectangle(
    line: Line_segment, rect_loc: Point, rect_size: Size
) -> Line_segment:
    """
    Given a line and a rectangle, compute and return the line segment that is the line
    clipped to the rectangle.
    """
    return gu.clip_line_to_rectange(line, rect_loc, rect_size)


def compute_center_intersecting_line(
    start_to_center: Line_segment, rect_size: Size, clipped_line: Line_segment
) -> bool:
    """
    Given a line from a point through the center of a rectangle, calculate
    the line that goes from the closest point of intersection on the rectangle
    to the center of the rectangle. Use for e.g. Fitts ID calculations.
    """
    return gu.compute_center_intersecting_line(start_to_center, rect_size, clipped_line)


def closest_distance(p: Point, rect_center: Point, rect_size: Size) -> float:
    """Compute the closest distance from p to the rectangle given by center, size"""
    return gu.closest_distance(p, rect_center, rect_size)


def cartesian_distance(p1: Point, p2: Point) -> float:
    """return the distance between two Points"""
    return gu.cartesian_distance(p1, p2)


def to_radians(theta_deg: float) -> float:
    """angle units conversion functions"""
    return gu.to_radians(theta_deg)


def to_degrees(theta_rad: float) -> float:
    """angle units conversion functions"""
    return gu.to_degrees(theta_rad)


def degrees_subtended(size_measure: float, distance_measure: float) -> float:
    """use these functions for things like visual angle per pixel"""
    return gu.degrees_subtended(size_measure, distance_measure)


def degrees_subtended_per_unit(
    units_per_measure: float, distance_measure: float
) -> float:
    """use these functions for things like visual angle per pixel"""
    return gu.degrees_subtended_per_unit(units_per_measure, distance_measure)


def units_per_degree_subtended(
    units_per_measure: float, distance_measure: float
) -> float:
    """use these functions for things like visual angle per pixel"""
    return gu.units_per_degree_subtended(units_per_measure, distance_measure)


if __name__ == "__main__":
    p1 = Point(10, 20)
    p2 = Point(-10, -20)
    ls = Line_segment(p1, p2)
    print(f"{p1=} {p2=}")
    print(f"{ls=}")
    print(f"{ls.get_size()=}")
