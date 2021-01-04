import numpy as np
import png
import re


class SVGRenderer:
    def __init__(self, size):
        """
        Constructor. Initializes a renderer for a image with the specified size. \n
        :param size: A tuple (height, width) \n
        """
        self.image = np.zeros((*size, 3), dtype='uint8')
        self.size = size
        self.strokeColor = (40, 255, 255)
        self.fillColor = (100, 20, 250)
        self.strokeWidth = 2
        self.max_x = 0
        self.max_y = 0
        self.min_x = self.size[0]
        self.min_y = self.size[1]

    # def setStyle(self, stroke, strokeWidth, fill):
    #
    #    self.style.update(
    #        {
    #        "stroke":stroke,
    #        "stroke-width":strokeWidth,
    #        "fill":fill
    #        }
    #    )

    def get_pixel_color(self, x, y):
        """
        Gets the color of a pixel.
        :param x: The x location of the pixel
        :param y: The y location of the pixel
        :return: (r,g,b) tuple of the color
        """
        return tuple(self.image[x][y])

    def put_pixel(self, x: int, y: int, **kwargs):
        """
        Sets the color of a pixel at position (x,y) to be the setted color strokeColor
        :param x: The x location of the pixel
        :param y: The y location of the pixel
        :return: None
        """
        if kwargs.get('color') == "fill":
            color = self.fillColor
            try:
                self.image[x][y][0] = color[0]
                self.image[x][y][1] = color[1]
                self.image[x][y][2] = color[2]
            except Exception:
                pass
        else:
            color = self.strokeColor
            # I draw instead of a point, a square like in
            x_start = x - self.strokeWidth // 2
            y_start = y - (self.strokeWidth - 1 - self.strokeWidth // 2)
            for i in range(self.strokeWidth):
                for j in range(self.strokeWidth):
                    if 0 <= x_start + i < self.size[1] and 0 <= y_start + j < self.size[0]:
                        try:
                            self.image[y_start + j][x_start + i][0] = color[0]
                            self.image[y_start + j][x_start + i][1] = color[1]
                            self.image[y_start + j][x_start + i][2] = color[2]
                            if self.max_x < x_start + i:
                                self.max_x = x_start + i
                            if self.max_y < y_start + j:
                                self.max_y = y_start + j
                            if self.min_x > x_start + i:
                                self.min_x = x_start + i
                            if self.min_y > y_start + j:
                                self.min_y = y_start + j
                        except Exception:
                            pass

    def is_in_bounds(self, point):
        if point[0] >= 0 and point[1] >= 0 \
                and point[0] < self.size[0] and point[1] < self.size[1]:
            return True
        return False

    def flood_fill(self, start_point):
        toFill = set([start_point])
        while len(toFill) != 0:
            position = toFill.pop()
            if self.get_pixel_color(*position) == self.fillColor:
                continue
            self.put_pixel(position[0], position[1], color='fill')
            for t in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                neighbor_pixel = (position[0] + t[0], position[1] + t[1])
                if self.is_in_bounds(neighbor_pixel) and \
                        self.get_pixel_color(*neighbor_pixel) != self.strokeColor \
                        and self.get_pixel_color(*neighbor_pixel) != self.fillColor:
                    toFill.add(neighbor_pixel)

    def bezier_line(self, x1, y1, x2, y2):
        """
        Draws a line from (x1, y2) to (x2, y2) \n
        :param x1: x val of the first \n
        :param y1: y val of the first \n
        :param x2: x val of the second \n
        :param y2: y val of the second \n
        :return: None
        """
        t = 0.0
        while t < 1:
            x = round((1 - t) * x1 + t * x2)
            y = round((1 - t) * y1 + t * y2)
            self.put_pixel(x, y)
            t += 0.001

    def quadratic_bezier_curve(self, x1, y1, xc, yc, x2, y2):
        """
        Draws a quadratic bezier curve \n
        :param x1: x val of the first point \n
        :param y1: y val of the first point \n
        :param xc: x val of the control point \n
        :param yc: y val of the control point \n
        :param x2: x val of the second point \n
        :param y2: y val of the second point \n
        :return:
        """
        t = 0.0
        while t < 1:
            x = round(((1 - t) ** 2) * x1 + 2 *
                      (1 - t) * t * xc + (t ** 2) * x2)
            y = round(((1 - t) ** 2) * y1 + 2 *
                      (1 - t) * t * yc + (t ** 2) * y2)
            self.put_pixel(x, y)
            t += 0.001

    def cubic_bezier_curve(self, x1, y1, xc1, yc1, xc2, yc2, x2, y2):
        """
        Draws a cubic bezier curve \n
        :param x1: x val of the first point \n
        :param y1: y val of the first point \n
        :param xc1: x val of the first control point \n
        :param yc1: y val of the first control point \n
        :param xc2: x val of the second control point \n
        :param yc2: y val of the second control point \n
        :param x2: x val of the second point \n
        :param y2: y val of the second point \n
        :return:
        """
        t = 0.0
        while t < 1:
            x = round(((1 - t) ** 3) * x1 + 3 * ((1 - t) ** 2) * t *
                      xc1 + 3 * (1 - t) * (t ** 2) * xc2 + (t ** 3) * x2)
            y = round(((1 - t) ** 3) * y1 + 3 * ((1 - t) ** 2) * t *
                      yc1 + 3 * (1 - t) * (t ** 2) * yc2 + (t ** 3) * y2)
            self.put_pixel(x, y)
            t += 0.001

    # https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    def line(self, x1, y1, x2, y2):
        """
        Draws a line from (x1, y2) to (x2, y2) \n
        :param x1: x val of the first \n
        :param y1: y val of the first \n
        :param x2: x val of the second \n
        :param y2: y val of the second \n
        :return: None
        """
        dx = abs(x2 - x1)
        sx = 1 if x1 < x2 else -1
        dy = -abs(y2 - y1)
        sy = 1 if y1 < y2 else -1
        err = dx + dy
        while True:
            self.put_pixel(x1, y1)

            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy

    def rectangle(self, x: int, y: int, width: int, height: int):
        """
        Draws a rectangle with the center at (x,y) for given width and height\n
        :param x: x val of the center\n
        :param y: x val of the center\n
        :param width: \n
        :param height: \n
        :return: none
        """
        x1 = x
        y1 = y
        x2 = x + width
        y2 = y + height
        self.line(x1, y1, x2, y1)
        self.line(x1, y1, x1, y2)
        self.line(x1, y2, x2, y2)
        self.line(x2, y2, x2, y1)

    # https://www.geeksforgeeks.org/midpoint-ellipse-drawing-algorithm/
    def ellipse(self, rx, ry, xc, yc):
        """
        Draws a ellipse with the center at (x,y) for given width rx  and height ry \n
        :param rx: x val of the center \n
        :param ry: x val of the center \n
        :param xc: width \n
        :param yc: height \n
        :return: none
        """
        x = 0
        y = ry

        # Initial decision parameter of region 1
        d1 = ((ry * ry) - (rx * rx * ry) +
              (0.25 * rx * rx))
        dx = 2 * ry * ry * x
        dy = 2 * rx * rx * y

        # For region 1
        while dx < dy:

            # Print points based on 4-way symmetry
            self.put_pixel(x + xc, y + yc)
            self.put_pixel(-x + xc, y + yc)
            self.put_pixel(x + xc, -y + yc)
            self.put_pixel(-x + xc, -y + yc)

            # Checking and updating value of
            # decision parameter based on algorithm
            if d1 < 0:
                x += 1
                dx = dx + (2 * ry * ry)
                d1 = d1 + dx + (ry * ry)
            else:
                x += 1
                y -= 1
                dx = dx + (2 * ry * ry)
                dy = dy - (2 * rx * rx)
                d1 = d1 + dx - dy + (ry * ry)

        # Decision parameter of region 2
        d2 = (((ry * ry) * ((x + 0.5) * (x + 0.5))) +
              ((rx * rx) * ((y - 1) * (y - 1))) -
              (rx * rx * ry * ry))

        # Plotting points of region 2
        while y >= 0:

            # printing points based on 4-way symmetry
            self.put_pixel(x + xc, y + yc)
            self.put_pixel(-x + xc, y + yc)
            self.put_pixel(x + xc, -y + yc)
            self.put_pixel(-x + xc, -y + yc)

            # Checking and updating parameter
            # value based on algorithm
            if d2 > 0:
                y -= 1
                dy = dy - (2 * rx * rx)
                d2 = d2 + (rx * rx) - dy
            else:
                y -= 1
                x += 1
                dx = dx + (2 * ry * ry)
                dy = dy - (2 * rx * rx)
                d2 = d2 + dx - dy + (rx * rx)

    def circle(self, cx, cy, r):
        """
        Draws a cincle at the center (cx, cy) with radius r \n
        :param cx: x val of center \n
        :param cy: x val of center \n
        :param r: val of radius \n
        :return: none
        """
        self.ellipse(r, r, cx, cy)

    def arc_ellipse(self, start_x, start_y, rx, ry, angle, large_arc_flag, sweep_flag, end_x, end_y):
        """
        Draws an elliptical arc from the current point to (x, y).
        The size and orientation of the ellipse are defined by two radii (rx, ry) 
        and an x-axis-rotation, which indicates how the ellipse as a whole is rotated, 
        in degrees, relative to the current coordinate system. The center (cx, cy) of the 
        ellipse is calculated automatically to satisfy the constraints imposed by the other parameters.
        large-arc-flag and sweep-flag contribute to the automatic calculations 
        and help determine how the arc is drawn. \n
        :param start_x: \n
        :param start_y: \n
        :param rx: \n
        :param ry: \n
        :param angle: \n
        :param large_arc_flag: \n
        :param sweep_flag: \n
        :param end_x: \n
        :param end_y: \n
        :return:
        """
        # Eroare crese pentru valori mari a unghiului, este precis numai in jur de 0 grade
        # https://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes

        p_prim = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle),
                                                            np.cos(angle)]]) @ np.array(
            [(start_x - end_x) / 2, (start_y - end_y) / 2])

        val = ((p_prim[0] / rx) ** 2) + ((p_prim[1] / ry) ** 2)
        if val > 1:
            rx = abs(val ** (1 / 2) * p_prim[0])
            ry = abs(val ** (1 / 2) * p_prim[1])
        c_prim = (abs(
            ((rx ** 2) * (ry ** 2) - (rx ** 2) * (p_prim[1] ** 2) - (ry ** 2) * (p_prim[0] ** 2)) / ((rx ** 2) * (
                p_prim[1] ** 2) + (ry ** 2) * (p_prim[0] ** 2)))) ** (1 / 2) * np.array(
            [rx * p_prim[1] / ry, -1 * ry * p_prim[0] / rx])

        c = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                     ) @ c_prim + np.array([(start_x + end_x) / 2, (start_y + end_y) / 2])

        xc = int(c[0])  # (startX + endX)//2
        yc = int(c[1])  # (startY + end_y)//2

        def relative_point_position(s_x, s_y, e_x, e_y, check_x, check_y):
            """
            Checks if a point is on the left side or the right side of a line. The line is 
            described as start point (s_x, s_y), end point (e_x, e_y).
            :param s_x: start point of the line\n
            :param s_y: start point of the line\n
            :param e_x: end point of the line\n
            :param e_y: end point of the line\n
            :param check_x: x of the point \n
            :param check_y: y of the point \n
            :return: left +  |  right -   | 0 on the line
            """
            return (e_x - s_x) * (check_y - s_y) - (e_y - s_y) * (check_x - s_x)
        x = 0
        y = ry

        # Initial decision parameter of region 1
        d1 = ((ry * ry) - (rx * rx * ry) +
              (0.25 * rx * rx))
        dx = 2 * ry * ry * x
        dy = 2 * rx * rx * y

        # For region 1
        while dx < dy:

            # Print points based on 4-way symmetry
            p1_x, p1_y = x + xc, y + yc
            p2_x, p2_y = -x + xc, y + yc
            p3_x, p3_y = x + xc, -y + yc
            p4_x, p4_y = -x + xc, -y + yc
            # add rotation
            p1_x, p1_y = self.rotate_point([p1_x, p1_y], [xc, yc], angle)
            p2_x, p2_y = self.rotate_point([p2_x, p2_y], [xc, yc], angle)
            p3_x, p3_y = self.rotate_point([p3_x, p3_y], [xc, yc], angle)
            p4_x, p4_y = self.rotate_point([p4_x, p4_y], [xc, yc], angle)

            if relative_point_position(start_x, start_y, end_x, end_y, p1_x, p1_y) >= 0:
                self.put_pixel(p1_x, p1_y)
            if relative_point_position(start_x, start_y, end_x, end_y, p2_x, p2_y) >= 0:
                self.put_pixel(p2_x, p2_y)
            if relative_point_position(start_x, start_y, end_x, end_y, p3_x, p3_y) >= 0:
                self.put_pixel(p3_x, p3_y)
            if relative_point_position(start_x, start_y, end_x, end_y, p4_x, p4_y) >= 0:
                self.put_pixel(p4_x, p4_y)

            # Checking and updating value of
            # decision parameter based on algorithm
            if d1 < 0:
                x += 1
                dx = dx + (2 * ry * ry)
                d1 = d1 + dx + (ry * ry)
            else:
                x += 1
                y -= 1
                dx = dx + (2 * ry * ry)
                dy = dy - (2 * rx * rx)
                d1 = d1 + dx - dy + (ry * ry)

        # Decision parameter of region 2
        d2 = (((ry * ry) * ((x + 0.5) * (x + 0.5))) +
              ((rx * rx) * ((y - 1) * (y - 1))) -
              (rx * rx * ry * ry))

        # Plotting points of region 2
        while y >= 0:

            # printing points based on 4-way symmetry
            p1_x, p1_y = x + xc, y + yc
            p2_x, p2_y = -x + xc, y + yc
            p3_x, p3_y = x + xc, -y + yc
            p4_x, p4_y = -x + xc, -y + yc
            # add rotation
            p1_x, p1_y = self.rotate_point([p1_x, p1_y], [xc, yc], angle)
            p2_x, p2_y = self.rotate_point([p2_x, p2_y], [xc, yc], angle)
            p3_x, p3_y = self.rotate_point([p3_x, p3_y], [xc, yc], angle)
            p4_x, p4_y = self.rotate_point([p4_x, p4_y], [xc, yc], angle)

            if relative_point_position(start_x, start_y, end_x, end_y, p1_x, p1_y) >= 0:
                self.put_pixel(p1_x, p1_y)
            if relative_point_position(start_x, start_y, end_x, end_y, p2_x, p2_y) >= 0:
                self.put_pixel(p2_x, p2_y)
            if relative_point_position(start_x, start_y, end_x, end_y, p3_x, p3_y) >= 0:
                self.put_pixel(p3_x, p3_y)
            if relative_point_position(start_x, start_y, end_x, end_y, p4_x, p4_y) >= 0:
                self.put_pixel(p4_x, p4_y)

            # Checking and updating parameter
            # value based on algorithm
            if d2 > 0:
                y -= 1
                dy = dy - (2 * rx * rx)
                d2 = d2 + (rx * rx) - dy
            else:
                y -= 1
                x += 1
                dx = dx + (2 * ry * ry)
                dy = dy - (2 * rx * rx)
                d2 = d2 + dx - dy + (rx * rx)

    @staticmethod
    def rotate_point(point, rotation_point, angle):
        """
        Calculated the position of a point after applying a rotation of a certain angle, by a certain point \n
        :param point: (x,y) of the point \n
        :param rotation_point: (x,y) of the center of rotation \n
        :param angle: angle in degrees \n
        :return: (x,y) coordinates of the rotated point
        """
        # rotation origin point
        rotation_point = np.array(rotation_point, dtype="int32")
        # the point i want to rotate
        point = np.array(point, dtype="int32")
        # the Rotation matrix

        rotation = np.zeros((2, 2))
        rotation[0][0] = np.cos(-angle)
        rotation[0][1] = -1 * np.sin(-angle)
        rotation[1][0] = np.sin(-angle)
        rotation[1][1] = np.cos(-angle)

        point = point - rotation_point
        point = rotation @ point
        point = point + rotation_point
        return int(point[0]), int(point[1])

    def draw_svg(self, name, root):
        """
        Draws the SVG into a PNG \n
        :param name: name of the file \n
        :param root: root of the XML tree of the SVG file \n
        :return: None
        """
        f = open(name, 'wb')
        w = png.Writer(self.size[1], self.size[0], greyscale=False)
        pen_y = 0
        pen_x = 0

        for child in root:
            # get style options
            if child.tag == 'rect':
                try:
                    width = int(float(child.attrib.get('width')))
                    height = int(float(child.attrib.get('height')))
                    assert width is not None, "Wrong parameters. No width found, skipping rectangle."
                    assert height is not None, "Wrong parameters. No height found, skipping rectangle."
                    width = int(float(width))
                    height = int(float(height))
                    x = pen_x if child.attrib.get('x') is None else int(
                        float(child.attrib.get('x')))
                    y = pen_y if child.attrib.get('y') is None else int(
                        float(child.attrib.get('y')))
                    self.rectangle(x, y, width, height)
                    if width >= 2 * self.strokeWidth - 1 \
                            and height >= 2 * self.strokeWidth - 1:
                        self.flood_fill(
                            (x + self.strokeWidth, y + self.strokeWidth))
                except AssertionError as e:
                    print(e)
            elif child.tag == 'test':
                self.rotate_point([50, 50], [70, 70], np.pi / 2)
            elif child.tag == 'circle':
                try:
                    r = child.attrib.get('r')
                    assert r is not None, "Wrong parameters. No radius found, skipping circle."
                    r = int(r)
                    cx = pen_x if child.attrib.get('cx') is None else int(
                        float(child.attrib.get('cx')))
                    cy = pen_y if child.attrib.get('cy') is None else int(
                        float(child.attrib.get('cy')))
                    self.circle(cx, cy, r)
                    self.flood_fill((cx, cy))
                except AssertionError as e:
                    print(e)
            elif child.tag == 'line':
                try:
                    x2 = child.attrib.get('x2')
                    y2 = child.attrib.get('y2')
                    assert x2 is not None, "Wrong parameters. No final point (x2,y2) found, skipping line."
                    assert y2 is not None, "Wrong parameters. No final point (x2,y2) found, skipping line."
                    x2 = int(float(x2))
                    y2 = int(float(y2))
                    x1 = pen_x if child.attrib.get('x1') is None else int(
                        float(child.attrib.get('x1')))
                    y1 = pen_y if child.attrib.get('y1') is None else int(
                        float(child.attrib.get('y1')))
                    self.bezier_line(x1, y1, x2, y2)
                    # pen_x = x2
                    # pen_y = y2
                except AssertionError as e:
                    print(e)
            elif child.tag == 'ellipse':
                try:
                    rx = child.attrib.get('rx')
                    ry = child.attrib.get('ry')
                    assert rx is not None, "Wrong parameters. No radius found, skipping ellipse."
                    assert ry is not None, "Wrong parameters. No radius found, skipping ellipse."
                    rx = int(float(rx))
                    ry = int(float(ry))
                    cx = pen_x if child.attrib.get('cx') is None else int(
                        float(child.attrib.get('cx')))
                    cy = pen_y if child.attrib.get('cy') is None else int(
                        float(child.attrib.get('cy')))
                    self.ellipse(rx, ry, cx, cy)
                    self.flood_fill((cx, cy))
                except AssertionError as e:
                    print(e)
            elif child.tag == 'polyline':
                try:
                    points = child.attrib.get('points')
                    assert points is not None, "Wrong parameters. No radius found, skipping ellipse."
                    points = points.split(' ')
                    assert len(points) > 2
                    assert len(points) % 2 == 0, "Point coordinate missing."
                    points = list(map(lambda num: int(num), points))
                    i = 0
                    while i < len(points) - 2:
                        self.line(points[i], points[i + 1],
                                  points[i + 2], points[i + 3])
                        i += 2
                except AssertionError as e:
                    print(e)
            elif child.tag == 'path':
                path_d_commands_look_ahead = {
                    "M": 2,
                    "m": 2,
                    "L": 2,
                    "l": 2,
                    "H": 1,
                    "h": 1,
                    "V": 1,
                    "v": 1,
                    "Z": 0,
                    "z": 0,
                    "C": 6,
                    "c": 6,
                    "Q": 4,
                    "q": 4,
                    "S": 4,
                    "s": 4,
                    "A": 7,
                    "a": 7
                }
                self.max_x = 0
                self.max_y = 0
                self.min_x = self.size[0]
                self.min_y = self.size[1]
                try:
                    d = child.attrib.get('d')
                    assert d is not None, "Wrong parameters. Skipping path."
                    regex = r"(?:[a-zA-Z])|(?:-[\d]+\.?[\d]+)|(?:[\d]+\.?[\d]+)|(?:[\d])"
                    d = re.findall(regex, d)
                    command_count = 0
                    initial_x = pen_x
                    initial_y = pen_y
                    cubic_bezier_last_point_x = None
                    cubic_bezier_last_point_y = None
                    command = ""
                    while len(d) > 0:
                        command_count += 1
                        if path_d_commands_look_ahead.get(d[0]) is None:
                            print("wow")
                        if path_d_commands_look_ahead.get(d[0]) is not None:
                            command = d.pop(0)

                        if command == "M" or command == "m":
                            # assert command_count == 1, "Wrong format for path d attribute for " + \
                            #                           command + " command. M can be just first in d attribute"
                            new_pen_x = d.pop(0)
                            new_pen_y = d.pop(0)
                            try:
                                initial_x = pen_x = int(
                                    float(new_pen_x.strip(',')))
                                initial_y = pen_y = int(
                                    float(new_pen_y.strip(',')))
                                cubic_bezier_last_point_x = pen_x
                                cubic_bezier_last_point_y = pen_y
                            except ValueError as ve:
                                print("Error in command " + command)
                                raise ve

                        # line with x y parameters (absolute)
                        elif command == "L":
                            new_x2 = d.pop(0).strip(',')
                            new_y2 = d.pop(0).strip(',')
                            try:
                                new_x2 = int(float(new_x2))
                                new_y2 = int(float(new_y2))
                            except ValueError as ve:
                                print("Error in command " + command)
                                raise ve

                            self.line(pen_x, pen_y, new_x2, new_y2)
                            pen_x = new_x2
                            pen_y = new_y2
                            cubic_bezier_last_point_x = pen_x
                            cubic_bezier_last_point_y = pen_y
                        # line with delta x delta y parameters (relative)
                        elif command == "l":
                            d_x2 = d.pop(0).strip(',')
                            d_y2 = d.pop(0).strip(',')
                            try:
                                d_x2 = int(float(d_x2))
                                d_y2 = int(float(d_y2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.line(pen_x, pen_y, pen_x + d_x2, pen_y + d_y2)
                            pen_x = pen_x + d_x2
                            pen_y = pen_y + d_y2
                            cubic_bezier_last_point_x = pen_x
                            cubic_bezier_last_point_y = pen_y
                        elif command == 'H':
                            new_x2 = d.pop(0).strip(',')
                            try:
                                new_x2 = int(float(new_x2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.line(pen_x, pen_y, new_x2, pen_y)
                            pen_x = new_x2
                            cubic_bezier_last_point_x = pen_x
                            cubic_bezier_last_point_y = pen_y
                        elif command == 'h':
                            d_x2 = d.pop(0).strip(',')
                            try:
                                d_x2 = int(float(d_x2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.line(pen_x, pen_y, pen_x + d_x2, pen_y)
                            pen_x = pen_x + d_x2
                            cubic_bezier_last_point_x = pen_x
                            cubic_bezier_last_point_y = pen_y
                        elif command == 'V':
                            new_y2 = d.pop(0).strip(',')
                            try:
                                new_y2 = int(float(new_y2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.line(pen_x, pen_y, pen_x, new_y2)
                            pen_y = new_y2
                            cubic_bezier_last_point_x = pen_x
                            cubic_bezier_last_point_y = pen_y
                        elif command == 'v':
                            d_y2 = d.pop(0).strip(',')
                            try:
                                d_y2 = int(float(d_y2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.line(pen_x, pen_y, pen_x, pen_y + d_y2)
                            pen_y = pen_y + d_y2
                            cubic_bezier_last_point_x = pen_x
                            cubic_bezier_last_point_y = pen_y
                        elif command == 'z' or command == 'Z':
                            self.line(pen_x, pen_y, initial_x, initial_y)
                            pen_x = initial_x
                            pen_y = initial_y
                            cubic_bezier_last_point_x = pen_x
                            cubic_bezier_last_point_y = pen_y
                        elif command == 'C':
                            control_x1 = d.pop(0).strip(',')
                            control_y1 = d.pop(0).strip(',')
                            control_x2 = d.pop(0).strip(',')
                            control_y2 = d.pop(0).strip(',')
                            new_x2 = d.pop(0).strip(',')
                            new_y2 = d.pop(0).strip(',')
                            try:
                                control_x1 = int(float(control_x1))
                                control_y1 = int(float(control_y1))
                                control_x2 = int(float(control_x2))
                                control_y2 = int(float(control_y2))
                                new_x2 = int(float(new_x2))
                                new_y2 = int(float(new_y2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.cubic_bezier_curve(
                                pen_x, pen_y, control_x1, control_y1, control_x2, control_y2, new_x2, new_y2)
                            # calculate reflexion of the control_point2 by the point (new_x2, new_y2)
                            cubic_bezier_last_point_x = 2 * new_x2 - control_x2
                            cubic_bezier_last_point_y = 2 * new_y2 - control_y2  # calculate reflexion
                            pen_x = new_x2
                            pen_y = new_y2
                        elif command == 'c':
                            control_d_x1 = d.pop(0).strip(',')
                            control_d_y1 = d.pop(0).strip(',')
                            control_d_x2 = d.pop(0).strip(',')
                            control_d_y2 = d.pop(0).strip(',')
                            d_x2 = d.pop(0).strip(',')
                            d_y2 = d.pop(0).strip(',')
                            try:
                                control_d_x1 = int(float(control_d_x1))
                                control_d_y1 = int(float(control_d_y1))
                                control_d_x2 = int(float(control_d_x2))
                                control_d_y2 = int(float(control_d_y2))
                                d_x2 = int(float(d_x2))
                                d_y2 = int(float(d_y2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.cubic_bezier_curve(pen_x, pen_y, pen_x + control_d_x1, pen_y + control_d_y1,
                                                    pen_x + control_d_x2, pen_y + control_d_y2, pen_x + d_x2,
                                                    pen_y + d_y2)
                            # calculate reflexion
                            cubic_bezier_last_point_x = 2 * \
                                (pen_x + d_x2) - (pen_x + control_d_x2)
                            cubic_bezier_last_point_y = 2 * \
                                (pen_y + d_y2) - (pen_y + control_d_y2)
                            pen_x = pen_x + d_x2
                            pen_y = pen_y + d_y2
                        elif command == "S":
                            assert cubic_bezier_last_point_x is not None and \
                                cubic_bezier_last_point_y is not None, "Can't use S command, no last point found. " \
                                "Use C before it. "
                            control_x2 = d.pop(0).strip(',')
                            control_y2 = d.pop(0).strip(',')
                            new_x2 = d.pop(0)
                            new_y2 = d.pop(0)
                            try:
                                control_x2 = int(float(control_x2))
                                control_y2 = int(float(control_y2))
                                new_x2 = int(float(new_x2))
                                new_y2 = int(float(new_y2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.cubic_bezier_curve(
                                pen_x, pen_y, cubic_bezier_last_point_x, cubic_bezier_last_point_y, control_x2,
                                control_y2,
                                new_x2, new_y2)
                            cubic_bezier_last_point_x = 2 * new_x2 - control_x2
                            cubic_bezier_last_point_y = 2 * new_y2 - control_y2
                            pen_x = new_x2
                            pen_y = new_y2
                        elif command == "s":
                            assert cubic_bezier_last_point_x is not None and cubic_bezier_last_point_y is not None,\
                                "Can't use S command, no last point found. Use C before it. "
                            control_d_x2 = d.pop(0).strip(',')
                            control_d_y2 = d.pop(0).strip(',')
                            d_x2 = d.pop(0).strip(',')
                            d_y2 = d.pop(0).strip(',')
                            try:
                                control_d_x2 = int(float(control_d_x2))
                                control_d_y2 = int(float(control_d_y2))
                                d_x2 = int(float(d_x2))
                                d_y2 = int(float(d_y2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.cubic_bezier_curve(pen_x, pen_y, cubic_bezier_last_point_x, cubic_bezier_last_point_y,
                                                    pen_x + control_d_x2, pen_y + control_d_y2, pen_x + d_x2,
                                                    pen_y + d_y2)
                            # calculate reflexion of the control point by the end point of the curve
                            cubic_bezier_last_point_x = 2 * \
                                (pen_x + d_x2) - (pen_x + control_d_x2)
                            cubic_bezier_last_point_y = 2 * \
                                (pen_y + d_y2) - (pen_y + control_d_y2)
                            pen_x = pen_x + d_x2
                            pen_y = pen_y + d_y2
                        elif command == "Q":
                            control_x1 = d.pop(0).strip(',')
                            control_y1 = d.pop(0).strip(',')
                            new_x2 = d.pop(0).strip(',')
                            new_y2 = d.pop(0).strip(',')
                            try:
                                control_x1 = int(float(control_x1))
                                control_y1 = int(float(control_y1))
                                new_x2 = int(float(new_x2))
                                new_y2 = int(float(new_y2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.quadratic_bezier_curve(
                                pen_x, pen_y, control_x1, control_y1, new_x2, new_y2)
                            pen_x = new_x2
                            pen_y = new_y2
                            cubic_bezier_last_point_x = pen_x
                            cubic_bezier_last_point_y = pen_y
                        elif command == 'q':
                            control_d_x1 = d.pop(0).strip(',')
                            control_d_y1 = d.pop(0).strip(',')
                            d_x2 = d.pop(0).strip(',')
                            d_y2 = d.pop(0).strip(',')
                            try:
                                control_d_x1 = int(float(control_d_x1))
                                control_d_y1 = int(float(control_d_y1))
                                d_x2 = int(float(d_x2))
                                d_y2 = int(float(d_y2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.quadratic_bezier_curve(
                                pen_x, pen_y, pen_x + control_d_x1, pen_y + control_d_y1, pen_x + d_x2, pen_y + d_y2)
                            pen_x = pen_x + d_x2
                            pen_y = pen_y + d_y2
                            cubic_bezier_last_point_x = pen_x
                            cubic_bezier_last_point_y = pen_y
                        elif command == 'A':
                            # https://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes
                            # functionality incomplete
                            # big error in drawing when the rotation angle is big.
                            r_x = d.pop(0).strip(',')
                            r_y = d.pop(0).strip(',')
                            axis_rotation = d.pop(0).strip(',')
                            large_arc_flag = d.pop(0).strip(',')
                            sweep_flag = d.pop(0).strip(',')
                            x2 = d.pop(0).strip(',')
                            y2 = d.pop(0).strip(',')
                            try:
                                r_x = int(float(r_x))
                                r_y = int(float(r_y))
                                # invert the angle because the axis is inverted
                                axis_rotation = -1 * int(float(axis_rotation))
                                large_arc_flag = int(large_arc_flag)
                                sweep_flag = int(sweep_flag)
                                x2 = int(float(x2))
                                y2 = int(float(y2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.arc_ellipse(pen_x, pen_y, r_x, r_y,
                                             axis_rotation, 1, 1, x2, y2)
                            pen_x = x2
                            pen_y = y2
                            cubic_bezier_last_point_x = pen_x
                            cubic_bezier_last_point_y = pen_y
                        elif command == 'a':
                            r_x = d.pop(0).strip(',')
                            r_y = d.pop(0).strip(',')
                            axis_rotation = d.pop(0).strip(',')
                            large_arc_flag = d.pop(0).strip(',')
                            sweep_flag = d.pop(0).strip(',')
                            d_x2 = d.pop(0).strip(',')
                            d_y2 = d.pop(0).strip(',')
                            try:
                                r_x = int(float(r_x))
                                r_y = int(float(r_y))
                                # invert the angle because the axis is inverted
                                axis_rotation = -1 * int(float(axis_rotation))
                                large_arc_flag = int(large_arc_flag)
                                sweep_flag = int(sweep_flag)
                                d_x2 = int(float(d_x2))
                                d_y2 = int(float(d_y2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.arc_ellipse(pen_x, pen_y, r_x, r_y,
                                             axis_rotation, 1, 1, pen_x + d_x2, pen_y + d_y2)
                            pen_x = pen_x + d_x2
                            pen_y = pen_y + d_y2
                            cubic_bezier_last_point_x = pen_x
                            cubic_bezier_last_point_y = pen_y
                        else:
                            print("Unsuported Command")

                    # cast rays and determine what is inside
                    intersect_count = 0
                    inside_countour = False
                    step = (self.max_x - self.min_x) // 5
                    for i in range(self.min_x + self.strokeWidth + 1, self.max_x, step):
                        intersect_count = 0
                        inside_countour = False
                        for j in range(self.min_y, self.max_y):
                            #self.put_pixel(j,i, color='fill')
                            current_pixel_color = self.get_pixel_color(j, i)
                            if current_pixel_color == self.strokeColor:
                                if inside_countour == False:
                                    intersect_count += 1
                                    inside_countour = True
                            else:
                                if inside_countour == True:
                                    inside_countour = False
                                    if intersect_count % 2 == 1 and current_pixel_color != self.fillColor:
                                        #self.put_pixel(j,i, color='fill')
                                        self.flood_fill((j,i))
                                        pass
                    # imprecise. 

                except Exception as e:
                    print(e)
        self.image = self.image.reshape(self.size[0], self.size[1] * 3)
        w.write(f, self.image)
        f.close()
