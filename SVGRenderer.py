import numpy as np
import png

class SVGRenderer:
    def __init__(self, size):
        self.image = np.zeros((*size, 3), dtype='uint8')
        self.size = size
        self.strokeColor = (55,45,166)
        

    #def setStyle(self, stroke, strokeWidth, fill):
#
    #    self.style.update(
    #        {
    #        "stroke":stroke,
    #        "stroke-width":strokeWidth,
    #        "fill":fill
    #        }
    #    )

    def putPixel(self, x: int, y: int):
        if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
            self.image[y][x][0] = self.strokeColor[0]
            self.image[y][x][1] = self.strokeColor[1]
            self.image[y][x][2] = self.strokeColor[2]

    def besierLine(self, x1, y1, x2, y2):
        t = 0.0
        while t < 1:
            x = round((1-t)*x1 + t*x2)
            y = round((1-t)*y1 + t*y2)
            self.putPixel(x, y)
            t += 0.001
    
    def quadraticBesierCurve(self, x1, y1, xc, yc, x2, y2):
        t = 0.0
        while t < 1:
            
            x = round( ((1-t)**2)*x1 + 2*(1-t)*t*xc + (t**2)*x2 )
            y = round( ((1-t)**2)*y1 + 2*(1-t)*t*yc +(t**2)*y2 )
            self.putPixel(x, y)
            t += 0.001

    def cubicBesierCurve(self, x1, y1, xc1, yc1, xc2, yc2, x2, y2):
        t = 0.0
        while t < 1:
            
            x = round( ((1-t)**3)*x1 + 3*((1-t)**2)*t*xc1 + 3*(1-t)*(t**2)*xc2 + (t**3)*x2 )
            y = round( ((1-t)**3)*y1 + 3*((1-t)**2)*t*yc1 + 3*(1-t)*(t**2)*yc2 + (t**3)*y2 )
            self.putPixel(x, y)
            t += 0.001         

# https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    def line(self, x1, y1, x2, y2):
        dx =  abs(x2-x1)
        sx = 1 if x1<x2 else -1
        dy = -abs(y2-y1)
        sy = 1 if y1<y2 else -1
        err = dx+dy
        while True:
            self.putPixel(x1, y1)

            #self.putPixel(x1+1, y1,255)
            #self.putPixel(x1-1, y1,255)
            #self.putPixel(x1, y1+1,255)
            #self.putPixel(x1, y1-1,255)
            if x1 == x2 and y1 == y2: break
            e2 = 2*err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy
            
    def rectangle(self, x: int, y: int, width: int, height: int):
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
  
        x = 0
        y = ry
    
        # Initial decision parameter of region 1  
        d1 = ((ry * ry) - (rx * rx * ry) +
                          (0.25 * rx * rx))
        dx = 2 * ry * ry * x
        dy = 2 * rx * rx * y
    
        # For region 1  
        while (dx < dy):  
        
            # Print points based on 4-way symmetry  
            self.putPixel( x + xc, y + yc)
            self.putPixel(-x + xc, y + yc)
            self.putPixel( x + xc,-y + yc)
            self.putPixel(-x + xc,-y + yc)
    
            # Checking and updating value of  
            # decision parameter based on algorithm  
            if (d1 < 0):  
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
        while (y >= 0): 
        
            # printing points based on 4-way symmetry  
            self.putPixel( x + xc, y + yc)
            self.putPixel(-x + xc, y + yc)
            self.putPixel( x + xc,-y + yc)
            self.putPixel(-x + xc,-y + yc)
    
            # Checking and updating parameter  
            # value based on algorithm  
            if (d2 > 0): 
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
        self.ellipse(r, r, cx, cy)

    def drawSVG(self, name, root):
        f = open( name, 'wb')
        w = png.Writer(self.size[0], self.size[1], greyscale=False)
        penY = 0
        penX = 0

        
        
        for child in root:
            # get style options
            if child.tag == 'rect':
                try:
                    width = int(child.attrib.get('width'))
                    height = int(child.attrib.get('height'))
                    assert width != None, "Wrong parameters. No width found, skipping rectangle."
                    assert height != None, "Wrong parameters. No height found, skipping rectangle."
                    width = int(width)
                    height = int(height)
                    x = penX if child.attrib.get('x') == None else int(child.attrib.get('x'))
                    y = penY if child.attrib.get('y') == None else int(child.attrib.get('y'))
                    self.rectangle(x, y, width, height)
                except AssertionError as e:
                    print(e)
            elif child.tag == 'circle':
                try:
                    r = child.attrib.get('r')
                    assert r != None, "Wrong parameters. No radius found, skipping circle."
                    r = int(r)
                    cx = penX if child.attrib.get('cx') == None else int(child.attrib.get('cx'))
                    cy = penY if child.attrib.get('cy') == None else int(child.attrib.get('cy'))
                    self.circle(cx, cy, r)
                except AssertionError as e:
                    print(e)
            elif child.tag == 'line':
                try:
                    x2 = child.attrib.get('x2')
                    y2 = child.attrib.get('y2')
                    assert x2 != None, "Wrong parameters. No final point (x2,y2) found, skipping line."
                    assert y2 != None, "Wrong parameters. No final point (x2,y2) found, skipping line."
                    x2 = int(x2)
                    y2 = int(y2)
                    x1 = penX if child.attrib.get('x1') == None else int(child.attrib.get('x1'))
                    y1 = penY if child.attrib.get('y1') == None else int(child.attrib.get('y1'))
                    self.besierLine(x1, y1, x2, y2)
                except AssertionError as e:
                    print(e)
            elif child.tag == 'ellipse':
                try:
                    rx = child.attrib.get('rx')
                    ry = child.attrib.get('ry')
                    assert rx!= None, "Wrong parameters. No radius found, skipping ellipse."
                    assert ry!= None, "Wrong parameters. No radius found, skipping ellipse."
                    rx = int(rx)
                    ry = int(ry)
                    cx = penX if child.attrib.get('cx') == None else int(child.attrib.get('cx'))
                    cy = penY if child.attrib.get('cy') == None else int(child.attrib.get('cy'))
                    self.ellipse(rx, ry, cx, cy)
                except AssertionError as e:
                    print(e)
            elif child.tag == 'polyline':
                try:
                    points = child.attrib.get('points')
                    assert points!= None, "Wrong parameters. No radius found, skipping ellipse."
                    points = points.split(' ')
                    assert len(points) > 2
                    assert len(points)%2 == 0, "Point coordinate missing."
                    points = list( map(lambda num: int(num), points) )
                    i = 0
                    while i < len(points) - 2:
                        self.line(points[i], points[i+1], points[i+2], points[i+3])
                        i += 2
                except AssertionError as e:
                    print(e)
            elif child.tag == 'path':
                pathDCommandsLookAhead = {
                    "M": 2,
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
                    "q": 4
                }
                try:
                    d = child.attrib.get('d')
                    assert d!= None, "Wrong parameters. Skipping path."
                    d = d.split(' ')
                    commandCount = 0
                    initialX = penX
                    initialY = penY
                    while len(d) > 0:
                        command = d.pop(0)
                        commandCount += 1
                        assert pathDCommandsLookAhead.get(command) != None, "Wrong or unsupported command: "+ command
                        if command == "M":
                            assert commandCount == 1, "Wrong format for path d attribute for " + command + " command. M can be just first in d attribute"
                            newPenX = d.pop(0)
                            newPenY = d.pop(0)
                            assert newPenX.isnumeric(), "Wrong format for path d attribute for " + command + " command"
                            assert newPenY.isnumeric(), "Wrong format for path d attribute for " + command + " command"
                            initialX = penX = int(newPenX)
                            initialY = penY = int(newPenY)
                        elif command == "L":  # line with x y parameters (absolute)
                            newX2 = d.pop(0)
                            newY2 = d.pop(0)
                            assert newX2.isnumeric(), "Wrong format for path d attribute for " + command + " command"
                            assert newY2.isnumeric(), "Wrong format for path d attribute for " + command + " command"
                            newX2 = int(newX2)
                            newY2 = int(newY2)
                            self.line(penX, penY, newX2, newY2)
                            penX = newX2
                            penY = newY2
                        elif command == "l":  # line with delta x delta y parameters (relative)
                            dX2 = d.pop(0)
                            dY2 = d.pop(0)
                            assert dX2.isnumeric(), "Wrong format for path d attribute for " + command + " command"
                            assert dY2.isnumeric(), "Wrong format for path d attribute for " + command + " command"
                            dX2 = int(dX2)
                            dY2 = int(dY2)
                            self.line(penX, penY, penX + dX2, penY + dY2)
                            penX = penX + dX2
                            penY = penY + dY2
                        elif command == 'H':
                            newX2 = d.pop(0)
                            assert newX2.isnumeric(), "Wrong format for path d attribute for " + command + " command"
                            newX2 = int(newX2)
                            self.line( penX, penY, newX2, penY )
                            penX = newX2
                        elif command == 'h':
                            dX2 = d.pop(0)
                            assert dX2.isnumeric(), "Wrong format for path d attribute for " + command + " command"
                            dX2 = int(dX2)
                            self.line( penX, penY, penX + dX2, penY )
                            penX = penX + dX2
                        elif command == 'V':
                            newY2 = d.pop(0)
                            assert newY2.isnumeric(), "Wrong format for path d attribute for " + command + " command"
                            newY2 = int(newX2)
                            self.line( penX, penY, penX, newY2 )
                            penY = newY2
                        elif command == 'v':
                            dY2 = d.pop(0)
                            assert dY2.isnumeric(), "Wrong format for path d attribute for " + command + " command"
                            dY2 = int(dY2)
                            self.line( penX, penY, penX, penY + dY2 )
                            penY = penY + dY2
                        elif command == 'z' or command == 'Z':
                            self.line(penX, penY, initialX, initialY)
                            penX = initialX
                            penY = initialY
                        elif command == 'C':
                            controlX1 = d.pop(0)
                            controlY1 = d.pop(0)
                            controlX2 = d.pop(0)
                            controlY2 = d.pop(0)
                            newX2 = d.pop(0)
                            newY2 = d.pop(0)
                            assert (controlX1.isnumeric() and
                                    controlY1.isnumeric() and
                                    controlX2.isnumeric() and
                                    controlY2.isnumeric() and
                                    newX2.isnumeric() and
                                    newY2.isnumeric()), "Wrong format for path d attribute for " + command + " command"
                            controlX1 = int(controlX1)
                            controlY1 = int(controlY1)
                            controlX2 = int(controlX2)
                            controlY2 = int(controlY2)
                            newX2 = int(newX2)
                            newY2 = int(newY2)
                            self.cubicBesierCurve(penX, penY, controlX1, controlY1, controlX2, controlY2, newX2, newY2)
                            penX = newX2
                            penY = newY2
                        #elif command == 'c':   I don't understand this
                        #    control_dX1 = d.pop(0)
                        #    control_dY1 = d.pop(0)
                        #    control_dX2 = d.pop(0)
                        #    control_dY2 = d.pop(0)
                        #    newX2 = d.pop(0)
                        #    newY2 = d.pop(0)
                        #    assert (control_dX1.isnumeric() and
                        #            control_dY1.isnumeric() and
                        #            control_dX2.isnumeric() and
                        #            control_dY2.isnumeric() and
                        #            newX2.isnumeric() and
                        #            newY2.isnumeric()), "Wrong format for path d attribute for " + command + " command"
                        #    control_dX1 = int(control_dX1)
                        #    control_dY1 = int(control_dY1)
                        #    control_dX2 = int(control_dX2)
                        #    control_dY2 = int(control_dY2)
                        #    newX2 = int(newX2)
                        #    newY2 = int(newY2)
                        #    self.cubicBesierCurve(penX, penY, control_dX1, control_dY1, control_dX2, control_dY2, newX2, newY2)
                        #    penX = newX2
                        #    penY = newY2
                        elif command == "Q":
                            control_X1 = d.pop(0)
                            control_Y1 = d.pop(0)
                            newX2 = d.pop(0)
                            newY2 = d.pop(0)
                            assert (control_X1.isnumeric() and
                                    control_Y1.isnumeric() and
                                    newX2.isnumeric() and
                                    newY2.isnumeric()), "Wrong format for path d attribute for " + command + " command"
                            control_X1 = int(control_X1)
                            control_Y1 = int(control_Y1)
                            newX2 = int(newX2)
                            newY2 = int(newY2)
                            self.quadraticBesierCurve(penX, penY, control_X1, control_Y1, newX2, newY2)
                            penX = newX2
                            penY = newY2
                except AssertionError as e:
                    print(e)
        self.image = self.image.reshape(self.size[0], self.size[1]*3)
        w.write(f, self.image)
        f.close()


if __name__ == "__main__":
    pass
    #r = SVGRenderer((1000,1000), colorSpace='grayscale')
    #r.drawSVG()
