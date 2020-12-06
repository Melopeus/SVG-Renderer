import numpy as np
import png

class SVGRenderer:
    def __init__(self, size):
        self.image = np.zeros((*size, 3), dtype='uint8')
        self.size = size
        self.strokeColor = (40,255,255)
        self.strokeWidth = 5
        

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
        x_start = x - self.strokeWidth//2    # I draw instead of a point, a square like in
        y_start = y - ( self.strokeWidth-1 - self.strokeWidth//2 ) # 
        for i in range(self.strokeWidth):
            for j in range(self.strokeWidth):
                if 0 <= x_start+i < self.size[1] and 0 <= y_start+j < self.size[0]:
                    
                    self.image[y_start+j][x_start+i][0] = self.strokeColor[0]
                    self.image[y_start+j][x_start+i][1] = self.strokeColor[1]
                    self.image[y_start+j][x_start+i][2] = self.strokeColor[2]

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
        w = png.Writer(self.size[1], self.size[0], greyscale=False)
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
            elif child.tag == 'test':
                self.putPixel(10,10)
                self.circle(50,50,30)
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
                    "q": 4,
                    "S": 4,
                    "s": 4
                }
                try:
                    d = child.attrib.get('d')
                    assert d!= None, "Wrong parameters. Skipping path."
                    d = d.split(' ')
                    commandCount = 0
                    initialX = penX
                    initialY = penY
                    cubicBesierLastPoint_X = None
                    cubicBesierLastPoint_Y = None
                    while len(d) > 0:
                        command = d.pop(0)
                        commandCount += 1
                        if pathDCommandsLookAhead.get(command) == None:
                            print("Wrong or unsupported command: "+ command)
                            continue
                        if command == "M":
                            assert commandCount == 1, "Wrong format for path d attribute for " + command + " command. M can be just first in d attribute"
                            newPenX = d.pop(0)
                            newPenY = d.pop(0)
                            try:
                                initialX = penX = int(float(newPenX.strip(',')))
                                initialY = penY = int(float(newPenY.strip(',')))
                                cubicBesierLastPoint_X = penX
                                cubicBesierLastPoint_Y = penY
                            except ValueError as ve:
                                print("Error in command " + command)
                                raise ve
                            
                        elif command == "L":  # line with x y parameters (absolute)
                            newX2 = d.pop(0).strip(',')
                            newY2 = d.pop(0).strip(',')
                            try:
                                newX2 = int(float(newX2))
                                newY2 = int(float(newY2))
                            except ValueError as ve:
                                print("Error in command " + command)
                                raise ve
                            
                            self.line(penX, penY, newX2, newY2)
                            penX = newX2
                            penY = newY2
                            cubicBesierLastPoint_X = penX
                            cubicBesierLastPoint_Y = penY
                        elif command == "l":  # line with delta x delta y parameters (relative)
                            dX2 = d.pop(0).strip(',')
                            dY2 = d.pop(0).strip(',')
                            try:
                                dX2 = int(float(dX2))
                                dY2 = int(float(dY2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.line(penX, penY, penX + dX2, penY + dY2)
                            penX = penX + dX2
                            penY = penY + dY2
                            cubicBesierLastPoint_X = penX
                            cubicBesierLastPoint_Y = penY
                        elif command == 'H':
                            newX2 = d.pop(0).strip(',')
                            try:
                                newX2 = int(float(newX2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.line( penX, penY, newX2, penY )
                            penX = newX2
                            cubicBesierLastPoint_X = penX
                            cubicBesierLastPoint_Y = penY
                        elif command == 'h':
                            dX2 = d.pop(0).strip(',')
                            try:
                                dX2 = int(float(dX2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.line( penX, penY, penX + dX2, penY )
                            penX = penX + dX2
                            cubicBesierLastPoint_X = penX
                            cubicBesierLastPoint_Y = penY
                        elif command == 'V':
                            newY2 = d.pop(0).strip(',')
                            try:
                                newY2 = int(float(newY2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.line( penX, penY, penX, newY2 )
                            penY = newY2
                            cubicBesierLastPoint_X = penX
                            cubicBesierLastPoint_Y = penY
                        elif command == 'v':
                            dY2 = d.pop(0).strip(',')
                            try:
                                dY2 = int(float(dY2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.line( penX, penY, penX, penY + dY2 )
                            penY = penY + dY2
                            cubicBesierLastPoint_X = penX
                            cubicBesierLastPoint_Y = penY
                        elif command == 'z' or command == 'Z':
                            self.line(penX, penY, initialX, initialY)
                            penX = initialX
                            penY = initialY
                            cubicBesierLastPoint_X = penX
                            cubicBesierLastPoint_Y = penY
                        elif command == 'C':
                            controlX1 = d.pop(0).strip(',')
                            controlY1 = d.pop(0).strip(',')
                            controlX2 = d.pop(0).strip(',')
                            controlY2 = d.pop(0).strip(',')
                            newX2 = d.pop(0).strip(',')
                            newY2 = d.pop(0).strip(',')
                            try:
                                controlX1 = int(float(controlX1))
                                controlY1 = int(float(controlY1))
                                controlX2 = int(float(controlX2))
                                controlY2 = int(float(controlY2))
                                newX2 = int(float(newX2))
                                newY2 = int(float(newY2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.cubicBesierCurve(penX, penY, controlX1, controlY1, controlX2, controlY2, newX2, newY2)
                            cubicBesierLastPoint_X = 2*newX2 - controlX2 # calculate reflexion of the control_point2 by the point (newX2, newY2)
                            cubicBesierLastPoint_Y = 2*newY2 - controlY2 # calculate reflexion 
                            penX = newX2
                            penY = newY2
                        elif command == 'c':
                            control_dX1 = d.pop(0).strip(',')
                            control_dY1 = d.pop(0).strip(',')
                            control_dX2 = d.pop(0).strip(',')
                            control_dY2 = d.pop(0).strip(',')
                            dX2 = d.pop(0).strip(',')
                            dY2 = d.pop(0).strip(',')
                            try:
                                control_dX1 = int(float(control_dX1))
                                control_dY1 = int(float(control_dY1))
                                control_dX2 = int(float(control_dX2))
                                control_dY2 = int(float(control_dY2))
                                dX2 = int(float(dX2))
                                dY2 = int(float(dY2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.cubicBesierCurve(penX, penY, penX + control_dX1, penY + control_dY1, penX + control_dX2, penY + control_dY2, penX + dX2, penY + dY2)
                            cubicBesierLastPoint_X = 2*(penX + dX2) - (penX + control_dX2) # calculate reflexion
                            cubicBesierLastPoint_Y = 2*(penY + dY2) - (penY + control_dY2) # calculate reflexion
                            penX = penX + dX2
                            penY = penY + dY2
                        elif command == "S":
                            assert cubicBesierLastPoint_X != None and cubicBesierLastPoint_Y != None, "Can't use S command, no last point found. Use C before it."
                            control_X2 = d.pop(0).strip(',')
                            control_Y2 = d.pop(0).strip(',')
                            newX2 = d.pop(0)
                            newY2 = d.pop(0)
                            try:
                                control_X2 = int(float(control_X2))
                                control_Y2 = int(float(control_Y2))
                                newX2 = int(float(newX2))
                                newY2 = int(float(newY2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.cubicBesierCurve(penX, penY, cubicBesierLastPoint_X, cubicBesierLastPoint_Y, control_X2, control_Y2, newX2, newY2)
                            cubicBesierLastPoint_X = 2*newX2 - control_X2
                            cubicBesierLastPoint_Y = 2*newY2 - control_Y2
                            penX = newX2
                            penY = newY2
                        elif command == "s":
                            assert cubicBesierLastPoint_X != None and cubicBesierLastPoint_Y != None, "Can't use S command, no last point found. Use C before it."
                            control_dX2 = d.pop(0).strip(',')
                            control_dY2 = d.pop(0).strip(',')
                            dX2 = d.pop(0).strip(',')
                            dY2 = d.pop(0).strip(',')
                            try:
                                control_dX2 = int(float(control_dX2))
                                control_dY2 = int(float(control_dY2))
                                dX2 = int(float(dX2))
                                dY2 = int(float(dY2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.cubicBesierCurve(penX, penY, cubicBesierLastPoint_X, cubicBesierLastPoint_Y, penX + control_dX2, penY + control_dY2, penX + dX2, penY + dY2)
                            cubicBesierLastPoint_X = 2*(penX + dX2) - (penX + control_dX2) # calculate reflexion of the control point by the end point of the curve
                            cubicBesierLastPoint_Y = 2*(penY + dY2) - (penY + control_dY2) # calculate reflexion 
                            penX = penX + dX2
                            penY = penY + dY2
                        elif command == "Q":
                            control_X1 = d.pop(0).strip(',')
                            control_Y1 = d.pop(0).strip(',')
                            newX2 = d.pop(0).strip(',')
                            newY2 = d.pop(0).strip(',')
                            try:
                                control_X1 = int(float(control_X1))
                                control_Y1 = int(float(control_Y1))
                                newX2 = int(float(newX2))
                                newY2 = int(float(newY2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.quadraticBesierCurve(penX, penY, control_X1, control_Y1, newX2, newY2)
                            penX = newX2
                            penY = newY2
                            cubicBesierLastPoint_X = penX
                            cubicBesierLastPoint_Y = penY
                        elif command == 'q':
                            control_dX1 = d.pop(0).strip(',')
                            control_dY1 = d.pop(0).strip(',')
                            dX2 = d.pop(0).strip(',')
                            dY2 = d.pop(0).strip(',')
                            try:
                                control_dX1 = int(float(control_dX1))
                                control_dY1 = int(float(control_dY1))
                                dX2 = int(float(dX2))
                                dY2 = int(float(dY2))
                            except Exception as ve:
                                print("Error in command " + command)
                                raise ve
                            self.quadraticBesierCurve(penX, penY, penX + control_dX1, penY + control_dY1, penX + dX2, penY + dY2)
                            penX = penX + dX2
                            penY = penY + dY2
                            cubicBesierLastPoint_X = penX
                            cubicBesierLastPoint_Y = penY
                except Exception as e:
                    print(e)
        self.image = self.image.reshape(self.size[0], self.size[1]*3)
        w.write(f, self.image)
        f.close()


if __name__ == "__main__":
    pass
    #r = SVGRenderer((1000,1000), colorSpace='grayscale')
    #r.drawSVG()
