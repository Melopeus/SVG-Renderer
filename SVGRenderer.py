import numpy as np
import png

class SVGRenderer:
    def __init__(self, size, colorSpace="rgb"):
        if colorSpace == 'grayscale':
            self.image = np.zeros(size, dtype='uint8')
        elif colorSpace == 'rgb':
            self.image = np.zeros((*size, 3), dtype='uint8')
        self.colorSpace = colorSpace
        self.size = size


    def putPixel(self, x: int, y: int, color):
        if 0 < x < self.size[0] and 0 < y < self.size[1]:
            if self.colorSpace == 'grayscale':
                self.image[y][x] = color
            else:
                self.image[x][y][0] = color[0]
                self.image[x][y][1] = color[1]
                self.image[x][y][2] = color[2]

    # https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    def line(self, x1, y1, x2, y2):
        dx =  abs(x2-x1)
        sx = 1 if x1<x2 else -1
        dy = -abs(y2-y1)
        sy = 1 if y1<y2 else -1
        err = dx+dy
        while True:
            self.putPixel(x1, y1,255)

            self.putPixel(x1+1, y1,255)
            self.putPixel(x1-1, y1,255)
            self.putPixel(x1, y1+1,255)
            self.putPixel(x1, y1-1,255)
            if x1 == x2 and y1 == y2: break
            e2 = 2*err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy
            
    def rectangle(self, x1, y1, x2, y2):
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
            self.putPixel( x + xc, y + yc , 255)
            self.putPixel(-x + xc, y + yc , 255)
            self.putPixel( x + xc,-y + yc , 255)
            self.putPixel(-x + xc,-y + yc , 255)
    
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
            self.putPixel( x + xc, y + yc , 255)
            self.putPixel(-x + xc, y + yc , 255)
            self.putPixel( x + xc,-y + yc , 255)
            self.putPixel(-x + xc,-y + yc , 255)
    
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

    def drawSVG(self):
        self.rectangle(10, 10, 300, 300)
        self.ellipse(700,200, 500, 500)
        f = open('swatch.png', 'wb')
        w = png.Writer(self.size[0], self.size[1], greyscale=True)
        
        w.write(f, self.image)
        f.close()

r = SVGRenderer((1000,1000), colorSpace='grayscale')
r.drawSVG()