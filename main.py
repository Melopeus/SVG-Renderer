from SVGParser import SVGParser
from SVGRenderer import SVGRenderer

class SVGConverter:
    @staticmethod
    def makePNG(name, SVGFile):
        root = SVGParser.parseXML('svgTest.svg')
        width = int(root.attrib['width'])
        height = int(root.attrib['height'])
        rendered = SVGRenderer((width,height), colorSpace='grayscale')
        rendered.drawSVG(name, root)


if __name__ == "__main__":
    SVGConverter.makePNG('Test.png', 'svgTest.svg')








