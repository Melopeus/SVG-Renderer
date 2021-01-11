from SVGParser import SVGParser
from SVGRenderer import SVGRenderer


class SVGConverter:
    @staticmethod
    def make_png(name, svg_file):
        """
        Creates a png file with the name given and from the svg file path provided
        :param name: string, name of the file
        :param svg_file: string, path to the svg file
        :return: None
        """
        root = SVGParser.parse_xml(svg_file)
        width = int(root.attrib['width'])
        height = int(root.attrib['height'])
        rendered = SVGRenderer((height, width))
        rendered.draw_svg(name, root)


if __name__ == "__main__":
    SVGConverter.make_png('Test.png', 'svgTest.svg')