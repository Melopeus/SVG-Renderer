import xml.etree.ElementTree as ET


class SVGParser:
    @staticmethod
    def parse_xml(svg_file):
        """
        This pasres a file and returns a tree of the XML
        :param svg_file:  String to the path of the file
        :return: root of XML tree of SVG
        """
        tree = ET.parse(svg_file)
        root = tree.getroot()
        return root


if __name__ == "__main__":
    SVGParser.parse_xml('svgTest.svg')
