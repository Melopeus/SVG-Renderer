import xml.etree.ElementTree as ET

class SVGParser:
    @staticmethod
    def parseXML(svgFile):
        tree = ET.parse( svgFile )
        root = tree.getroot()
        return root


if __name__ == "__main__":
    
    SVGParser.parseXML('svgTest.svg')