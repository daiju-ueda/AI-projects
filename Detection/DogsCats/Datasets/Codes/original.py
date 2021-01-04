import numpy as np
import pandas as pd

from PIL import Image
from glob import glob
import xml.etree.ElementTree as ET

class xml2list(object):

    def __init__(self, classes):
        self.classes = classes

    def __call__(self, xml_path):

        ret = []
        xml = ET.parse(xml_path).getroot()

        for size in xml.iter("size"):
            width = float(size.find("width").text)
            height = float(size.find("height").text)

        for obj in xml.iter("object"):
            difficult = int(obj.find("difficult").text)
            if difficult == 1:
                continue
            bndbox = [width, height]
            name = obj.find("name").text.lower().strip()
            bbox = obj.find("bndbox")
            pts = ["xmin", "ymin", "xmax", "ymax"]
            for pt in pts:
                cur_pixel =  float(bbox.find(pt).text)
                bndbox.append(cur_pixel)
            bndbox.append(name)
            ret += [bndbox]
        return np.array(ret)

xml_paths = glob("../Original/Annotations/*.xml")
#img_paths = glob("../Original/Images/*.xml")
classes = ["dog", "cat"]

transform_anno = xml2list(classes)

df = pd.DataFrame(columns=["image_id", "width", "height", "xmin", "ymin", "xmax", "ymax", "class"])

for path in xml_paths:
    image_id = path.split("/")[-1].split(".")[0]
    bboxs = transform_anno(path)

    for bbox in bboxs:
        tmp = pd.Series(bbox, index=["width", "height", "xmin", "ymin", "xmax", "ymax", "class"])
        tmp["image_id"] = image_id
        df = df.append(tmp, ignore_index=True)

df = df.sort_values(by="image_id", ascending=True)
df.to_csv('../Original/' + 'summary.csv', index=False)
