from paquo.projects import QuPathProject
from paquo.images import QuPathImageType
from openslide import open_slide
import openslide
from PIL import Image


qp = QuPathProject('/Users/lelandling/Documents/Stanford work/neuroblastoma - shimada project/Shimada/test/', mode='r')  # open project for appending

slide = open_slide('/Users/lelandling/Documents/Stanford work/neuroblastoma - shimada project/908998 A4.svs')
# openslidetest.read_region((round(16810.0), round(32125.0)), 0, (round(5190.0), round(8234.0)))
# prop = slide.properties
# print(prop["openslide.level[0].tile-height"]) 

tb = slide.read_region((13438, (10704-1959)), 0, (3547, 1959))
tb.show()

# for property in slide.properties :
#     print(property)
# # slide.show()


# slidetb = slide.get_thumbnail(size=(600,600))
# slidetb.show()

# dims = slide.level_dimensions

# print(dims)

# roi = slide.read_region((38129, 62412), 0, (10000, 10000))
# roi.show()

qp.images  # <- access images via this
# print(qp.is_readable())
image = qp.images[0]  # get the first image
# print(image.hierarchy.image)  # annotations are stored in a set like proxy object

# QuPathPathAnnotationObjectSet(n=2)
# x = image.hierarchy.annotations[1].roi.exterior.coords.xy[0].tolist()
# y = image.hierarchy.annotations[1].roi.exterior.coords.xy[1].tolist()
# print(list(zip(x,y)))
# for annotation in image.hierarchy.annotations:
#     print(annotation.name, annotation.path_class, annotation.roi)
