from PIL import Image
import sys

if __name__=='__main__':

    PIC_PATH = sys.argv[1]

    # jpgfile = Image.open("./westbrook.jpg")
    jpgfile = Image.open(PIC_PATH)
    pix = jpgfile.load()

    width,height = jpgfile.size

    for i in range(width):
        for j in range(height):
            rgb=tuple(int(el/2) for el in pix[i,j])
            pix[i,j]=rgb
    jpgfile.save("./Q2.jpg")        