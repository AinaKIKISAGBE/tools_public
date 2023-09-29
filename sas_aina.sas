

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$@$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$@$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$8oohkho8$$$$$$$$$$Mooo@$$$@oooW$$$$$$$*oM$@$$$*bbB$$$$$okhhoohkhk%$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$@$z       "n@$$$$$@$?   f$B$/   [$@$$$$$, <$$$J' ,UB$$$$$!         d$@$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$@$c  p%8ML  _$$$$$@$- .  M@#  . [$@$$$$$" ~$Zl .c@$$$$$$$B8%b  f%8%$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$@$c  k$$$$u  d$@$$@$? -/ 1${ f+ }$@$$$$$" ,~  'W$@@$$$$$$$$$o  x$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$@$c  b$$@$c  q$@$$@$? ~W  C  &i }$@$$$$$^   n! ^w$@$$$$$$$@$a  r$@$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$@$c  k$$Bb. i@$$$$@$? i$}   ($l }$@$$$$$" IB$%?  u$$$$$$$$$$a  r$@$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$@$v   ..   \%$$$$$@$- i$o   #$l [$@$$$$$^ <$B$$t  _M$$$$$$$$a  j$@$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$Wpdwmmp*$$$$$$$$$$opa$$hpa$$ap*$$$$$$$kpa$$$$$kdZM$$$$$$$$BdpM$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$@$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$@$@@@$$$$$$$$$$$$$@$$$$@$$$$@$$$$$$$$$@$$$$$$$@$$$$$$$$$$$$@$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$








oooooooooo.   ooo        ooooo oooo    oooo ooooooooooooo 
`888'   `Y8b  `88.       .888' `888   .8P'  8'   888   `8 
 888      888  888b     d'888   888  d8'         888      
 888      888  8 Y88. .P  888   88888[           888      
 888      888  8  `888'   888   888`88b.         888      
 888     d88'  8    Y     888   888  `88b.       888      
o888bood8P'   o8o        o888o o888o  o888o     o888o     







.sSSSSs.    .sSSSsSS SSsSSSSS .sSSS  SSSSS     .sSSSSSSSSs.   
SSSSSSSSSs. SSSSS  SSS  SSSSS SSSSS  SSSSS  .sSSSSSSSSSSSSSs. 
S SSS SSSSS S SSS   S   SSSSS S SSS SSSSS   SSSSS S SSS SSSSS 
S  SS SSSSS S  SS       SSSSS S  SS SSSSS   SSSSS S  SS SSSSS 
S..SS SSSSS S..SS       SSSSS S..SSsSSSSS   `:S:' S..SS `:S:' 
S:::S SSSSS S:::S       SSSSS S:::S SSSSS         S:::S       
S;;;S SSSSS S;;;S       SSSSS S;;;S  SSSSS        S;;;S       
S%%%S SSSS' S%%%S       SSSSS S%%%S  SSSSS        S%%%S       
SSSSSsS;:'  SSSSS       SSSSS SSSSS   SSSSS       SSSSS       










###########################################################
######### python generate asic art 1 from image png, jpg, ...  ########################
import sys
import numpy as np
from PIL import Image

# Contrast on a scale -10 -> 10
contrast = 10
density = ('$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|'
           '()1{}[]?-_+~<>i!lI;:,"^`\'.            ')
density = density[:-11+contrast]
n = len(density)


try:
    img_name = sys.argv[1]
    width = int(sys.argv[2])
except IndexError:
    # Default ASCII image width.
    width = 100
    img_name="C:/Users/user_distant/Downloads/DMKT.PNG"

# Read in the image, convert to greyscale.
img = Image.open(img_name)
img = img.convert('L')
# Resize the image as required.
orig_width, orig_height = img.size
r = orig_height / orig_width
# The ASCII character glyphs are taller than they are wide. Maintain the aspect
# ratio by reducing the image height.
height = int(width * r * 0.5)
img = img.resize((width, height), Image.ANTIALIAS)

# Now map the pixel brightness to the ASCII density glyphs.
arr = np.array(img)
for i in range(height):
    for j in range(width):
        p = arr[i,j]
        k = int(np.floor(p/256 * n))
        print(density[n-1-k], end='')
    print()

###########################################################
######### python generate asic art 2 from text ######################

def no_use():
    # -*- coding: utf-8 -*-
    """
    Created on Thu Sep 28 01:21:52 2023

    @author: user_distant
    """

    from art import *

    tprint("DMKT",font="rnd-large")


    tprint("DMKT","rnd-xlarge")



    text2art("DMKT",font="black") 

    import pyfiglet

    pyfiglet.figlet_format("DMKT",font='isometric1')


    pyfiglet.figlet_format("DMKT", font="3-d")

    pyfiglet.figlet_format("DMKT", font="alligator")


    DMKT DTM


















