import cv2
from PIL import Image,ImageDraw,ImageFont
import numpy as np
img_OpenCV = cv2.imread('./data/testimages/00002.jpg')
# 图像从OpenCV格式转换成PIL格式
img_PIL = Image.fromarray(cv2.cvtColor(img_OpenCV, cv2.COLOR_BGR2RGB))

font = ImageFont.truetype('simsun.ttc', 40)
# 文字输出位置
position = (100, 100)
# 输出内容
str = '在图片上输出中文'

# 需要先把输出的中文字符转换成Unicode编码形式
# str = str.encode('utf8')

draw = ImageDraw.Draw(img_PIL)
x,y = draw.textsize(str,font)
draw.rectangle(((100,100),(100+x,100+y)),fill='white')
draw.text(position, str, font=font, fill=(255, 0, 0))
# 使用PIL中的save方法保存图片到本地
# img_PIL.save('02.jpg', 'jpeg')

# 转换回OpenCV格式
img_OpenCV = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
cv2.imshow("print chinese to image", img_OpenCV)
cv2.waitKey()
    
