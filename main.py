try:
  from PIL import Image
except ImportError:
  import Image
import pytesseract
import cv2
import numpy as np
from bentley_ottmann.planar import segments_intersections
from matplotlib import pyplot as plt
from pdftabextract import imgproc
from pdftabextract.clustering import find_clusters_1d_break_dist

img = cv2.imread('test.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel_size = 5
blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur, low_threshold, high_threshold)
cv2.imwrite('edges.png', edges)

rho = 1
theta = np.pi / 180
threshold = 35
min_line_length = 50
max_line_gap = 20
line_image = np.copy(img) * 0
lines = cv2.HoughLinesP(
  edges,
  rho,
  theta,
  threshold,
  np.array([]),
  min_line_length,
  max_line_gap
)

points = []
for line in lines:
  for x1,y1,x2,y2 in line:
    points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
line_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

intersections = segments_intersections(points)
#for idx, inter in enumerate(intersections):
#  a, b = inter
#  match = 0
#  for other_inter in intersections[idx:]:
#    c, d = other_inter
#    if abs(c - a) < 15 and abs(d - b) < 15:
#      match = 1
#      intersections[idx] = ((c + a) / 2, (d + b) / 2)
#      intersetctions.remove(other_inter)
#  if match == 0:
#    intersections.remove(inter)
for inter in intersections:
  a, b = inter
  for i in range(3):
    for j in range(3):
      line_edges[int(b) + i, int(a) + j] = [0, 255, 0]

cv2.imwrite('lines2.jpg', line_edges)

#iproc_obj = imgproc.ImageProc('test.png')
#lines_hough = iproc_obj.detect_lines(
#  canny_kernel_size=3,
#  canny_low_thresh=150,
#  canny_high_thresh=200,
#  hough_rho_res=1,
#  hough_theta_res=np.pi/180,
#  hough_votes_thresh=round(0.2*iproc_obj.img_w)
#)
#print("> found %d lines" % len(lines_hough))
#img_lines = iproc_obj.draw_lines(orig_img_as_background=True)
#cv2.imwrite('lines.png', img_lines)
#
#vertical_clusters = iproc_obj.find_clusters(
#  imgproc.DIRECTION_VERTICAL,
#  find_clusters_1d_break_dist,
#  dist_thresh=30
#)
#print("> found %d clusters" % len(vertical_clusters))
#img_w_clusters = iproc_obj.draw_line_clusters(imgproc.DIRECTION_VERTICAL, vertical_clusters)
#cv2.imwrite('clusters.png', img_w_clusters)

#im = cv2.imread('test.png')
#gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray,100,200,apertureSize=3)
#
#while True:
#  img = im.copy()
#  k = cv2.waitKey(0)
#  lines = cv2.HoughLines(edges,1,np.pi/180,250)
#  for line in lines:
#    for rho,theta in line:
#      a = np.cos(theta)
#      b = np.sin(theta)
#      x0 = a*rho
#      y0 = b*rho
#      x1 = int(x0 + 1000*(-b))
#      y1 = int(y0 + 1000*(a))
#      x2 = int(x0 - 1000*(-b))
#      y2 = int(y0 - 1000*(a))
#      cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
#  cv2.imshow('houghlines',img)
#
#  if k == 27:
#    break
#
#cv2.destroyAllWindows()

#img = cv2.imread('test.png')
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray,100,200,apertureSize=3)
#plt.subplot(121),plt.imshow(img,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#plt.show()

#table = Image.open('test.png')
#gray = table.convert('L')
#blackwhite = gray.point(lambda x: 0 if x < 90 else 255, '1')
#blackwhite.save('bw.jpg')

#print(pytesseract.image_to_string(Image.open('bw.jpg')))
#print(pytesseract.image_to_boxes(Image.open('bw.jpg')))
#print(pytesseract.image_to_data(Image.open('bw.jpg')))
