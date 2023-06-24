import matplotlib.pyplot as plt
import numpy as np
import cv2
import pydicom as dicom 
import os
from skimage.segmentation import clear_border

path = ".\scans"
segmentation = []
imagens = []
seg_aux_paths = []
paths_seg = []
paths_img = []

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(display_list[i])
    plt.axis('off')
  plt.show()

def sort_list(list_str):
  x=[]
  dig="0123456789"
  for i in list_str:
      p=""
      for j in i:
          if j in dig:
              p+=j
      x.append(int(p))
  y=[]
  y.extend(x)
  x.sort()
  res=[]
  for i in x:
      res.append(list_str[y.index(i)])
  return res

def get_images_dir(path):
  for root, dirs, files in os.walk(path):
    files.sort()
    for file in files:
      relativePath = os.path.join(root, file)
      if('lung_mask' in relativePath):
        seg_aux_paths.append(relativePath)
      else:
        paths_img.append(relativePath)

    if('lung_mask' in root):
      aux_name = sort_list(seg_aux_paths)
      seg_aux_paths.clear()
      for filepath in aux_name:
        paths_seg.append(filepath)

def get_images(path):
      get_images_dir(path)
      print(len(paths_seg), len(paths_img))
      for i in range(len(paths_seg)):
        path_image, path_mask = paths_seg[i], paths_img[i]
        segmentation.append(dicom.dcmread(path_image).pixel_array)
        imagens.append(dicom.dcmread(path_mask).pixel_array)

get_images(path)

pred = []

def normalize_canais(input_image):
    input_image = np.stack((input_image,)*1, axis=-1)
    return input_image

kernel = np.ones((14, 14), 'uint8')
kernel2 = np.ones((12, 12), 'uint8')
for i in range(len(segmentation)):
    
    img = normalize_canais(imagens[i])

    filtro = cv2.medianBlur(img, 5)
    erode_img = cv2.erode(filtro, kernel2, iterations=1)
    dilate_img = cv2.dilate(erode_img, kernel, iterations=1)
    # Aplica uma limiarização para binarizar a imagem
    ret, thresh = cv2.threshold(dilate_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    mask = np.vectorize(clear_border, signature='(n,m)->(n,m)')(thresh)

    pred.append(mask)
    #para ver de maneira visual a segmentacao
    #display([imagens[i], segmentation[i], mask])

def get_metrics_pixel_to_pixel(true, predict):
  fn = 0
  tp = 0
  tn = 0
  fp = 0
  for index in range(len(predict)):
    for indexInside in range(len(predict[index])):
       for indexDeep in range(len(predict[index][indexInside])):
          if true[index][indexInside][indexDeep] == 255 and predict[index][indexInside][indexDeep] == 255:
             tp += 1
          if true[index][indexInside][indexDeep] == 0 and predict[index][indexInside][indexDeep] == 0:
             tn += 1
          if true[index][indexInside][indexDeep] == 255 and predict[index][indexInside][indexDeep] == 0:
             fn += 1
          if true[index][indexInside][indexDeep] == 0 and predict[index][indexInside][indexDeep] == 255:
             fp += 1
          
  return fn, tp, tn, fp

fn, tp, tn, fp = get_metrics_pixel_to_pixel(segmentation, pred)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
F1_score = 2 * (precision * recall) / (precision + recall)
dice_Coefficient = 2 * tp / (2*tp + fp + fn)
acuracia = (tp + tn) / (tp + fp + fn + tn)
print("False Negative: " + str(fn) + " True Negative: " + str(fp))
print("False Positive: " + str(tp) + " True Positive: " + str(tn))
print("Precision: " + str(precision))
print("recall: " + str(recall))
print("F1_score: " + str(F1_score))
print("Dice Coefficient: " + str(dice_Coefficient))
print("acuracia: " + str(acuracia))


