import cv2

# Draw line:
def draw_line(image):
    im_height, im_width = image.shape[:2]
    center = (im_width/2, im_height/2)
    cv2.rectangle(image, (460, int(center[1])), (im_width - 250, int(center[1] + 1)), [0,0,255], 1)
    return center

def put_number_moto(image, count):
    text = "Number of motobike: {}".format(count)
    cv2.putText(image, text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def put_objectID_into_object(image, centroid, objectID):
    text = "{}".format(objectID)
    cv2.putText(image, text, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
