import cv2


# Create our body classifier
classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('WhatsApp Video 2023-10-21 at 16.44.09_70c7d9fc.mp4')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    scale = classifier.detectMultiScale(gray,1.1,5)
    # Pass frame to our body classifier
    for (x,y,w,h) in scale:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),5)
    
    # Extract bounding boxes for any bodies identified
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
