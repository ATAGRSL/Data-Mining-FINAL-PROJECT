import numpy as np  # Importing the numpy library for numerical operations
import cv2 as cv  # Importing the OpenCV library for computer vision tasks

class Person:
    total_persons = 0  # Class variable to keep track of the total number of persons

    def __init__(self, xi, yi):
        self.x = xi  # X-coordinate of the person
        self.y = yi  # Y-coordinate of the person
        self.tracks = []  # List to store the historical tracks of the person
        Person.total_persons += 1  # Incrementing the total number of persons

    def updateCoords(self, xn, yn):
        self.tracks.append([self.x, self.y])  # Adding current coordinates to the track history
        self.x = xn  # Updating the x-coordinate of the person
        self.y = yn  # Updating the y-coordinate of the person

video_path = 'TestVideo.mp4'  # Path to the input video file

height = 480  # Height of the video frame
width = 640  # Width of the video frame
frame_area = height * width  # Total area of the video frame
area_threshold = frame_area / 250  # Threshold for contour area to filter out small detections

fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=True)  # Creating a background subtractor object

opening_kernel = np.ones((3, 3), np.uint8)  # Kernel for morphological opening operation
closing_kernel = np.ones((11, 11), np.uint8)  # Kernel for morphological closing operation

persons = []  # List to store the detected persons

cap = cv.VideoCapture(video_path)  # Opening the video capture object

while cap.isOpened():
    ret, frame = cap.read()  # Reading a frame from the video capture object

    if not ret or frame is None:
        break  # If no frame is read or the video capture is finished, break the loop

    foreground_mask = fgbg.apply(frame)  # Applying background subtraction to get the foreground mask
    foreground_mask2 = fgbg.apply(frame)

    _, binarized_mask = cv.threshold(foreground_mask, 200, 255, cv.THRESH_BINARY)  # Thresholding the foreground mask
    _, binarized_mask2 = cv.threshold(foreground_mask2, 200, 255, cv.THRESH_BINARY)
    opening_mask = cv.morphologyEx(binarized_mask, cv.MORPH_OPEN, opening_kernel)  # Morphological opening on the thresholded mask
    opening_mask2 = cv.morphologyEx(binarized_mask2, cv.MORPH_OPEN, opening_kernel)
    mask = cv.morphologyEx(opening_mask, cv.MORPH_CLOSE, closing_kernel)  # Morphological closing on the opened mask
    mask2 = cv.morphologyEx(opening_mask2, cv.MORPH_CLOSE, closing_kernel)

    contours, hierarchy = cv.findContours(mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Finding contours in the mask

    current_persons = 0  # Counter for the current number of persons in the frame

    for cnt in contours:
        area = cv.contourArea(cnt)  # Calculating the area of the contour
        if area > area_threshold:  # Checking if the contour area exceeds the threshold
            M = cv.moments(cnt)  # Calculating the moments of the contour
            cx = int(M['m10'] / M['m00'])  # Calculating the centroid x-coordinate
            cy = int(M['m01'] / M['m00'])  # Calculating the centroid y-coordinate
            x, y, w, h = cv.boundingRect(cnt)  # Getting the bounding rectangle coordinates

            new_person = True  # Flag indicating if a new person is detected
            for person in persons:
                if abs(x - person.x) <= w and abs(y - person.y) <= h:
                    new_person = False  # If the bounding rectangle overlaps with an existing person, it is not a new person
                    person.updateCoords(cx, cy)  # Updating the coordinates of the existing person
                    break
            if new_person:
                p = Person(cx, cy)  # Creating a new Person object
                persons.append(p)  # Adding the new person to the list

            cv.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # Drawing a circle at the centroid of the person
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)  # Drawing a rectangle around the person

            current_persons += 1  # Incrementing the current number of persons

    str_all_persons = 'All persons: ' + str(Person.total_persons)  # String for displaying the total number of persons
    str_current_persons = 'Current persons: ' + str(current_persons)  # String for displaying the current number of persons

    cv.putText(frame, str_all_persons, (10, 190), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv.LINE_AA)  # Displaying the total number of persons
    cv.putText(frame, str_current_persons, (10, 220), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv.LINE_AA)  # Displaying the current number of persons

    cv.imshow('Frame', frame)  # Displaying the frame

    k = cv.waitKey(30) & 0xff  # Waiting for a key press with a delay of 30 milliseconds
    if k == 27:
        break  # If the 'Esc' key is pressed, break the loop

cap.release()  # Releasing the video capture object
cv.destroyAllWindows()  # Closing all the windows