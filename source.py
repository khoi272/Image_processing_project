import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

### Dictionary to look up traffic signs
SIGNS_LOOKUP_RED = {
    (1, 1, 0, 1): 'Signs for tractors',
    (1, 0, 0, 1): 'No-horn sign',
    (0, 1, 1, 1): 'No go straight sign',
    (0, 1, 0, 1): 'No-taxi sign',
    (0, 1, 0, 0): 'No-parking sign',
    (0, 0, 1, 0): 'No-right-turn sign',
    (1, 1, 1, 1): 'Speed-Limit 20 km/h sign',
    (0, 0, 0, 1): 'No U-Turn sign',
    (0, 1, 1, 0): 'Height-Limit-3.7m sign',
    (0, 0, 1, 1): 'No Left Turn Sign',
}

SIGNS_LOOKUP_RED_LOWER = {
    (1, 0, 1, 0): 'Stop-sign',
    (1, 1, 1, 0): 'No-Entry Sign',
    (1, 1, 1, 1): 'No-cycling sign',
    (0, 0, 0, 0): 'No-Entry sign'
}

### function used to transform perspective by transforming four specified points.
def four_point_transform(image, pts):
    # Initialize an array to hold the ordered points (top-left, top-right, bottom-right, bottom-left)
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum the x and y coordinates and find the top-left (smallest sum) and bottom-right (largest sum) points
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left point
    rect[2] = pts[np.argmax(s)]  # Bottom-right point
    
    # Compute the difference between the x and y coordinates and find the top-right (smallest difference)
    # and bottom-left (largest difference) points
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right point
    rect[3] = pts[np.argmax(diff)]  # Bottom-left point
    
    # Optional: Apply a scaling ratio (not needed here, so ratio is 1)
    ratio = 1
    rect *= ratio
    
    # Unpack the ordered points
    (tl, tr, br, bl) = rect
    
    # Compute the width of the new image, which will be the maximum distance between the bottom-right and
    # bottom-left x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute the height of the new image, which will be the maximum distance between the top-right and
    # bottom-right y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]],
        dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

###  Traffic sign recognition function
def identifyTrafficSign(image, mask_red, blue):
    # Invert the colors of the image (this operation reverses black to white and white to black, etc.)
    image = cv2.bitwise_not(image)

    # Divide the image dimensions by 10 to get the size of the blocks
    (subHeight, subWidth) = np.divide(image.shape, 10)
    subHeight = int(subHeight)  # Convert to integer
    subWidth = int(subWidth)  # Convert to integer

    # Define blocks from the image
    leftBlock = image[4 * subHeight:9 * subHeight, subWidth:3 * subWidth]  # Block on the left side
    centerBlock = image[4 * subHeight:9 * subHeight, 4 * subWidth:6 * subWidth]  # Center block
    rightBlock = image[4 * subHeight:9 * subHeight, 7 * subWidth:9 * subWidth]  # Block on the right side
    topBlock = image[2 * subHeight:4 * subHeight, 3 * subWidth:7 * subWidth]  # Block on the top

    # Calculate the fraction of the white pixels in each block
    leftFraction = np.sum(leftBlock) / (leftBlock.shape[0] * leftBlock.shape[1])
    centerFraction = np.sum(centerBlock) / (centerBlock.shape[0] * centerBlock.shape[1])
    rightFraction = np.sum(rightBlock) / (rightBlock.shape[0] * rightBlock.shape[1])
    topFraction = np.sum(topBlock) / (topBlock.shape[0] * topBlock.shape[1])

    # Combine the fractions into a tuple
    segments = (leftFraction, centerFraction, rightFraction, topFraction)
    print(segments)  # Print out the segment fractions

    # Determine the threshold and lookup dictionary based on the color (blue or red)
    if blue:
        THRESHOLD = 100  # Threshold for detecting blue traffic signs
        SIGNS_LOOKUP = SIGNS_LOOKUP_BLUE  # Lookup dictionary for blue signs
    else:
        THRESHOLD = 196  # Threshold for detecting red traffic signs
        SIGNS_LOOKUP = SIGNS_LOOKUP_RED  # Lookup dictionary for red signs

        # Adjust the threshold and lookup dictionary if all segments are below the initial threshold
        if leftFraction < THRESHOLD and centerFraction < THRESHOLD and rightFraction < THRESHOLD and topFraction < THRESHOLD:
            print('Lower')
            THRESHOLD = 80  # Lower threshold for red traffic signs
            SIGNS_LOOKUP = SIGNS_LOOKUP_RED_LOWER  # Lookup dictionary for red signs with lower threshold

    # Convert the fractions into binary values (1 if above threshold, 0 if below)
    segments = tuple(1 if segment > THRESHOLD else 0 for segment in segments)
    print(segments)  # Print out the binarized segments

    # Check if the binarized segments match any entry in the lookup dictionary
    if segments in SIGNS_LOOKUP:
        return SIGNS_LOOKUP[segments]  # Return the identified traffic sign
    else:
        return None  # Return None if no match is found


### Traffic sign detection function 
def findTrafficSign(image_path, output_folder, blue):
    print(image_path)  # Print out the image paths
    bb = 'box'
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 60])
    upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([90, 110, 80])
    upper_blue = np.array([255, 255, 255])

    # Read image
    frame = cv2.imread(image_path)
    frame = cv2.resize(frame, (250, 250))  # Resize the image
    frameArea = frame.shape[0] * frame.shape[1]

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)

    # Segmentation based on color
    if blue:
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    else:
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask_red1, mask_red2)

    # Transform morphology
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Edge detection
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largestArea = 0
    largestRect = None
    if len(cnts) > 0:
        for cnt in cnts:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            sideOne = np.linalg.norm(box[0] - box[1])
            sideTwo = np.linalg.norm(box[0] - box[3])
            area = sideOne * sideTwo
            if area > largestArea:
                largestArea = area
                largestRect = box

    # Condition: If the area is greater than a threshold
    if largestArea > frameArea * 0.02:
        cv2.drawContours(frame, [largestRect], 0, (0, 0, 255), 3)
        output_folder_bb = output_folder + bb
        if not os.path.exists(output_folder_bb):
            os.makedirs(output_folder_bb)
        output_folder_bb = os.path.join(output_folder_bb, os.path.basename(image_path))
        cv2.imwrite(output_folder_bb, frame)

        # Perspective transformation
        warped = four_point_transform(mask, [largestRect][0])

        # Traffic sign recognition
        detectedTrafficSign = identifyTrafficSign(warped, mask, blue)

        # Adding text to an image
        text_position = (30, 30)
        cv2.putText(frame, detectedTrafficSign, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save output image
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, frame)

### Function to display images in a folder
def display_images_in_folder(folder_path, num_cols=3):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    num_images = len(image_files)
    num_rows = (num_images + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    fig.set_facecolor('black')  # Set the background of the figure to black
    axes = axes.ravel()
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(image)
        axes[idx].axis('off')
        axes[idx].set_facecolor('black')  # Set the background of the axis to black

    for i in range(idx + 1, len(axes)):
        axes[i].axis('off')
        axes[i].set_facecolor('black')  # Set the background of the axis to black
    plt.tight_layout()
    plt.show()

### Function to read images from a folder and detect traffic signs
def read_image(folder_red_path, output_folder, blue=False):
    image_files = os.listdir(folder_red_path)
    for image_file in image_files:
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(folder_red_path, image_file)
            findTrafficSign(image_path, output_folder, blue)

### Main function
def main():
    folder_path = 'image_red'
    output_folder = 'image_output'
    bb = 'box'
    output_folder_bb = output_folder + bb
    read_image(folder_path, output_folder)
    display_images_in_folder(output_folder_bb)
    display_images_in_folder(output_folder)

if __name__ == '__main__':
    main()
