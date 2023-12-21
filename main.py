import cv2
import matplotlib.pyplot as plt
import numpy as np


def find_house(img, inner = None):
    bw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(bw, 100, 200)
    circles = cv2.HoughCircles(
        canny,
        cv2.HOUGH_GRADIENT,
        dp=1,
        param1=100,
        param2=100,
        minDist=40,
        minRadius=40
    ).astype(np.int64)

    values = np.atleast_2d(np.squeeze(circles))[0]
    center = (values[0], values[1])
    radius = values[-1]

    out = img.copy()

    cv2.circle(
        out,
        center=(center[0], center[1]),
        radius=radius,
        color=(255, 0, 0),
        thickness=2
    )

    for smaller in inner:
        cv2.circle(
            out,
            center=(center[0], center[1]),
            radius=int(radius * smaller),
            color=(255, 0, 0),
            thickness=2
        )

    return out, center



if __name__ == '__main__':
    file = "data/some_top.png"
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(
        bw,
        (3, 3),
        0
    )
    canny = cv2.Canny(bw, 100, 200)
    contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    circles_info = []

    # Iterate through detected contours
    for contour in contours:
        # Fit an ellipse to the contour (assuming it's a circle)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            # Extract ellipse information: center, axes lengths (major and minor), and angle
            center, axes, angle = ellipse

            # Calculate the diameter of the circle (average of major and minor axes)
            diameter = int((axes[0] + axes[1]) / 2.0)

            # Filter out non-circular shapes based on aspect ratio
            min_axis_length = min(axes)
            if min_axis_length > 0 and diameter > 50:
                aspect_ratio = max(axes) / min_axis_length
                if aspect_ratio < 1.2:
                    # Store circle information as a tuple (center, diameter)
                    circles_info.append((center, diameter))

    # Calculate the average diameter for each circle
    average_diameters = {}

    # Iterate through detected circles and calculate the average diameter
    for _, diameter in circles_info:
        # Find circles with similar diameters
        similar_diameters = [d for _, d in circles_info if abs(d - diameter) < 10]

        # Calculate the average diameter
        average_diameter = sum(similar_diameters) / len(similar_diameters)

        # Store the average diameter with the diameter value as the key
        average_diameters[average_diameter] = similar_diameters

    # Draw the circles and their average diameters on the frame
    for center, diameter in circles_info:
        # Convert the center coordinates to integers
        center = (int(center[0]), int(center[1]))

        # Draw the circle
        cv2.circle(img, center, int(diameter / 2), (0, 255, 0), 3)

        # Calculate the topmost point on the circle
        topmost_point = center[1] - int(diameter / 2)

        # Find the corresponding average diameter
        for avg_diameter, similar_diameters in average_diameters.items():
            if diameter in similar_diameters:
                # Draw the average diameter as text next to the circle
                cv2.putText(img, f"Diameter: {avg_diameter:.2f}", (center[0], topmost_point),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                break

    plt.subplot(121)
    plt.imshow(img)

    plt.subplot(122)
    plt.imshow(canny)



