"""
This program processes images of a jigsaw pieces.
It can process a single image, or a folder full of image.
It provides functionality for the following processes:
    - Remove the background
    - Crop to only the jigsaw piece
    - Resize to desired size
    - Denoise
    - Center jigsaw piece within the image
    - Add a solid background colour (optional)
Also accepts the following command line arguments:
    - `--quiet` - Only print warnings/errors.
    - `--show-path` - Print the final filepath when complete.
    - `--replace` - Replace original file.
    - You must also give a path to a file or folder to be processed.
        - This can occur anywhere within the arguments.
        - Make sure to enclose within quotes if the path contains spaces.
To process images from another Python script, import the script using
`import image_processing` and call the `image_processor.process_from_code(...)` method
with the command line arguments as a string, e.g.
```
import image_processor as ip
import os
ip.process_from_code(os.path.join(os.getcwd(), "test.png") + " --quiet")
```
Example usage:
    - `python image_processor.py --quiet "C:/some/valid/path/to/an/image.png"`
        - Processes a single image and does not output anything to the command line unless a warning is triggered.
    - `python image_processor.py "C:/another/path/to/a/folder/containing/images"`
        - Processes an entire folder full of images.
"""
import math
import os
import shutil
import sys
import shlex

import cv2
import numpy as np
from PIL import Image
from rembg import remove

# Desired output image size in X by X pixels.
desired_size = 256

# Amount of background removed to trigger an error.
# E.g. if more than 35% of the image has been removed, then the background removal has failed.
background_removal_error_threshold_percent = 0.20

# Percentage of size of bounding area compared to total area to trigger an error.
# E.g. if the area of the bounding rect is less than 20% of the original size, trigger an error.
bounding_factor_error = 0.1

# Clamp values for threshold (used in bounding rect calculation)
# To increase sensitivity, increase the distance between the two.
bounding_clamp_lower_value = 5
bounding_clamp_upper_value = 70

# Background colour used when adding the background.
# If the alpha value is max (255), no background is added.
background_colour = (128, 0, 0, 255)  # RGBA

# Input path to the image or folder containing images to process.
# Can either point to an image, or a folder.
input_path = ""

# Logging level. Will only output warnings if true.
quiet = False

# Should output final filepath when complete?
should_output_path = False

# If original image should be replaced by processed image
replace = False

# Return output path(s) when calling from code.
output_paths = []


def main():
    """
    Begin execution.
    """

    if not quiet:
        print("Starting...")

    # Check type of input...
    if os.path.isdir(input_path):
        # is a path.
        if not quiet:
            print("Processing directory...")

        # Process each file in the path.
        for file_path in os.listdir(input_path):
            if os.path.isdir(file_path):
                continue
            process(os.path.join(input_path, file_path))

    else:
        # is a file.
        if input_path.endswith(".png") or input_path.endswith(".jpg"):
            process(input_path)


def process_from_code(args_as_string):
    """
    Call the image processor script from code rather than command line.
    Takes the same args as command line but in the form of a string.
    Returns a list containing all output paths as strings. If only a single file is processed, the list will only contain 1 item.
    """

    # https://docs.python.org/3/library/shlex.html
    args = shlex.split(args_as_string)
    parse_args(args)

    main()

    return output_paths


def parse_args(args):
    """
    Parse the arguments given to the program.
    """
    global quiet, input_path, should_output_path, output_paths, replace

    # Clean up previous values.
    quiet = False
    should_output_path = False
    input_path = ""
    output_paths = []

    for arg in args:
        # Flag
        if arg.startswith("-"):
            if arg == "--quiet":
                quiet = True
            elif arg == "--show-path":
                should_output_path = True
            if arg == "--replace":
                replace = True
        # Else, input path
        else:
            input_path = arg


def process(file_path):
    """
    Process the file at the given file path.
    """

    # Retrieve the base file name
    file_name = os.path.basename(file_path)
    if not quiet:
        print("Processing file: " + str(file_name))

    # Copy the file to memory to be worked on.
    file = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    # Apply the filters in order.
    # fast_denoise()
    file, mask = remove_background(file)
    # draw_debug_rect(calc_bounding_rect())
    file = crop_image(file, calc_bounding_rect(file, mask))
    file = resize(file)
    file = center(file)
    file = add_background_colour(file)

    # Calculate the output path.
    output_path = "unknown.png"

    append = ""
    # To stop file being replaced -out will be added to file name
    if not replace:
        append = "-piece"

    if os.path.isfile(input_path):
        # If just a file, save in the same folder as the input.
        if ".png" in input_path:
            idx = input_path.index(".png")
        else:
            idx = input_path.index(".jpg")
        
        output_path = input_path[:idx] + append + input_path[idx:]
    else:
        # Same as above, but respects path inputs.
        basename = os.path.basename(file_path).replace(".png", "")
        basepath = file_path.replace(os.path.basename(file_path), "")
        output_path = os.path.join(basepath, basename + append + ".png")

    cv2.imwrite(output_path, file)

    if not quiet:
        print("Done!")

    output_paths.append(output_path)

    if should_output_path:
        print(output_path)


def draw_debug_rect(rect_array):
    """
    Draw a rectangle on the image containing where we think the bounds of the piece are.
    Should be used for debugging only.
    """

    if not quiet:
        print("Drawing debug rect...")

    # Decompose input array.
    x = rect_array[0]
    y = rect_array[1]
    w = rect_array[2]
    h = rect_array[3]

    if not quiet:
        print("Start position: " + str(x) + ", " + str(y))
        print("Dimensions: " + str(w) + ", " + str(h))

    # Read the image and draw the rectangle.
    img = cv2.imread(temp_path)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # Write the image.
    cv2.imwrite(temp_path, img)


def add_background_colour(file):
    """
    Adds a simple solid background colour to the image.
    No processing is applied if the colour is transparent.
    The `background_colour` variable defines the colour to use.
    Can be used to remove transparency if needed.
    """

    # If fully transparent, skip processing.
    if background_colour[3] == 255:
        return file

    if not quiet:
        print("Adding background colour...")

    # Create a new image with the background colour.
    colour = Image.new("RGB", (desired_size, desired_size), background_colour)

     # Convert cv2 to PIL image
    img = cv2.cvtColor(file, cv2.COLOR_BGRA2RGBA)
    piece = Image.fromarray(img)

    # Combine the two, and save.
    colour.paste(piece, (0, 0), piece)
    
    # Convert back
    colour = np.asarray(colour)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGRA2RGBA)
    return colour


def crop_image(file, rect_array):
    """
    Crop the image to the given rectangle in the form of an array containing 4 elements (x start, y start, width, height).
    """

    if not quiet:
        print("Cropping...")

    # Decompose input array.
    x = rect_array[0]
    y = rect_array[1]
    w = rect_array[2]
    h = rect_array[3]

    # Crop the image using numpy array.
    crop_img = file[y:y + h, x:x + w]
    return crop_img


def resize(file):
    """
    Resize the image, preserving aspect ratio.
    The image will never be bigger than the desired output size.
    """

    if not quiet:
        print("Resizing...")

    # Read the image and get height and width.
    width = file.shape[1]
    height = file.shape[0]

    # Assuming square only
    desired_size_x = desired_size
    desired_size_y = desired_size

    if not quiet:
        print('Before Resizing...')
        print('Image Width is', width)
        print('Image Height is', height)

    # Preserving the aspect ratio
    if width > height:
        desired_size_y *= (height / width)
    elif height > width:
        desired_size_x *= (width / height)

    # Actually perform the resizing.
    img_resized = cv2.resize(file, (int(desired_size_x), int(desired_size_y)))

    if not quiet:
        print('Resized!')
        print('Image Width is now', img_resized.shape[1])
        print('Image Height is now', img_resized.shape[0])
    
    return img_resized


def center(file):
    """
    Center the image by adding transparent borders to fit the desired output size.
    """

    if not quiet:
        print("Centering...")

    # Read the image and width and height.
    width = file.shape[1]
    height = file.shape[0]

    # Calculate space that needs to be added horizontally and vertically.
    space_vertical = max(desired_size - height, 0)
    space_horizontal = max(desired_size - width, 0)

    # Split either side (to center image).
    border_top = math.floor(space_vertical / 2)
    border_bottom = math.ceil(space_vertical / 2)
    border_left = math.floor(space_horizontal / 2)
    border_right = math.ceil(space_horizontal / 2)

    # Add the border.
    img_centered = cv2.copyMakeBorder(src=file,
        top=border_top, bottom=border_bottom, left=border_left, right=border_right,
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0, 0]
    )

    return img_centered


def remove_background(file):
    """
    Remove the background from the image using the "rembg" library.
    Also saves a temporary mask.
    https://github.com/danielgatis/rembg
    """

    if not quiet:
        print("Removing background...")

    # Convert cv2 to PIL image
    img = cv2.cvtColor(file, cv2.COLOR_BGRA2RGBA)
    input = Image.fromarray(img)

    # Call the remove method of "rembg" twice (one mask, one not.)
    output = remove(input, only_mask=False)
    output_mask = remove(input, only_mask=True)

    # Error calculation
    total_filled = 0.0
    # Loop through every pixel.
    for x in range(output_mask.width):
        for y in range(output_mask.height):
            this_pixel = output_mask.getpixel((x, y))
            # Get total value of pixels.
            total_filled += this_pixel  # All greyscale

    # Calculate average value
    percent_visible = (total_filled / (output.width * output.height) / 100)
    if not quiet:
        print("Percent visible: " + str(percent_visible))

    # Error check
    if percent_visible <= background_removal_error_threshold_percent:
        warning("Total visible is below allowed threshold! (" + str(percent_visible) + ")")

    # Convert back to cv2
    output = np.asarray(output)
    output = cv2.cvtColor(output, cv2.COLOR_RGBA2BGRA)
    output_mask = np.asarray(output_mask)
    output_mask = cv2.cvtColor(output_mask, cv2.COLOR_RGBA2BGRA)
    return (output, output_mask)


def calc_bounding_rect(file, mask):
    """
    Calculates the bounding rectangle of the piece and returns it in the form of an array.
    https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
    """

    if not quiet:
        print("Calculating bounding rect...")

    # Load as greyscale only, use mask image as it is greyscale.
    img = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)

    # Code from https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
    ret, thresh = cv2.threshold(img, bounding_clamp_lower_value, bounding_clamp_upper_value, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    # Open mask to PIL for bound calculation
    mask_colour = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGBA)
    mask_image = Image.fromarray(mask_colour)
    error_area = (mask_image.width * mask_image.height) * bounding_factor_error

    # Find most significant contour that meets criteria
    area = 0
    index = 0
    x, y, w, h = [0, 0, 0, 0]
    while (area <= error_area) and not (index >= len(contours)):
        cnt = contours[index]
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        index += 1

    # Error check
    # Check if the area of the rectangle is an appropriate size relative to the size of the image.
    if area <= error_area:
        warning("Bounding rect size is too small! (" + str(w * h) + " vs " + str(
            (mask_image.width * mask_image.height) * bounding_factor_error) + ")")

    # Return the rectangle as an array.
    return [x, y, w, h]


def warning(msg):
    """
    Output a warning message regardless of quiet value.
    """

    # Print warning message
    print("Warning: " + str(msg))


def fast_denoise():
    """
    Use a fast denoising algorithm provided by OpenCV.
    """

    if not quiet:
        print("Denoising...")

    # Load image.
    img = cv2.imread(temp_path, cv2.IMREAD_UNCHANGED)

    # Perform fast denoising...
    # https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html
    output = cv2.fastNlMeansDenoisingColored(
        src=img
    )

    # Output image.
    cv2.imwrite(temp_path, output)


if __name__ == "__main__":
    parse_args(sys.argv[1:])
    main()