import cv2
import numpy as np

# Create a Macbeth ColorChecker (6x4 color palette)
def create_macbeth_colorchecker():
    colorchecker = np.zeros((400, 600, 3), dtype=np.uint8)
    colors = [
        [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67],
        [133, 128, 177], [103, 189, 170], [214, 126, 44], [80, 91, 166],
        [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
        [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31],
        [187, 86, 149], [8, 133, 161], [243, 243, 242], [200, 200, 200],
        [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52]
    ]
    k = 0
    for i in range(4):
        for j in range(6):
            colorchecker[i*100:(i+1)*100, j*100:(j+1)*100] = colors[k]
            k += 1
    return colorchecker

# Add noise to a specific channel
def add_noise_to_channel(image, channel):
    noisy_image = image.copy()
    noise = np.random.normal(0, 20, noisy_image[:,:,channel].shape).astype(np.uint8)  # Reduced noise
    noisy_image[:,:,channel] = cv2.add(noisy_image[:,:,channel], noise)
    return noisy_image

# Add text with a dark edge
def put_text_with_edge(image, text, position, font_scale=1.2, thickness=3, color=(255, 255, 255), edge_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, font_scale, edge_color, thickness+2, cv2.LINE_AA)
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

# Video creation
def create_noisy_video():
    colorchecker = create_macbeth_colorchecker()
    height, width, _ = colorchecker.shape
    output_size = (width*2, height*2)  # For the final split-screen section

    # Video writer
    fps = 30
    out = cv2.VideoWriter('noisy_channels.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Maximize the display window
    cv2.namedWindow('ColorChecker Noise Simulation', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('ColorChecker Noise Simulation', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Show original image for 15 seconds
    for _ in range(fps * 10):
        cv2.imshow('ColorChecker Noise Simulation', colorchecker)
        out.write(colorchecker)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Noise on R channel + label
    noisy_R = add_noise_to_channel(colorchecker, 2)
    for _ in range(fps * 10):
        frame = noisy_R.copy()
        put_text_with_edge(frame, "Noise added to the Red channel", (50, 50), font_scale=1.0)
        cv2.imshow('ColorChecker Noise Simulation', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Noise on G channel + label
    noisy_G = add_noise_to_channel(colorchecker, 1)
    for _ in range(fps * 10):
        frame = noisy_G.copy()
        put_text_with_edge(frame, "Noise added to the Green channel", (50, 50), font_scale=1.0)
        cv2.imshow('ColorChecker Noise Simulation', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Noise on B channel + label
    noisy_B = add_noise_to_channel(colorchecker, 0)
    for _ in range(fps * 10):
        frame = noisy_B.copy()
        put_text_with_edge(frame, "Noise added to the Blue channel", (50, 50), font_scale=1.0)
        cv2.imshow('ColorChecker Noise Simulation', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Show all 4 images (split screen) for 20 seconds
    for _ in range(fps * 10):
        frame = np.zeros((height*2, width*2, 3), dtype=np.uint8)
        frame[:height, :width] = colorchecker      # Original top-left
        frame[:height, width:] = noisy_R          # R-noisy top-right
        frame[height:, :width] = noisy_G          # G-noisy bottom-left
        frame[height:, width:] = noisy_B          # B-noisy bottom-right
        cv2.imshow('ColorChecker Noise Simulation', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Show question with 4 images for 20 seconds
    for _ in range(fps * 10):
        frame = np.zeros((height*2, width*2, 3), dtype=np.uint8)
        frame[:height, :width] = colorchecker
        frame[:height, width:] = noisy_R
        frame[height:, :width] = noisy_G
        frame[height:, width:] = noisy_B
        put_text_with_edge(frame, "Which image has the least noise?", (50, height*2 // 2), font_scale=1.9)
        cv2.imshow('ColorChecker Noise Simulation', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()

create_noisy_video()
