import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from numpy import ones, vstack
from pandas import DataFrame
from numpy.linalg import lstsq
from PIL import Image, ImageDraw, ImageFont
from itertools import combinations
import random
import math
import time
import scipy.stats as st
import cv2
import warnings

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Defining global variables
PWB_D_true = []
PWB_D_pixel = []
Rectangle_list = []
wsk_points_data = []
brg_rec_data = []
wsk_num_data = []
num_simulations = int()
max_wsk_one_trial = int()
DF1 = DataFrame
DF2 = DataFrame
DF3 = DataFrame


def main_MCS(length_PWB, width_PWB, TWNum, SimNum, length_mu, length_sigma, thickness_mu, thickness_sigma, filename):
    start = time.time()

    # Function to process circuit board image
    def contours_detecting(image_path, PWB_W_True, PWB_L_True):
        # Reading image
        font_ct = cv2.FONT_HERSHEY_COMPLEX
        img2 = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Reading same image in another variable and converting to gray scale.
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img2 is not None:
            print('Original image size (width, height, color): ' + str(img2.shape))
        else:
            print('img2 Is None')

        # Resizing the ratio between L and W in pixels to the same as the ratio in true dimension
        if img.shape[1] >= img.shape[0]:
            # The length in pixels will keep unchanged while only resize the width pixels
            w_scale = ((img.shape[1] / img.shape[0]) / (PWB_length_true / PWB_width_true))
            img = cv2.resize(img, None, fx=1, fy=w_scale)
            img2 = cv2.resize(img2, None, fx=1, fy=w_scale)

        else:
            # The width in pixels will keep unchanged while only resize the length pixels
            l_scale = ((img.shape[0] / img.shape[1]) / (PWB_width_true / PWB_length_true))
            img = cv2.resize(img, None, fx=l_scale, fy=1)
            img2 = cv2.resize(img2, None, fx=l_scale, fy=1)

        # Converting image to a binary image (black and white only image).
        _, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

        # Detecting contours in image.
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)

        if img2 is not None:
            print('Resized image size (width, height, color): ' + str(img2.shape))
        else:
            print('img2 Is None')

        # Drawing a plotting scale
        d = img2.shape

        # Defining circuit board width and length in pixels
        PWB_W_pixel = d[0]
        PWB_L_pixel = d[1]
        # Calculating the ratio between microns and pixels
        ratio_w = round(PWB_W_True * 1000 / PWB_W_pixel, 3)
        ratio_l = round(PWB_L_True * 1000 / PWB_L_pixel, 3)
        # Averaging these two ratio to calculate 1 pixel = ? of microns
        ratio_avg = round((ratio_w + ratio_l) / 2, 0)  # needs to be integer

        ratio_subsection = [0.5625 * 2 ** i for i in range(10)]
        scale_subsection = [25 * 2 ** i for i in range(10)]
        color = (0, 128, 255)
        for z in range(10):
            if ((z == 0) and (ratio_avg < ratio_subsection[z])) \
                    or (ratio_avg >= ratio_subsection[z - 1]) and (ratio_avg < ratio_subsection[z]):
                # Length of the plotting scale 50 microns = ?# of pixel
                l_scale = round(scale_subsection[z] / ratio_avg, 0)
                l_start = (int(PWB_L_pixel - l_scale - l_scale), int(PWB_W_pixel - l_scale))
                l_end = (int(PWB_L_pixel - l_scale), int(PWB_W_pixel - l_scale))

                l_start_seg = (int(PWB_L_pixel - l_scale - l_scale), int(PWB_W_pixel - 1.1 * l_scale))
                l_end_seg = (int(PWB_L_pixel - l_scale), int(PWB_W_pixel - 1.1 * l_scale))

                thickness = math.ceil(9 * l_scale / 200)
                cv2.line(img2, l_start, l_end, color, thickness)
                cv2.line(img2, l_start, l_start_seg, color, thickness)
                cv2.line(img2, l_end, l_end_seg, color, thickness)

                text_x = int(round(l_start[0] - 0.4 * l_scale, 0))
                text_y = int(round(int(PWB_W_pixel - 1.6 * l_scale)))
                cv2.putText(img2, '{} microns'.format(scale_subsection[z]), (text_x, text_y),
                            font_ct, 3 * l_scale / 400, color)

                count_rec = 0
                # Going through every contour found in the image.
                for cnt in contours:
                    if (len(cnt) != 4) or (cnt[0][0][0] != cnt[1][0][0]):
                        count_rec += 1
                        x, y, w, h = cv2.boundingRect(cnt)
                        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
                        # draws boundary of contours.
                        cv2.drawContours(img2, [approx], 0, (0, 0, 255), 2)
                        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        cv2.putText(img2, str(count_rec), (x, y), font_ct, l_scale/200, (0, 255, 0))
                        temp_index = contours.index(cnt)
                        contours[temp_index] = np.array([[[x, y]], [[x, y + h]], [[x + w, y + h]], [[x + w, y]]])

                    else:
                        count_rec += 1
                        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
                        # draws boundary of contours.
                        cv2.drawContours(img2, [approx], 0, (0, 0, 255), 2)
                        # Used to flat the array containing the co-ordinates of the vertices.
                        n = approx.ravel()
                        num = 0
                        for _ in n:
                            if num % 2 == 0:
                                x = n[num]
                                y = n[num + 1]

                                if num == 0:
                                    # text on topmost co-ordinate.
                                    cv2.putText(img2, str(count_rec), (x, y),
                                                font_ct, l_scale/200, (0, 255, 0))
                            num += 1

        cv2.imwrite('Circuit Board Layout with Conductor Numbers.png', img2)
        return contours

    # Function to randomly generate whisker on circuit board
    def whisker_generator():
        # Defining whisker length and thickness parameter (unit: micron)
        # mu_length, sigma_length = 5.0093, 1.1519
        # mu_thickness, sigma_thickness = 1.1685, 0.6728
        mu_length, sigma_length = length_mu, length_sigma
        mu_thickness, sigma_thickness = thickness_mu, thickness_sigma

        # Randomly generating whisker start points, angle
        x_start = random.randint(1, PWB_length_pixel - 1)
        y_start = random.randint(1, PWB_width_pixel - 1)
        angle = random.uniform(0, 2 * math.pi)

        # Calculating the ratio between microns and pixels
        ratio_w = round(PWB_width_true * 1000 / PWB_width_pixel, 3)
        ratio_l = round(PWB_length_true * 1000 / PWB_length_pixel, 3)
        # Averaging these two ratio to calculate 1 pixel = ? microns
        ratio_avg = round((ratio_w + ratio_l) / 2, 0)  # needs to be integer

        # Generating whisker length based on lognormal distribution
        wsk_length = np.random.lognormal(mu_length, sigma_length) / ratio_avg
        wsk_thickness = np.random.lognormal(mu_thickness, sigma_thickness) / ratio_avg
        x_end = 0
        y_end = 0

        # Calculating the position of end point
        if angle < math.pi / 2:
            x_end = int(round(x_start + wsk_length * math.sin(angle)))
            y_end = int(round(y_start + wsk_length * math.cos(angle)))
        elif angle < math.pi:
            x_end = int(round(x_start + wsk_length * math.sin(math.pi - angle)))
            y_end = int(round(y_start - wsk_length * math.cos(math.pi - angle)))
        elif angle < math.pi * 3 / 2:
            x_end = int(round(x_start - wsk_length * math.sin(angle - math.pi)))
            y_end = int(round(y_start - wsk_length * math.cos(angle - math.pi)))
        elif angle < math.pi * 2:
            x_end = int(round(x_start - wsk_length * math.sin(2 * math.pi - angle)))
            y_end = int(round(y_start + wsk_length * math.cos(2 * math.pi - angle)))

        # The whisker passes through the left boundary of the PWB
        if x_end <= 0:
            y_end = int(round(y_start + (y_end - y_start) * (0 - x_start) / (x_end - x_start)))
            x_end = 0
        # The whisker passes through the bottom boundary of the PWB
        if y_end <= 0:
            x_end = int(round(x_start + (x_end - y_end) * (0 - y_start) / (y_end - y_start)))
            y_end = 0
        # The whisker passes through the right boundary of the PWB
        if x_end >= PWB_length_pixel:
            y_end = int(round(y_start + (y_end - y_start) * (PWB_length_pixel - x_start) / (x_end - x_start)))
            x_end = PWB_length_pixel
        # The whisker passes through the top boundary of the PWB
        if y_end >= PWB_width_pixel:
            x_end = int(round(x_start + (x_end - y_end) * (PWB_width_pixel - y_start) / (y_end - y_start)))
            y_end = PWB_width_pixel

        return x_start, y_start, x_end, y_end, wsk_length, wsk_thickness, ratio_avg

    # Function to compute region code for a point(x, y)
    def computeCode(x, y, x_min, y_min, x_max, y_max):
        code = INSIDE
        if x < x_min:  # to the left of rectangle
            code |= LEFT
        elif x > x_max:  # to the right of rectangle
            code |= RIGHT
        if y < y_min:  # below the rectangle
            code |= BOTTOM
        elif y > y_max:  # above the rectangle
            code |= TOP

        return code

    # Cohen-Sutherland algorithm
    def cohenSutherland(x1, y1, x2, y2, m, c, x_min, y_min, x_max, y_max, bridged_rec, rec_num):
        # Compute region codes for P1, P2
        code1 = computeCode(x1, y1, x_min, y_min, x_max, y_max)
        code2 = computeCode(x2, y2, x_min, y_min, x_max, y_max)

        # If both endpoints lie within rectangle
        if code1 == 0 and code2 == 0:
            rectangle_status = 1  # both endpoints are in region

        # If both endpoints are outside rectangle
        elif (code1 & code2) != 0:
            rectangle_status = 0  # outside region

        else:
            if ((((m * x_min) + c) <= y_max) & (((m * x_min) + c) >= y_min)) \
                    or ((((m * x_max) + c) <= y_max) & (((m * x_max) + c) >= y_min)) \
                    or (((y_min - c) / m >= x_min) & ((y_min - c) / m <= x_max)) \
                    or (((y_max - c) / m >= x_min) & ((y_max - c) / m <= x_max)):
                rectangle_status = 0.5
                bridged_rec.append(rec_num)
            else:
                rectangle_status = 0

        return rectangle_status, bridged_rec

    # Function to determine if a whisker is bridging
    def check_Bridged():
        # Generating a whisker from P1 = (x_s, y_s) to P2 = (x_e, y_e)
        x_s, y_s, x_e, y_e, wsk_length, wsk_thickness, ratio_avg = whisker_generator()
        rec_status = []
        # Creating a list to contain the rectangles which are bridged
        bridged_rectangle = []

        # Obtaining the equation of the line
        endpoints = [(x_s, y_s), (x_e, y_e)]
        x_coords, y_coords = zip(*endpoints)
        A = vstack([x_coords, ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords, rcond=None)[0]

        for rectangle in Rectangle_list:
            # Defining x_min, y_min, x_max, y_max for rectangle
            x_min = rectangle[0]
            y_min = rectangle[1]
            x_max = rectangle[2]
            y_max = rectangle[3]
            rectangle_num = Rectangle_list.index(rectangle) + 1

            rec_st, bridged_rectangle = cohenSutherland(x_s, y_s, x_e, y_e, m, c, x_min, y_min, x_max, y_max,
                                                        bridged_rectangle, rectangle_num)
            rec_status.append(rec_st)

        if rec_status.count(0.5) >= 2:
            bridged = True
        else:
            bridged = False

        return bridged, x_s, y_s, x_e, y_e, rec_status, wsk_length, wsk_thickness, bridged_rectangle, ratio_avg

    # Function to calculate resistance of whisker
    def calculate_resistance(wsk_length, wsk_thickness, ratio_avg):
        resistivity_tin_micron = 0.11  # 1.1e-7 Ohm meter = 0.11 Ohm micron
        resistivity_tin_pixel = resistivity_tin_micron / ratio_avg  # 0.11/ratio_avg Ohm pixel
        A = math.pi * wsk_thickness ** 2 / 4
        resistance = resistivity_tin_pixel * wsk_length / A

        return resistance

    # Function to store the conductor pairs data as Dataframe
    def conductor_pair_DF():
        # Creating dataframe to store conductor pairs data
        run_list = []
        pair_list = []
        wsk_n_list = []
        for i in range(num_simulations):
            brg_pairs = brg_rec_data[i]
            wsk_n = wsk_num_data[i]
            for index, pairs in enumerate(brg_pairs):
                if len(pairs) == 2:
                    pair_list.append(pairs)
                    wsk_n_list.append(wsk_n[index])
                    run_list.append(i + 1)
                else:
                    list_combination = list(combinations(pairs, 2))
                    for t in list_combination:
                        pair_list.append(list(t))
                        wsk_n_list.append(wsk_n[index])
                        run_list.append(i + 1)

        # Sheet 1 -------------------------------------------
        data1 = {'Run Num.': run_list,
                 'Whisker Num.': wsk_n_list,
                 '1st Conductor': [i[0] for i in pair_list],
                 '2nd Conductor': [i[1] for i in pair_list],
                 'Bridging Pairs': pair_list}
        df1 = DataFrame(data1)
        df1['Bridging Pairs'] = df1['Bridging Pairs'].agg(lambda x: ', '.join(map(str, x)))

        # Sheet 2 -------------------------------------------
        data2 = {'Bridging Pairs': pair_list}
        df2 = DataFrame(data2)
        df2['Bridging Pairs'] = df2['Bridging Pairs'].agg(lambda x: ', '.join(map(str, x)))
        df2 = df2.groupby(['Bridging Pairs'])['Bridging Pairs'].count()
        df2 = df2.sort_values(0, ascending=False)
        df2 = df2.reset_index(name='Count')
        # Calculating frequency
        df2['Frequency (%)'] = [round(x / df2['Count'].sum(), 5) * 100 for x in df2['Count'].to_list()]

        # Sheet 3 -------------------------------------------
        s3_1 = df1['1st Conductor']
        s3_2 = df1['2nd Conductor']
        s3_3 = pd.concat([s3_1, s3_2], axis=0, ignore_index=True)
        data3 = {'Bridging Conductor Num.': s3_3}
        df3 = DataFrame(data3)
        df3 = df3.groupby(['Bridging Conductor Num.'])['Bridging Conductor Num.'].count()
        df3 = df3.sort_values(0, ascending=False)
        df3 = df3.reset_index(name='Count')
        # Calculating frequency
        df3['Frequency (%)'] = [round(x / df3['Count'].sum(), 5) * 100 for x in df3['Count'].to_list()]

        return df1, df2, df3

    # =============================Circuit Board Processing==========================================
    # True dimensions of circuit board
    PWB_width_true = width_PWB  # mm
    PWB_length_true = length_PWB  # mm
    global PWB_D_true
    PWB_D_true = [PWB_length_true, PWB_width_true]

    # Processing the image and getting coordinate
    ct = contours_detecting(filename, PWB_width_true, PWB_length_true)

    # Creating a list to include all rectangle
    global Rectangle_list
    for a in range(len(ct)):
        rec_x_min = ct[a][0][0][0]
        rec_y_min = ct[a][0][0][1]
        rec_x_max = ct[a][2][0][0]
        rec_y_max = ct[a][2][0][1]
        Rectangle_list.append([rec_x_min, rec_y_min, rec_x_max, rec_y_max])

    # Read circuit board
    image = cv2.imread(filename)

    # Resizing the ratio between L and W in pixels to the same as the ratio in true dimension
    # To do so, the longer side in pixels will keep unchanged while only the shorter side in pixels will be changed
    # image.shape[1] is the length of PWB in pixels, image.shape[0] is the width (height) of PWB in pixels
    if image.shape[1] > image.shape[0]:
        # The length in pixels stays unchanged while the width pixels changes
        width_scale = ((image.shape[1] / image.shape[0]) / (PWB_length_true / PWB_width_true))
        image = cv2.resize(image, None, fx=1, fy=width_scale)

    else:
        # The width in pixels stays unchanged while the length pixels changes
        length_scale = ((image.shape[0] / image.shape[1]) / (PWB_width_true / PWB_length_true))
        image = cv2.resize(image, None, fx=length_scale, fy=1)

    dimensions = image.shape

    # Defining circuit board width and length in pixels
    PWB_width_pixel = dimensions[0]
    PWB_length_pixel = dimensions[1]
    global PWB_D_pixel
    PWB_D_pixel = [PWB_length_pixel, PWB_width_pixel]

    # Defining region codes
    INSIDE = 0  # 0000
    LEFT = 1  # 0001
    RIGHT = 2  # 0010
    BOTTOM = 4  # 0100
    TOP = 8  # 1000

    # ============================= Monte Carlo Simulation ==================================
    # Input
    global num_simulations
    num_simulations = SimNum
    global max_wsk_one_trial
    max_wsk_one_trial = TWNum
    trial = 0
    trial_list = []

    # Tracking
    probability_each_trial = []
    probability_cumulative = []

    # Create data lists for gui
    treeview_data = []

    # =========== Visualizing the change in P ====================
    # Counter variables for the number of throws and value of the x-axis to be plotted on img3. Note, the first
    # This lists will contain all the points plotted on the graph "img3"
    points = []

    # This is a counter variable that will be updated each time an element is popped from the "points" list.
    # This will allow us to shift the graph to the right.
    N_popped = 0

    # plotted value will be at 50 throws.
    throws = 0
    throws_x = 50
    total_num_bridged = 0

    # define variables
    y_value = 0
    P_est = 0
    P_est_y = 0

    # This maps the values between 0 and 0.050 (used for y-axis in img3) to pixel values between 0 and 600.
    # Because pixel 0,0 is the top left corner, we have to reverse the mapping so the value 0.05 is on the
    # top and 0 is on the bottom of the y-axis.
    mappedList = np.around(np.linspace(0.050, 0, 600, dtype=float), decimals=8)

    # We set the font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Create tick marks/labels for the y-axis graph in img3.
    ticks = [i / 1000 for i in range(0, 52, 5)]
    labels = ['{}'.format(i / 1000) for i in range(0, 52, 5)]

    print("====================== Bridged Whisker Points ======================")
    # While loop to run for the number of simulations desired
    while True:
        trial += 1
        probability = [0]
        num_whiskers = [1]
        num_bridged = 0
        whisker_length = []
        whisker_thickness = []
        whisker_resistance = []
        whisker_points_oneTrial = []
        total_brg_rec_oneTrial = []
        brg_rec_oneTrial = []
        wsk_num_oneTrial = []

        print("======================For Run #" + str(trial) + "============================")
        # Run until max number of whiskers
        while num_whiskers[-1] <= max_wsk_one_trial:
            # Increase the number of throws by 1
            throws += 1

            # Generate a whisker and check if it is bridged
            Bridged, x_st, y_st, x_ed, y_ed, wsk_s, wsk_l, wsk_t, brg_rec, ratio_mean = check_Bridged()
            num_whiskers.append(num_whiskers[-1] + 1)
            whisker_length.append(wsk_l)
            whisker_thickness.append(wsk_t)
            whisker_resistance.append(calculate_resistance(wsk_l, wsk_t, ratio_mean))
            # List for rectangles that are bridged; only needs those include two or more conductors in one list
            total_brg_rec_oneTrial.append(brg_rec)
            wsk_num_oneTrial = [num + 1 for num in range(len(total_brg_rec_oneTrial)) if
                                len(total_brg_rec_oneTrial[num]) >= 2]
            brg_rec_oneTrial = [total_brg_rec_oneTrial[i] for i in range(len(total_brg_rec_oneTrial)) if
                                len(total_brg_rec_oneTrial[i]) >= 2]

            # Test if the whisker is bridging
            if Bridged:
                print("Start point: ({}, {})    |   End point: ({}, {})".format(x_st, y_st, x_ed, y_ed))
                # Drawing the bridging whiskers
                # draw.line((x_st, y_st, x_ed, y_ed), fill=(255, 0, 0), width=5)
                image = cv2.line(image, (x_st, y_st), (x_ed, y_ed), (0, 0, 255), 5)
                whisker_points_oneTrial.append([x_st, y_st, x_ed, y_ed, 'red'])
                num_bridged += 1
                probability.append(num_bridged / max_wsk_one_trial)

                # Update the estimate for the value of P after each throw
                total_num_bridged += 1
                P_est = total_num_bridged / throws
                P_est_y = round(P_est, 6)

                if P_est_y <= 0.05:
                    y_value = np.where(mappedList >= P_est_y)[0][-1]
                elif P_est_y > 0.05:
                    y_value = int(600)

            else:
                # Drawing the non-bridging whiskers
                # draw.line((x_st, y_st, x_ed, y_ed), fill=(0, 255, 0), width=5)
                image = cv2.line(image, (x_st, y_st), (x_ed, y_ed), (0, 255, 0), 5)
                whisker_points_oneTrial.append([x_st, y_st, x_ed, y_ed, 'green'])
                probability.append(num_bridged / max_wsk_one_trial)

            # Create/update img3 every 50 throws.
            if throws % 50 == 0:
                throws_x += 1
                points.append((throws_x, y_value))
                img3 = np.zeros((630, 630, 3), np.uint8)

                # Create an x- and y-axes
                cv2.line(img3, (50, 600), (630, 600), (255, 255, 255), 2)
                cv2.line(img3, (50, 0), (50, 600), (255, 255, 255), 2)

                # Draw tick marks
                for tick, label in zip(ticks, labels):
                    y_tick = np.where(mappedList >= tick)[0][-1]
                    cv2.line(img3, (47, y_tick), (53, y_tick), (255, 255, 255), 1)
                    cv2.putText(img3, label, (0, y_tick + 3), font, 0.5, (255, 255, 255), 1)

                # if there are more than 600 points on the graph, i.e., the length of "points" is 600,
                # then remove the first element in the list and increase the "N_popped" counter by 1.
                # The "N_popped" counter will allow us to shift the graph to the right so the plotted points
                # do not go off the window/screen.
                if throws_x > 600:
                    points.pop(0)
                    N_popped += 1

                # Update img3 by putting the respective texts and plotting a circle based on the
                # current estimate of P.
                for point in points:
                    # # draw a horizontal line at the value P_est
                    if len(np.where(mappedList >= P_est)[0]) != 0:
                        cv2.line(img3, (50, np.where(mappedList >= P_est)[0][-1]),
                                 (630, np.where(mappedList >= P_est)[0][-1]), (255, 255, 255), 1)
                    else:
                        cv2.line(img3, (50, np.where(mappedList < P_est)[0][-1]),
                                 (630, np.where(mappedList < P_est)[0][-1]), (255, 255, 255), 1)
                    cv2.rectangle(img3, (int(630 * 0.75), 0), (int(630), 60), (0, 0, 0), -1)    # (B,G,R)3
                    cv2.rectangle(img3, (0, 0), (100, 40), (0, 0, 0), -1)
                    cv2.putText(img3, "# of whiskers = " + str(throws), (420, 620), font, 0.5, (0, 255, 0), 2)
                    cv2.putText(img3, "Press 'Q' to quit", (480, 20), font, 0.5, (0, 255, 0), 2)
                    cv2.putText(img3, 'Est P: ' + str(P_est_y), (10, 20), font, 0.6, (0, 255, 0), 2)
                    cv2.circle(img3, (point[0] - N_popped, point[1]), 1, (0, 255, 0), -1)
                    cv2.imshow('Graph', img3)

        if num_bridged == 0:
            print('     No Bridging Whisker in this run\n')
            print("Number of Detached Whiskers (N): " + str(max_wsk_one_trial))
            print("Number of Bridging Whiskers (n): " + str(num_bridged))
            print("Probability of Bridging (n/N): " + str(probability[-1]) + '\n')
        else:
            print("Number of Detached Whiskers (N): " + str(max_wsk_one_trial))
            print("Number of Bridging Whiskers (n): " + str(num_bridged))
            print("Probability of Bridging (n/N): " + str(probability[-1]) + '\n')

        # Store tracking variables and add line to figure
        probability_each_trial.append(num_bridged / max_wsk_one_trial)
        probability_cumulative.append(sum(probability_each_trial) / len(probability_each_trial))
        trial_list.append(trial)

        n_bridgedPairs_oneTrial = 0
        for i in brg_rec_oneTrial:
            if len(i) == 2:
                n_bridgedPairs_oneTrial += 1
            else:
                n_bridgedPairs_oneTrial += len(list(combinations(i, 2)))

        # Bridging whisker No. for gui
        global wsk_num_data
        wsk_num_data.append(wsk_num_oneTrial)

        # Treeview data for gui
        treeview_data.append([trial, num_bridged, max_wsk_one_trial,
                              round(num_bridged / max_wsk_one_trial, 5), n_bridgedPairs_oneTrial])

        # Wsk points for gui
        global wsk_points_data
        wsk_points_data.append(whisker_points_oneTrial)

        # Bridged rectangles for gui
        global brg_rec_data
        brg_rec_data.append(brg_rec_oneTrial)

        # Creating Figures for Simulation
        f1 = plt.figure('Percentage of 1 or More Whiskers Bridging after {} simulations'.format(num_simulations))
        f1.set_figwidth(10)
        f1.set_figheight(5)
        plt.title("Percentage of 1 or More Whiskers Bridging in Each Run (%) "
                  "[" + str(num_simulations) + " simulations]")
        plt.xlabel("Run Number")
        plt.ylabel("Percentage of 1 or More Whiskers Bridging (%)")
        plt.xlim([0, num_simulations])
        plt.plot(trial_list, [p * 100 for p in probability_each_trial], '-ok',
                 color='k', markersize=1.5, linewidth=1)
        ax1 = plt.gca()
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        f2 = plt.figure('Cumulative Probability after {} simulations'.format(num_simulations))
        f2.set_figwidth(10)
        f2.set_figheight(5)
        plt.title("Cumulative Probability of Whiskers Bridging [" + str(num_simulations) + " simulations]")
        plt.xlabel("Run Number")
        plt.ylabel("Cumulative Probability of Whiskers Bridging")
        plt.xlim([0, num_simulations])
        plt.plot(trial_list, probability_cumulative, '-ok',
                 color="k", markersize=1.5, linewidth=1)
        ax2 = plt.gca()
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Continue updating/running the simulation until the letter "q" is pressed
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
        if trial == num_simulations:
            break
    print("====================================================================\n")

    # Averaging bridged probability
    overall_bridged_probability = round(sum(probability_each_trial) / len(probability_each_trial), 5)

    # Calculating the number of Trials with 1 or more bridging whiskers
    num_run_bridging = len([1 for x in probability_each_trial if x != 0])

    # Calculating 95% confidence interval
    CI = st.t.interval(0.95, df=len(probability_each_trial) - 1, loc=np.mean(probability_each_trial),
                       scale=st.sem(probability_each_trial))

    # Create conductor pair / conductor bridging frequency dataframe
    global DF1
    global DF2
    global DF3
    DF1, DF2, DF3 = conductor_pair_DF()

    # 1st Figure --------------------------------------------
    # Add a line to represent the average value of probability
    plt.figure('Percentage of 1 or More Whiskers Bridging after {} simulations'.format(num_simulations))
    plt.axhline(y=overall_bridged_probability * 100, color='orange', linestyle='dashed')
    # Save the figure
    plt.savefig('Percentage of 1 or More Whiskers Bridging after {} simulations.png'.format(num_simulations))

    # 2nd Figure --------------------------------------------
    plt.figure('Cumulative Probability after {} simulations'.format(num_simulations))
    plt.savefig('Cumulative Probability after {} simulations.png'.format(num_simulations))

    # 3rd Figure --------------------------------------------
    f3 = plt.figure(3)
    ax3 = plt.gca()
    f3.set_figwidth(80)
    f3.set_figheight(40)
    plt.title("Conductor Pairs Bridging Frequency [" + str(num_simulations) + " simulations]", fontsize=50)
    plt.xlabel("Bridging Conductor Pair Number", fontsize=50)
    plt.ylabel("Frequency of Conductor Pair Bridging (#)", fontsize=50)
    if len(DF2['Bridging Pairs']) != 0:
        DF2.plot.bar(x='Bridging Pairs', y='Count', rot=0, ax=ax3)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=30)
        f3.autofmt_xdate()
        plt.figure(3)
        plt.savefig('Conductor Pairs Bridging Frequency after {} simulations.png'.format(num_simulations))
    else:
        plt.text(0.5, 0.5, "No Bridging Whisker", size=100, rotation=0,
                 ha='center', va='center',
                 bbox=dict(boxstyle='round',
                           ec=(1., 0.5, 0.5),
                           fc=(1., 0.8, 0.8)))
        plt.figure(3)
        plt.savefig('Conductor Pairs Bridging Frequency after {} simulations.png'.format(num_simulations))

    # 4th Figure --------------------------------------------
    f4 = plt.figure(4)
    ax4 = plt.gca()
    f4.set_figwidth(80)
    f4.set_figheight(40)
    plt.title("Conductor Bridging Frequency [" + str(num_simulations) + " simulations]", fontsize=50)
    plt.xlabel("Bridging Conductor Number", fontsize=50)
    plt.ylabel("Frequency of Conductor Bridging (#)", fontsize=50)
    if len(DF3['Bridging Conductor Num.']) != 0:
        DF3.plot.bar(x='Bridging Conductor Num.', y='Count', rot=0, ax=ax4)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=30)
        f4.autofmt_xdate()
        # manager = plt.get_current_fig_manager()
        # manager.window.showMaximized()
        plt.figure(4)
        plt.savefig('Conductor Bridging Frequency after {} simulations.png'.format(num_simulations))
    else:
        plt.text(0.5, 0.5, "No Bridging Whisker", size=100, rotation=0,
                 ha='center', va='center',
                 bbox=dict(boxstyle='round',
                           ec=(1., 0.5, 0.5),
                           fc=(1., 0.8, 0.8)))
        plt.figure(4)
        plt.savefig('Conductor Bridging Frequency after {} simulations.png'.format(num_simulations))

    # Displaying the results in console
    print("Total Number of Detached Whiskers (N): " + str(trial * max_wsk_one_trial))
    print("Total Number of Bridging Whiskers (n): " + str(total_num_bridged))
    print("Probability of Bridging after " + str(trial) + " trials: " + str(overall_bridged_probability))
    print("95% Confidence Interval is " + str(CI))

    # Results data for gui
    results_data = [trial, num_run_bridging, trial * max_wsk_one_trial, total_num_bridged]

    # Calculating the time
    end = time.time()
    print('Time in seconds: ' + str(round(end) - round(start)) + 's')

    # Showing the plot after the simulations are finished
    # plt.show()

    # Closing all plots and windows
    plt.close('all')
    cv2.destroyAllWindows()

    return treeview_data, results_data


def draw_img(imageNum):
    # Drawing the PWB
    im = Image.new('RGB', (PWB_D_pixel[0], PWB_D_pixel[1]), (0, 0, 0))
    draw = ImageDraw.Draw(im)

    # Defining circuit board width and length in pixels
    PWB_L_pixel = PWB_D_pixel[0]
    PWB_W_pixel = PWB_D_pixel[1]
    # Defining circuit board true width and length in mm
    PWB_L_True = PWB_D_true[0]
    PWB_W_True = PWB_D_true[1]
    # Calculating the ratio between microns and pixels
    ratio_w = round(PWB_W_True * 1000 / PWB_W_pixel, 3)
    ratio_l = round(PWB_L_True * 1000 / PWB_L_pixel, 3)
    # Averaging these two ratio to calculate 1 pixel = ? microns
    ratio_avg = round((ratio_w + ratio_l) / 2, 0)  # needs to be integer

    # Generating lists for the thickness of lines and the value of scale plotting to be drawn on the PWB
    ratio_subsection = [0.5625 * 2 ** i for i in range(10)]
    scale_subsection = [25 * 2 ** i for i in range(10)]
    color = (255, 128, 0)
    for z in range(10):
        if ((z == 0) and (ratio_avg < ratio_subsection[z])) or \
                ((ratio_avg >= ratio_subsection[z - 1]) and (ratio_avg < ratio_subsection[z])):
            # Length of the plotting scale 25*2^z ([25, 50, 100, 200, ...]) microns = ?# of pixel
            l_scale = round(scale_subsection[z] / ratio_avg, 0)

            font_s = ImageFont.truetype(font="arial.ttf", size=math.ceil(3*l_scale/25))
            font_m = ImageFont.truetype(font="arial.ttf", size=math.ceil(9*l_scale/25))
            # Drawing the conductors on circuit board
            for i in Rectangle_list:
                draw.rectangle((i[0], i[1], i[2], i[3]), fill=(255, 255, 255), outline=(255, 255, 255))
                draw.text((round((i[0] + i[2]) / 2, 0), round((i[1] + i[3]) / 2, 0)),
                          "{}".format(Rectangle_list.index(i) + 1),
                          (255, 128, 0), font=font_s)
            # Drawing the whiskers on circuit board
            for j in wsk_points_data[int(imageNum - 1)]:
                if j[4] == 'red':
                    draw.line((j[0], j[1], j[2], j[3]), fill=(255, 0, 0), width=math.ceil(3 * l_scale / 40))
                elif j[4] == 'green':
                    draw.line((j[0], j[1], j[2], j[3]), fill=(0, 255, 0), width=math.ceil(3 * l_scale / 40))

            l_start = (int(PWB_L_pixel - l_scale - l_scale), int(PWB_W_pixel - l_scale))
            l_end = (int(PWB_L_pixel - l_scale), int(PWB_W_pixel - l_scale))
            l_start_seg = (int(PWB_L_pixel - l_scale - l_scale), int(PWB_W_pixel - 1.1 * l_scale))
            l_end_seg = (int(PWB_L_pixel - l_scale), int(PWB_W_pixel - 1.1 * l_scale))

            # Drawing a plotting scale
            draw.line((l_start[0], l_start[1], l_end[0], l_end[1]), fill=color, width=math.ceil(3 * l_scale / 40))
            draw.line((l_start[0], l_start[1], l_start_seg[0], l_start_seg[1]), fill=color,
                      width=math.ceil(3 * l_scale / 40))
            draw.line((l_end[0], l_end[1], l_end_seg[0], l_end_seg[1]), fill=color, width=math.ceil(3 * l_scale / 40))

            # Drawing the text label
            text_x = int(round(l_start[0] - 0.4 * l_scale, 0))
            text_y = int(round(int(PWB_W_pixel - 1.6 * l_scale)))
            draw.text((text_x, text_y), '{} microns'.format(scale_subsection[z]), color, font=font_m)

    im.show()


def export_Pairs_Data():
    # Save the dataframe to excel------------------------
    with pd.ExcelWriter("Output after {} runs ({} whiskers in each run).xlsx".format(
            num_simulations, max_wsk_one_trial)) as writer:
        DF1.style.set_properties(**{'text-align': 'center'}).to_excel(writer, sheet_name='Bridge Report', index=False)
        DF2.style.set_properties(**{'text-align': 'center'}).to_excel(writer, sheet_name='Bridging Pairs Frequency',
                                                                      index=False)
        DF3.style.set_properties(**{'text-align': 'center'}).to_excel(writer,
                                                                      sheet_name='Bridging Conductors Frequency',
                                                                      index=False)
        No_Bridging_Whisker = False
        if DF1.empty:
            No_Bridging_Whisker = True
        else:
            # Auto-adjust columns' width
            for column in DF1:
                column_width = max(DF1[column].astype(str).map(len).max(), len(column))
                col_idx = DF1.columns.get_loc(column)
                writer.sheets['Bridge Report'].set_column(col_idx, col_idx, column_width)

            for column in DF2:
                column_width = max(DF2[column].astype(str).map(len).max(), len(column))
                col_idx = DF2.columns.get_loc(column)
                writer.sheets['Bridging Pairs Frequency'].set_column(col_idx, col_idx, column_width)

            for column in DF3:
                column_width = max(DF3[column].astype(str).map(len).max(), len(column))
                col_idx = DF3.columns.get_loc(column)
                writer.sheets['Bridging Conductors Frequency'].set_column(col_idx, col_idx, column_width)

            writer.sheets['Bridge Report'].insert_image('G2', 'Circuit Board Layout with conductor numbers.png')
            writer.sheets['Bridging Pairs Frequency'].insert_image(
                'E2', 'Conductor Pairs Bridging Frequency after {} simulations.png'.format(num_simulations))
            writer.sheets['Bridging Conductors Frequency'].insert_image(
                'E2', 'Conductor Bridging Frequency after {} simulations.png'.format(num_simulations))

    writer.save()

    return No_Bridging_Whisker
