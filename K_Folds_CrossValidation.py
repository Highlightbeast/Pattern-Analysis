import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from fore_back_estimation import Density

sample_num = 500000
n_estimators = 30
max_depth = 30
img = misc.face(gray=True)
foreground = Density(img, sample_num)
img_height = foreground.img_shape[0]
img_width = foreground.img_shape[1]
pixel_num = img_width * img_height

# get sample position(x, y) of foreground
fore_idx = foreground.get_idx()
fore_coord = foreground.get_position(fore_idx)
fore_label = np.ones(sample_num).reshape(-1, 1)

# generate a background density, the samples that are used in background estimation
# should be at least the same as in foreground
back_idx = np.random.randint(pixel_num, size=sample_num)
back_coord = foreground.get_position(back_idx)
back_label = np.zeros(sample_num).reshape(-1, 1)

# set up a new data set which combines both foreground(training data) and background(reference data)
data = np.concatenate((fore_coord, back_coord), axis=0)
label = np.concatenate((fore_label, back_label), axis=0)
# get the coordinates of all the pixels, [x, y]
coord = foreground.generate_coordinate(img_height, img_width)
# x, y = np.mgrid[0:img_height:1, 0:img_width:1]
# x_coord = y.reshape(-1, 1)
# y_coord = x.reshape(-1, 1)
# coord = np.concatenate((x_coord, y_coord), axis=1)

# Random Forest Regression
regr = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
regr.fit(data, label)
# ExtraTreesRegression
rege = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth)
rege.fit(data, label)
# making regression prediction for each pixel of the raccoon image and reshape vector back to image form for imshow
pre_re = regr.predict(coord).reshape(img_height, img_width)
pre_ex = rege.predict(coord).reshape(img_height, img_width)
# plot pre_re and pre_ex
fig = plt.figure(figsize=(30, 30))
re = fig.add_subplot(1, 2, 1)
ex = fig.add_subplot(1, 2, 2)
re.imshow(pre_re, cmap="gray")
ex.imshow(pre_ex, cmap="gray")
plt.show()







