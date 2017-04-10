__author__ = 'Dhananjay Mehta and Swapnil Kumar'
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage import restoration
from skimage import measure
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb
import matplotlib.patches as mpatches

#Reference : http://scikit-image.org/docs/dev/auto_examples/plot_label.html

class ProcessImage():
    
    def __init__(self, image_file):
        self.image = imread(image_file, as_grey=True)
        image = restoration.denoise_tv_chambolle(self.image, weight=0.1)
        thresh = threshold_otsu(image)
        self.bw = closing(image > thresh, square(2))
        self.cleared = self.bw.copy()
        self.clear_border = clear_border(self.cleared)

    def get_text_candidates(self):
        label_image = measure.label(self.cleared)   
        borders = np.logical_xor(self.bw, self.cleared)
        label_image[borders] = -1

        image_label_overlay = label2rgb(label_image, image=self.image)

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(image_label_overlay)
        i = 0
        for region in regionprops(label_image):
            if region.area > 10:
                minr, minc, maxr, maxc = region.bbox
                margin = 4
                minr, minc, maxr, maxc = minr-margin, minc-margin, maxr+margin, maxc+margin
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
                image_portion = self.image[minr:maxr, minc:maxc]
                if image_portion.shape[0]*image_portion.shape[1] == 0:
                    continue
                else:
                    if i == 0:
                        samples = resize(image_portion, (28,28))
                        i += 1
                    elif i == 1:
                        image_portion_small = resize(image_portion, (28,28))
                        samples = np.concatenate((samples[None, :, :], image_portion_small[None, :, :]), axis=0)
                        i += 1
                    else:
                        image_portion_small = resize(image_portion, (28,28))
                        samples = np.concatenate((samples[:, :, :], image_portion_small[None, :, :]), axis=0)

                ax.add_patch(rect)

        self.candidates = {
                    'fullscale': samples,          
                    'flattened': samples.reshape((samples.shape[0], -1)),
                    }
        plt.show()

        return self.candidates

