import cv2
import time

# Read image
img = cv2.imread("cat.jpg")
# Selective Search Algorithm.
# Step1. Create Selective Search Segmentation Object using default parameters
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
# Step2. Set input image on which we will run segmentation
ss.setBaseImage(img)
# Step3. Set switchToSelectiveSearchFast() or switchToSelectiveSearchQuality()
#ss.switchToSelectiveSearchFast()
ss.switchToSelectiveSearchQuality()
# Step4. Run selective search on the image
start_time = time.time()
region_proposals = ss.process()
elapsed_time = time.time() - start_time
print(f"Number of Region Proposals: {len(region_proposals)}; Elapsed time:{elapsed_time:0.3f}ms.")
# Visualize the region proposals
numShowRects = 50 # number of region proposals to show
# Itereate over all the region proposals
for i, rect in enumerate(region_proposals):
    # draw rectangle for region proposal till numShowRects
    if (i < numShowRects):
        x, y, w, h = rect
        cv2.rectangle(img, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 0), 1, cv2.LINE_AA)
    else:
        break
cv2.imshow("Selective Search Demo", img)
cv2.waitKey()
cv2.destroyAllWindows()