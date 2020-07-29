import kmeans_radec
import numpy as np

def segment_data(RA, DEC):
    
    coord = np.vstack([RA, DEC]).T
    #centers of the jacknife regions
    centers = np.loadtxt("new_DR4_9bandnoAwr_uniform_centers.txt")
    #number of jacknife regions
    NJK = centers.shape[0]
    print("Segmentation begins!")
    labels_jk = kmeans_radec.find_nearest(coord, centers)
    print("Done with assigning jacknife labels to galaxies")
    #final_cat = {"RA": RA, 
    #	    "DEC": DEC, 
	#    "JK_LABEL": labels_jk}
    return labels_jk
