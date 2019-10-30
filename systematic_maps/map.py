import numpy as np
import healpy as hp
import fitsio

def hpix2radec(nside, pix):
    
    theta, phi = hp.pixelfunc.pix2ang(nside, pix)
    
    return np.degrees(phi), 90-np.degrees(theta)


def good_fraction(nside):
    
    """
    Given the KiDS mask with nside = 4096, 
    this function returns the pixels on a healpix 
    map with lower nside (e.g. 256) that pass the KiDS 
    mask. Furthermore, it returns the fraction of area 
    of each pixel that is covered by KiDS.

    input: nisde (resolution of a healpix map with 
    lower resolution than that of KiDS)
    """
    nside_low = nside
    npix_low = hp.nside2npix(nside_low)

    nside_high = 4096
    npix_high = hp.nside2npix(nside_high)

    ratio = npix_high/npix_low

    pixel_low = np.arange(npix_low)
    pixel_high = np.arange(npix_high)

    kids_mask = hp.fitsfunc.read_map("data/KiDS_K1000_healpix.fits")
    kids_binary = kids_mask.copy()
    kids_binary[kids_binary>0] = 1
    
    kids_theta, kids_phi = hp.pixelfunc.pix2ang(nside_high, pixel_high)
    kids_theta_good, kids_phi_good = kids_theta[kids_binary == 1], kids_phi[kids_binary == 1]
    kids_low_pixels = hp.ang2pix(nside_low, kids_theta, kids_phi)
    kids_low_pixels_good = hp.ang2pix(nside_low, kids_theta_good, kids_phi_good)

    y = np.bincount(kids_low_pixels_good)  #this one returns the unique values
    ii = np.nonzero(y)[0]  
    pixfreq = np.vstack((ii,y[ii])).T
    pixel, freq = pixfreq[:,0], pixfreq[:,1]
    
    return pixel, freq

def apply_window(random_catalog_name):

    rand = pd.read_csv(random_catalog_name)
    print rand.keys()
    ra_rand , dec_rand = rand.ALPHA_J2000.as_matrix(), rand.DELTA_J2000.as_matrix()
    ra_rand[ra_rand > 300] -= 360
    theta = (90. - dec_rand) * np.pi/180. 
    phi = ra_rand * np.pi / 180.
    nside = 4096
    pixel_id = hp.ang2pix(nside, theta, phi)
    
    x = hp.fitsfunc.read_map("data/KiDS_K1000_healpix.fits")

    bad_mask = np.where(x[pixel_id] == 0)[0]
    rand = rand.drop(rand.index[bad_mask])

    return rand
    
def load_sys(nside):
    """
    This function loads the healpix maps of systematic 	
    quantities and puts them in a numpy array
    """
    columns = ['nstar', 'ell','fwhm', 'BackGr','extinction', 'threshold', 'ulim', 'glim', 'rlim','ilim','zlim','ylim','jlim','hlim','klim']
    npix = hp.nside2npix(nside)
    sys_arrays = np.zeros((npix, len(columns)))
    fname = 'dr4_gaia_nstar_nside_'+str(nside)+'.fits'
    sys_arrays[:,0] = hp.fitsfunc.read_map(fname)
    i = 1
    for col in columns[1:]:
        fname  = 'dr4_masked_'+str(col)+'_map_nside_'+str(nside)+'.fits'
        sys_arrays[:,i] = hp.fitsfunc.read_map(fname)
        i = i + 1
    return sys_arrays
