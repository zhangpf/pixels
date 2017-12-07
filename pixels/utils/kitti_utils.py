import numpy as np

ABS_THRESH = 3.0
REL_THRESH = 0.05

# def disp_3pe(gt, orig, ipol, mapping):
# 	assert gt.shape == orig.shape
# 	assert gt.shape == ipol.shape
# 	assert gt.shape == mapping.shape
# 	assert np.all(orig >= 0), "The estimated results is not full for all pixels."

# 	valid = (gt >= 0).astype(float)
# 	err = np.abs(gt - ipol) > ABS_THRESH | 
# 		np.abs(gt - ipol) / np.abs(gt) > REL_THRESH

# 	bg = (mapping == 0).astype(float)
# 	fg = (mapping != 0).astype(float)
# 	orig_valid = (orig >= 0).astype(float)
# 	orig_valid_bg = ((orig >= 0) & bg).astype(float)
# 	orig_valid_fg = ((orig >= 0) & fg).astype(float)

# 	err_bg = bg * err
# 	err_bg_result = err_bg * orig_valid_bg
# 	err_fg = fg * err
# 	err_fg_result = err_fg * orig_valid_fg

# 	err_result = err * orig_valid

# 	return [ np.sum(err_bg), np.sum(bg), np.sum(err_bg_result), np.sum(orig_valid_bg),
# 			 np.sum(err_fg), np.sum(fg), np.sum(err_fg_result), np.sum(orig_valid_fg),
# 			 np.sum(err), float(gt.size), np.sum(err_result), np.sum(orig_valid) ]

def disp_3pe_all(gt, est):
	assert gt.shape == est.shape
	assert np.all(est >= 0), "The estimated results is not full for all pixels."

	err = np.abs(gt - est) > ABS_THRESH | 
		np.abs(gt - est) > REL_THRESH * np.abs(gt)

	return gt.size, np.sum(err)

def disp_eval(gt_noc, gt_occ, est, mapping):
	return [ disp_err(gt_noc, est, est, mapping), 
			 disp_err(gt_occ, est, est, mapping)
			]
