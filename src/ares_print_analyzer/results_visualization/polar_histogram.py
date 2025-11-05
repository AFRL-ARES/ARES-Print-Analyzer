import matplotlib.pyplot as plt
import numpy as np

def plot_polar_histogram(hist1:np.ndarray,
                         hist2:np.ndarray,
                         row_inds:np.ndarray,
                         col_inds:np.ndarray,
                         angle_bins:int=12,
                         dist_bins:int=5):
    # this function takes the histogram data, and the score assignment data 
    #get the sizes of the histograms so we can pair up the row/columun values propperly
    size1 = hist1.shape[0]
    size2 = hist2.shape[0]
    worker_size= row_inds[-1]+1

    if size1 == worker_size:
        worker = hist1
        task = hist2
    elif size2 == worker_size:
        worker = hist2
        task=hist1

    # Sorts and prunes both histograms such that each indiex corresponds to the most similar point.
    worker = worker[row_inds,:]
    task = task[col_inds,:]

    sum_ = worker+task
    diff = worker-task

    disp_hist = diff**2/(sum_ + 1e-16)
    mean_diff = np.mean(disp_hist,axis=0)

    props = dict(boxstyle='round', facecolor='gray', alpha=0.5)
    text_str = 'Difference Score: {:.2f}'.format(np.sum(mean_diff)/2)
    fig, ax = plt.subplots()
    ax.bar(np.arange(0,60),mean_diff)
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax.set_xlabel(r'Bin #')
    fig.show()

