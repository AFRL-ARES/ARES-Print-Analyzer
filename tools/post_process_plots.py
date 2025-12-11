import cv2 as cv
import pandas as pd
import numpy as np
from pathlib import Path
from ares_print_analyzer import pose_and_render
from ares_print_analyzer.analysis import *
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib
import itertools
#%% Sources

image_folder = '../athena-demo-resources/time_lapse/images'
data_file = '../athena-demo-resources/time_lapse/ATHENA_RESULTS.xlsx'
model_file = "../athena-demo-resources/processed files/bunny_head_0/bunny_head_0_marked.stl"
config_json = "../athena-demo-resources/time_lapse/post_process_config.json"
model_json = "../athena-demo-resources/processed files/bunny_head_0/bunny_head_0_marked.json"
output_folder = '../athena-demo-resources/time_lapse/output'
debug_folder = '../athena-demo-resources/debug'
data_bar_height = 800
video_width = 3840
#%% Setup
matplotlib.use('Agg')
image_folder = Path(image_folder)
data = pd.read_excel(data_file)
# data = data.iloc[:3,:]
file_list = list(image_folder.glob('*.jpg'))
file_names_list = [f.stem.split('_')[0] for f in file_list]
image_file_dict = dict(zip(file_names_list,file_list))

#%% Rerun analysis on each experimenta image in order
new_scores = []
for i, e in enumerate(data['Experiment ID'].to_list()):
    img_file = image_file_dict[e]
    input_image = e_img = cv.imread(str(img_file))
    experiment_name = f'Experiment_{i}'
    output_path = Path(output_folder)/experiment_name
    experiment_image, synthetic_image, roi_min, roi_max, obj_center, marker_contours = pose_and_render(input_image,
                                                                                                        model_file,
                                                                                                        config_json,
                                                                                                        model_json,
                                                                                                        str(output_path),
                                                                                                        experiment_name,skip_distortion_correction=True,debug=True)

  

    exp_crop = experiment_image[roi_min[1]:roi_max[1],roi_min[0]:roi_max[0],:]
    syn_crop = synthetic_image[roi_min[1]:roi_max[1],roi_min[0]:roi_max[0],:]
    local_center = obj_center - roi_min
    local_markers = tuple((i-roi_min for i in marker_contours))
    try:
        exp_contour, syn_contour ,e,s = get_contours(exp_crop,syn_crop,local_center,local_markers,debug=True)
        cv.imwrite(str(output_path/"ex_cont.jpg"),e)
        cv.imwrite(str(output_path/"sy_cont.jpg"),s)

        # rescale the (square) ROI images to 800px x 800 p
        # add the labels 'Experiment' and 'Rendered' to their respective images
        font                   = cv.FONT_HERSHEY_SIMPLEX
        fontScale              = 2
        fontColor              = (0,0,0)
        thickness              = 3
        lineType               = 8
        e800 = cv.resize(e,(data_bar_height,data_bar_height),interpolation=cv.INTER_CUBIC)
        cv.putText(e800,"Experiment", (30,60), font,fontScale,fontColor,thickness,lineType)
        s800 = cv.resize(s,(data_bar_height,data_bar_height),interpolation=cv.INTER_CUBIC)
        cv.putText(s800,"Render", (30,60), font,fontScale,fontColor,thickness,lineType)
        out_img = np.column_stack((e800,s800))
        cv.imwrite(str(output_path/"cont_compare.jpg"),out_img)

    except Exception as e:
        print("An error occured during contour extration: {}".format(e))
        score = 10000
    # 5. Get the histograms for both contours
    try: 
        exp_hist = get_histogram(exp_contour)
        syn_hist = get_histogram(syn_contour)
    except Exception as e: 
        print("An error occured during histogram calculation: {}".format(e))
        score=10000

    # 6. Score the contours on how similar they are
    try:
        stats = get_chi_statistic(syn_hist,exp_hist)
        row_ind, col_ind = linear_sum_assignment(stats)
        score = stats[row_ind, col_ind].sum()/len(row_ind)
    except Exception as e:
        print("An error occured during scoring: {}".format(e))
        score = 10000
    
    new_scores.append(score)
data['New Analysis Result'] = new_scores

#%% Now to make the campaign results plots
base_speed = 60 # mm/s
base_fan = .7
param_names = ['Nozzle Temp', 'Fan Speed Mod', 'Print Speed Mod']
param_bounds = ((200,245),(0.0,1.2),(0.7,1.3))
param_bounds = dict(zip(param_names,param_bounds))

def make_conditions_plots(data, param_names, param_bounds,plot_shape=(video_width-1600,data_bar_height)):

    parameter_values = [] 
    normalized_values =  []
    for name in param_names:
        values = data[name].to_numpy()
        bounds = param_bounds[name]
        norm_values = (values - bounds[0]) / (bounds[1]-bounds[0])
        parameter_values.append(values)
        normalized_values.append(norm_values)
    parameter_values = np.array(parameter_values)
    normalized_values = np.array(normalized_values)
    n_iter = parameter_values.shape[1]

    score = data['New Analysis Result'].to_numpy()
    norm_score = (score)/(400)
    inv_score = 1-norm_score
    best_score = []
    for i, s in enumerate(inv_score):
        if i == 0:
            best_score.append(s)
        else:
            if s > best_score[-1]:
                best_score.append(s)
            else:
                best_score.append(best_score[-1])

    best_score = np.array(best_score)



    
    images = []
    fig_size =(plot_shape[0]/300,plot_shape[1]/300)
    for i in range(n_iter+1):
        marker_cycle = itertools.cycle(('o','+','.','*','^','s','x','D')) 
        fig,(ax0,ax1) = plt.subplots(2,1,sharex=True,height_ratios=(1,1),figsize=fig_size)
        # top plot: The planner score
        ax0.set_ylabel('Print Quality',fontsize=10,fontweight='bold')
        ax0.set_ylim(-0.1,1.1)
        ax0.set_yticks([0,0.5,1])
        ax0.plot(np.arange(0,i),inv_score[:i],label='Most Recent',marker='^')
        ax0.plot(np.arange(0,i),best_score[:i],label='Best',marker='s')
        ax0.legend(loc='center right',fontsize=9)

        # bottom plot: the changes in parameters
        ax1.set_xlim(-0.5,n_iter+5)
        ax1.set_ylim(-0.1,1.1)
        ax1.set_xticks(np.arange(0,n_iter))
        ax1.set_yticks([0,0.5,1])
        ax1.set_yticklabels([r'$x_{min}$','',r'$x_{max}$'])
        ax1.set_xlabel('Iteration',fontsize=10,fontweight='bold')
        ax1.set_ylabel('Param. Value',fontsize=10,fontweight='bold')
        if i >= n_iter:
            for j, p in enumerate(param_names):
                ax1.plot(np.arange(0,i),normalized_values[j,:],label=p,marker=next(marker_cycle))
        else:
            for j, p in enumerate(param_names):
                ax1.plot(np.arange(0,i+1),normalized_values[j,:i+1],label=p,marker=next(marker_cycle))
        ax1.legend(loc='center right',fontsize=9)
        fig.set_dpi(300)
        fig.tight_layout()
        fig.canvas.draw()
        img = np.array(fig.canvas.buffer_rgba())
        img = cv.cvtColor(img, cv.COLOR_RGBA2BGR)
        images.append(img)
    
    return images

plots = make_conditions_plots(data,param_names,param_bounds)
for i, p in enumerate(plots):
    experiment_name = f'Experiment_{i}'
    output_path = Path(output_folder)/experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(output_path/"plot.jpg"),p)
    if i == 0:
        cont_img = np.zeros((data_bar_height,data_bar_height*2,3),dtype=np.uint8)
    else:
        last_expt = f'Experiment_{i-1}'
        read_path = Path(output_folder)/last_expt
        cont_img = cv.imread(str(read_path/'cont_compare.jpg'))
    
    data_img = np.column_stack((cont_img,p))
    cv.imwrite(str(output_path/"cont_plot.jpg"),data_img)


#%% 