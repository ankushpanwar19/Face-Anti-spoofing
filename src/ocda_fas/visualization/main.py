import glob
import os
import numpy as np
from PIL import Image

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from bokeh.plotting import figure, curdoc
import pandas as pd

from bokeh.models import ColumnDataSource, ImageURL,Range1d
from bokeh.layouts import row
from bokeh.models import WheelZoomTool, SaveTool, LassoSelectTool


N_BINS_CHANNEL = 50

# Fetch the number of images using glob
#my_root = "C:/Users/Pratyush/SWITCHdrive/tsne/static"
my_root =  "C:/Users/Pratyush/SWITCHdrive/samples"
my_exts = ['*.jpg', 'jpeg', '*.png']
files = [glob.glob(my_root + '/**/'+ x, recursive=True) for x in my_exts]  
flattened = [val for sublist in files for val in sublist]

N = len(flattened)
#N = len(glob.glob("static/*.jpg"))

# root directory of  app to generate the image URL for the bokeh server
ROOT = os.path.split(os.path.abspath("."))[1] + "/"

# Number of bins per color for the 3D color histograms
N_BINS_COLOR = 16
# Number of bins per channel for the channel histograms
N_BINS_CHANNEL = 50

# Define an array containing the 3D color histograms. We have one histogram per image each having N_BINS_COLOR^3 bins.
# i.e. an N * N_BINS_COLOR^3 array
color_hist = np.zeros((N, N_BINS_COLOR**3))

# Define an array containing the channel histograms, there is one per image each having 3 channel and N_BINS_CHANNEL
# bins i.e an N x 3 x N_BINS_CHANNEL array
channel_hist = np.zeros((N, 3, N_BINS_CHANNEL))
# initialize an empty list for the image file paths
url = []
image_list = [[] for i in range(N)]
# Compute the color and channel histograms
for idx, f in enumerate(glob.glob(my_root + '/**/'+ "*.jpg", recursive=True)):
#for idx, f in enumerate(glob.glob("static/*.jpg")):
    image_list[idx].append (f)
    image = Image.open(os.path.abspath(f))
    image_list[idx].append (image)
    # Convert the image into a numpy array and reshape it such that we have an array with the dimensions (N_Pixel, 3)
    image_np = (np.array(image)).reshape(-1,3)
    # Compute a multi dimensional histogram for the pixels, which returns a cube
    h, e = np.histogramdd(image_np, bins=N_BINS_COLOR)
    # However, later used methods do not accept multi dimensional arrays, so reshape it to only have columns and rows
    # (N_Images, N_BINS^3) and add it to the color_histograms array you defined earlier
    color_hist[idx] = h.reshape(-1)
    # Append the image url to the list for the server
    url.append(ROOT + f)
    # Compute a "normal" histogram for each color channel (rgb)
    # and add them to the channel_histograms
    for channel_id in range(3):
        histogram, bin_edges = np.histogram(image_np[:, channel_id], bins=N_BINS_CHANNEL, range=(0, 256), density = True)
        channel_hist[idx][channel_id] = histogram
    
#calculating tSNE dimensionality reduction
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=250)
tsne_results = tsne.fit_transform(color_hist)

#calculating pca dimensionality reduction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(color_hist)

# Creating a dataframe with fitted values of tsne and pca
df = pd.DataFrame()
df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 

CDS_color_histo = ColumnDataSource(dict(url = url,tsne_1  = df['tsne-2d-one'].to_numpy(),tsne_2  = df['tsne-2d-two'].to_numpy(),pca_1  =  df['pca-one'].to_numpy(),pca_2  =  df['pca-two'].to_numpy()))
# Calculate height and width range for image plots
range_tsne_one = max(df['tsne-2d-one']) - min(df['tsne-2d-one'])
range_tsne_two = max(df['tsne-2d-two']) - min(df['tsne-2d-two'])

# setting image width to somewhat double of image height
width_tsne = 3
height_tsne = 1.5


#t-SNE Plot
xdr = Range1d(start=min(df['tsne-2d-one']), end = max(df['tsne-2d-one'])+width_tsne)  
ydr = Range1d(start=min(df['tsne-2d-two']), end = max(df['tsne-2d-two']))
p1 = figure(x_range=xdr, y_range=ydr, plot_width=1000, plot_height=1000,
    min_border=0, tools="wheel_zoom,reset, pan")
image = ImageURL(url="url", x="tsne_1", y="tsne_2", w=width_tsne, h=height_tsne)
#p1.add_glyph(CDS_color_histo,image)#adding image glyphs
p1.circle('tsne_1', 'tsne_2', source=CDS_color_histo, line_color="red",alpha=1)
p1.title.text ='t-SNE' #plot title and axes names
p1.xaxis.axis_label = 'x'
p1.yaxis.axis_label = 'y'

aggre_channel_histo = channel_hist.sum(axis=0)
# Construct a datasource containing the channel histogram data. Default value should be the selection of all images.
# Normalizing the data 

CDS_channel_histo = ColumnDataSource(dict(
    idx = np.arange(N_BINS_CHANNEL),
    ch1  = (aggre_channel_histo[0] - min(aggre_channel_histo[0]))/max(aggre_channel_histo[0] - min(aggre_channel_histo[0])),
    ch2  = (aggre_channel_histo[1] - min(aggre_channel_histo[1]))/max(aggre_channel_histo[1] - min(aggre_channel_histo[1])),
    ch3  = (aggre_channel_histo[2] - min(aggre_channel_histo[2]))/max(aggre_channel_histo[2] - min(aggre_channel_histo[2])),
))

# Connect the on_change routine of the selected attribute of the dimensionality reduction ColumnDataSource with a
# callback/update function to recompute the channel histogram.
#calling the onchange functionality
#CDS_color_histo.selected.on_change('indices',update_callback)
# Construct a layout and use curdoc() to add it to your document.
curdoc().add_root(row(p1))

#TO RUN---
# use the command below in the folder of your python file to start a bokeh directory app
# python file must be named main.py and provide the path to variable my_root. 

# bokeh serve --show .
# python -m bokeh serve --show .
# wait fr it to run, takes time depending on no. of samples.
# the no. of iterations and perplexity needs to be tuned as per need and no. of samples on line 110

#doubt
#dont know why but KL divergence value of 251th iter is very very high. But it should nopt be a problem since only 250 iters are taken into account