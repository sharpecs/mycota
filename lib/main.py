# MYCOTA: A Regression Analysis on Fungi using Python.

"""
1. LOAD DATA
"""
import pandas as pd

d = pd.read_json('../data/2023-observations-eudoria.json')
species = pd.DataFrame(d['observation'][1]['species'])
records = pd.DataFrame(d['observation'][3]['records'])

observations = records[records['s'].isin(species['i'])]     # union set


"""
2. PROFILE & CLEAN
"""
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from itertools import cycle
from matplotlib import cm

features = observations.iloc[:, [1, 2]].values              # segment with long/lat for all observations.
n_clusters = 10
kmeans = KMeans(n_init=1, n_clusters= n_clusters, init='k-means++', random_state=0)
kmeans.fit(features)
y_kmeans = kmeans.predict(features)
centers = kmeans.cluster_centers_
observations['zone'] = kmeans.labels_                       # establish observational zones.

plt.figure(figsize=(10,10))                                 # visualisation to id zones.
colors = cycle(cm.tab10.colors)
for i in range(0, 10):
    label = "Zone {0}".format(10 if i == 0 else i)
    # plot one cluster for each iteration
    color = next(colors)
    # find indeces corresponding to cluser i
    idx = y_kmeans == i
    # plot cluster
    plt.scatter(features[idx, 0], features[idx, 1], color=color, s=50, label=label, alpha=0.25)
    # plot center
    plt.scatter(centers[i, 0], centers[i, 1], edgecolors="k", linewidth=2, color=color, s=200, alpha=1)
    plt.annotate(label, xy=(centers[i, 0], centers[i, 1]))

plt.title('Means Clustering Map\n',fontweight="bold", fontsize=18)
plt.show()

observations['zone'] = observations['zone'].replace(0,10)
fungi = species[species['k'].str.contains('FUNGI')]
oFungi = observations[observations['s'].isin(fungi['i'])]

# build feature set.
def set_genus(id):
    return species.loc[species['i'] == id]['g'].to_string(index=False)

def set_subtrate(features):    
    return list(filter(lambda x: x in ['wood','mulch','soil','leaf','tree'], features))[0]

def set_role(features):    
    return list(filter(lambda x: x in ['recycler','parasite','symbiont'], features))[0]

def set_basidios_group(features):
    return list(filter(lambda x: x in ['mushroom','jelly','shelf','rust','bracket','puffball','patch'], features))[0]

oFungi['date'] = pd.to_datetime(pd.to_datetime(observations['t']*1000).dt.strftime('%d/%m/%Y'), format= '%d/%m/%Y')
oFungi['genus'] = oFungi['s'].apply(set_genus)
oFungi['subtrate'] = oFungi['c'].apply(set_subtrate)
oFungi['subtrate_no'], s_index = pd.Series(oFungi['subtrate'].factorize())
oFungi['role'] = oFungi['c'].apply(set_role)
oFungi['role_no'], s_index = pd.Series(oFungi['role'].factorize())

myxos = oFungi[oFungi['c'].apply(lambda x: ', '.join(x)).str.contains('myxos')]
ascos = oFungi[oFungi['c'].apply(lambda x: ', '.join(x)).str.contains('ascos')]

basidios = oFungi[oFungi['c'].apply(lambda x: ', '.join(x)).str.contains('basidios')]
basidios['group'] = basidios['c'].apply(set_basidios_group)
basidios['group_no'], s_index = pd.Series(basidios['group'].factorize())

# look at the NaN value distribution among different column 
# print(basidios.isna().sum())

fGroups = basidios.groupby(['zone','group']).size().reset_index(name='counts')


"""
3. EXPLORATORY
"""
import numpy as np

from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from perth_parks import MCWetland


# Figure A. Map out the observational hotspots.
figA = plt.figure(figsize=(12,12))
perth = Basemap(projection='lcc', lat_0=-32.081559, lon_0=116.001270, width=800, height=800, resolution='h')
perth.scatter(oFungi['y'], oFungi['x'], latlon=True, marker = '', c='r', zorder=1, alpha=0.2)
mcw = MCWetland(perth)

max_perc = 50
plt.colorbar(label=r'Percentage')
plt.set_cmap('Reds')  # RdBu_r
plt.clim(0, max_perc)
cmap = plt.get_cmap()
colors = cmap(np.linspace(0, 1, max_perc))

# add Basins
smBasin = Polygon(mcw.getSmBasinMap(), facecolor='#3282f6', edgecolor='#3282f6',linewidth=3, alpha=0.4, label='')
plt.gca().add_patch(smBasin)
lgBasin = Polygon(mcw.getLgBasinMap(), facecolor='#3282f6', edgecolor='#3282f6',linewidth=3, alpha=0.4, label='')
plt.gca().add_patch(lgBasin)

zone_count = fGroups.groupby('zone', as_index=False).agg({'counts': 'sum'})
zone_count_tot = zone_count.shape[0] + 1
zone_count_sum = zone_count['counts'].sum()

for x in range(1, zone_count_tot):
    zone = mcw.getZone(x)

    zoneVal = ((zone_count[zone_count['zone'] == x]['counts'] / zone_count_sum) * 100).astype(int)
    zonePol = Polygon(zone['map'], facecolor=colors[zoneVal], edgecolor='#3282f6',linewidth=3, alpha=0.4, label='')
    plt.gca().add_patch(zonePol)

    x, y = perth(zone['mark'][0], zone['mark'][1]) 
    plt.text(x, y, zone['name'], fontsize=12) 

plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='lower right', fontsize=12)
plt.title('Choropleth of Fungi by Zone\n', fontsize=20)
figA.show()

# Figure B. Chart the fungi groups showing most prolific.
figB, ax = plt.subplots(figsize=(10,7))
rank = fGroups.groupby('group', as_index=False).agg({'counts': 'sum'}).sort_values(by = 'counts', ascending = True)

y = np.array(rank['group'])
x = np.array(rank['counts'])

ax.barh(y, x)
ax.set_title('Ranking of Fungi Groups')
ax.set_xlabel('Number of Observations')
figB.show()

# Figure C. Chart the fungi subtrates by zones. 
figC, ax = plt.subplots(figsize=(11,7))

fSubtrates = basidios.groupby(['zone','subtrate']).size().reset_index(name='counts')
zones = np.array(fSubtrates.groupby('zone', as_index=False).agg({'counts': 'count'})['zone'])
names = np.array(fSubtrates.groupby(['subtrate'], as_index=False).agg({'counts': 'count'})['subtrate'])
subtrates = fSubtrates.groupby(['zone','subtrate'], as_index=False).agg({'counts': 'count'})

subtrate_list = {}
for n in names :
    nList = np.zeros(len(zones), dtype=int)
    s_df = subtrates[subtrates['subtrate'] == n]
    for x in range(1, len(nList) + 1):
        l_idx = x - 1
        nList[l_idx] = s_df[s_df['zone'] == x].agg({'counts':'sum'})['counts']

    subtrate_list[n] = nList

bottom = np.zeros(len(zones), dtype=int)
for item, item_count in subtrate_list.items():
    p = ax.bar(zones, item_count, label=item, bottom=bottom)
    bottom += item_count

ax.set_title('Subtrate types by Zone')
ax.legend()
ax.set_xlabel('Zones')
figC.show()

# Figure D. Chart fungi groups by zones.
figD, ax = plt.subplots(figsize=(11,7))

names = np.array(fGroups.groupby(['group'], as_index=False).agg({'counts': 'count'})['group'])
groups = fGroups.groupby(['zone','group'], as_index=False).agg({'counts': 'count'})

group_list = {}
for n in names :
    nList = np.zeros(len(zones), dtype=int)
    s_df = groups[groups['group'] == n]
    for x in range(1, len(nList) + 1):
        l_idx = x - 1
        nList[l_idx] = s_df[s_df['zone'] == x].agg({'counts':'sum'})['counts']

    group_list[n] = nList

bottom = np.zeros(len(zones), dtype=int)
for item, item_count in group_list.items():
    p = ax.bar(zones, item_count, label=item, bottom=bottom)
    bottom += item_count

ax.set_title('Groups Types by Zone')
ax.set_xlabel('Zones')
ax.legend()
figD.show()


"""
4. CORRELATION
"""
import seaborn as sns

# Adding additional variables beyond simple bivariate correlation can improve both the explanatory and 
# predictive value of a model.

r_squared = basidios.drop(['t','s','x','y','o','i','l','c','date','role','group','subtrate','genus'], axis=1).corr()['zone']**2
r_squared.round(3)
r_squared = r_squared.sort_values(ascending=False)
print(r_squared)

corr = r_squared = basidios.drop(['t','s','x','y','o','i','l','c','date','role','group','subtrate','genus'], axis=1).corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, vmax=1, square=True, annot=True)
plt.title('Correlation Matrix')


"""
5. STANDARDISATION
"""
dependent = ['zone']
independents = ['role_no','subtrate_no','group_no']

model_data = basidios[dependent + independents + ['q', 'x', 'y']]
model_data = model_data.dropna()
model_data.hist()

transform_vars = independents + ['q']
# One is added to all values because some variables have values of zero and a logarithm of zero is undefined.
model_data[transform_vars] = np.log(model_data[transform_vars] + 1)
model_data.hist()

standardize = independents
model_data.drop("q", axis='columns', inplace=True)
# Because the different variables measure different phenomena and different scales, we can standardize the data 
# values by converting them to z-scores that can then be compared with each other.

# A z-score is the number of standard deviations that a value differs from the mean for the whole data set.
model_data[standardize] = (model_data[standardize] - model_data[standardize].mean()) / model_data[standardize].std()
model_data.hist()


"""
6. MODELLING
"""
import statsmodels.api as sm
import geopandas as gpd
import libpysal

from pysal.model import spreg
from scipy import stats
from shapely.geometry import Point

ols_model = spreg.OLS(model_data[dependent].values,
        model_data[independents].values,
        name_y = dependent, name_x = independents)

# Residuals are the differences between predicted modeled values and actual values.
residuals = model_data
residuals["residuals"] = ols_model.u
residuals['actual'] = 0.0
residuals['predicted'] = 0.0

y = model_data[dependent] 
x = model_data[independents]

# add placeholder (column of 1s) for independents ie. Y = 1a + bX constant 
# is required to get prediction for Y.
x = sm.add_constant(x)

#fit linear regression model
model = sm.OLS(y, x).fit()
mu, std = stats.norm.fit(model.resid)

fig, ax = plt.subplots(figsize=(11,7))
# plot the residuals
sns.histplot(x=model.resid, ax=ax, stat="density", linewidth=0, kde=True)
ax.set(title="Distribution of residuals", xlabel="residual")

# plot corresponding normal curve
xmin, xmax = plt.xlim() # the maximum x values from the histogram above
x = np.linspace(xmin, xmax, 100) # generate some x values
p = stats.norm.pdf(x, mu, std) # calculate the y values for the normal curve
sns.lineplot(x=x, y=p, color="orange", ax=ax)
plt.show()

print(ols_model.summary)


# Map out the nearest neighbours to confirm weights matrix is correctly shaped with zones. 
from shapely import Polygon

knn_data = basidios[dependent + independents + ['x', 'y']]
knn_data = knn_data.dropna()

zone_df = pd.DataFrame({
    'y' : knn_data['y'], 
    'x' : knn_data['x'], 
    'subtrate_no' : knn_data['subtrate_no'],
    'group_no' : knn_data['group_no'],
    'role_no' : knn_data['role_no'],
    'zone'  : knn_data['zone'],
    'coords': list(zip(knn_data['y'], knn_data['x']))
})

# A EPSG code defines what coordinate system the geo shape file is using. Its important to use a correct Coordinate
# Reference System for accuracy when applying weighted regression. Use epsg.io to search for correct code.
# https://epsg.io/8031

code = 'EPSG:8031'

geo_df = gpd.GeoDataFrame(
    zone_df, crs  ={'init': code},
    geometry = zone_df['coords'].apply(Point)
).to_crs(epsg=8031)

ax = geo_df.plot(figsize= (12, 12), alpha  = 1)

# Add map features to axis
smBasin_gm = Polygon(zip([x[0] for x in mcw.getSmBasinXY()], [y[1] for y in mcw.getSmBasinXY()]))
smBasin_po = gpd.GeoDataFrame(zone_df, index=[0], crs=code, geometry=[smBasin_gm])
smBasin_po.plot(ax=ax, facecolor='#ADCDFB', edgecolor='#3282f6')

lgBasin_gm = Polygon(zip([x[0] for x in mcw.getLgBasinXY()], [y[1] for y in mcw.getLgBasinXY()]))
lgBasin_po = gpd.GeoDataFrame(zone_df, index=[0], crs=code, geometry=[lgBasin_gm])
lgBasin_po.plot(ax=ax, facecolor='#ADCDFB', edgecolor='#3282f6')

# add Zones
for x in range(1, 11):
    zone = mcw.getZone(x)
    coordinates = zone['xy']
    gpd.GeoDataFrame(zone_df, index=[0], crs=code, geometry=[
        Polygon(zip([x[0] for x in coordinates], [y[1] for y in coordinates]))
    ]).plot(ax=ax, edgecolor='lightgray', facecolor='none')

w = libpysal.weights.KNN.from_dataframe(geo_df, k=4)                            # weights matrix
geo_df["index"] = geo_df.index
w.plot(gdf=geo_df, indexed_on="index", ax=ax, color='#808080')

ax.set_title('Nearest Neighbours Map\n',fontweight="bold", fontsize=18)

# The weights matrix can be passed to the OLS() function along with the spat_diag and moran parameters to 
# display spatial autocorrelation statistics at the bottom of the model summary.

ols_model = spreg.OLS(model_data[dependent].values,
        model_data[independents].values, w, spat_diag = True, moran=True,
        name_y = dependent, name_x = independents)
print(ols_model.summary)

lag_model = spreg.ML_Lag(model_data[dependent].values,
        model_data[independents].values, w,
	name_y = dependent[0], name_x = independents)
print(lag_model.summary)

err_model = spreg.ML_Error(model_data[dependent].values,
        model_data[independents].values, w,
	name_y = dependent[0], name_x = independents)
print(err_model.summary)

# Map out the residuals from the model.
from matplotlib.patches import Polygon

# model_data["Residuals"] = err_model.u
residuals = model_data
residuals["residuals"] = ols_model.u
residuals['actual'] = 0.0
residuals['predicted'] = 0.0

ols = plt.figure(figsize=(12,13))
perth = Basemap(projection='lcc', lat_0=-32.081559, lon_0=116.001270, width=800, height=800, resolution='h')
perth.scatter(oFungi['y'], oFungi['x'], latlon=True, marker = '', c='r', zorder=1, alpha=0.2)
mcw = MCWetland(perth)

max_perc = 50
plt.colorbar(label=r'Change Percentage')
plt.set_cmap('Reds')  # RdBu_r
plt.clim(0, max_perc)
cmap = plt.get_cmap()
colors = cmap(np.linspace(0, 1, max_perc))

# add Basins
smBasin = Polygon(mcw.getSmBasinMap(), facecolor='#3282f6', edgecolor='#3282f6',linewidth=3, alpha=0.4, label='')
plt.gca().add_patch(smBasin)
lgBasin = Polygon(mcw.getLgBasinMap(), facecolor='#3282f6', edgecolor='#3282f6',linewidth=3, alpha=0.4, label='')
plt.gca().add_patch(lgBasin)

zone_count = residuals.groupby(['zone'], as_index=False).agg({'actual':'count','residuals':'mean'})
zone_count['predicted'] = (zone_count['actual'] - zone_count['residuals']).abs().round()
zone_count['difference'] = (zone_count['predicted'] - zone_count['actual']).abs()
zone_count_tot = zone_count.shape[0]
zone_count_sum = zone_count['difference'].sum()

# add Zones
for x in range(1, zone_count_tot + 1):
    zone = mcw.getZone(x)

    zoneVal = ((zone_count[zone_count['zone'] == x]['difference'] / zone_count_sum) * 100).astype(int)
    zonePol = Polygon(zone['map'], facecolor=colors[zoneVal], edgecolor='#3282f6',linewidth=3, alpha=0.4, label='')
    plt.gca().add_patch(zonePol)

    x, y = perth(zone['mark'][0], zone['mark'][1]) 
    plt.text(x, y, zone['name'], fontsize=12) 

plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='lower right', fontsize=12)
plt.title('Residuals from Spatial Error Model\n',fontweight="bold", fontsize=18)
ols.show()

# Apply the geographically weighted model to map out the predicted change for each independent.
import mgwr.gwr
import mgwr.sel_bw
from matplotlib.patches import Polygon

coordinates = model_data[['y', 'x']]
gwr_selector = mgwr.sel_bw.Sel_BW(coordinates, model_data[dependent].values,model_data[independents].values)
gwr_bw = gwr_selector.search(bw_min=2)
gwr_model = mgwr.gwr.GWR(coordinates, model_data[dependent].values,
        model_data[independents].values, bw=82).fit()
gwr_model.summary()

geo_wr = model_data

for z in range(1, len(independents) + 1):
    geo_wr["feature"] = gwr_model.params[:,z]   # independents
    geo_wr['actual'] = 0.0
    geo_wr['predicted'] = 0.0

    gwr = plt.figure(figsize=(12,12))
    perth = Basemap(projection='lcc', lat_0=-32.081559, lon_0=116.001270, width=800, height=800, resolution='h')
    perth.scatter(oFungi['y'], oFungi['x'], latlon=True, marker = '', c='r', zorder=1, alpha=0.2)
    mcw = MCWetland(perth)
    max_perc = 50
    plt.colorbar(label=r'Change Percentage')
    plt.set_cmap('Reds')  # RdBu_r
    plt.clim(0, max_perc)
    cmap = plt.get_cmap()
    colors = cmap(np.linspace(0, 1, max_perc))

    # add Basins
    smBasin = Polygon(mcw.getSmBasinMap(), facecolor='#3282f6', edgecolor='#3282f6',linewidth=3, alpha=0.4, label='')
    plt.gca().add_patch(smBasin)
    lgBasin = Polygon(mcw.getLgBasinMap(), facecolor='#3282f6', edgecolor='#3282f6',linewidth=3, alpha=0.4, label='')
    plt.gca().add_patch(lgBasin)

    zone_count = geo_wr.groupby(['zone'], as_index=False).agg({'actual':'count','feature':'mean'})
    zone_count['predicted'] = (zone_count['actual'] - zone_count['feature']).abs().round()
    zone_count['difference'] = (zone_count['predicted'] - zone_count['actual']).abs() + 0.001
    zone_count_tot = zone_count.shape[0]
    zone_count_sum = zone_count['difference'].sum()

    # add Zones
    for x in range(1, zone_count_tot + 1):
        zone = mcw.getZone(x)

        zoneVal = ((zone_count[zone_count['zone'] == x]['difference'] / zone_count_sum) * 100).astype(int)
        zonePol = Polygon(zone['map'], facecolor=colors[zoneVal], edgecolor='#3282f6',linewidth=3, alpha=0.4, label='')
        plt.gca().add_patch(zonePol)

        x, y = perth(zone['mark'][0], zone['mark'][1]) 
        plt.text(x, y, zone['name'], fontsize=12) 

    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='lower right', fontsize=12)
    plt.title("Predicted Change for {} by Zone".format(independents[z - 1].replace('_no','s').capitalize()),fontweight="bold", fontsize=18)
    gwr.show()


