<a id="readme-top"></a>

# MYCOTA: A Regression Analysis on Fungi using Python.

<figure>
    <center>
    <img src="../assets/fungi-01.png">
    <center>
    Fungal groups [L-R] : Patch, Rust, Puffball, Bracket, Jelly, Shelf, Mushroom
</figure>


## Mary Carroll Park is a seasonal clay-based wetland with two basins that dry to an impervious state in late autumn. With the winter rains, the park hydrates and erupts with distinctive fruit bodies from a diverse range of fungi. The data used in this analysis was collected from this location in 2023.

The purpose of this exercise is to understand the principles behind the following two techniques with Python:

* Ordinary Least Squares (OLS) is the best known of the regression techniques. It's a great starting point for all spatial regression analyses. It provides a generic model of the features you are trying to understand creating a single regression equation to describe it.

* Geographically Weighted Regression (GWR) is another tool used to model spatially varying relationships. The main difference is that it creates separate models for each independent feature which may provide insights into additional factors related to that feature.

Darwin's blind spot is a hypothesis that symbiosis is insignificant in nature. This analysis will investigate
the hypothesis using the regression techniques and the observations collected.


### Content
<ol>
  <li><a href="#segmentation">Segmentation</a></li>
  <li><a href="#correlation">Correlation</a></li>
  <li><a href="#standardisation">Standardisation</a></li>
  <li><a href="#regression">Regression</a></li>
  <li><a href="#spatial-regression">Spatial Regression</a></li>
  <li><a href="#geographically-weighted-regression">Geographically Weighted Regression</a></li>
  <li><a href="#references">References</a></li>
</ol>


<!-- ABOUT THE PROJECT -->
### Segmentation

The data used for this purpose has been collected over a twelve-month period and consists of observations from all biological kingdoms. To target distinct features in the park, the data has been segmented into smaller groups for more tailored insights.

<figure>
    <center>
    <img src="../assets/mycota-01.png">
    <center>
</figure>


K-Means Clustering is the technique used to group the observational points by assigning each point to the nearest cluster centroid. These centroids are used to identify the hotspots or zones in the park. Within each zone, the concentration of fungi observations can then be identified, showing the distribution across the park. In this case, Zone 7 has the most observations, while Zone 10 has the least.

<figure>
    <center>
    <img src="../assets/mycota-02.png">
    <center>
</figure>


<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Correlation

An approach to evaluate the influence of multiple factors is to apply a regression method that considers multiple explanatory variables known as independents to predict the outcome of a response variable called the dependent. From the data, the variables available for consideration are the number of fungi observed , the substrate it was seen on, the type of fungi, and its purpose in nature. The accuracy of this information is uncertain.

<figure>
    <center>
    <img src="../assets/mycota-03.png">
    <center>
</figure>

A correlation matrix can be used to visualise the strength a variable may have in relation to the zones created. In this case, looking at the zone column, the role (purpose in nature) has the strongest relationship. Lower correlated variables are still important in this situation, as they may add weight to the model when the relationship between the independents is spurious.

<figure>
    <center>
    <img src="../assets/mycota-04.png">
    <center>
</figure>

The most prolific group of fungi observed is the mushroom. The total number far exceeds that of the others and may skew the results inappropriately. The groups have been defined by the fruiting structure of the fungi for the purpose of this exercise.

| Group | Features |
| ----------- | ----------- |
| Mushroom | A typical structure with cap, gills and stem. |
| Patch | Attached to a wood surface like a crusty sheet or patch growing flat.|
| Bracket | Firm or hard bracket-like structures on trees and wood. |
| Shelf | Thin, leathery, tiered structures usually found on wood. |
| Jelly | Soft gelatinous fungi on wood. |
| Rust | Usually brown or yellow spots on plant also can be powdery rust coloured. |
| Puffball | Sac-like with unusal appearance. Sac will eventually rupture or split. |


<figure>
    <center>
    <img src="../assets/mycota-05.png">
    <center>
</figure>

The bar chart shows Zone 7 having the most diverse groups in the park. The only group not in the zone is the Jelly. The main point is that a group does exist in each zone , making it a good candidate for the model.

<figure>
    <center>
    <img src="../assets/mycota-06.png">
    <center>
</figure>

Since there are groups in every zone, there must be substrates in every zone as well. As expected, the chart shows a greater range of substrates observed in Zone 7. Being coupled with groups , the substrates will have a similar correlation and also be good candidates . The matrix also identifies group and role as having the strongest correlation among all variables. Since the group is selected, the role will also be selected. There are still some reservations regarding quantity, as this variable has the weakest correlation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Standardisation

It was noted earlier that the quantity of mushrooms observed may impact the results from the model. To ensure this variable does not influence the model, all variables can be standardised by converting them to z-scores. A z-score is the number of standard deviations that a value differs from the mean for all variables.

<figure>
    <center>
    <img src="../assets/mycota-07.png"> 
    <center>
    [ Histograms showing variable distributions ]
</figure>

Looking at the statistics available for the chosen variables, the quantity, as expected, is skewed to the left, showing the impact of mushrooms . This variable can be modified to attempt to bring its distribution closer to normality. A transformation modification can be achieved using the same logarithmic transformation applied to the variables.

<figure>
    <center>
    <img src="../assets/mycota-08.png"> 
    <center>
    [ Histograms showing transformed variables ]
</figure>

The chart for transformed variables shows little impact on Quantity. The matrix previously indicated that Quantity had the least strength in the correlation of variables, making this variable unfit for the model, which will be dropped from standardisation . Although the chosen variables have similar scales, they will be converted to z-scores to facilitate comparison.

<figure>
    <center>
    <img src="../assets/mycota-09.png"> 
    <center>
    [ Histograms showing standardised distributions ]
</figure>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Regression

Like the correlation, the regression is evaluated with the variable values squared. Higher values indicate a better fit. The chosen modeling technique applies the 'least- squares ' line, which is the distance from the 'best-fit' squared and then added. This results in values having a smaller range. The main interest in any modeling result is data phenomena. To identify whether the chosen variables are adequate for the model, the difference between the predicted and actual values is charted.

<figure>
    <center>
    <img src="../assets/mycota-10.png">
    <center>
</figure>

The residual is an attribute that can be easily extracted from the model. It shows the difference between modeled values and actual values. By plotting the residuals as a weight ( Kernel Density Estimation) and overlaying its normal curve with the mean and standard deviation curve, any changes can be visualized . In this case, there is a shift indicating some type of phenomenon worth investigating.

```
REGRESSION RESULTS
------------------

SUMMARY OF OUTPUT: ORDINARY LEAST SQUARES
-----------------------------------------
Data set            :     unknown
Weights matrix      :        None
Dependent Variable  :    ['zone']                Number of Observations:          82
Mean dependent var  :      5.1098                Number of Variables   :           4
S.D. dependent var  :      2.3571                Degrees of Freedom    :          78
R-squared           :      0.0212
Adjusted R-squared  :     -0.0164
Sum squared residual:     440.456                F-statistic           :      0.5641
Sigma-square        :       5.647                Prob(F-statistic)     :      0.6403
S.E. of regression  :       2.376                Log likelihood        :    -185.278
Sigma-square ML     :       5.371                Akaike info criterion :     378.555
S.E of regression ML:      2.3176                Schwarz criterion     :     388.182

------------------------------------------------------------------------------------
            Variable     Coefficient       Std.Error     t-Statistic     Probability
------------------------------------------------------------------------------------
            CONSTANT         5.10976         0.26242        19.47168         0.00000
             role_no         0.28044         0.29280         0.95777         0.34114
         subtrate_no         0.25556         0.27054         0.94461         0.34777
            group_no        -0.18625         0.29917        -0.62257         0.53538
------------------------------------------------------------------------------------

REGRESSION DIAGNOSTICS
MULTICOLLINEARITY CONDITION NUMBER           1.666

TEST ON NORMALITY OF ERRORS
TEST                             DF        VALUE           PROB
Jarque-Bera                       2          4.022           0.1338

DIAGNOSTICS FOR HETEROSKEDASTICITY
RANDOM COEFFICIENTS
TEST                             DF        VALUE           PROB
Breusch-Pagan test                3          1.450           0.6939
Koenker-Bassett test              3          2.701           0.4400
================================ END OF REPORT =====================================

```

The summary report produced for this model shows weak quantified results. The R-squared value, which is interpreted as the percentage of variance with the independents, is quite low compared to the values displayed in the correlation matrix. The strongest correlation with the zone in the matrix is with the role. Also worth noting is that the coefficient for role shows a positive correlation, as does substrate . However, group has a negative correlation. A positive correlation indicates that the strength of the relationship increases towards the higher numbered zones, while a negative correlation indicates a weaker relationship towards the higher zones. Since the residual chart showed some type of change, the modeled results should still be mapped.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Spatial Regression

One of the reasons for a weak summary result is that even though a positive and negative correlation direction exist, there is no correlation (neagtive Adjusted R-squared) between the sequence of values at different parts of the distribution. Meaning the zones used to identify as a dependent values must have been autocorrelated when processed by the model. The ordinary least squares algorithm assumes the variables are not aurocorrelated. A problem with applying clustering to geospatial variables, in this case to create zones, is that the variables would have been autocorrelated as part of that process. Spatial autocorrelation causes multivariate model coeffients and output to be bias and untrustworthy.

Fortunately there are spatial regression functions that compensate for autocorrelation so that the coefficients and outputs are more realistic. To apply these functions we need to scale the zones using a weights matrix created with a tool that calculates the neighbours of each data feature with a Coordinate Reference System (CRS) for mapping.

<figure>
    <center>
    <img src="../assets/mycota-11.png">
    <center>
</figure>

The quality of the weight matrix can then be visually checked before applying it to the model. The nearest neighbor map is different from the clustering map used for segmentation; mainly, there are fewer groups, and the observational points are connected. Having fewer groups will not affect the results because the clustered groups are only for labeling purposes. The weight matrix is added to the model along with other spatial parameters set to display spatial autocorrelation statistics with the summary.

```
REGRESSION RESULTS
------------------

SUMMARY OF OUTPUT: ORDINARY LEAST SQUARES
-----------------------------------------
Data set            :     unknown
Weights matrix      :     unknown
Dependent Variable  :    ['zone']                Number of Observations:          82
Mean dependent var  :      5.1098                Number of Variables   :           4
S.D. dependent var  :      2.3571                Degrees of Freedom    :          78
R-squared           :      0.0212
Adjusted R-squared  :     -0.0164
Sum squared residual:     440.456                F-statistic           :      0.5641
Sigma-square        :       5.647                Prob(F-statistic)     :      0.6403
S.E. of regression  :       2.376                Log likelihood        :    -185.278
Sigma-square ML     :       5.371                Akaike info criterion :     378.555
S.E of regression ML:      2.3176                Schwarz criterion     :     388.182

------------------------------------------------------------------------------------
            Variable     Coefficient       Std.Error     t-Statistic     Probability
------------------------------------------------------------------------------------
            CONSTANT         5.10976         0.26242        19.47168         0.00000
             role_no         0.28044         0.29280         0.95777         0.34114
         subtrate_no         0.25556         0.27054         0.94461         0.34777
            group_no        -0.18625         0.29917        -0.62257         0.53538
------------------------------------------------------------------------------------

REGRESSION DIAGNOSTICS
MULTICOLLINEARITY CONDITION NUMBER           1.666

TEST ON NORMALITY OF ERRORS
TEST                             DF        VALUE           PROB
Jarque-Bera                       2          4.022           0.1338

DIAGNOSTICS FOR HETEROSKEDASTICITY
RANDOM COEFFICIENTS
TEST                             DF        VALUE           PROB
Breusch-Pagan test                3          1.450           0.6939
Koenker-Bassett test              3          2.701           0.4400

DIAGNOSTICS FOR SPATIAL DEPENDENCE
- SARERR -
TEST                           MI/DF       VALUE           PROB
Moran's I (error)              0.7500       11.027           0.0000
Lagrange Multiplier (lag)         1        115.226           0.0000
Robust LM (lag)                   1         16.117           0.0001
Lagrange Multiplier (error)       1        107.286           0.0000
Robust LM (error)                 1          8.177           0.0042
Lagrange Multiplier (SARMA)       2        123.403           0.0000

- Spatial Durbin -
TEST                              DF       VALUE           PROB
LM test for WX                    3         18.831           0.0003
Robust LM WX test                 3         10.891           0.0123
Lagrange Multiplier (lag)         1        115.226           0.0000
Robust LM Lag - SDM               1        107.286           0.0000
Joint test for SDM                4        126.117           0.0000
================================ END OF REPORT =====================================

```

The PROB column in the diagnostics for spatial dependence serves as indicators of spatial autocorrelation, with low values of less than 0.05 indicating evidence of spatial dependence , meaning the value of the variable is related to its neighbors . In this case, as expected, spatial autocorrelation will have an impact. Another value to note is the Akaike Information Criterion, which is an estimate of model prediction error, with lower values representing a better fit. Having spatial dependence requires a spatial lag regression that models the influence of nearest neighbors on the dependent variable (zone).

```
REGRESSION RESULTS
------------------

SUMMARY OF OUTPUT: MAXIMUM LIKELIHOOD SPATIAL LAG (METHOD = FULL)
-----------------------------------------------------------------
Data set            :     unknown
Weights matrix      :     unknown
Dependent Variable  :        zone                Number of Observations:          82
Mean dependent var  :      5.1098                Number of Variables   :           5
S.D. dependent var  :      2.3571                Degrees of Freedom    :          77
Pseudo R-squared    :      0.7813
Spatial Pseudo R-squared:  0.0003
Log likelihood      :   -137.9607
Sigma-square ML     :      1.3240                Akaike info criterion :     285.921
S.E of regression   :      1.1506                Schwarz criterion     :     297.955

------------------------------------------------------------------------------------
            Variable     Coefficient       Std.Error     z-Statistic     Probability
------------------------------------------------------------------------------------
            CONSTANT         1.03660         0.26098         3.97199         0.00007
             role_no         0.12832         0.14178         0.90509         0.36542
         subtrate_no        -0.05867         0.13106        -0.44768         0.65438
            group_no         0.04161         0.14492         0.28716         0.77399
              W_zone         0.20491         0.01009        20.30898         0.00000
------------------------------------------------------------------------------------

SPATIAL LAG MODEL IMPACTS
Impacts computed using the 'simple' method.
            Variable         Direct        Indirect          Total
             role_no         0.1283          0.0331          0.1614
         subtrate_no        -0.0587         -0.0151         -0.0738
            group_no         0.0416          0.0107          0.0523
================================ END OF REPORT =====================================
```

The R-squared value is much higher than the previous, indicating a better-fitting model. The coefficients for group and role have a positive correlation, while substrate has a negative direction. The AIC for this model is also lower than the previous. In contrast to the spatial lag, spatial error regression models spatial interactions with the independent variables, assuming that errors are correlated with nearest neighbors.

```
REGRESSION RESULTS
------------------

SUMMARY OF OUTPUT: ML SPATIAL ERROR (METHOD = full)
---------------------------------------------------
Data set            :     unknown
Weights matrix      :     unknown
Dependent Variable  :        zone                Number of Observations:          82
Mean dependent var  :      5.1098                Number of Variables   :           4
S.D. dependent var  :      2.3571                Degrees of Freedom    :          78
Pseudo R-squared    :      0.0030
Log likelihood      :   -137.9057
Sigma-square ML     :      1.3233                Akaike info criterion :     283.811
S.E of regression   :      1.1504                Schwarz criterion     :     293.438

------------------------------------------------------------------------------------
            Variable     Coefficient       Std.Error     z-Statistic     Probability
------------------------------------------------------------------------------------
            CONSTANT         5.69163         0.70338         8.09187         0.00000
             role_no         0.02680         0.12443         0.21541         0.82945
         subtrate_no        -0.11136         0.13256        -0.84007         0.40087
            group_no         0.13053         0.13940         0.93640         0.34907
              lambda         0.20471         0.01020        20.07840         0.00000
------------------------------------------------------------------------------------
================================ END OF REPORT =====================================

```

The R-squared value is much higher than the previous, indicating a better-fitting model. The coefficients for group and role have a positive correlation, while substrate has a negative direction. The AIC for this model is also lower than the previous. In contrast to the spatial lag, spatial error regression models spatial interactions with the independent variables, assuming that errors are correlated with nearest neighbors .

<figure>
    <center>
    <img src="../assets/mycota-12.png">
    <center>
</figure>

The map shows the model predicting changes around the park, with some zones having a darker shade. When compared to the actual map, significant changes are occurring in Zone 5, with a reduction, and in Zone 10, with an increase. The results show changes , but the factors contributing to the change are unknown.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Geographically Weighted Regression

A geographically weighted model applies the weighted matrix to construct separate equations for each independent explanatory variable by incorporating its nearest neighbor . The autocorrelation patterns in the residuals showed some changes around the park; this model can provide insights into which variable is having a greater impact.

<figure>
    <center>
    <img src="../assets/mycota-13.png">
    <center>
</figure>

It was noted previously that substrates having a negative correlation direction with lesser changes towards the higher zones. The model indicates that Zone 1 is predicted to have the most change with an increase in the types of substrates found with fungi. Zone 8 is moving in the opposite direction.

<figure>
    <center>
    <img src="../assets/mycota-14.png">
    <center>
</figure>

Groups have a positive correlation direction with a stronger relationship toward the higher zones. The model predicts an increase in groups for Zone 7. This zone currently consists of six group types, leaving Jelly as the only type predicted to be located in this zone. Zone 3 is predicted to show the largest reduction in groups.

<figure>
    <center>
    <img src="../assets/mycota-15.png">
    <center>
</figure>

The characteristics that define the role of fungi in this analysis are either symbiont, recycler, or parasite. The role had the strongest relationship with zones in the correlation matrix and is also strongly coupled with the group. The model has predicted no changes for this variable. The only parasitic group in the dataset is Rust, which has little impact (could not be quantified) on the park.

A symbiotic relationship with fungi is critical for keeping ecosystems in balance and is therefore seen as ' Darwin's Blind Spot' within the patterns of evolution. However, this mutual partnership can shift to parasitism when conditions change. In this analysis , the model predicts no change in the role of fungi, suggesting that the park is in a healthy condition with a balanced ecosystem.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### References
<ol>
  <li>Minn, M. (2023). Regression Analysis in Python. https://michaelminn.net/tutorials/python-regression/index.html Accessed 1 May 2024.</li>
  <li>Seifert, K. (2022). The hidden kingdom of fungi. Australia: University of Queensland Press.</li>
  <li>Robinson, R. (2022). Fungi of the south-west forests. Australia: Department of Biodiversity Conservation and Attractions.</li>
  <li>SJ. Rey, D Arribas-Bel, LJ. Wolf (2020). Geographic Data Science with Python. https://geographicdata.science/book/intro.html. Accessed 12 Dec 2024.</li>
</ol>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

