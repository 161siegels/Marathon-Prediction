# Marathon-Prediction
Bayesian Hierarchical Models for Marathon Prediction

The Boston Marathon qualifying standards have received quite a bit of scrutiny over the years. Many members of the running community question the fairness of the qualifying times for certain male and female age groups. Others have suggested that certain races which are eligible to qualify a runner are much easier than others. The current Boston qualifying standard for males and females are split into 11 age groups with standards that increase at an increasing rate.

As a senior at UNC Chapel Hill in 2020 (with guidance from Dr. Richard Smith), I began a study which utilized a linear mixed-effects model to produce aging curves for male and female marathon runners [1]. This research modeled each marathon location separately which inhibited the reliability and generalizability of the results. Furthermore, it is unable to control for the location of the race in evaluating the strength of a runner's performance.

This study extends my prior research by fitting a Bayesian Hierarchical Model (BHM) to predict marathon performance from 11 of the most popular marathons in the United States. The input to the algorithm is a tabular dataset of runner performances in these races from 2000-2019. The dataset includes gender, age, marathon location, and marathon year. I fit a separate model for male and females, treating a cubic spline on age as a fixed effect to account for the non-linear relationship between age and finish time. I model the individual runner ability, the marathon location, and the marathon location-year as random effects. With this specification, I fit a BHM using the numpyro package to output a distribution of predicted marathon times (in minutes) for each runner performance. For baseline comparisons, I also fit a ridge regression and CatBoost regressor on this same dataset.