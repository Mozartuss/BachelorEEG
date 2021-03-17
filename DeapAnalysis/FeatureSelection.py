# Feature Extraction with PCA
from os import sep

from pandas import read_csv
from sklearn.decomposition import PCA

# load data
data = ".." + sep + ".." + sep + ".." + sep + "DEAP Dataset" + sep + "Preprocessed Datasets" + sep + "data_preprocessed_python" + sep + "Participants" + sep + "CompleteFeatures" + sep + "CompleteFeatures.csv"
label = ".." + sep + ".." + sep + ".." + sep + "DEAP Dataset" + sep + "Preprocessed Datasets" + sep + "data_preprocessed_python" + sep + "Participants" + sep + "CompleteFeatures" + sep + "CompleteFeaturesLabels.csv"

data_names = ["Mean", "Std", "Afd", "Norm_Afd", "Asd", "Norm_Asd", "Variance", "Peak-to-peak", "Skewness", "Kurtosis",
         "Aprox_BP_Theta", "Aprox_BP_LowerAlpha", "Aprox_BP_UpperAlpha", "Aprox_BP_Beta", "Aprox_BP_Gamma",
         "Linelength", "Hjorth_Activity", "Hjorth_Morbidity", "Hjorth_Complexity", "Petrosian", "Hurst"]
label_names = ["arousal", "valence", "dominance", "liking"]
features = read_csv(data, names=data_names)
features.columns = features.iloc[1]
features.drop(features.index[1])
labels = read_csv(label, names=label_names)
labels.columns = labels.iloc[1]
labels.drop(labels.index[1])
X = features.values[1:,:]
Y = labels.values[1:, 0]

# feature extraction
pca = PCA(n_components=30)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)
