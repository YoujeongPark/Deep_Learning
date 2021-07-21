import pandas as pd
red = pd.read_csv("winequality-red.csv");
white = pd.read_csv("winequality-white.csv");
red['type'] = 0;
white['type'] = 0;

print(red.describe());
print(white.describe());


wineData = pd.concat([red,white]);
print(wineData.describe());

wineShuffleData = wineData.sample(frac=1);
wineNumpyData = wineShuffleData.to_numpy();
print(wineNumpyData[:5])

