import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Total satır:', len(df))
print('Eğitim satırı:', len(X_train))
print('Test satırı:', len(X_test))
