from src.preprocess import load_split_data
from src.models import get_models,train_evalute_models


def main():
    X_train,X_test,y_train,y_test = load_split_data(r'C:\Users\Murat\OneDrive - hacettepe.edu.tr\Masaüstü\titanic_project\data\titanic_data_cleaned.csv',
                                                    targetcolumn='Survived')
    model = get_models()

    results = train_evalute_models(model,X_train,X_test,y_train,y_test)
    for name, acc in results.items():
        print(f'{name} Model Accuracy is %{acc*100}')


if __name__ == "__main__":
    main()