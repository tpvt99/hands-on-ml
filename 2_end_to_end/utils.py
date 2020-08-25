from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def find_metrics(model, data, labels):
    predictions = model.predict(data)
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mean_squared_error(labels, predictions))

    custom_mae = np.mean(np.abs(labels - predictions))
    custom_rmse = np.sqrt(np.mean(np.square(labels - predictions)))
    print("MAE: {:.3f}, RMSE: {:.3f}, Custom MAE: {:.3f}, Custom RMSE: {:.3f}".format(
        mae, rmse, custom_mae, custom_rmse
    ))
    return mae, rmse

def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", np.mean(scores))
    print("Standard deviation: ", np.std(scores))