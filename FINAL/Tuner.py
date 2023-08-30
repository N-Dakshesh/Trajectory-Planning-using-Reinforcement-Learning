"""Use optuna to test values of different hyperparameters"""

import optuna
import Training_loop as rl



def objective(trial):
    learning_rate = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    entropy_coeff = trial.suggest_float("entropy_coeff", 1e-2, 0.5, log=True)
    hidden = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128])
    layers = trial.suggest_categorical("num_layers", [1, 2, 3, 4, 5])
    gamma = trial.suggest_float("gamma", 0.5, 0.999)
    end_reward = trial.suggest_float("end_reward", 0, 5)
    stop_time = trial.suggest_float("cuttoff_time", 10, 20)
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 3, 4, 5, 6, 7, 8])

    final_time = rl.training(
        learning_rate,
        entropy_coeff,
        hidden,
        layers,
        gamma,
        end_reward,
        stop_time,
        batch_size,
    )

    #function returns final time, best inputs and best outputs.
    return final_time[0]


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=192, n_jobs=-1)

print("Number of finished trials: ", len(study.trials))
print("Best trial: ")
trial = study.best_trial

print("Value: ", trial.value)
print("Params: ")

for key, value in trial.params.items():
    print(f"   {key}: {value}")

file = open("Tuner_ideal.txt", "w")
file.write(trial.value)
file.close()
