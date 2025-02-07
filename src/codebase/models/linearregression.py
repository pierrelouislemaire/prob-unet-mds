import numpy as np
import torch
import time
from sklearn.linear_model import LinearRegression
import codebase.data_utils as du
import codebase.train_utils as tu

def get_lr_preds(data_path="data/"):

    args = tu.get_args()
    trainset = du.climex2torch(datadir=args.datadir, years=args.years_train, coords=args.coords, 
                               lowres_scale=args.lowres_scale, transfo=True, type="lr_to_hr")
    valset = du.climex2torch(datadir=args.datadir, years=args.years_val, coords=args.coords, 
                               lowres_scale=args.lowres_scale, transfo=True, type="lr_to_hr")
    megatrainset = du.climex2torch(datadir=args.datadir, years=args.years_megatrain, coords=args.coords,
                                   lowres_scale=args.lowres_scale, transfo=True, type="lr_to_hr")
    testset = du.climex2torch(datadir=args.datadir, years=args.years_test, coords=args.coords, 
                               lowres_scale=args.lowres_scale, transfo=True, type="lr_to_hr")
    
    hr_res = args.resolution[0]
    lr_res = hr_res // args.lowres_scale

    x_train = np.zeros(((365*len(args.years_train)), 1+lr_res*lr_res*3), dtype=np.float32)
    y_train = np.zeros(((365*len(args.years_train)), hr_res*hr_res*3), dtype=np.float32)

    x_val = np.zeros(((365*len(args.years_val)), 1+lr_res*lr_res*3), dtype=np.float32)
    y_val = np.zeros(((365*len(args.years_val)), hr_res*hr_res*3), dtype=np.float32)

    x_megatrain = np.zeros(((365*len(args.years_megatrain)), 1+lr_res*lr_res*3), dtype=np.float32)
    y_megatrain = np.zeros(((365*len(args.years_megatrain)), hr_res*hr_res*3), dtype=np.float32)

    x_test = np.zeros(((365*len(args.years_test)), 1+lr_res*lr_res*3), dtype=np.float32)
    y_test = np.zeros(((365*len(args.years_test)), hr_res*hr_res*3), dtype=np.float32)

    for t in range(365*len(args.years_train)):

        data_t = trainset.__getitem__(t)
        timestep, input, target = data_t["timestamps"].numpy(), data_t["inputs"].numpy(), data_t["targets"].numpy()

        input_pr, input_tasmin, input_tasmax = input[0], input[1], input[2]
                
        x_train[t, 0] = timestep
        x_train[t, 1:1+lr_res*lr_res] = input_pr.flatten()
        x_train[t, 1+1*lr_res*lr_res:1+2*lr_res*lr_res] = input_tasmin.flatten()
        x_train[t, 1+2*lr_res*lr_res:1+3*lr_res*lr_res] = input_tasmax.flatten()
        y_train[t] = target.flatten()


    for t in range(365*len(args.years_val)):

        data_t = valset.__getitem__(t)
        timestep, input, target = data_t["timestamps"].numpy(), data_t["inputs"].numpy(), data_t["targets"].numpy()

        input_pr, input_tasmin, input_tasmax = input[0], input[1], input[2]
   
        x_val[t, 0] = timestep
        x_val[t, 1:1+lr_res*lr_res] = input_pr.flatten()
        x_val[t, 1+lr_res*lr_res:1+2*lr_res*lr_res] = input_tasmin.flatten()
        x_val[t, 1+2*lr_res*lr_res:1+3*lr_res*lr_res] = input_tasmax.flatten()
        y_val[t] = target.flatten()

    for t in range(365*len(args.years_megatrain)):

        data_t = megatrainset.__getitem__(t)
        timestep, input, target = data_t["timestamps"].numpy(), data_t["inputs"].numpy(), data_t["targets"].numpy()

        input_pr, input_tasmin, input_tasmax = input[0], input[1], input[2]
                
        x_megatrain[t, 0] = timestep
        x_megatrain[t, 1:1+lr_res*lr_res] = input_pr.flatten()
        x_megatrain[t, 1+1*lr_res*lr_res:1+2*lr_res*lr_res] = input_tasmin.flatten()
        x_megatrain[t, 1+2*lr_res*lr_res:1+3*lr_res*lr_res] = input_tasmax.flatten()
        y_megatrain[t] = target.flatten()

    for t in range(365*len(args.years_test)):

        data_t = testset.__getitem__(t)
        timestep, input, target = data_t["timestamps"].numpy(), data_t["inputs"].numpy(), data_t["targets"].numpy()

        input_pr, input_tasmin, input_tasmax = input[0], input[1], input[2]
   
        x_test[t, 0] = timestep
        x_test[t, 1:1+lr_res*lr_res] = input_pr.flatten()
        x_test[t, 1+lr_res*lr_res:1+2*lr_res*lr_res] = input_tasmin.flatten()
        x_test[t, 1+2*lr_res*lr_res:1+3*lr_res*lr_res] = input_tasmax.flatten()
        y_test[t] = target.flatten()

    print("Training data shape: ", x_train.shape, y_train.shape)
    print("Validation data shape: ", x_val.shape, y_val.shape)
    print("Megatrain data shape: ", x_megatrain.shape, y_megatrain.shape)
    print("Test data shape: ", x_test.shape, y_test.shape)

    # Defining the linear model

    lr = LinearRegression(n_jobs=-1)

    # Training the linear models

    print("Training the linear model")
    t = time.time()
    lr.fit(x_megatrain, y_megatrain)
    pr_time = time.time()-t
    print("Time to train the linear model: ", pr_time, "s")

    y_train = y_train.reshape(-1, 3, hr_res, hr_res)
    y_val = y_val.reshape(-1, 3, hr_res, hr_res)
    y_megatrain = y_megatrain.reshape(-1, 3, hr_res, hr_res)
    y_test = y_test.reshape(-1, 3, hr_res, hr_res)

    train_preds = lr.predict(x_train).reshape(-1, 3, hr_res, hr_res)
    val_preds = lr.predict(x_val).reshape(-1, 3, hr_res, hr_res)
    megatrain_preds = lr.predict(x_megatrain).reshape(-1, 3, hr_res, hr_res)
    test_preds = lr.predict(x_test).reshape(-1, 3, hr_res, hr_res)

    train_preds = trainset.invstand_residual(torch.tensor(train_preds))
    val_preds = valset.invstand_residual(torch.tensor(val_preds))
    megatrain_preds = trainset.invstand_residual(torch.tensor(megatrain_preds))
    test_preds = testset.invstand_residual(torch.tensor(test_preds))

    return {"train_preds": torch.tensor(train_preds), 
            "val_preds": torch.tensor(val_preds),
            "megatrain_preds": torch.tensor(megatrain_preds),
            "test_preds": torch.tensor(test_preds)}
