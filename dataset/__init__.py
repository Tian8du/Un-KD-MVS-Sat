import importlib


def find_dataset_def(dataset, dataset_name):
    if dataset_name == "US3D":
        if dataset == "rpc":
            module_name = 'dataset.US3D_dataset'
            module = importlib.import_module(module_name)
        elif dataset == "pinhole":
            module_name = 'dataset.virdataset'
            module = importlib.import_module(module_name)
        else:
            raise Exception("Not implemented yet")
        return getattr(module, "US3DDataset")
    else:
        if dataset == "rpc":
            module_name = 'dataset.satmvsdataset'
            module = importlib.import_module(module_name)
        elif dataset == "pinhole":
            module_name = 'dataset.virdataset'
            module = importlib.import_module(module_name)
        else:
            raise Exception("Not implemented yet")
        return getattr(module, "MVSDataset")
