
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader

def create_dataloader_mars(opt):
    from data.mars_data_loader import MarsDatasetDataLoader
    data_loader = MarsDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader

def create_comp_data_loader(opt, img_path):
    # use the img_path to read comp image
    from data.comp_data_dataloader import CompImageDataLoader
    data_loader = CompImageDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt, img_path)
    return data_loader


def create_dataloader_mars_nonorm(opt):
    from data.mars_data_loader import MarsDatasetDataLoaderNoNorm
    data_loader = MarsDatasetDataLoaderNoNorm()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader

