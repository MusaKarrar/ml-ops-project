from tests import _N
import torch, os

def check_dataset(data_file, data_group_name, is_train_file):
    data_file_shape = data_file.shape

    #assert but also print filename as error message
    if is_train_file:
        assert len(data_file_shape) == 4, "Expected 4 dimensions, got %s in %s" % (len(data_file_shape), data_group_name)
        assert data_file_shape[1:] == (4, 160, 106), "Expected (4, 160, 106), got %s in %s" % (data_file_shape[1:], data_group_name)
    else:
        assert len(data_file_shape) == 1, "Expected 1 dimensions, got %s in %s" % (len(data_file_shape), data_group_name)
    
    assert data_file.dtype == torch.float32, "Expected float32, got %s in %s" % (data_file.dtype, data_group_name)

    return data_file_shape[0]

def test_data():
    """
    test that data is loaded correctly
    """
    dataset_names = os.listdir("data/processed/")
    train_images = [torch.load("data/processed/" + dataset_names[i]) for i in range(len(dataset_names)) if 'train_images' in dataset_names[i]]
    train_targets = [torch.load("data/processed/" + dataset_names[i]) for i in range(len(dataset_names)) if 'train_targets' in dataset_names[i]]
    test_images = [torch.load("data/processed/" + dataset_names[i]) for i in range(len(dataset_names)) if 'test_images' in dataset_names[i]]
    test_targets = [torch.load("data/processed/" + dataset_names[i]) for i in range(len(dataset_names)) if 'test_targets' in dataset_names[i]]


    for index, data_group in enumerate([train_images,test_images]):
        N = _N[index]
        group_name = ['training images', 'test images'][index]

        observations = 0
        for data_file in data_group:
            observations += check_dataset(data_file, group_name, is_train_file=True)
        
        assert observations == N, "Expected %s observations, got %s in %s" % (N, observations, data_group)
    
    for index, data_group in enumerate([train_targets,test_targets]):
        N = _N[index]
        group_name = ['training targets', 'test targets'][index]

        observations = 0
        for data_file in data_group:
            observations += check_dataset(data_file, group_name, is_train_file=False)
        
        assert observations == N, "Expected %s observations, got %s in %s" % (N, observations, data_group)

