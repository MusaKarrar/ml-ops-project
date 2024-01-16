if __name__ == "__main__":
    # Get the data and process it
    import torch, os, time
    from torchvision import transforms
    import numpy as np

    import torchvision.transforms.functional as TF
    from torchvision import transforms
    from PIL import Image 
    number_of_bands = 4
    # Specify the path to the folder containing your image data

    # training1 test1
    # finaltraining finaltest

    # Uncomment below for non-mac 
    # data_traning = r'data\raw\training5'
    # data_test = r'data\raw\test5'

    # Make it work for mac, remember to be in main directory
    data_training = 'data/raw/training5'
    data_test = 'data/raw/test5'


    # Create empty lists to store the data and labels
    X_train, X_test = np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    all_type_img_train,all_type_img_test = np.array([[]], dtype=np.float32), np.array([[]], dtype=np.float32)
    #all_type_img_train,all_type_img_test = np.array([[]], dtype=np.float32), np.array([[]], dtype=np.float32)
    imageCatagory = 0
    y_train, y_test = np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # Loop through the image files in the folder
    for filename in os.listdir(data_training):

        imageCatagory = imageCatagory+1
        if filename.endswith('.tif'):  # Adjust the file format to match your data
            # Load the image using OpenCV
            all_type_img_train = np.array([[]], dtype=np.float16)

            img = Image.open(os.path.join(data_training, filename))
            TF.to_tensor(img)
            transforms.ToTensor()(img)


            if img is not None:
                all_type_img_train = np.append(
                    all_type_img_train, img)

                del img
            else:
                print("train", filename)
            # Extract the label from the filename or use a different method to assign labels
            # For example, you can name your files like "class_label_image1.tif"
            # Extract the class label from the filename¨
        X_train = np.append(X_train, (all_type_img_train))
        del all_type_img_train

        if imageCatagory == number_of_bands:
            #        print("the last 4 images have been catagolize")
            imageCatagory = 0
            if 'zero' in filename:
                y_train = np.append(y_train, 0.)
            elif '100kg' in filename:
                y_train = np.append(y_train, 100.) #100kg
            elif '200kg' in filename:
                y_train = np.append(y_train, 200.) #200kg
            elif '300kg' in filename:
                y_train = np.append(y_train, 300.) #300kg

    # print("the training set have been loaded into an array")
    for filename in os.listdir(data_test):
        all_type_img_test = np.array([[]], dtype=np.float16)

        imageCatagory = imageCatagory+1

        if filename.endswith('.tif'):  # Adjust the file format to match your data
            # Load the image using OpenCV

            img = Image.open(os.path.join(data_test, filename))
            TF.to_tensor(img)
            transforms.ToTensor()(img)
            if img is not None:

                all_type_img_test = np.append(
                    all_type_img_test, img)
    #            print(f"picture {filename} has been loaded")
                del img

            else:
                print("test", filename)

        X_test = np.append(X_test, (all_type_img_test ))
        del all_type_img_test

        if imageCatagory == number_of_bands:
            imageCatagory = 0
    #        print("the last 4 images have been catagolize")
            if 'zero' in filename:
                y_test = np.append(y_test, 0.)
            elif '100kg' in filename:
                y_test = np.append(y_test, 100.) #100kg
            elif '200kg' in filename:
                y_test = np.append(y_test, 200.) #200kg
            elif '300kg' in filename:
                y_test = np.append(y_test, 300.) #300kg

    #convert to tensor
    X_train = torch.from_numpy(X_train).reshape(120,4,160,106)
    X_test = torch.from_numpy(X_test).reshape(40,4,160,106)
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    torch.save(X_train, "data/processed/train_images.pt")
    torch.save(X_test, "data/processed/test_images.pt")
    torch.save(y_train, "data/processed/train_targets.pt")
    torch.save(y_test, "data/processed/test_targets.pt")
    

    pass
