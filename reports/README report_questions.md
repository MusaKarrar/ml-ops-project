**Group information**

**Question 1**

**Enter the group number you signed up on <learn.inside.dtu.dk>**

Answer:

--- Group 25 ---

**Question 2**

**Enter the study number for each member in the group**

Example:

Answer:

--- S204161, s220044 ,s230432 ,s223092, s184213 ---

**Question 3**

**What framework did you choose to work with and did it help you complete the project?**

Answer length: 100-200 words.

Example: *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the* *package to do ... and ... in our project*.

Answer:

--- We use pyTorch for training our model. We were able to create neural networks (a CNN and vision transformer (structure from <https://arxiv.org/abs/2010.11929>)). We used the package nn from torch for creating the neural networks. With Torch, we were able to create Transformer blocks and then let the number of transformer blocks be a hyperparameter - this would have been much more difficult if coding a ViT using numpy only. Optimizers, loss functions & dataloaders were also predefined from torch. Einops was used for the vision transformer. We also used W&B for logging data we get from the models. ---



**Coding environment**

In the following section we are interested in learning more about you local development environment.

**Question 4**

**Explain how you managed dependencies in your project? Explain the process a new team member would have to go** **through to get an exact copy of your environment.**

Answer length: 100-200 words

Example: *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a* *complete copy of our development environment, one would have to run the following commands*

Answer:

--- We used conda for managing our dependencies. The first thing everybody did was to create a conda environment for this project & this helped with avoiding conflict errors. We used ‘pipreqs –force’ to generate a requirements.txt, which we then went through to make sure that the dependencies looked OK.. The new group member needs to have access to the git repository first, then clone the project. Then for the new group member to get a copy, we would use ‘pip install -r requirements.txt’ to ensure they have the right dependencies to run all the code. 

The exact commands are:

#create conda env

conda create –name ml-ops-env python=3.10

conda activate ml-ops-env

git clone https://github.com/MusaKarrar/ml-ops-project.git

pip install -r requirements.txt

Alternatively, to get the EXACT copy one would pull the docker images we have built (on Cloud build).

\---

**Question 5**

**We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your** **code. Did you fill out every folder or only a subset?**

Answer length: 100-200 words

Example: *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder* *because we did not use any ... in our project. We have added an ... folder that contains ... for running our* *experiments.* Answer:

--- The scripts we run to perform the experiment are in a src folder, while the data and results are outside of it such as figures from visualizations and confidential raw and processed data. This data cannot be acquired through the public github repository and can only be acquired with an agreement with the company providing the data. The results from wandb runs are also outside the src folder. We filled out configs, tests, updated dockerfiles, added folder for W&B info… The folders we did not fill out are notebooks, documentation for src (docs folder). We have also added a “bfg.jar” file outside the src folder, which is the repo cleaner progam just in case that the confidential data gets committed to the github repository to remove past commits. ---

**Question 6**

**Did you implement any rules for code quality and format? Additionally, explain with your own words why these** **concepts matters in larger projects.**

Answer length: 50-100 words.

Answer:

\---

We agreed that we would try to be PEP8 compliant (but we are mere mortals…). Because it can cause problems when many people work on a big project. If unlucky, one can spend a whole day debugging and find out that the error is because a capital S should have been used somewhere instead of a lowercase s amongst thousands of lines of code. We implemented typing in CNN class but ran out of time.   ---

**Version control**

In the following section we are interested in how version control was used in your project during development to corporate and increase the quality of your code.

**Question 7**

**How many tests did you implement and what are they testing in your code?**

Answer length: 50-100 words.

Example: *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our* *application but also ... .*

Answer:

--- We have implemented 3 tests. In test\_data,py we test whether the data has been loaded in correctly, by asserting if the shape is (N\_obs,4,160,106). Where N\_obs is a list defined in the config file [N\_train,N\_test] =  [120,40]. The test is not hardcoded to a single file, so assuming all training\_images are named with the prefix ‘training\_images’, then test\_data.py would make sure that all N\_train observations exist across all the training files. same for test files. test\_construction.py makes sure that the image shape is divisible by the patch shape in both image dimensions (these shapes are defined in config\_model.yaml). test\_training\_model.py makes sure that the loss of the trained model on test set is not unrealistically bad. ---

**Question 8**

**What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close** **to), would you still trust it to be error free? Explain you reasoning.**

Answer length: 100-200 words.

Example: \*The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our \*\* *code and even if we were then...*

Answer:

--- Total coverage is 76%. 2 testfiles have 100% and the 3. tesfile has 95%. ‘coverage report -m’ reveals that the line in test\_training.py, which is missing is line 41 where the CNN is defined inside an if-statement. Line 41 is not used because ViT is chosen and so line 41 is not triggered and coverage drops. model.py has coverage of 70% (both models are defined in same python script, CNN function not triggered) and predict\_model.py has coverage 27%. If coverage was 100%, then it does not mean code is error free - it only means that all lines of codes are triggered and used for something. Coverage would still be 100% for example if the test accuracy was incorrectly divided by a random number. ---

**Question 9**

**Did you workflow include using branches and pull requests? If yes, explain how? If not, explain how branches and** **pull request can help improve version control.**

Answer length: 100-200 words.

Example: *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in* *addition to the main branch. To merge code we ...*

Answer:

--- Everyone had a branch they worked on, so we had 6 branches in total including the master branch. Some used the website to merge with after pulling and pushing to their own branch, while others used VS Code for pulling, pushing and merging. VS Code had a merging tool that was utilized by a few members. One member used github desktop for pulling, pushing and merging. For too complicated/big changes, the terminal or VS Code was used for merging. For everyone to be up to date, everyone will be pulling from the master branch, as that is the branch we ensure to be our up to date that does not have any major issues with the files and code. ---

**Question 10**

**Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version** **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**

Answer length: 100-200 words.

Example: *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our* *pipeline*

Answer:

---  We did make use of DVC to manage the data and models in our project. It helped us resolve the issue of uploading confidential data to the github repository, since we wanted the github repository to be publicly available, considering it is a project assignment, but not the data. Therefore the data is stored through Google Drive privately through DVC. It was however not needed for storage efficiency, which is what DVC is great for. Well, it does help in terms of reproducibility, as it ensures the same data is used all the time throughout the experiments of models we have. For the future, it can also help us track the data, in case of changes of data, that is going to be made, or perhaps when we have a lot more models or data with larger file sizes, DVC is pretty useful. We then used google cloud buckets instead. ---

**Question 11**

**Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test** **multiple operating systems, python versions etc. Do you make use of caching? Feel free to insert a link to one of** **your github actions workflow.**

Answer length: 200-300 words.

Example: *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running* *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here:*

Answer:

--- We have* tested different operating systems. MacOS and Windows were tested and both operating systems could run the code without problems. We tried to test it on Ubuntu, but there are some errors, as it should have been done within minutes. Currently, we can conclude that it is not working properly on Ubuntu for now., Our workflows are defined in our YAML file. We did not upload the data to github, which is why some of the workflow tests failed, specifically it tries to find the data in test\_data.py, which is one of our 3 unit testing files (we also have test\_construction.py and test\_training.py). The YAML of our problem can be found here.

We used linting (with ruff-package) on our src files & fixed most of the linting issues.

Docker is also set up with version control, so every time an update/commit is made & when a VM is enabled in the Compute Engine, then a new docker image is built and saved to the docker registry in Cloud build.

We only tested for python version 3.10/3.11, since most of the members either had python version 3.10 or 3.11. The most important aspect is to ensure that everyone could work on their python environment.

` `---

**Running code and tracking experiments**

In the following section we are interested in learning more about the experimental setup for running your code and especially the reproducibility of your experiments.

**Question 12**

**How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would** **run a experiment.**

Answer length: 50-100 words.

Example: *We used a simple argparser, that worked in the following way: python my\_script.py --lr 1e-3 --batch\_size 25*

Answer:

--- While the approach to be followed by default was to use an argparser, we created a configuration file called config\_model.yaml, created for listing the hyperparameters of both models used in our code i.e. the Vision Transformer as well as the CNN. While some hyperparameters were kept specific to each model and defined under their respective sections i.e. hyperparameters\_ViT and hyperparameters\_CNN, there were a few that could be defined under defaults as they were common to both. OmegaConf was then utilized in the training code(training\_model.py) to call each one of these variables/constants whenever necessary.

The following code snippet illustrates how OmegaConf was utilized for using hyperparameters while decoupling them from the mainstream model definition code:

`  `![](Aspose.Words.25da072a-7e40-419c-b7c8-8fb9d2a49757.001.png)                                        ---

**Question 13**

**Reproducibility of experiments are important. Related to the last question, how did you secure that no information** **is lost when running experiments and that your experiments are reproducible?**

Answer length: 100-200 words.

Example: *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment* *one would have to do ...*

Answer:

--- We made use of OmegaConf to collect all information about hyperparameters and utilize them via the YAML configuration files to a format acceptable in the python training code. W&B was used to log these runs and the config file is also supplied to W&B for information storage. To reproduce the experiment, random seeds are used & the experiment should just be run using the supplied YAML file. 

This approach of utilizing the configuration  files ensured that any experiments done were not lost, hence fulfilling the reproducibility criteria. W&B also came in handy when being used for the aforementioned purpose for storing the information from the configuration file and hence this problem was tackled quite efficiently as per our beliefs.

\---

**Question 14**

**Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking** **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take** **inspiration from [this figure](https://github.com/SkafteNicki/dtu_mlops/blob/main/reports/figures/wandb.png). Explain what metrics you are tracking and why they are** **important.**

Answer length: 200-300 words + 1 to 3 screenshots.

Example: *As seen in the first image when we have tracked ... and ... which both inform us about ... in our experiments.* *As seen in the second image we are also tracking ... and ...*

Answer:

--- ![](Aspose.Words.25da072a-7e40-419c-b7c8-8fb9d2a49757.002.png)The most important metrics to track are the training and validation loss, which can be seen in the two figures below the one big figure, since the training loss is decreasing over several iterations/steps, it tells us that the model is learning to fit better with the training data, which is a positive sign. With the validation loss also decreasing for each epoch (up to 5), it means it is doing better each time to generalize with unseen data. We can assess the difficulty of the problem in terms of classification by encoding in t-distributed stochastic neighbor encoding(TSNE), reducing the dimensionality to 2 and then looking at the overlap between the different classes in the first big picture (nitrogen content - 0,100,200,300), which is the image we have logged.

![](Aspose.Words.25da072a-7e40-419c-b7c8-8fb9d2a49757.003.png)

As we can see in the second image with the bands of predicted data of nitrogen levels, the bands give us a clear picture of how all the numbers are clumb together within their respective nitrogen levels, which are at 0, 100, 200 and 300 and help us classify them. 

We did not perform hyperparameters sweep, since we stumbled upon issues performing them. 

` `---

**Question 15**

**Docker is an important tool for creating containerized applications. Explain how you used docker in your** **experiments? Include how you would run your docker images and include a link to one of your docker files.**

Answer length: 100-200 words.

Example: *For our project we developed several images: one for training, inference and deployment. For example to run the* *training docker image:. Link to docker file:* 

Answer:

--- We developed a docker image for training, one for inference in the cloud using FastAPI and one for training in the cloud (cloud\_build.yaml). We setup the train dockerfile in the cloud, such that it every time ‘git commit’ is ran on main branch, then docker builds and sends to the container registry. To get pull the docker image one can pull from the docker registry and run:

docker pull gcr.io/nifty-atlas-410710/train:latest

docker run --name EX1 gcr.io/nifty-atlas-410710/train:latest

However, since the docker image includes the data, which is under an NDA, the docker images are set to private. It does work, documentation:

![](Aspose.Words.25da072a-7e40-419c-b7c8-8fb9d2a49757.004.png)

As seen above, the pull works if one is permitted access to the docker image. Below, an unauthorized user tries to pull and it doesn’t work,![](Aspose.Words.25da072a-7e40-419c-b7c8-8fb9d2a49757.005.png)

` `---

**Question 16**

**When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you** **try to profile your code or do you think it is already perfect?**

Answer length: 100-200 words.

Example: *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling* *run of our main code at some point that showed ...*

Answer:

--- Debugging was dependent on group member. One used the debug&run in VS Code, defining breakpoints & inspecting variables. We did some profiling on the train\_model.py file both with cprofiling and pytorch profiling. Through the tensorboard, the model performance could be visualized, which can be launched directly through VS code. We can then view the results and analyze the performance of the model and identify the bottlenecks and issues that may affect the training speed. For example, on an apple m2 chip that utilities CPU, we found out that it had convolution\_backward called 48 times, and a total time spent was at 34.77 seconds aka (34765607 microseconds), which is something that can be optimized, by looking into the convolutional layers. Optimization was not highly prioritized as the time was spent on working on the other parts of the project, but it proves that profiling is a great tool for model optimization. ---

**Working in the cloud**

In the following section we would like to know more about your experience when developing in the cloud.

**Question 17**

**List all the GCP services that you made use of in your project and shortly explain what each service does?**

Answer:

\--- 

*1.Compute Engine: Engine is used for  running the virtual machines (VM) on Google Cloud platform and it  provides a flexible VM’s base one specific requirements including choice of CPU, memory and storage it also to customs images and it allows to package and deploy our software configuration and application to package.* 

*2.Storages, Buckets: Storage is the service used to store/retrieve data and bucket is the container used for* storing our data in the cloud (GCP)

*3.Cloud Build: Is the google cloud build that contains our continuous integration and automates the build, test and deployment of our stuff.*

*4.Cloud Triggers: The cloud triggers are the google cloud functions. These are used for example in cases, where we upload the data into the storage bucket, it will automatically trigger the data processing tasks, which generates our processed data. This data is confidential, which is why it is stored here.*  

*5. Vertex AI: it customs the job by running our custom training code including worker pool, machine types and do setting related to our Python training application and custom container*

\--- 

**Question 18**

**The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs** **you used?**

Answer length: 100-200 words.

Example: *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the* *using a custom container: ...*

Answer:

\---

We rented out VMs using Compute Engine. We used a low-power/low-cost VM because our model size & dataset is not very big, so we did not need much compute and memory. For our project, we used series E2 (linux-amd64) which is a low cost and day to day computing with a small memory range between 1-128GB. It also has a feature of confidential service and is able to deploy a container image in this VM instance. We utilized features of VM instances tailored to our project requirements; we opted VM instance with e2-medium 1-2 vCPU and memory 4GB, as it mentioned before our dataset is small, so we used the lowest memory range. We did make sure to support GPU usage in our script if it was available, so we would have been able to utilize the computing power of the cloud, however there was no need for this, because our small model and dataset size.

\---

**Question 19**

**Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.** **You can take inspiration from [this figure](https://github.com/SkafteNicki/dtu_mlops/blob/main/reports/figures/bucket.png).**

Answer:

\---

` `![](Aspose.Words.25da072a-7e40-419c-b7c8-8fb9d2a49757.006.png)

![](Aspose.Words.25da072a-7e40-419c-b7c8-8fb9d2a49757.007.png)

one example of the dataset we have in google cloud as test5 file is below:

![](Aspose.Words.25da072a-7e40-419c-b7c8-8fb9d2a49757.008.png)


\---

**Question 20**

**Upload one image of your GCP container registry, such that we can see the different images that you have stored.** **You can take inspiration from [this figure](https://github.com/SkafteNicki/dtu_mlops/blob/main/reports/figures/registry.png).**

Answer:

\--- 

![](Aspose.Words.25da072a-7e40-419c-b7c8-8fb9d2a49757.009.png)

To show more in GCP Container Registry in images/train file show look like: 

![](Aspose.Words.25da072a-7e40-419c-b7c8-8fb9d2a49757.010.png)

` `---

**Question 21**

**Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in** **your project. You can take inspiration from [this figure](https://github.com/SkafteNicki/dtu_mlops/blob/main/reports/figures/build.png).**

Answer:


---![](Aspose.Words.25da072a-7e40-419c-b7c8-8fb9d2a49757.011.png) ---

**Question 22**

**Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and** **preferably how you invoke your deployed service?**

Answer length: 100-200 words.

Example: *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which* *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call* *curl -X POST -F "file=@file.json"<weburl>*

Answer:

--- We managed to deploy using FastAPI locally, by wrapping our model into an application using fastAPI. Every time a change is made in the python script which we use the ‘uvicorn’ command on, then the page is refreshed as inference on the test dataset is run again. Multiple tabs can be opened and inference is run independently for each of the tabs. What you see below is the predicted nitrogen content vs the ground truth with the ConvNet2C model (GT only shown if it exists, one can change the path of the test folder in the config file.)

![](Aspose.Words.25da072a-7e40-419c-b7c8-8fb9d2a49757.012.png) 

The specific command is ‘uvicorn src.predict\_model\_fastapi:app --reload’

\---

**Question 23**

**Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how** **monitoring would help the longevity of your application.**

Answer length: 100-200 words.

Example: *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could* *measure ... and ... that would inform us about this ... behaviour of our application.*

Answer:

--- Sadly, we didn’t. If we had monitoring, then one of the things we would be able to spot, if we monitored memory usage on the deployed model, is cache-buildup or memory leak. ---

**Question 24**

**How many credits did you end up using during the project and what service was most expensive?**

Answer length: 25-100 words.

Example: *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service* *costing the most was ... due to ...*

Answer:

\--- 

s204161 used 0.01 USD (Creating a bucket, which was not used)

Karrar Adam (s230432), since I am the one in the group who used Compute Engine in GCP to build the buckets, using container registry and cloud build. I used $5.34 (the dataset  and the model we build were not that big, so we didn’t used much credits)

In total, the total amount of credits spent was $5.35

\--- 

**Overall discussion of project**

In the following section we would like you to think about the general structure of your project.

**Question 25**

**Include a figure that describes the overall architecture of your system and what services that you make use of.** **You can take inspiration from [this figure](https://github.com/SkafteNicki/dtu_mlops/blob/main/reports/figures/overview.png). Additionally in your own words, explain the** **overall steps in figure.**

Answer length: 200-400 words

Example:

*The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.* *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*

Answer:

—

![](Aspose.Words.25da072a-7e40-419c-b7c8-8fb9d2a49757.013.png)

So there are 2 use-cases of our system architecture - the developer role and the user role. For a user, they can either pull the repository and run the code or pull an image from docker (needs authorization because the docker image contains the private data) to get an exact copy of the environment.

When the user runs docker, the experiment results should be available in W&B.

For a developer, they might also use W&B to log experiment results - and we were able to set up Vertex AI in order to train the model in the cloud (not really needed however because of the small scale of the model and dataset). In this case, results are also sent to W&B.

Some used Github Desktop, while some version control in VSCode & pull request on the website - these changes were linked to a GCP trigger to automatically build docker images and save them to the docker registry in the cloud after every git commit. Specifically, GCP buckets are used by the developer for data version control. 

As a developer, linting & code formatting is also used with the ruff library. We strongly believe that the key objectives have been fulfilled satisfactorily and hence conclude the project on this note.

`  `---

**Question 26**

**Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these** **challenges?**

Answer length: 200-400 words.

Example: *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*

Answer:

` `--- 

One of the major issues we had was time and task management, considering we did not finish the tasks within week 3, which is about monitoring, data drifting and distributed data and model loading. This was due to both time constraints and delegating the work all with different work loads, which meant we underestimated the time it takes to do some tasks and overestimated the time it took to do some other tasks, and also a lot of focus went to fixing up some of the prior issues we had with prior tasks.

A big struggle was setting up GCP properly & especially the cloud deployment, we kept running into different errors. Also building docker with W&B. 

Another big struggle we had was, that we had to be cautious of the data we have acquired from the company of one of our members, since we have signed the NDA. Specifically, we got authorization errors (One of the different errors we had, explained above) when trying to deploy our fastAPI app in the cloud (also when trying this on the owner of the GCP project). We also tried to create service accounts so multiple people could work on the GCP project, but could not make this work. 

` `---

**Question 27**

**State the individual contributions of each team member. This is required information from DTU, because we need to** **make sure all members contributed actively to the project**

Answer length: 50-200 words.

Example: *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the* *docker containers for training our applications.* *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.* *All members contributed to code by...*

Answer:

\--- 

Student s204161 was in charge of setting up the initial cookiecutter project, creating the ViT model & unit testing.

Student s223092 was in charge of contacting the company Aerial Tools for the data, and for the workflow of this project. Student s223092 also created code for visualization and preprocessing the data.

Student s220044 was responsible for creating the YAML configuration file for listing out all hyperparameters belonging to both models and using Hydra and OmegaConf to load them while decoupling them from the mainstream model definition code. 

Besides, he also dealt with typing for the CNN model to ensure good coding practice and PEP8 compliance as required in the course.

Student s184213 was in charge of initial DVC, profiling the train model, logging the data onto wandb and working together with s204161 to set up deployment of FastAPI.

Student s230432 was in charge of formatting, saving the processed data, setting up the docker and getting some continuous integration to work.

Also see the MLOps Work Plan for the list we used to delegate different tasks.

(<https://docs.google.com/document/d/1lLLNnOMxulvgJ_XtQBqTcpUVArmR-BCf2veG_gLhQXo/edit?usp=sharing>) 

` `—

Student: s230432 was in charge of cloud setup tasks such as creating the project and organized most of Google cloud project,building buckets, test running in Vertex AI, and Cloud building for our repository 




![](Aspose.Words.25da072a-7e40-419c-b7c8-8fb9d2a49757.014.png)



