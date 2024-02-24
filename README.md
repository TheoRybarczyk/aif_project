# Recommender project

Recommender double feature:
- recommend some movie posters from an input image
- recommend movies based from a text description

1 - First download this project on your local drive using `git clone https://github.com/TheoRybarczyk/aif_project.git`

2 - Then, please download all the data from https://drive.google.com/drive/folders/11O-a1nV0abWnvHGWfyk3r54bpG8E2gWT?usp=drive_link. Google drive should zip all the contents within a couple of minutes and you will end up downloading an approximataly 600Mb archive. \
Then unzip the archive **so the AIF_data folder end up in the project root folder**. It has to look like this:

![image](https://github.com/TheoRybarczyk/aif_project/assets/83536996/55131762-888e-4518-81e6-95975bfb80f3)

3 - To build and run the docker images, please run `docker compose up` **from the project /docker folder**. \
This step is long because of the required Python modules to install. \
If you have some issues, you can run instead `docker compose up --no-deps --build`.

4 - Go to http://127.0.0.1:7860/ and enjoy!

5 - Run `docker compose down` when finished.
