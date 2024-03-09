Dusk Till Dawn: Self-supervised Nighttime Stereo Depth Estimation using Visual Foundation Models
====================================
This repository holds all the code and data for the paper:

[**Dusk Till Dawn: Self-supervised Nighttime Stereo Depth Estimation using Visual Foundation Models**, ICRA 2024.]()\
[Madhu Vankadari][Madhu], [Samuel Hodgson][Sam], [Sangyun Shin][Sangyun], [Kaichen Zhou][Kaichen], [Andrew Markham][Andrew],[Niki Trigoni][Niki]

If you find this code useful, please consider citing:  
```text
@inProceedings{vankadari2024dtd,
  title={{Dusk Till Dawn: Self-supervised Nighttime Stereo Depth Estimation using Visual Foundation Models}},
  author={Madhu Vankadari, Samuel Hodgson, Sangyun Shin, Kaichen Zhou, Andrew Markham, and Niki Trigoni}
  booktitle={ICRA},
  year={2024},
}
```
if you have any questions, please email madhu.vankadari@cs.ox.ac.uk
#
Usage Instructions
------------------
We use serveral other repositories, so please use the following command that automatically downloads all the included submodules for easy use:
```
git clone --recurse-submodules https://github.com/madhubabuv/dtd.git
```
Datasets
------
**RobotCar Dataset** 

Please create an account on the official [RobotCar dataset page][robot_car_reg] before downloading the dataset. 
```
cd datasets/robotcar/RobotCarDataset-Scraper

python scrape_mrgdatashare.py --username <placeholder> --password <placehodler> --datasets_file ../datasets.csv --downloads_dir <placehodler>
```
This should create a folder structure like the following
```
<dataset root>
    |- stereo.timestamps
    |- lms_front.timestamps
    |- stereo
        |- left
            |- timestamp1.png
            |- timestamp2.png
            ...
        |- right
            |- timestamp1.png
            |- timestamp2.png
            ...
    |- lms_front
        |- timestamp1.png
        |- timestamp2.png
        ...
    |- vo
        |- vo.csv
    |- gps
        |- gps.csv
        |- ins.csv
```

Now the downloded dataset is in Bayer format and we should convert the images to RGB format. To do that,
```
cd datasets/robotcar/sdk
python convert_images.py --data_path <placeholder>
```
Please convert images from the both left and right directories. This process going to take a bit of time (10x of minutes).

Testing
--------
Please download the pretrained checkpoints from [here][dtd_checkpoint] on RobotCarDataset. If you want to test it on the robotcar dataset test split, use the follwoing
```
python test.py --test_file_path datasets/robotcar/files/2014-12-16-18-44-24_test.txt --checkpoint_path <downloaded checkpoint> --save_dir <whereever you want to save>
```

Evaluation
-----------
We need to download the ground depths that are calculated using RTK data released along with the dataset using the robotcar-sdk from [here][dtd_gt].

```
python evaluation/eval_depth_weighted.py --gt_depths_path <ground-truth depth path> --pred_disp_path <prediction path>
```

If you see a small change in the results from the paper, that is mainly because of convertion of depths to `fp16` while sharing on google drive. 

Training
---------
We are going to use the splits proposed in [When the sun goes down][wgsd] for training. The training files are already there on `datasets/robotcar/files`. The original file was for monocular training, hence the 3 columns, we are going to use the middle column
```
python train.py --data_path <place holder> --checkpoint_dir <place holder> --num_epochs 20 --learning_rate 0.0001
```

Licence
--------
This repository is released under the MIT Licence license as found in the LICENSE file.

Acknowledgements
---------
This project would not have been possible without replying the awesome repo [Unimatch][unimatch_git]. We thank the original others for their excellent work

[Madhu]: https://www.cs.ox.ac.uk/people/madhu.vankadari/
[Sam]: https://www.cs.ox.ac.uk/people/samuel.hodgson/
[Sangyun]:https://www.cs.ox.ac.uk/people/sangyun.shin/
[Kaichen]:https://www.cs.ox.ac.uk/people/kaichen.zhou/
[Andrew]:https://www.cs.ox.ac.uk/people/Andrew.Markham/
[Niki]:https://www.cs.ox.ac.uk/people/niki.trigoni/
[robot_car_reg]:https://mrgdatashare.robots.ox.ac.uk/accounts/login/
[wgsd]:https://arxiv.org/abs/2206.13850
[dtd_checkpoint]:https://drive.google.com/file/d/1dgUKBf-UKpOZp_3681iS_v3hgBs4Ip9a/view?usp=drive_link
[dtd_predictions]:https://drive.google.com/file/d/1aQfNbn5VmrHjrp6cUwelz64N-M1UmH0s/view?usp=drive_link
[dtd_gt]:https://drive.google.com/file/d/1nSV97qH-D7yI7AkD3B1pFfdt03UU3MoA/view?usp=drive_link

[unimatch_git]:https://github.com/autonomousvision/unimatch
