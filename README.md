# 可交互式遥感图像切割技术前端代码

### 总的目录结构

```bash
pyqt_demo_
|---assets # 编译配置
|
|---inetracrive_demo 
| |---canvas.py 
|-|--|----def 
|-|--|----class 
| |---controller.py 
| |---wrappers.py 
|
|---isegm
| |---inference
|   |---clicker.py # 点击类Clicker
|
|
|---models
|
|---notebooks
|
|---scripts
|
|---weights
|
|---app.py # 窗口类，组件类
|--|--|---def _load_image_callback(self): #加载图片的函数
|--|--|---def _update_image(self, reset_canvas=False):#更新图片
|--|--|---def _update_image(self, reset_canvas=False):#更新图片
|
|---main.py # 主程序入口
|
|---config # 项目编译配置
|
|---README.md #文档
|
|---requirements # 项目软件包
```


**Controls**:

| Key                                                           | Description                        |
| ------------------------------------------------------------- | ---------------------------------- |
| <kbd>Left Mouse Button</kbd>                                  | Place a positive click             |
| <kbd>Right Mouse Button</kbd>                                 | Place a negative click             |
| <kbd>Scroll Wheel</kbd>                                       | Zoom an image in and out           |
| <kbd>Right Mouse Button</kbd> + <br> <kbd>Move Mouse</kbd>    | Move an image                      |
| <kbd>Space</kbd>                                              | Finish the current object mask     |
```
@inproceedings{ritm2022,
  title={Reviving iterative training with mask guidance for interactive segmentation},
  author={Sofiiuk, Konstantin and Petrov, Ilya A and Konushin, Anton},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={3141--3145},
  year={2022},
  organization={IEEE}
}

@inproceedings{fbrs2020,
   title={f-brs: Rethinking backpropagating refinement for interactive segmentation},
   author={Sofiiuk, Konstantin and Petrov, Ilia and Barinova, Olga and Konushin, Anton},
   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
   pages={8623--8632},
   year={2020}
}
```