# pyqt


### 总的目录结构

```bash
pyqt_demo_
|---assets # 编译配置
|
|---inetracrive_demo #  
| |---canvas.py #   Display and zoom image  还有很多要改。。。
|-|--|--def reload_image(self, image, reset_can #原来的代码是 显示图片和监听图片  然后我把显示图片放在了appdef _show_imag
|-|--|--class MyEventFilter(QObject):                            #处理事件的类
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

**目前主要在修改app.py和controller.py和canvas.py**