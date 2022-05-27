### Environment

You will have to initalize a blender environment (2.83 with python 3.7.4). The version must be exact this one.

You need to relink blender's default python environment to anaconda environment. You can check this source for how to do this:

[https://stackoverflow.com/questions/70639689/how-to-use-the-anaconda-environment-on-blender]()

### Run

`blender --background --python data_generator.py`

If you don't want to add visualization, remember to set the enable_vis as False.
