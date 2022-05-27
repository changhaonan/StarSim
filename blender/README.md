You will have to initalize a blender environment (2.83 with python 3.7)

You can simply run the data_generator.py with this command:

`blender --background --python data_generator.py`

The config file helps you control what you want in the output from the animate_render.py. For the data_generator all we need is the depth, hence we can only keep return the depths in the config.


## Design

What do we want to do? A unifed sampling & data-generating pipeline, espeicially for embedded graph data generating.

Usage:

- DeformingThings4D: Deformable node graph
- FlyingThings: Topolog-aware graph

We have two major parts: renderer & sampler. The core abstract layer here is mesh data / mesh representation. Those mesh things here are universal. (Vertex, Face, Offset)

Relationship:

- Mesh2Render: to depth, rgb things
- Mesh2Graph: sampling graph from mesh data

Mesh is a blender centric thing. Vertex things is more general. (Mesh: vertex, face id)