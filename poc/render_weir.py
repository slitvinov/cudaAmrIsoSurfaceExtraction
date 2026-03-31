import bpy
import math

bpy.ops.wm.read_factory_settings(use_empty=True)

bpy.ops.wm.obj_import(filepath="/tmp/weir.obj")
obj = bpy.context.selected_objects[0]
obj.rotation_euler = (0, 0, math.radians(-90))
bpy.ops.object.transform_apply(rotation=True)

mat = bpy.data.materials.new("Water")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

output = nodes.new("ShaderNodeOutputMaterial")
glass = nodes.new("ShaderNodeBsdfGlass")
glass.inputs["Color"].default_value = (0.6, 0.8, 1.0, 1.0)
glass.inputs["Roughness"].default_value = 0.05
glass.inputs["IOR"].default_value = 1.33
links.new(glass.outputs["BSDF"], output.inputs["Surface"])
obj.data.materials.append(mat)

bpy.ops.object.shade_smooth()

bpy.ops.mesh.primitive_plane_add(size=5, location=(0.35, 0.5, -0.05))
floor = bpy.context.active_object
fmat = bpy.data.materials.new("Checker")
fmat.use_nodes = True
fnodes = fmat.node_tree.nodes
flinks = fmat.node_tree.links
checker = fnodes.new("ShaderNodeTexChecker")
checker.inputs["Scale"].default_value = 20
checker.inputs["Color1"].default_value = (0.9, 0.9, 0.9, 1)
checker.inputs["Color2"].default_value = (0.6, 0.6, 0.6, 1)
flinks.new(checker.outputs["Color"], fnodes["Principled BSDF"].inputs["Base Color"])
floor.data.materials.append(fmat)

bpy.ops.mesh.primitive_plane_add(size=3, location=(-0.05, 0.5, 0.5),
                                  rotation=(0, math.radians(90), 0))
wall = bpy.context.active_object
wmat = bpy.data.materials.new("Wall")
wmat.use_nodes = True
wnodes = wmat.node_tree.nodes
wlinks = wmat.node_tree.links
wcheck = wnodes.new("ShaderNodeTexChecker")
wcheck.inputs["Scale"].default_value = 20
wcheck.inputs["Color1"].default_value = (1.0, 1.0, 1.0, 1)
wcheck.inputs["Color2"].default_value = (0.7, 0.7, 0.7, 1)
wlinks.new(wcheck.outputs["Color"], wnodes["Principled BSDF"].inputs["Base Color"])
wall.data.materials.append(wmat)

bpy.ops.object.light_add(type='AREA', location=(1.5, 1.5, 2.5))
light = bpy.context.active_object
light.data.energy = 300
light.data.size = 3

bpy.ops.object.light_add(type='AREA', location=(-1, -0.5, 1.5))
fill = bpy.context.active_object
fill.data.energy = 80
fill.data.size = 2

world = bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
wnodes = world.node_tree.nodes
wnodes["Background"].inputs["Color"].default_value = (0.95, 0.95, 1.0, 1)
wnodes["Background"].inputs["Strength"].default_value = 0.5

cam = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj
cam_obj.location = (1.5, -1.0, 0.7)
cam_obj.rotation_euler = (math.radians(70), 0, math.radians(55))
cam.lens = 28

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 256
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.filepath = "/tmp/weir_render.png"
scene.render.image_settings.file_format = 'PNG'

bpy.ops.render.render(write_still=True)
