import bpy
import math
import mathutils

bpy.ops.wm.read_factory_settings(use_empty=True)

rot = (0, 0, math.radians(-90))

bpy.ops.wm.obj_import(filepath="/tmp/weir.obj")
water = bpy.context.selected_objects[0]
water.name = "Water"
water.rotation_euler = rot
bpy.ops.object.transform_apply(rotation=True)

bb = [water.matrix_world @ mathutils.Vector(c) for c in water.bound_box]
xlo = min(v.x for v in bb)
xhi = max(v.x for v in bb)
ylo = min(v.y for v in bb)
yhi = max(v.y for v in bb)
zlo = min(v.z for v in bb)
zhi = max(v.z for v in bb)
print("water: x[%.3f,%.3f] y[%.3f,%.3f] z[%.3f,%.3f]" %
      (xlo, xhi, ylo, yhi, zlo, zhi))

mat = bpy.data.materials.new("Water")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()
output = nodes.new("ShaderNodeOutputMaterial")
glass = nodes.new("ShaderNodeBsdfGlass")
glass.inputs["Color"].default_value = (0.3, 0.5, 0.9, 1.0)
glass.inputs["Roughness"].default_value = 0.02
glass.inputs["IOR"].default_value = 1.33
links.new(glass.outputs["BSDF"], output.inputs["Surface"])
water.data.materials.append(mat)
bpy.context.view_layer.objects.active = water
bpy.ops.object.shade_smooth()

bpy.ops.wm.obj_import(filepath="/tmp/wall.obj")
wall_obj = bpy.context.selected_objects[0]
wall_obj.name = "Weir"
wall_obj.rotation_euler = rot
bpy.ops.object.transform_apply(rotation=True)

def make_brick_material(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    n = m.node_tree.nodes
    l = m.node_tree.links
    bsdf = n["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.9
    tex = n.new("ShaderNodeTexCoord")
    mapping = n.new("ShaderNodeMapping")
    l.new(tex.outputs["Object"], mapping.inputs["Vector"])
    brick = n.new("ShaderNodeTexBrick")
    brick.inputs["Color1"].default_value = (0.55, 0.18, 0.12, 1)
    brick.inputs["Color2"].default_value = (0.45, 0.14, 0.10, 1)
    brick.inputs["Mortar"].default_value = (0.75, 0.73, 0.68, 1)
    brick.inputs["Scale"].default_value = 30
    brick.inputs["Mortar Size"].default_value = 0.02
    l.new(mapping.outputs["Vector"], brick.inputs["Vector"])
    l.new(brick.outputs["Color"], bsdf.inputs["Base Color"])
    bump = n.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.2
    l.new(brick.outputs["Fac"], bump.inputs["Height"])
    l.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return m

brick_mat = make_brick_material("Brick")
wall_obj.data.materials.append(brick_mat)
bpy.context.view_layer.objects.active = wall_obj
bpy.ops.object.shade_smooth()

h = zhi + 0.1
m = 0.02
cx = (xlo + xhi) / 2
cy = (ylo + yhi) / 2
sx = xhi - xlo + 0.1
sy = yhi - ylo + 0.1

def add_wall(name, loc, rot_euler, sx, sz):
    bpy.ops.mesh.primitive_plane_add(size=1, location=loc)
    w = bpy.context.active_object
    w.name = name
    w.rotation_euler = rot_euler
    w.scale = (sx / 2, sz / 2, 1)
    bpy.ops.object.transform_apply(scale=True, rotation=True)
    w.data.materials.append(brick_mat)
    return w

add_wall("WallXlo", (xlo - m, cy, h / 2), (0, math.radians(90), 0), sy, h)
add_wall("WallXhi", (xhi + m, cy, h / 2), (0, math.radians(-90), 0), sy, h)
add_wall("WallYlo", (cx, ylo - m, h / 2), (math.radians(-90), 0, 0), sx, h)
add_wall("WallYhi", (cx, yhi + m, h / 2), (math.radians(90), 0, 0), sx, h)

bpy.ops.mesh.primitive_plane_add(size=5, location=(cx, cy, zlo - 0.01))
floor = bpy.context.active_object
floor.name = "Floor"
fmat = bpy.data.materials.new("Checker")
fmat.use_nodes = True
fn = fmat.node_tree.nodes
fl = fmat.node_tree.links
checker = fn.new("ShaderNodeTexChecker")
checker.inputs["Scale"].default_value = 20
checker.inputs["Color1"].default_value = (0.9, 0.9, 0.9, 1)
checker.inputs["Color2"].default_value = (0.6, 0.6, 0.6, 1)
fl.new(checker.outputs["Color"], fn["Principled BSDF"].inputs["Base Color"])
floor.data.materials.append(fmat)

bpy.ops.object.light_add(type='AREA', location=(1.5, -0.5, 2.5))
light = bpy.context.active_object
light.data.energy = 300
light.data.size = 3

bpy.ops.object.light_add(type='AREA', location=(-0.5, 1.0, 1.5))
fill = bpy.context.active_object
fill.data.energy = 80
fill.data.size = 2

world = bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
wn = world.node_tree.nodes
wn["Background"].inputs["Color"].default_value = (0.95, 0.95, 1.0, 1)
wn["Background"].inputs["Strength"].default_value = 0.5

cam = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj
cam_obj.location = (1.4, -1.3, 0.6)
cam_obj.rotation_euler = (math.radians(72), 0, math.radians(50))
cam.lens = 26

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 256
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.filepath = "/tmp/weir_render.png"
scene.render.image_settings.file_format = 'PNG'

bpy.ops.render.render(write_still=True)
