import os
import glob
import math
import xml.etree.ElementTree as ET
import bpy
import bmesh
from mathutils import Vector, Matrix, Euler, Quaternion


def find_rootlinks(joints):
    """Return all links that don't occur as child in any joint"""
    parents = []
    children = []
    for joint in joints:
        parents.append(joint.find('parent').attrib['link'])
        children.append(joint.find('child').attrib['link'])

    rootlinks = list(set(parents) - set(children))
    return rootlinks


def find_childjoints(joints, link):
    """Returns all joints that contain the link as parent"""
    childjoints = []
    for joint in joints:
        if joint.find('parent').attrib['link'] == link:
            childjoints.append(joint)
    return childjoints


def get_joint_origin_offset(joint):
    origin = joint.find('origin')
    if origin is not None and 'xyz' in origin.attrib:
        return Vector([float(s) for s in origin.attrib['xyz'].split()])
    return Vector((0.0, 0.0, 0.0))


def get_origin_matrix(element):
    origin = element.find('origin')
    if origin is None:
        return Matrix.Identity(4)

    translation = Vector((0.0, 0.0, 0.0))
    rotation_matrix = Matrix.Identity(4)

    if 'xyz' in origin.attrib:
        translation = Vector([float(s) for s in origin.attrib['xyz'].split()])
    if 'rpy' in origin.attrib:
        roll, pitch, yaw = [float(s) for s in origin.attrib['rpy'].split()]
        rotation_matrix = Euler((roll, pitch, yaw), 'XYZ').to_matrix().to_4x4()

    return Matrix.Translation(translation) @ rotation_matrix


def parse_name_filters(raw):
    if not raw:
        return []
    raw = raw.replace(';', ',')
    parts = [part.strip() for part in raw.split(',') if part.strip()]
    return [part.lower() for part in parts]


def name_matches_filters(name, filters):
    if not filters:
        return False
    lowered = name.lower()
    return any(token in lowered for token in filters)


def should_add_ik_for_joint(joint, ik_settings):
    if not ik_settings or not ik_settings.get('enabled'):
        return False
    driver_names = ik_settings.get('driver_names', [])
    if driver_names:
        return joint.attrib.get('name', '') in driver_names
    filters = ik_settings.get('driver_filters', [])
    if filters:
        return name_matches_filters(joint.attrib.get('name', ''), filters)
    return False


def find_alignment_joint(child_joints, align_names, align_filters):
    if not child_joints:
        return None
    for candidate in child_joints:
        candidate_name = candidate.attrib.get('name', '')
        child_elem = candidate.find('child')
        child_name = child_elem.attrib.get('link', '') if child_elem is not None else ''
        if align_names:
            if candidate_name in align_names or child_name in align_names:
                return candidate
        elif align_filters:
            if name_matches_filters(candidate_name, align_filters) or name_matches_filters(child_name, align_filters):
                return candidate
    return None


def get_bone_axis_locks(armature, bone_name, axis_world):
    if axis_world.length < 1e-6:
        axis_world = Vector((0.0, 1.0, 0.0))
    else:
        axis_world = axis_world.normalized()

    try:
        axis_armature = armature.matrix_world.to_3x3().inverted() @ axis_world
    except (AttributeError, RuntimeError, ValueError):
        axis_armature = axis_world

    bone = armature.data.bones.get(bone_name)
    if bone is not None:
        axis_local = bone.matrix_local.to_3x3().inverted() @ axis_armature
    else:
        axis_local = axis_armature

    abs_axis = [abs(axis_local.x), abs(axis_local.y), abs(axis_local.z)]
    max_index = abs_axis.index(max(abs_axis))
    lock_axis = [True, True, True]
    lock_axis[max_index] = False
    return lock_axis, max_index


def get_joint_limit_values(joint):
    limit = joint.find('limit')
    if limit is None:
        return None, None
    if 'lower' not in limit.attrib or 'upper' not in limit.attrib:
        return None, None
    try:
        lower = float(limit.attrib['lower'])
        upper = float(limit.attrib['upper'])
    except (ValueError, TypeError):
        return None, None
    return lower, upper


def apply_revolute_constraints(posebone, axis_index, lower, upper):
    limit_location = posebone.constraints.get('URDF_LimitLocation')
    if limit_location is None:
        limit_location = posebone.constraints.new('LIMIT_LOCATION')
        limit_location.name = 'URDF_LimitLocation'
    limit_location.influence = 0.0
    if hasattr(limit_location, 'mute'):
        limit_location.mute = True
    limit_location.owner_space = 'LOCAL'
    limit_location.use_min_x = True
    limit_location.use_min_y = True
    limit_location.use_min_z = True
    limit_location.use_max_x = True
    limit_location.use_max_y = True
    limit_location.use_max_z = True
    limit_location.min_x = 0.0
    limit_location.min_y = 0.0
    limit_location.min_z = 0.0
    limit_location.max_x = 0.0
    limit_location.max_y = 0.0
    limit_location.max_z = 0.0

    if lower is None or upper is None:
        return

    existing = posebone.constraints.get('URDF_LimitRotation')
    if existing:
        posebone.constraints.remove(existing)

    limit_rotation = posebone.constraints.get('URDF_LimitRotation')
    if limit_rotation is None:
        limit_rotation = posebone.constraints.new('LIMIT_ROTATION')
        limit_rotation.name = 'URDF_LimitRotation'
    limit_rotation.influence = 0.0
    if hasattr(limit_rotation, 'mute'):
        limit_rotation.mute = True
    limit_rotation.owner_space = 'LOCAL'
    limit_rotation.use_limit_x = axis_index == 0
    limit_rotation.use_limit_y = axis_index == 1
    limit_rotation.use_limit_z = axis_index == 2
    if axis_index == 0:
        limit_rotation.min_x = lower
        limit_rotation.max_x = upper
    elif axis_index == 1:
        limit_rotation.min_y = lower
        limit_rotation.max_y = upper
    else:
        limit_rotation.min_z = lower
        limit_rotation.max_z = upper


def apply_transform_rotation_mapping(
    posebone,
    target_name,
    axis_index,
    armature,
    from_min,
    from_max,
    to_min,
    to_max,
):
    if from_min is None or from_max is None or to_min is None or to_max is None:
        return
    if abs(from_max - from_min) < 1e-6:
        return

    existing = posebone.constraints.get('URDF_TransformRotation')
    if existing:
        posebone.constraints.remove(existing)

    transform = posebone.constraints.new('TRANSFORM')
    transform.name = 'URDF_TransformRotation'
    transform.influence = 0.0
    if hasattr(transform, 'mute'):
        transform.mute = True
    transform.target = armature
    transform.subtarget = target_name
    transform.owner_space = 'LOCAL'
    transform.target_space = 'LOCAL'
    transform.map_from = 'ROTATION'
    transform.map_to = 'ROTATION'
    transform.from_rotation_mode = 'XYZ'
    transform.to_euler_order = 'XYZ'
    transform.map_to_x_from = 'X'
    transform.map_to_y_from = 'Y'
    transform.map_to_z_from = 'Z'
    transform.mix_mode_rot = 'REPLACE'

    for axis in ('x', 'y', 'z'):
        setattr(transform, f'from_min_{axis}_rot', 0.0)
        setattr(transform, f'from_max_{axis}_rot', 0.0)
        setattr(transform, f'to_min_{axis}_rot', 0.0)
        setattr(transform, f'to_max_{axis}_rot', 0.0)

    axis_labels = ('x', 'y', 'z')
    axis = axis_labels[axis_index]
    setattr(transform, f'from_min_{axis}_rot', from_min)
    setattr(transform, f'from_max_{axis}_rot', from_max)
    setattr(transform, f'to_min_{axis}_rot', to_min)
    setattr(transform, f'to_max_{axis}_rot', to_max)


def apply_transform_rotation_offset(
    driven_pose,
    control_name,
    axis_index,
    armature,
    control_min,
    control_max,
    lower,
    upper,
):
    apply_transform_rotation_mapping(
        driven_pose,
        control_name,
        axis_index,
        armature,
        control_min,
        control_max,
        lower,
        upper,
    )


def select_only(blender_object):
    """Selects and actives a Blender object and deselects all others"""
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = blender_object
    blender_object.select_set(True)


def select_objects(objects, active=None):
    bpy.ops.object.select_all(action='DESELECT')
    if not objects:
        return
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = active or objects[0]


def delete_objects(objects):
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)


def get_fixed_bone_shape():
    shape_name = 'URDF_FIXED_BONE_SHAPE'
    shape_obj = bpy.data.objects.get(shape_name)
    if shape_obj:
        mesh = shape_obj.data
    else:
        mesh = bpy.data.meshes.new(f'{shape_name}_MESH')
        shape_obj = bpy.data.objects.new(shape_name, mesh)
        bpy.context.scene.collection.objects.link(shape_obj)

    thickness = 0.02
    verts = [
        (-thickness, 0.0, -thickness),
        (thickness, 0.0, -thickness),
        (thickness, 0.0, thickness),
        (-thickness, 0.0, thickness),
        (-thickness, 1.0, -thickness),
        (thickness, 1.0, -thickness),
        (thickness, 1.0, thickness),
        (-thickness, 1.0, thickness),
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]

    try:
        mesh.clear_geometry()
    except AttributeError:
        pass
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    shape_obj.hide_render = True
    if hasattr(shape_obj, 'hide_viewport'):
        shape_obj.hide_viewport = True
    if hasattr(shape_obj, 'hide_select'):
        shape_obj.hide_select = True
    if hasattr(shape_obj, 'display_type'):
        shape_obj.display_type = 'SOLID'
    return shape_obj


def get_standard_bone_shape():
    shape_name = 'URDF_STANDARD_BONE_SHAPE'
    shape_obj = bpy.data.objects.get(shape_name)
    if shape_obj:
        mesh = shape_obj.data
    else:
        mesh = bpy.data.meshes.new(f'{shape_name}_MESH')
        shape_obj = bpy.data.objects.new(shape_name, mesh)
        bpy.context.scene.collection.objects.link(shape_obj)

    thickness = 0.01
    verts = [
        (-thickness, 0.0, -thickness),
        (thickness, 0.0, -thickness),
        (thickness, 0.0, thickness),
        (-thickness, 0.0, thickness),
        (-thickness, 1.0, -thickness),
        (thickness, 1.0, -thickness),
        (thickness, 1.0, thickness),
        (-thickness, 1.0, thickness),
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]

    try:
        mesh.clear_geometry()
    except AttributeError:
        pass
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    shape_obj.hide_render = True
    if hasattr(shape_obj, 'hide_viewport'):
        shape_obj.hide_viewport = True
    if hasattr(shape_obj, 'hide_select'):
        shape_obj.hide_select = True
    if hasattr(shape_obj, 'display_type'):
        shape_obj.display_type = 'SOLID'
    return shape_obj


def get_control_bone_shape():
    shape_name = 'URDF_CONTROL_BONE_SHAPE'
    shape_obj = bpy.data.objects.get(shape_name)
    if shape_obj:
        mesh = shape_obj.data
    else:
        mesh = bpy.data.meshes.new(f'{shape_name}_MESH')
        shape_obj = bpy.data.objects.new(shape_name, mesh)
        bpy.context.scene.collection.objects.link(shape_obj)

    bar_length = 1.02
    bar_thickness = 0.02
    knob_radius = 0.06

    try:
        mesh.clear_geometry()
    except AttributeError:
        pass

    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(
        bm,
        verts=bm.verts,
        vec=(bar_thickness, bar_length, bar_thickness),
    )
    bmesh.ops.translate(
        bm,
        verts=bm.verts,
        vec=(0.0, bar_length * 0.5, 0.0),
    )

    sphere = bmesh.ops.create_icosphere(
        bm,
        subdivisions=2,
        radius=knob_radius,
    )
    bmesh.ops.translate(
        bm,
        verts=sphere['verts'],
        vec=(0.0, bar_length + knob_radius, 0.0),
    )

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    shape_obj.hide_render = True
    if hasattr(shape_obj, 'hide_viewport'):
        shape_obj.hide_viewport = True
    if hasattr(shape_obj, 'hide_select'):
        shape_obj.hide_select = True
    if hasattr(shape_obj, 'display_type'):
        shape_obj.display_type = 'SOLID'
    return shape_obj


def apply_bone_custom_shape(posebone, shape_obj):
    if shape_obj is None:
        return
    posebone.custom_shape = shape_obj
    if hasattr(posebone, 'use_custom_shape_bone_size'):
        posebone.use_custom_shape_bone_size = True


def apply_bone_custom_color(posebone, normal, select, active):
    bone_color = getattr(posebone.bone, 'color', None)
    if bone_color and hasattr(bone_color, 'palette'):
        try:
            bone_color.palette = 'CUSTOM'
        except (AttributeError, RuntimeError, TypeError):
            pass
        custom = getattr(bone_color, 'custom', None)
        if custom:
            for attr, value in (
                ('normal', normal),
                ('select', select),
                ('active', active),
            ):
                if hasattr(custom, attr):
                    try:
                        setattr(custom, attr, value)
                    except (AttributeError, RuntimeError, TypeError):
                        pass


def apply_fixed_bone_display(armature, bone_name):
    select_only(armature)
    try:
        bpy.ops.object.mode_set(mode='POSE')
    except (AttributeError, RuntimeError):
        return

    posebone = armature.pose.bones.get(bone_name)
    if posebone is None:
        bpy.ops.object.mode_set(mode='OBJECT')
        return

    shape_obj = get_fixed_bone_shape()
    if shape_obj is not None:
        posebone.custom_shape = shape_obj
        if hasattr(posebone, 'use_custom_shape_bone_size'):
            posebone.use_custom_shape_bone_size = True

    bone_color = getattr(posebone.bone, 'color', None)
    if bone_color and hasattr(bone_color, 'palette'):
        try:
            bone_color.palette = 'CUSTOM'
        except (AttributeError, RuntimeError, TypeError):
            pass
        custom = getattr(bone_color, 'custom', None)
        if custom:
            for attr, value in (
                ('normal', (0.2, 0.2, 0.2)),
                ('select', (0.25, 0.25, 0.25)),
                ('active', (0.3, 0.3, 0.3)),
            ):
                if hasattr(custom, attr):
                    try:
                        setattr(custom, attr, value)
                    except (AttributeError, RuntimeError, TypeError):
                        pass

    bpy.ops.object.mode_set(mode='OBJECT')


def apply_transforms_and_rest_pose(armature, apply_transforms=True, apply_rest_pose=True):
    if apply_transforms:
        deform_objects = [
            o for o in bpy.data.objects
            if 'LINK__' in o.name and getattr(o, 'parent_type', '') != 'BONE'
        ]
        select_objects([armature] + deform_objects, active=armature)
        try:
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        except (AttributeError, RuntimeError) as exc:
            print(f'Failed to apply transforms: {exc}')

    if not apply_rest_pose:
        return

    select_only(armature)
    try:
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.armature_apply()
        for posebone in armature.pose.bones:
            for constraint in posebone.constraints:
                if constraint.name in (
                    'URDF_LimitRotation',
                    'URDF_LimitLocation',
                    'URDF_TransformRotation',
                    'URDF_IK',
                ):
                    try:
                        constraint.influence = 1.0
                    except (AttributeError, RuntimeError, TypeError):
                        pass
                    if hasattr(constraint, 'mute'):
                        try:
                            constraint.mute = False
                        except (AttributeError, RuntimeError, TypeError):
                            pass
    except (AttributeError, RuntimeError) as exc:
        print(f'Failed to apply rest pose: {exc}')
    finally:
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except (AttributeError, RuntimeError) as exc:
            print(f'Failed to return to OBJECT mode: {exc}')


def set_midpoint_rest_pose(armature, joints):
    select_only(armature)
    try:
        bpy.ops.object.mode_set(mode='POSE')
    except (AttributeError, RuntimeError):
        return

    for joint in joints:
        if joint.attrib.get('type') not in ('revolute', 'continuous'):
            continue
        lower, upper = get_joint_limit_values(joint)
        if lower is None or upper is None:
            continue
        joint_name = joint.attrib['name']
        posebone = armature.pose.bones.get(joint_name)
        if posebone is None:
            continue
        axis_index = next(
            (i for i, locked in enumerate(posebone.lock_rotation) if not locked),
            None,
        )
        if axis_index is None:
            continue
        midpoint = (lower + upper) * 0.5
        posebone.rotation_euler = (0.0, 0.0, 0.0)
        posebone.rotation_euler[axis_index] = midpoint

        ik_target = armature.pose.bones.get(f'IK_{joint_name}')
        if ik_target is not None:
            bpy.context.view_layer.update()
            offset = posebone.tail - ik_target.tail
            ik_target.location = ik_target.location + offset

    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except (AttributeError, RuntimeError):
        pass


def align_ik_targets(armature):
    select_only(armature)
    try:
        bpy.ops.object.mode_set(mode='EDIT')
    except (AttributeError, RuntimeError):
        return

    edit_bones = armature.data.edit_bones
    for ik_bone in edit_bones:
        if not ik_bone.name.startswith('IK_'):
            continue
        driven_name = ik_bone.name[3:]
        driven_bone = edit_bones.get(driven_name)
        if driven_bone is None:
            continue

        ik_vec = ik_bone.tail - ik_bone.head
        dir_vec = ik_vec
        if dir_vec.length < 1e-6:
            dir_vec = driven_bone.tail - driven_bone.head
        if dir_vec.length < 1e-6:
            dir_vec = Vector((0.0, 0.0, 1.0))
        ik_len = ik_vec.length if ik_vec.length > 1e-6 else 0.05
        dir_vec = dir_vec.normalized()

        ik_bone.tail = driven_bone.tail
        ik_bone.head = driven_bone.tail - dir_vec * ik_len

    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except (AttributeError, RuntimeError):
        pass


def create_calibration_action(armature, joints, frame_start=0, frame_end=50, fps=50):
    scene = bpy.context.scene
    if scene is None:
        return

    auto_key = None
    if hasattr(scene, 'tool_settings') and hasattr(scene.tool_settings, 'use_keyframe_insert_auto'):
        auto_key = scene.tool_settings.use_keyframe_insert_auto
        scene.tool_settings.use_keyframe_insert_auto = False

    scene.render.fps = fps
    scene.frame_start = frame_start
    scene.frame_end = frame_end

    armature.animation_data_create()
    action_name = 'Calibration'
    action = bpy.data.actions.get(action_name)
    if action is None:
        action = bpy.data.actions.new(action_name)
    else:
        action.fcurves.clear()
    armature.animation_data.action = action

    ik_constraints = []
    for posebone in armature.pose.bones:
        for constraint in posebone.constraints:
            if constraint.type != 'IK' or constraint.name != 'URDF_IK':
                continue
            ik_constraints.append(
                (
                    constraint,
                    getattr(constraint, 'mute', None),
                    constraint.influence,
                )
            )

    pose_bones = armature.pose.bones
    joint_bones = []
    for joint in joints:
        if joint.attrib.get('type') not in ('revolute', 'continuous'):
            continue
        posebone = pose_bones.get(joint.attrib.get('name', ''))
        if posebone is None:
            continue
        lower, upper = get_joint_limit_values(joint)
        if lower is None or upper is None:
            continue
        axis_index = next(
            (i for i, locked in enumerate(posebone.lock_rotation) if not locked),
            None,
        )
        if axis_index is None:
            continue
        midpoint = (lower + upper) * 0.5
        centered_min = lower - midpoint
        centered_max = upper - midpoint
        joint_bones.append((posebone, axis_index, centered_min, centered_max))

    scene.frame_set(frame_start)
    for constraint, _old_mute, _old_influence in ik_constraints:
        constraint.influence = 0.0
        constraint.keyframe_insert(data_path='influence')
        if hasattr(constraint, 'mute'):
            constraint.mute = True
            constraint.keyframe_insert(data_path='mute')
    for posebone, axis_index, lower, _upper in joint_bones:
        posebone.rotation_mode = 'XYZ'
        posebone.rotation_euler = (0.0, 0.0, 0.0)
        posebone.rotation_euler[axis_index] = lower
        posebone.keyframe_insert(data_path='rotation_euler')

    base_pose = pose_bones.get('base')
    if base_pose is not None:
        base_pose.rotation_mode = 'QUATERNION'
        base_pose.rotation_quaternion = Quaternion((0.0, 0.0, 1.0), 0.0)
        base_pose.keyframe_insert(data_path='rotation_quaternion')

    scene.frame_set(frame_end)
    for constraint, _old_mute, _old_influence in ik_constraints:
        constraint.influence = 0.0
        constraint.keyframe_insert(data_path='influence')
        if hasattr(constraint, 'mute'):
            constraint.mute = True
            constraint.keyframe_insert(data_path='mute')
    for posebone, axis_index, _lower, upper in joint_bones:
        posebone.rotation_mode = 'XYZ'
        posebone.rotation_euler = (0.0, 0.0, 0.0)
        posebone.rotation_euler[axis_index] = upper
        posebone.keyframe_insert(data_path='rotation_euler')

    if base_pose is not None:
        base_pose.rotation_mode = 'QUATERNION'
        base_pose.rotation_quaternion = Quaternion((0.0, 0.0, 1.0), math.radians(90.0))
        base_pose.keyframe_insert(data_path='rotation_quaternion')

    for constraint, old_mute, old_influence in ik_constraints:
        constraint.influence = old_influence
        if hasattr(constraint, 'mute') and old_mute is not None:
            constraint.mute = old_mute

    if auto_key is not None:
        scene.tool_settings.use_keyframe_insert_auto = auto_key


def create_idle_action(armature, frame=0):
    scene = bpy.context.scene
    if scene is None:
        return

    auto_key = None
    if hasattr(scene, 'tool_settings') and hasattr(scene.tool_settings, 'use_keyframe_insert_auto'):
        auto_key = scene.tool_settings.use_keyframe_insert_auto
        scene.tool_settings.use_keyframe_insert_auto = False

    armature.animation_data_create()
    action_name = 'Idle'
    action = bpy.data.actions.get(action_name)
    if action is None:
        action = bpy.data.actions.new(action_name)
    else:
        action.fcurves.clear()
    armature.animation_data.action = action

    ik_constraints = []
    for posebone in armature.pose.bones:
        for constraint in posebone.constraints:
            if constraint.type != 'IK' or constraint.name != 'URDF_IK':
                continue
            ik_constraints.append(
                (
                    constraint,
                    getattr(constraint, 'mute', None),
                    constraint.influence,
                )
            )

    scene.frame_set(frame)
    for constraint, _old_mute, _old_influence in ik_constraints:
        constraint.influence = 1.0
        constraint.keyframe_insert(data_path='influence')
        if hasattr(constraint, 'mute'):
            constraint.mute = False
            constraint.keyframe_insert(data_path='mute')

    for posebone in armature.pose.bones:
        mode = posebone.rotation_mode
        if mode == 'QUATERNION':
            posebone.keyframe_insert(data_path='rotation_quaternion')
        elif mode == 'AXIS_ANGLE':
            posebone.keyframe_insert(data_path='rotation_axis_angle')
        else:
            posebone.keyframe_insert(data_path='rotation_euler')
        posebone.keyframe_insert(data_path='location')

    for constraint, old_mute, old_influence in ik_constraints:
        constraint.influence = old_influence
        if hasattr(constraint, 'mute') and old_mute is not None:
            constraint.mute = old_mute

    if auto_key is not None:
        scene.tool_settings.use_keyframe_insert_auto = auto_key


def ensure_armature_parenting(armature, objects, use_bone_parenting=False):
    for obj in objects:
        if obj.type != 'MESH':
            continue
        bone_name = None
        if use_bone_parenting and '__' in obj.name:
            parts = obj.name.split('__')
            if len(parts) >= 2:
                bone_name = parts[1]
        pose_bone = armature.pose.bones.get(bone_name) if bone_name else None
        if use_bone_parenting and pose_bone is not None:
            world_matrix = obj.matrix_world.copy()
            obj.parent = armature
            obj.parent_type = 'BONE'
            if hasattr(obj, 'parent_bone'):
                obj.parent_bone = bone_name
            try:
                parent_matrix = armature.matrix_world @ pose_bone.matrix
                obj.matrix_parent_inverse = parent_matrix.inverted()
                obj.matrix_world = world_matrix
            except (AttributeError, RuntimeError, ValueError):
                obj.matrix_world = world_matrix
            for mod in list(obj.modifiers):
                if mod.type == 'ARMATURE':
                    obj.modifiers.remove(mod)
            continue

        obj.parent = armature
        obj.parent_type = 'OBJECT'
        if hasattr(obj, 'parent_bone'):
            obj.parent_bone = ''
        arm_mod = next((m for m in obj.modifiers if m.type == 'ARMATURE'), None)
        if arm_mod is None:
            arm_mod = obj.modifiers.new(name='Armature', type='ARMATURE')
        arm_mod.object = armature
        if hasattr(arm_mod, 'use_vertex_groups'):
            arm_mod.use_vertex_groups = True


def set_object_smooth_shading(blender_object, angle_degrees=30.0):
    """Use the most compatible smooth shading API available."""
    if getattr(blender_object, 'type', None) != 'MESH':
        return

    select_only(blender_object)

    angle = math.radians(angle_degrees)
    auto_smooth_applied = False

    if 'shade_smooth_by_angle' in dir(bpy.ops.object):
        try:
            bpy.ops.object.shade_smooth_by_angle(angle=angle)
            auto_smooth_applied = True
        except (AttributeError, RuntimeError):
            pass

    if not auto_smooth_applied and 'shade_auto_smooth' in dir(bpy.ops.object):
        try:
            bpy.ops.object.shade_auto_smooth(angle=angle)
            auto_smooth_applied = True
        except (AttributeError, RuntimeError):
            pass

    if not auto_smooth_applied and 'shade_smooth' in dir(bpy.ops.object):
        try:
            bpy.ops.object.shade_smooth()
        except (AttributeError, RuntimeError):
            pass

    mesh = getattr(blender_object, 'data', None)
    if mesh and hasattr(mesh, 'use_auto_smooth'):
        mesh.use_auto_smooth = True
        if hasattr(mesh, 'auto_smooth_angle'):
            mesh.auto_smooth_angle = angle

    if not auto_smooth_applied and mesh and 'edges_select_sharp' in dir(bpy.ops.mesh):
        try:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_mode(type='EDGE')
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.mesh.edges_select_sharp(sharpness=angle)
            bpy.ops.mesh.mark_sharp(clear=False)
        except (AttributeError, RuntimeError):
            pass
        finally:
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except (AttributeError, RuntimeError):
                pass

def add_next_empty(empty, joint):
    """Duplicates the empty and applies the transform specified in the joint"""
    select_only(empty)

    bpy.ops.object.duplicate()
    new_empty = bpy.context.active_object
    new_empty.name = 'TF_' + joint.attrib['name']

    origin_matrix = get_origin_matrix(joint)
    new_empty.matrix_world = empty.matrix_world @ origin_matrix

    bpy.context.view_layer.update()
    return new_empty


def parse_mesh_filename(mesh_filename, base_dir=None):
    """This function will return the mesh path if it can be found, else None"""
    if not mesh_filename:
        return None

    if mesh_filename.startswith('file://'):
        mesh_filename = mesh_filename[7:]

    if os.path.exists(mesh_filename):
        return mesh_filename

    if base_dir:
        candidate = os.path.normpath(os.path.join(base_dir, mesh_filename))
        if os.path.exists(candidate):
            return candidate

    if 'package://' in mesh_filename:
        ros_package_paths = os.environ.get('ROS_PACKAGE_PATH')
        if ros_package_paths is None:
            error_msg = (
                'Your urdf file references a mesh file from a ROS package: \n'
                f'{mesh_filename}\n'
                'However, the ROS_PACKAGE_PATH environment variable is not set '
                'so we cannot find it.'
            )
            print(error_msg)
            return None

        ros_package_paths = [p for p in ros_package_paths.split(os.pathsep) if p]

        filepath_package = mesh_filename.replace('package://', '', 1)
        filepath_package = filepath_package.replace('\\', '/')
        filepath_split = filepath_package.split('/')
        package_name = filepath_split[0]
        filepath_in_package = os.path.join(*filepath_split[1:]) if len(filepath_split) > 1 else ''

        for ros_package_path in ros_package_paths:
            ros_package_path = os.path.normpath(ros_package_path)
            package_search = os.path.join(ros_package_path, '**', package_name)
            for package_path in glob.glob(package_search, recursive=True):
                filepath = os.path.join(package_path, filepath_in_package)
                if os.path.exists(filepath):
                    return filepath

    print('Cant find the mesh file :(')
    return None

def get_collada_asset_info(mesh_path):
    try:
        tree = ET.parse(mesh_path)
    except (ET.ParseError, OSError, ValueError):
        print(f'Failed to parse Collada asset: {mesh_path}')
        return None, None

    root = tree.getroot()
    namespace = ''
    if root.tag.startswith('{'):
        namespace = root.tag.split('}')[0].strip('{')

    def find_tag(tag):
        if namespace:
            return root.find(f'.//{{{namespace}}}{tag}')
        return root.find(f'.//{tag}')

    up_axis_element = find_tag('up_axis')
    unit_element = find_tag('unit')

    up_axis = None
    if up_axis_element is not None and up_axis_element.text:
        up_axis = up_axis_element.text.strip().upper()

    unit_meter = None
    if unit_element is not None and 'meter' in unit_element.attrib:
        try:
            unit_meter = float(unit_element.attrib['meter'])
        except ValueError:
            pass

    print(f'Collada asset info: path={mesh_path}, up_axis={up_axis}, unit_meter={unit_meter}')
    return up_axis, unit_meter


def collada_correction_matrix(mesh_path):
    if not mesh_path or not mesh_path.lower().endswith('.dae'):
        return None

    up_axis, _unit_meter = get_collada_asset_info(mesh_path)
    if up_axis == 'Y_UP':
        return Matrix.Rotation(math.pi / 2, 4, 'X')
    if up_axis == 'X_UP':
        return Matrix.Rotation(-math.pi / 2, 4, 'Y')

    return None


def load_mesh(mesh, base_dir=None):
    mesh_filename = mesh.attrib['filename']
    mesh_path = parse_mesh_filename(mesh_filename, base_dir=base_dir)
    if not mesh_path:
        print(f'Skipping mesh import; file not found: {mesh_filename}')
        return []

    fix_orientation = True
    correction = None if fix_orientation else collada_correction_matrix(mesh_path)
    print(
        f'Importing mesh: name={mesh_filename}, path={mesh_path}, '
        f'fix_orientation={fix_orientation}, correction={correction}'
    )

    bpy.ops.wm.collada_import(filepath=mesh_path, fix_orientation=fix_orientation)
    imported_objects = list(bpy.context.selected_objects)
    mesh_objects = [o for o in imported_objects if o.type == 'MESH']

    if imported_objects and 'parent_clear' in dir(bpy.ops.object):
        select_objects(imported_objects)
        try:
            bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
        except (AttributeError, RuntimeError):
            pass

    if correction:
        for obj in mesh_objects:
            obj.matrix_world = correction @ obj.matrix_world

    if mesh_objects:
        select_objects(mesh_objects)
        bpy.ops.object.transform_apply(location=False, rotation=bool(correction), scale=True)

    objects_to_delete = [o for o in imported_objects if o.type in ('CAMERA', 'LIGHT')]
    delete_objects(objects_to_delete)
    objects = [o for o in imported_objects if o.type not in ('CAMERA', 'LIGHT')]
    return objects

def load_geometry(visual, base_dir=None):
    geometry = visual.find('geometry')

    mesh = geometry.find('mesh')
    if mesh is not None:
        return load_mesh(mesh, base_dir=base_dir)

    cylinder = geometry.find('cylinder')
    if cylinder is not None:
        length = float(cylinder.attrib['length'])
        radius = float(cylinder.attrib['radius'])
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=64,
            radius=radius,
            depth=length,
            align='WORLD',
            location=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
        )
        return [bpy.context.active_object]

    box = geometry.find('box')
    if box is not None:
        x, y, z = [float(s) for s in box.attrib['size'].split()]
        bpy.ops.mesh.primitive_cube_add(
            align='WORLD',
            location=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
        )
        cube = bpy.context.active_object
        cube.dimensions = Vector((x, y, z))
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        return [cube]

    sphere = geometry.find('sphere')
    if sphere is not None:
        radius = float(sphere.attrib['radius'])
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=3,
            radius=radius,
            align='WORLD',
            location=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
        )
        return [bpy.context.active_object]

    return []

def add_revolute_joint_bone(
    armature,
    joint,
    parent_empty,
    child_empty,
    parent_bone_name,
    tail_position=None,
    parent_driver_name=None,
    ik_settings=None,
):
    axis_element = joint.find('axis')
    if axis_element is not None and 'xyz' in axis_element.attrib:
        axis = Vector([float(s) for s in axis_element.attrib['xyz'].split()])
    else:
        axis = Vector((1.0, 0.0, 0.0))
    axis_world = child_empty.matrix_world.to_3x3() @ axis

    select_only(armature)
    bpy.ops.object.mode_set(mode='EDIT')
    driven_bone = armature.data.edit_bones.new(joint.attrib['name'])

    head = child_empty.location.copy()
    if tail_position is not None:
        tail = tail_position.copy()
    else:
        tail = child_empty.location + (child_empty.location - parent_empty.location)
    if (tail - head).length < 1e-6:
        axis_dir = axis_world.normalized() if axis_world.length else Vector((0.0, 0.0, 1.0))
        tail = head + axis_dir * 0.1

    driven_bone.head = head
    driven_bone.tail = tail
    driven_bone.head_radius = 0.001
    driven_bone.tail_radius = 0.001
    driven_bone.use_deform = True
    bone_name = driven_bone.name

    parent_bone = armature.data.edit_bones[parent_bone_name]

    driven_bone.parent = parent_bone
    driven_bone.use_connect = False

    ik_target_name = None
    joint_name = joint.attrib.get('name', '')
    if should_add_ik_for_joint(joint, ik_settings):
        ik_bone = armature.data.edit_bones.new(f'IK_{bone_name}')
        axis_dir = driven_bone.tail - driven_bone.head
        if axis_dir.length < 1e-6:
            axis_dir = axis_world.normalized() if axis_world.length else Vector((0.0, 1.0, 0.0))
        else:
            axis_dir = axis_dir.normalized()
        forward_dir = axis_dir
        ik_bone.tail = driven_bone.tail
        ik_bone.head = driven_bone.tail - forward_dir * 0.05
        ik_bone.use_deform = False
        ik_bone.use_connect = False
        ik_target_name = ik_bone.name

    bpy.ops.object.mode_set(mode='POSE')

    driven_pose = armature.pose.bones[bone_name]
    driven_pose.rotation_mode = 'XYZ'
    lock_axis, axis_index = get_bone_axis_locks(armature, bone_name, axis_world)

    lower, upper = get_joint_limit_values(joint)
    centered_min = None
    centered_max = None
    if (
        joint.attrib.get('type') in ('revolute', 'continuous')
        and lower is not None
        and upper is not None
    ):
        midpoint = (lower + upper) * 0.5
        centered_min = lower - midpoint
        centered_max = upper - midpoint
        driven_pose.bone['urdf_min'] = lower
        driven_pose.bone['urdf_max'] = upper

    apply_bone_custom_shape(driven_pose, get_control_bone_shape())
    if hasattr(driven_pose.bone, 'show_wire'):
        driven_pose.bone.show_wire = True
    apply_bone_custom_color(
        driven_pose,
        normal=(0.8, 0.1, 0.1),
        select=(0.9, 0.2, 0.2),
        active=(1.0, 0.3, 0.3),
    )

    driven_pose.lock_rotation[0] = lock_axis[0]
    driven_pose.lock_rotation[1] = lock_axis[1]
    driven_pose.lock_rotation[2] = lock_axis[2]
    driven_pose.lock_ik_x = lock_axis[0]
    driven_pose.lock_ik_y = lock_axis[1]
    driven_pose.lock_ik_z = lock_axis[2]

    driven_pose.use_ik_limit_x = False
    driven_pose.use_ik_limit_y = False
    driven_pose.use_ik_limit_z = False
    if centered_min is not None and centered_max is not None:
        if axis_index == 0:
            driven_pose.use_ik_limit_x = True
            driven_pose.ik_min_x = centered_min
            driven_pose.ik_max_x = centered_max
        elif axis_index == 1:
            driven_pose.use_ik_limit_y = True
            driven_pose.ik_min_y = centered_min
            driven_pose.ik_max_y = centered_max
        else:
            driven_pose.use_ik_limit_z = True
            driven_pose.ik_min_z = centered_min
            driven_pose.ik_max_z = centered_max

    if ik_target_name:
        existing_ik = driven_pose.constraints.get('URDF_IK')
        if existing_ik:
            driven_pose.constraints.remove(existing_ik)
        ik_constraint = driven_pose.constraints.new('IK')
        ik_constraint.name = 'URDF_IK'
        ik_constraint.influence = 0.0
        if hasattr(ik_constraint, 'mute'):
            ik_constraint.mute = True
        ik_constraint.target = armature
        ik_constraint.subtarget = ik_target_name
        ik_constraint.chain_count = 2

    if joint.attrib.get('type') in ('revolute', 'continuous'):
        apply_revolute_constraints(driven_pose, axis_index, centered_min, centered_max)

    bpy.ops.object.mode_set(mode='OBJECT')
    return bone_name, None, bone_name


def position_link_objects(visual, objects, empty, joint_name):
    import_matrices = []
    origin_matrix = get_origin_matrix(visual)
    for i, object in enumerate(objects):
        select_only(object)
        #print(bpy.context.selected_objects)
        import_matrix = object.matrix_world.copy()
        import_matrices.append(import_matrix.copy())
        object.matrix_world = empty.matrix_world @ origin_matrix @ import_matrix
        object.name = 'LINK__' + joint_name + '__' + str(i)
        bpy.context.view_layer.update()
    return import_matrices


def position_collision_objects(
    collision,
    objects,
    empty,
    joint_name,
    armature=None,
    import_matrix_override=None,
):
    origin_matrix = get_origin_matrix(collision)
    for i, object in enumerate(objects):
        select_only(object)
        import_matrix = import_matrix_override or object.matrix_world.copy()
        object.matrix_world = empty.matrix_world @ origin_matrix @ import_matrix
        object.name = 'COLLISION__' + joint_name + '__' + str(i)
        object.hide_render = True
        if hasattr(object, 'display_type'):
            object.display_type = 'WIRE'
        bpy.context.view_layer.update()
        if armature is not None:
            pose_bone = armature.pose.bones.get(joint_name)
            if pose_bone is not None:
                world_matrix = object.matrix_world.copy()
                parent_matrix = armature.matrix_world @ pose_bone.matrix
                object.parent = armature
                object.parent_type = 'BONE'
                object.parent_bone = joint_name
                try:
                    object.matrix_parent_inverse = parent_matrix.inverted()
                    object.matrix_world = world_matrix
                except (AttributeError, RuntimeError):
                    pass


def add_childjoints(
    armature,
    joints,
    links,
    link,
    empty,
    parent_bone_name,
    parent_driver_name=None,
    base_dir=None,
    ik_settings=None,
):
    childjoints = find_childjoints(joints, link)
    for childjoint in childjoints:
        new_empty = add_next_empty(empty, childjoint)

        bone_name = parent_bone_name

        # Find the childlink xml object
        childlink_name = childjoint.find('child').attrib['link']
        for childlink in links:
            if childlink.attrib['name'] == childlink_name:
                break

        tail_position = None
        childlink_joints = find_childjoints(joints, childlink_name)
        if len(childlink_joints) == 1:
            offset = get_joint_origin_offset(childlink_joints[0])
            tail_position = new_empty.matrix_world @ offset
        elif should_add_ik_for_joint(childjoint, ik_settings):
            align_names = ik_settings.get('align_names', []) if ik_settings else []
            align_filters = ik_settings.get('align_filters', []) if ik_settings else []
            align_joint = find_alignment_joint(childlink_joints, align_names, align_filters)
            if align_joint is None and childlink_joints:
                align_joint = max(
                    childlink_joints,
                    key=lambda joint: get_joint_origin_offset(joint).length,
                )
            if align_joint is not None:
                offset = get_joint_origin_offset(align_joint)
                tail_position = new_empty.matrix_world @ offset

        joint_type = childjoint.attrib.get('type', '')
        driver_name = parent_driver_name
        if joint_type in ('revolute', 'continuous', 'prismatic'):
            bone_name, _control_name, driver_name = add_revolute_joint_bone(
                armature,
                childjoint,
                empty,
                new_empty,
                parent_bone_name,
                tail_position=tail_position,
                parent_driver_name=parent_driver_name,
                ik_settings=ik_settings,
            )
        elif joint_type == 'fixed':
            bone_name = parent_bone_name
        else:
            bone_name, _control_name, driver_name = add_revolute_joint_bone(
                armature,
                childjoint,
                empty,
                new_empty,
                parent_bone_name,
                tail_position=tail_position,
                parent_driver_name=parent_driver_name,
                ik_settings=ik_settings,
            )

        visual_objects = []
        visual = childlink.find('visual')
        if visual is not None:
            visual_objects = load_geometry(visual, base_dir=base_dir)
            position_link_objects(visual, visual_objects, new_empty, bone_name)

        for collision in childlink.findall('collision'):
            collision_objects = load_geometry(collision, base_dir=base_dir)
            position_collision_objects(
                collision,
                collision_objects,
                new_empty,
                bone_name,
                armature=armature,
            )

        add_childjoints(
            armature,
            joints,
            links,
            childlink_name,
            new_empty,
            bone_name,
            parent_driver_name=driver_name,
            base_dir=base_dir,
            ik_settings=ik_settings,
        )

def assign_vertices_to_group(object, groupname):
    select_only(object)
    group = object.vertex_groups.get(groupname)
    if group is None:
        group = object.vertex_groups.new(name=groupname)
    indices = [v.index for v in bpy.context.selected_objects[0].data.vertices]
    group.add(indices, 1.0, type='ADD')


def import_urdf(
    filepath,
    ik_enabled=False,
    ik_driver_joints=None,
    ik_align_joints=None,
    ik_driver_filter='',
    ik_align_filter='',
):
    if not os.path.exists(filepath):
        print('File does not exist')
        return
    if not os.path.isfile(filepath):
        print('Selected path is not a file')
        return

    driver_names = [name for name in (ik_driver_joints or []) if name]
    align_names = [name for name in (ik_align_joints or []) if name]
    ik_settings = {
        'enabled': bool(ik_enabled),
        'driver_names': driver_names,
        'align_names': align_names,
        'driver_filters': parse_name_filters(ik_driver_filter),
        'align_filters': parse_name_filters(ik_align_filter),
    }
    if (
        ik_settings['enabled']
        and not ik_settings['driver_names']
        and not ik_settings['driver_filters']
    ):
        print('IK enabled but no driver joint filter provided; skipping IK creation.')

    base_dir = os.path.dirname(os.path.abspath(filepath))

    tree = ET.parse(filepath)
    xml_root = tree.getroot()

    links = xml_root.findall('link')
    joints = xml_root.findall('joint')

    if joints:
        rootlinks = find_rootlinks(joints)
    else:
        rootlinks = [link.attrib['name'] for link in links]

    for rootlink in rootlinks:
        bpy.context.scene.cursor.location = Vector((0.0, 0.0, 0.0))
        bpy.context.scene.cursor.rotation_euler = Vector((0.0, 0.0, 0.0))

        bpy.ops.object.empty_add(type='ARROWS', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        bpy.context.object.empty_display_size = 0.2
        empty = bpy.context.active_object
        empty.name = 'TF_' + rootlink

        bpy.ops.object.armature_add(radius=0.05, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        armature = bpy.context.active_object
        armature.name = 'robot'
        armature.data.name = 'skeleton'
        if hasattr(armature, 'show_in_front'):
            armature.show_in_front = True

        bone_name = 'base'
        select_only(armature)
        bpy.ops.object.mode_set(mode='EDIT')
        root_bone = armature.data.edit_bones[0]
        root_bone.name = bone_name
        root_bone.head = empty.location.copy() + Vector((0.0, 0.0, 0.05))
        root_bone.tail = empty.location.copy()
        bpy.ops.object.mode_set(mode='OBJECT')

        select_only(armature)
        bpy.ops.object.mode_set(mode='POSE')
        armature.pose.bones[bone_name].lock_ik_x = True
        armature.pose.bones[bone_name].lock_ik_y = True
        armature.pose.bones[bone_name].lock_ik_z = True
        bpy.ops.object.mode_set(mode='OBJECT')


        for link in links:
            if link.attrib['name'] == rootlink:
                break

        visual_objects = []
        visual = link.find('visual')
        if visual is not None:
            visual_objects = load_geometry(visual, base_dir=base_dir)
            for object in visual_objects:
                set_object_smooth_shading(object)
            position_link_objects(visual, visual_objects, empty, bone_name)

        for collision in link.findall('collision'):
            collision_objects = load_geometry(collision, base_dir=base_dir)
            position_collision_objects(
                collision,
                collision_objects,
                empty,
                bone_name,
                armature=armature,
            )


        add_childjoints(
            armature,
            joints,
            links,
            rootlink,
            empty,
            bone_name,
            parent_driver_name=None,
            base_dir=base_dir,
            ik_settings=ik_settings,
        )

        ## Skinning
        select_only(armature)

        for object in bpy.data.objects:
            if 'LINK__' in object.name:
                object.select_set(True)


        link_objects = [o for o in bpy.data.objects if 'LINK__' in o.name]
        ensure_armature_parenting(armature, link_objects, use_bone_parenting=True)

        for object in bpy.data.objects:
            if 'LINK__' in object.name:
                set_object_smooth_shading(object)

        for object in bpy.data.objects:
            if 'LINK__' in object.name:
                groupname = object.name.split('__')[1]
                assign_vertices_to_group(object, groupname)

        apply_transforms_and_rest_pose(armature, apply_transforms=True, apply_rest_pose=False)
        set_midpoint_rest_pose(armature, joints)
        apply_transforms_and_rest_pose(armature, apply_transforms=False, apply_rest_pose=True)
        align_ik_targets(armature)
        create_idle_action(armature, frame=0)
        create_calibration_action(armature, joints, frame_start=0, frame_end=50, fps=50)
        idle_action = bpy.data.actions.get('Idle')
        if idle_action is not None:
            armature.animation_data_create()
            armature.animation_data.action = idle_action

        # Delete the empties
        bpy.ops.object.select_all(action='DESELECT')
        for object in bpy.data.objects:
            if 'TF_' in object.name:
                object.select_set(True)
        bpy.ops.object.delete()
        select_only(armature)


if __name__ == '__main__':
    filepath = '/home/idlab185/ur10.urdf'
    import_urdf(filepath)
