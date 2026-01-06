from . import import_urdf
import bpy
import os
import xml.etree.ElementTree as ET
from bpy_extras.io_utils import ImportHelper
from bpy.types import Operator, OperatorFileListElement, PropertyGroup, UIList
from bpy.props import BoolProperty, CollectionProperty, IntProperty, StringProperty


def populate_joint_collection(operator):
    selected = getattr(operator, "selected_filepath", "")
    filepath = selected or resolve_filepath(operator)
    if not filepath or not os.path.isfile(filepath):
        filepath = capture_selected_filepath(operator, bpy.context)
    if filepath and filepath != selected:
        operator.selected_filepath = filepath

    if filepath == operator.last_filepath and operator.joint_items:
        operator.parse_error = ''
        return

    operator.joint_items.clear()
    operator.parse_error = ''

    if not filepath or not os.path.exists(filepath) or not os.path.isfile(filepath):
        if filepath:
            operator.parse_error = 'Please select a valid URDF file.'
        operator.last_filepath = filepath
        return

    try:
        tree = ET.parse(filepath)
        xml_root = tree.getroot()
    except (ET.ParseError, OSError, ValueError) as exc:
        operator.parse_error = f'Failed to parse URDF: {exc}'
        operator.last_filepath = filepath
        return

    joints = xml_root.findall('joint')
    names = sorted(
        {joint.attrib.get('name', '') for joint in joints if joint.attrib.get('name')}
    )
    for name in names:
        item = operator.joint_items.add()
        item.name = name
        if "_calf_" in name:
            item.use_driver = True
            item.use_align = True
        else:
            item.use_driver = False
            item.use_align = False

    operator.last_filepath = filepath


def resolve_filepath(operator):
    filepath = bpy.path.abspath(operator.filepath) if operator.filepath else ''
    if filepath and os.path.isfile(filepath):
        return filepath

    directory = filepath if filepath and os.path.isdir(filepath) else ''
    if not directory:
        directory = bpy.path.abspath(operator.directory) if getattr(operator, "directory", "") else ''

    files = getattr(operator, "files", [])
    if directory and files:
        name = getattr(files[0], "name", "")
        if name:
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                return candidate

    return filepath if filepath else directory


def capture_selected_filepath(operator, context=None):
    raw = operator.filepath or ''
    if raw and os.path.isfile(raw):
        return raw
    if raw:
        raw_stripped = raw.rstrip("\\/")
        if raw_stripped and os.path.isfile(raw_stripped):
            return raw_stripped

    directory = operator.directory or (raw if raw and os.path.isdir(raw) else '')
    if directory:
        files = getattr(operator, "files", [])
        if files:
            name = getattr(files[0], "name", "")
            if name:
                candidate = os.path.join(directory, name)
                if os.path.isfile(candidate):
                    return candidate
        name = getattr(operator, "filename", "") or os.path.basename(raw)
        if name:
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                return candidate

    abs_raw = bpy.path.abspath(raw) if raw else ''
    if abs_raw and os.path.isfile(abs_raw):
        return abs_raw
    if abs_raw:
        abs_stripped = abs_raw.rstrip("\\/")
        if abs_stripped and os.path.isfile(abs_stripped):
            return abs_stripped

    if directory:
        dir_abs = bpy.path.abspath(directory)
        files = getattr(operator, "files", [])
        if files:
            name = getattr(files[0], "name", "")
            if name:
                candidate = os.path.join(dir_abs, name)
                if os.path.isfile(candidate):
                    return candidate
        name = getattr(operator, "filename", "") or os.path.basename(raw)
        if name:
            candidate = os.path.join(dir_abs, name)
            if os.path.isfile(candidate):
                return candidate

    if context is not None:
        candidate = filepath_from_context(context)
        if candidate:
            return candidate

    return abs_raw or raw or directory


def filepath_from_context(context):
    space = getattr(context, "space_data", None)
    if not space or getattr(space, "type", "") != 'FILE_BROWSER':
        return ''
    params = getattr(space, "params", None)
    if not params:
        return ''
    for attr in ("filepath",):
        value = getattr(params, attr, "")
        if value and os.path.isfile(value):
            return value
    directory = getattr(params, "directory", b"") or getattr(params, "directory", "")
    filename = getattr(params, "filename", "")
    if isinstance(directory, (bytes, bytearray)):
        try:
            directory = directory.decode(errors="ignore")
        except (AttributeError, UnicodeDecodeError):
            directory = str(directory)
    if directory and filename:
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate):
            return candidate
    return ''


class URDFJointItem(PropertyGroup):
    name: StringProperty(default="")
    use_driver: BoolProperty(default=False)
    use_align: BoolProperty(default=False)


class URDF_UL_JointList(UIList):
    def draw_item(
        self,
        context,
        layout,
        data,
        item,
        icon,
        active_data,
        active_propname,
        index,
    ):
        split = layout.split(factor=0.6)
        split.label(text=item.name)
        row = split.row(align=True)
        row.prop(item, "use_driver", text="Driver")
        row.prop(item, "use_align", text="Align")


class URDF_OT_FilebrowserImporter(Operator, ImportHelper):
    bl_idname = "import.urdf_filebrowser"
    bl_label = "Import URDF"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ".urdf"

    filter_glob: StringProperty(
        default="*.urdf",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )
    directory: StringProperty(
        subtype='DIR_PATH',
        default="",
        options={'HIDDEN'},
    )
    files: CollectionProperty(
        type=OperatorFileListElement,
        options={'HIDDEN'},
    )

    add_ik: BoolProperty(
        name="Add IK Targets",
        description="Create IK target bones and constraints during import",
        default=True,
    )
    joint_items: CollectionProperty(type=URDFJointItem)
    joint_index: IntProperty(default=0)
    selected_filepath: StringProperty(
        default="",
        options={'HIDDEN'},
    )
    last_filepath: StringProperty(
        default="",
        options={'HIDDEN'},
    )
    parse_error: StringProperty(
        default="",
        options={'HIDDEN'},
    )

    def draw(self, context):
        populate_joint_collection(self)
        layout = self.layout
        layout.prop(self, "add_ik")
        if self.add_ik:
            filename = os.path.basename(self.selected_filepath) if self.selected_filepath else '(no file selected)'
            layout.label(text=f'URDF: {filename}')
            layout.template_list(
                "URDF_UL_JointList",
                "",
                self,
                "joint_items",
                self,
                "joint_index",
                rows=8,
            )
            if self.parse_error:
                layout.label(text=self.parse_error, icon='ERROR')
                if self.selected_filepath:
                    layout.label(text=self.selected_filepath)
            elif not self.joint_items:
                layout.label(text='No joints found in the selected URDF.', icon='INFO')

    def invoke(self, context, event):
        self.selected_filepath = ""
        self.last_filepath = ""
        self.parse_error = ""
        self.joint_items.clear()
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        self.selected_filepath = capture_selected_filepath(self, context)
        if self.selected_filepath:
            self.filepath = self.selected_filepath
        filepath = self.selected_filepath or resolve_filepath(self)
        if not filepath or not os.path.isfile(filepath):
            self.report({'ERROR'}, 'Please select a valid URDF file.')
            return {'CANCELLED'}
        populate_joint_collection(self)
        driver_joints = [item.name for item in self.joint_items if item.use_driver]
        align_joints = [item.name for item in self.joint_items if item.use_align]
        import_urdf.import_urdf(
            filepath,
            ik_enabled=self.add_ik,
            ik_driver_joints=driver_joints,
            ik_align_joints=align_joints,
        )
        return {'FINISHED'}

