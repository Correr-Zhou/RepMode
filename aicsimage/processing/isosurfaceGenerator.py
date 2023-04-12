# author: Zach Crabtree zacharyc@alleninstitute.org

from skimage import measure
import os
from .aicsImage import AICSImage

class Mesh:
    """

    A cell mesh class contains the necessary information to generate a display a 3D isosurface.

    Example:
        image = AICSImage("some/bio/image.ome.tif")
        mesh0 = generate_mesh(image, isovalue=0, channel=0)
        # will generate a different mesh due to the different isovalue
        mesh1 = generate_mesh(image, isovalue=1, channel=0)
        mesh0.save_as_obj("some/bio/image/mesh.obj")
        # mesh.obj can be imported into 3D viewers and represents a 3D rendering of image.ome.tif
    """

    def __init__(self, verts, faces, normals, values):
        self.verts = verts
        self.faces = faces
        self.normals = normals
        self.values = values

    def save_as_obj(self, file_path):
        """
        Save a mesh object as an .obj file
        :param file_path: The filepath to the saved file
        """
        if not file_path.endswith(".obj"):
            if file_path.rfind('.') != -1:
                file_path = os.path.splitext(file_path)[0] + ".obj"
            else:
                file_path += ".obj"
        with open(file_path, "w") as writer:
            writer.write("# OBJ file\n")
            writer.write("g Object001\n")
            for v in self.verts:
                writer.write("v  {:.6f}  {:.6f}  {:.6f}\n".format(v[0], v[1], v[2]))
            for n in self.normals:
                writer.write("vn  {:.6f}  {:.6f}  {:.6f}\n".format(n[0], n[1], n[2]))
            for f in self.faces:
                # obj file vertex arrays are not 0-indexed :( must add 1 in order to reference the right vertices
                writer.write("f  {}  {}  {}\n".format(f[0]+1, f[1]+1, f[2]+1))


def generate_mesh(image, isovalue=0, channel=0):
    """
    Creates and returns a Mesh object
    :param image: an AICSImage object
    :param isovalue: The value that is used to pick the isosurface returned by the marching cubes algorithm
                     For more info: https://www.youtube.com/watch?v=5fNbCFjqWao @ 40:00 mins
    :param channel: The channel in the image that is used to extract the isosurface
    :return: A Mesh object
    """
    if not isinstance(image, AICSImage):
        raise ValueError("Meshes can only be generated with AICSImage objects!")
    if channel >= image.size_c:
        raise IndexError("Channel provided for mesh generation is out of bounds for image data!")
    image_stack = image.get_image_data("ZYX", C=channel)
    # Use marching cubes to obtain the surface mesh of the membrane wall
    verts, faces, normals, values = measure.marching_cubes(image_stack, isovalue, allow_degenerate=False)
    return Mesh(verts, faces, normals, values)

