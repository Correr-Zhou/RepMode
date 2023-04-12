import tifffile


class TifReader:
    """This class is used to open and process the contents of a tif file.

    Examples:
        reader = tifReader.TifReader(path="file.tif")
        file_image = reader.load()

        with tifReader.TifReader(path="file2.tif") as reader:
            file2_image = reader.load()

    The load function will get a 3D ZYX array from a tif file.
    """

    def __init__(self, file_path):
        # nothing yet!
        self.filePath = file_path
        self.tif = tifffile.TiffFile(self.filePath)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.tif.close()

    def load(self):
        """This will get an entire z stack from a tif file.

        :return: A 3D ZYX slice from the tif file.
        """
        return self.tif.asarray()

    def load_slice(self, z=0, c=0, t=0):
        """This will get a single slice out of the z stack of a tif file.

        :param z: The z index within the tiff stack
        :param c: An arbitrary c index that does nothing
        :param t: An arbitrary t index that does nothing
        :return: A 2D YX slice from the tiff file.
        """
        index = z
        data = self.tif.asarray(key=index)
        return data

    def get_metadata(self):
        return None

    def size_z(self):
        return len(self.tif.pages)

    def size_c(self):
        return 1

    def size_t(self):
        return 1

    def size_x(self):
        return self.tif.pages[0].shape[1]

    def size_y(self):
        return self.tif.pages[0].shape[0]

    def dtype(self):
        return self.tif.pages[0].dtype
