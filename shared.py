"""sets this repo up for processing, unzips"""

import os
import pathlib
import tempfile
from zipfile import ZipFile
import shutil


def unzip(
    zippath: os.PathLike, pathto: os.PathLike | None = None, flatten: bool = False
) -> str:
    """unzips a file and returns the path to the file

    args:
        - zippath (PathLike)
        - pathto (Pathlike | None): inferred as the zip name strip the ext into /tmp
        - flatten (bool): ignores structure
    returns:
        - pathto zipfile
    """
    pathto = pathto or os.path.join(
        tempfile.gettempdir(), pathlib.Path(zippath).name.split(".")[0]
    )

    def _unzip(_zip_file: ZipFile):
        """unzip wrapper to break out of `with`"""
        if not flatten:
            _zip_file.extractall(pathto)
            return
        for member in _zip_file.namelist():
            filename = os.path.basename(member)
            # skip directories
            if not filename:
                continue

            # copy file (taken from zipfile's extract)
            source = _zip_file.open(member)
            target = open(os.path.join(pathto, filename), "wb")
            with source, target:
                shutil.copyfileobj(source, target)

    with ZipFile(zippath) as zip_file:
        _unzip(zip_file)
    return pathto
