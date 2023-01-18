# Die files from: https://github.com/garykac/3d-cubes.
# Cylinder file from: https://github.com/jlamarche/Old-Blog-Code/blob/master/Wavefront%20OBJ%20Loader/Models/Cylinder.obj.

import os
import pandas as pd
import sys

from PIL import Image
from renderer import Renderer
from renderer_settings import *
from scipy.spatial.transform import Rotation

IMG_SIZE = 224


def main(which_obj):
    # Render object in default pose.
    renderer = Renderer(
        camera_distance=8,
        angle_of_view=ANGLE_OF_VIEW / 2,
        dir_light=DIR_LIGHT,
        dif_int=DIF_INT,
        amb_int=AMB_INT,
        default_width=WINDOW_SIZE,
        default_height=WINDOW_SIZE,
        cull_faces=CULL_FACES,
    )
    obj_mtl_path = f"{which_obj}_obj/{which_obj}"
    renderer.set_up_obj(f"{obj_mtl_path}.obj", f"{obj_mtl_path}.mtl")
    image = renderer.render(0.0, 0.0, 0.0).resize(
        (IMG_SIZE, IMG_SIZE), resample=Image.LANCZOS
    )
    image.show()

    data_dir = which_obj
    dataset2samples = {"train": 500000, "valid": 10000, "test": 10000}
    for (dataset, samples) in dataset2samples.items():
        imgs_dir = f"{data_dir}/images/{dataset}"
        os.makedirs(imgs_dir)
        z_len = len(str(samples - 1))

        rs = Rotation.random(samples)
        Rs = rs.as_matrix()
        yprs = rs.as_euler("YXZ")

        data = []
        for samp in range(samples):
            if samp % 500 == 0:
                print(samp)

            renderer.prog["R_obj"].write(Rs[samp].T.astype("f4").tobytes())
            image = renderer.render(0.0, 0.0, 0.0).resize(
                (IMG_SIZE, IMG_SIZE), resample=Image.LANCZOS
            )
            img_f = f"{str(samp).zfill(z_len)}.png"
            image.save(f"{imgs_dir}/{img_f}")
            (yaw, pitch, roll) = yprs[samp]
            data.append({"img_f": img_f, "yaw": yaw, "pitch": pitch, "roll": roll})

        data = pd.DataFrame(data)
        data.to_csv(f"{data_dir}/metadata_{dataset}.csv", index=False)


if __name__ == "__main__":
    which_obj = sys.argv[1]
    main(which_obj)
