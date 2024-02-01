import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt

from smpl.model import SMPLModel
from typing import Tuple


class SMPLRender:
    def __init__(self, size: Tuple[int, int], smpl_model: SMPLModel):
        self.size = size
        self.smpl_model = smpl_model

        # # Create a scene
        self.scene = pyrender.Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))

        # Create a camera
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 5.0)
        self.camera_pose = np.eye(4)

        self.scene.add(self.camera, pose=self.camera_pose)

        # Create a light
        self.light = pyrender.SpotLight(
            color=np.ones(3),
            intensity=3.0,
            innerConeAngle=np.pi / 16.0,
            outerConeAngle=np.pi / 6.0,
        )

        self.scene.add(self.light, pose=self.camera_pose)

    def render(self, beta=None, pose=None):
        if beta is None:
            beta = np.random.randn(self.smpl_model.shapedirs.shape[-1]) * 0.03
        if pose is None:
            pose = np.random.randn(24, 3) * 0.2
            pose[0] = 0

        v = self.smpl_model.set_params(beta=beta, pose=pose)

        mesh = pyrender.Mesh.from_trimesh(
            trimesh.Trimesh(v, self.smpl_model.faces), smooth=False
        )

        mesh_pose = np.array(
            [
                [0.2, 0.0, 0.0, 0.0],
                [0.0, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.2, -1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        self.scene.add(mesh, pose=mesh_pose)
        r = pyrender.OffscreenRenderer(*self.size)
        color, depth = r.render(self.scene)
        plt.imshow(color, cmap=plt.cm.gray_r)
        plt.show()


if __name__ == "__main__":
    smpl_model = SMPLModel("models/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
    smpl_render = SMPLRender((1080, 720), smpl_model)
    smpl_render.render()
