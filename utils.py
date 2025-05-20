import typing as tp
import numpy as np
import torch
import clip
from pathlib import Path
from scipy.spatial.transform import Rotation


class MiniCam:
    def __init__(self, width: int, height: int, fovx: float, fovy: float, world_view_transform: np.ndarray) -> None:
        self.image_width = width
        self.image_height = height
        self.FoVx = fovx
        self.FoVy = fovy
        self.z_near = 0.01
        self.z_far = 100.0
        self.world_view_transform = torch.tensor(world_view_transform, dtype=torch.float32).transpose(0, 1).cuda()
        self.projection_matrix = self.get_projection_matrix(z_near=self.z_near, z_far=self.z_far, fovx=self.FoVx,
                                                            fovy=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = self.world_view_transform \
                                       .unsqueeze(0) \
                                       .bmm(self.projection_matrix.unsqueeze(0)) \
                                       .squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @staticmethod
    def get_projection_matrix(z_near: float, z_far: float, fovx: float, fovy: float) -> torch.Tensor:
        tan_half_fovy = np.tan((fovy / 2))
        tan_half_fovx = np.tan((fovx / 2))

        top = tan_half_fovy * z_near
        bottom = -top
        right = tan_half_fovx * z_near
        left = -right

        proj_mtx = torch.zeros(4, 4)

        z_sign = 1.0

        proj_mtx[0, 0] = 2.0 * z_near / (right - left)
        proj_mtx[1, 1] = 2.0 * z_near / (top - bottom)
        proj_mtx[0, 2] = (right + left) / (right - left)
        proj_mtx[1, 2] = (top + bottom) / (top - bottom)
        proj_mtx[3, 2] = z_sign
        proj_mtx[2, 2] = z_sign * z_far / (z_far - z_near)
        proj_mtx[2, 3] = -(z_far * z_near) / (z_far - z_near)
        return proj_mtx


def load_poses(path: Path) -> tuple[np.ndarray, float]:
    poses = []

    pose_data = np.loadtxt(path, delimiter=' ', dtype=np.unicode_)
    pose_vecs = pose_data[:, 1:].astype(np.float32)
    tstamp = pose_data[:, 0].astype(np.float64)

    for pose_vec in pose_vecs:
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pose_vec[3:]).as_matrix()
        pose[:3, 3] = pose_vec[:3]
        poses.append(pose)

    return poses, tstamp


def get_world2view(R: np.ndarray,
                   t: np.ndarray,
                   translate: np.ndarray = np.array([.0, .0, .0]),
                   scale: float = 1.0) -> np.ndarray:
    C2W = np.eye(4)
    C2W[:3, :3] = R
    C2W[:3, 3] = t

    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center

    W2C = np.linalg.inv(C2W)

    return np.float32(W2C)

def focal2fov(focal: float, pixels: int) -> float:
    return 2 * np.arctan(pixels / (2 * focal))


def build_text_embedding(categories, dino_model, device="cuda"):
    """Build text embeddings for given categories."""
    tokens = []
    templates = [
        "itap of a {}.",
        "a bad photo of a {}.",
        "a origami {}.", 
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}.",
    ]
    
    for category in categories:
        tokens.append(
            clip.tokenize([template.format(category) for template in templates])
        )
    tokens = torch.stack(tokens)
    text_emb = dino_model.build_text_embedding(tokens)
    return text_emb
