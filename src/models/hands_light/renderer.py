# Modified from Yufei Ye's ihoi code - https://github.com/JudyYe/ihoi

import math
import pickle
import os
import numpy as np
import torch
from pytorch3d.structures import Meshes as MeshesBase
from pytorch3d.renderer import (MeshRasterizer, RasterizationSettings, PerspectiveCameras, BlendParams, MeshRenderer, SoftSilhouetteShader)

class Meshes(MeshesBase):

    def __init__(
        self,
        verts=None,
        faces=None,
        textures=None,
        *,
        verts_normals=None,
    ) -> None:
        """
        Args:
            verts:
                Can be either

                - List where each element is a tensor of shape (num_verts, 3)
                  containing the (x, y, z) coordinates of each vertex.
                - Padded float tensor with shape (num_meshes, max_num_verts, 3).
                  Meshes should be padded with fill value of 0 so they all have
                  the same number of vertices.
            faces:
                Can be either

                - List where each element is a tensor of shape (num_faces, 3)
                  containing the indices of the 3 vertices in the corresponding
                  mesh in verts which form the triangular face.
                - Padded long tensor of shape (num_meshes, max_num_faces, 3).
                  Meshes should be padded with fill value of -1 so they have
                  the same number of faces.
            textures: Optional instance of the Textures class with mesh
                texture properties.
            verts_normals:
                Optional. Can be either

                - List where each element is a tensor of shape (num_verts, 3)
                  containing the normals of each vertex.
                - Padded float tensor with shape (num_meshes, max_num_verts, 3).
                  They should be padded with fill value of 0 so they all have
                  the same number of vertices.
                Note that modifying the mesh later, e.g. with offset_verts_,
                can cause these normals to be forgotten and normals to be recalculated
                based on the new vertex positions.

        Refer to comments above for descriptions of List and Padded representations.
        """
        super().__init__(verts, faces, textures, verts_normals=verts_normals)
        # reset shape for empty meshes
        if self.isempty():

            # Identify type of verts and faces.
            if isinstance(verts, list) and isinstance(faces, list):
                if self._N > 0:
                    if not (
                        all(v.device == self.device for v in verts)
                        and all(f.device == self.device for f in faces)
                    ):
                        self._num_verts_per_mesh = torch.tensor(
                            [len(v) for v in self._verts_list], device=self.device
                        )
                        self._num_faces_per_mesh = torch.tensor(
                            [len(f) for f in self._faces_list], device=self.device
                        )
            elif torch.is_tensor(verts) and torch.is_tensor(faces):
                if self._N > 0:
                    # Check that padded faces - which have value -1 - are at the
                    # end of the tensors
                    faces_not_padded = self._faces_padded.gt(-1).all(2)
                    self._num_faces_per_mesh = faces_not_padded.sum(1)

                    self._num_verts_per_mesh = torch.full(
                        size=(self._N,),
                        fill_value=self._V,
                        dtype=torch.int64,
                        device=self.device,
                    )

            else:
                raise ValueError(
                    "Verts and Faces must be either a list or a tensor with \
                        shape (batch_size, N, 3) where N is either the maximum \
                        number of verts or faces respectively."
                )

            # if self.isempty():
            #     self._num_verts_per_mesh = torch.zeros(
            #         (0,), dtype=torch.int64, device=self.device
            #     )
            #     self._num_faces_per_mesh = torch.zeros(
            #         (0,), dtype=torch.int64, device=self.device
            #     )

            # Set the num verts/faces on the textures if present.
            if textures is not None:
                shape_ok = self.textures.check_shapes(self._N, self._V, self._F)
                if not shape_ok:
                    msg = "Textures do not match the dimensions of Meshes."
                    raise ValueError(msg)

                self.textures._num_faces_per_mesh = self._num_faces_per_mesh.tolist()
                self.textures._num_verts_per_mesh = self._num_verts_per_mesh.tolist()
                self.textures.valid = self.valid

            if verts_normals is not None:
                self._set_verts_normals(verts_normals)
        else:
            pass


class DiffRenderer(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args
        # settings taken from Yufei's ihoi code
        blend_params = BlendParams(sigma=1e-5, gamma=1e-4)
        dist_eps = 1e-6
        raster_settings = RasterizationSettings(
            image_size=args.img_res,
            blur_radius=math.log(1. / dist_eps - 1.) * blend_params.sigma,
            faces_per_pixel=10,
            perspective_correct=False,
        )
        self.rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=SoftSilhouetteShader())

    def forward(self, meshes, cameras):
        ##### this code block also works, but slower #####
        # fragments = self.rasterizer(meshes, cameras=cameras)
        # out = {}
        # out['frag'] = fragments
        # shader = SoftPhongShader(cameras.device, lights=ambient_light(meshes.device, cameras))
        # image = shader(fragments, meshes, cameras=cameras, )
        # rgb, mask = flip_transpose_canvas(image)
        # out['image'] = rgb # all 1s if SoftSilhouetteShader is used, doesn't matter since only care about mask
        # out['mask'] = mask
        ###### end of code block ######

        # render image
        image = self.renderer(meshes, cameras=cameras)
        rgb, mask = flip_transpose_canvas(image) # align the coordinate system of pytorch3d and camera
        out = {}
        out['image'] = rgb # all 1s if SoftSilhouetteShader is used, doesn't matter since only care about mask
        out['mask'] = mask
        
        return out

class MANORenderer(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args
        self.renderer = DiffRenderer(args)

        # change this to default MANO faces or use the faces from hamer (https://github.com/geopavlakos/hamer/blob/dc19e5686198a7c3fc3938bff3951f238a85fd11/hamer/utils/renderer.py#L136)
        # use build_mano_aa() from common.body_models.py to get mano and mano.faces to get the faces for left and right hands
        default_mano_faces_file = 'default_mano_faces.pkl'
        with open(default_mano_faces_file, 'rb') as f:
            default_mano_faces = pickle.load(f)
        self.mano_faces_r = torch.from_numpy(default_mano_faces['right'].astype(np.int16)).long()
        self.mano_faces_l = torch.from_numpy(default_mano_faces['left'].astype(np.int16)).long()

        self.intrx_to_ndc = torch.FloatTensor([
                                [2 / args.img_res, 0, -1],
                                [0, 2 / args.img_res, -1],
                                [0, 0, 1],
                            ])

    def forward(self, mano_output, meta_info, is_right=True):
        bz = len(meta_info['imgname'])
        if is_right:
            vertices = mano_output['mano.v3d.cam.r']
            faces = torch.stack([self.mano_faces_r]*bz, dim=0).to(vertices.device)
        else:
            vertices = mano_output['mano.v3d.cam.l']
            faces = torch.stack([self.mano_faces_l]*bz, dim=0).to(vertices.device)


        intrx_to_ndc = torch.stack([self.intrx_to_ndc]*bz, dim=0).to(vertices.device)
        K = torch.matmul(intrx_to_ndc, meta_info['intrinsics'])
        all_focal = torch.diagonal(K, dim1=-1, dim2=-2)[:,:2]
        all_principal = K[:, :2, 2]
        cameras = PerspectiveCameras(all_focal, all_principal, device=vertices.device)

        # create Meshes object
        meshes = Meshes(verts=vertices, faces=faces)
        meshes.textures = torch.zeros_like(vertices) # dummy texture
        
        out = self.renderer(meshes, cameras)
        return out

def flip_transpose_canvas(image, rgba=True):
    image = torch.flip(image, dims=[1, 2])  # flip up-down, and left-right
    image = image.transpose(-1, -2).transpose(-2, -3)  # H, 4, W --> 4, H, W
    if rgba:
        rgb, mask = torch.split(image, [image.size(1) - 1, 1], dim=1)  # [0-1]
        return rgb, mask
    else:
        return image