from argparse import Namespace
from typing import Optional
import torch


def get_model(
        arch: str,
        patch_size: Optional[int] = None,
        training_method: Optional[str] = None,
        configs: Optional[Namespace] = None,
        **kwargs
):
    if arch == "maskformer":
        assert configs is not None
        from networks.maskformer.maskformer import MaskFormer
        model = MaskFormer(
            n_queries=configs.n_queries,
            n_decoder_layers=configs.n_decoder_layers,
            learnable_pixel_decoder=configs.learnable_pixel_decoder,
            lateral_connection=configs.lateral_connection,
            return_intermediate=configs.loss_every_decoder_layer,
            scale_factor=configs.scale_factor,
            abs_2d_pe_init=configs.abs_2d_pe_init,
            use_binary_classifier=configs.use_binary_classifier,
            arch=configs.arch,
            training_method=configs.training_method,
            patch_size=configs.patch_size
        )

        for n, p in model.encoder.named_parameters():
            p.requires_grad_(True)

    elif "vit" in arch:
        import networks.vision_transformer as vits
        import networks.timm_deit as timm_deit
        if training_method == "dino":
            arch = arch.replace("vit", "deit") if arch.find("small") != -1 else arch
            model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
            load_model(model, arch, patch_size)

        elif training_method == "deit":
            assert patch_size == 16
            model = timm_deit.deit_small_distilled_patch16_224(True)

        elif training_method == "supervised":
            assert patch_size == 16
            state_dict: dict = torch.load(
                "/users/gyungin/selfmask/networks/pretrained/deit_small_patch16_224-cd65a155.pth"
            )["model"]
            for k in list(state_dict.keys()):
                if k in ["head.weight", "head.bias"]:  # classifier head, which is not used in our network
                    state_dict.pop(k)

            model = get_model(arch="vit_small", patch_size=16, training_method="dino")
            model.load_state_dict(state_dict=state_dict, strict=True)

        else:
            raise NotImplementedError
        print(f"{arch}_p{patch_size}_{training_method} is built.")

    elif arch == "resnet50":
        from networks.resnet import ResNet50
        assert training_method in ["mocov2", "swav", "supervised"]
        model = ResNet50(training_method)

    else:
        raise ValueError(f"{arch} is not supported arch. Choose from [maskformer, resnet50, vit, dino]")
    return model


def load_model(model, arch: str, patch_size: int) -> None:
    url = None
    if arch == "deit_small" and patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    elif arch == "deit_small" and patch_size == 8:
        # model used for visualizations in our paper
        url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
    elif arch == "vit_base" and patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    elif arch == "vit_base" and patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    if url is not None:
        print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        model.load_state_dict(state_dict, strict=True)
    else:
        print("There is no reference weights available for this model => We use random weights.")