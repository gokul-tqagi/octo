# finetune_xarm_config.py: Octo finetune config for xarm place-object-in-toolbox dataset.
# finetune_xarm_config.py: Uses front+wrist cameras, 7-dim delta EEF actions, language conditioning.

from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from octo.utils.spec import ModuleSpec


def get_config(config_string="full,language_conditioned"):
    mode, task = config_string.split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    # xarm place-object-in-toolbox dataset
    # Extracted from mcap bags via experiments/data/mcap_to_rlds.py
    FINETUNING_KWARGS = {
        "name": "xarm_place_toolbox",
        "data_dir": placeholder(str),  # path to RLDS output dir
        "image_obs_keys": {"primary": "image_0", "wrist": "image_1"},
        "proprio_obs_key": "proprio",
        "language_key": "language_instruction",
        "action_proprio_normalization_type": "normal",
        # Normalize pos+rot deltas, but NOT gripper (last dim)
        "action_normalization_mask": [True, True, True, True, True, True, False],
        # Standardization transform: maps state -> proprio
        "standardize_fn": ModuleSpec.create(
            "scripts.xarm_standardization_transforms:xarm_place_toolbox_transform",
        ),
    }

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    elif mode == "head_mlp_only":
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )
    else:
        raise ValueError("Invalid mode")

    # 37 episodes at 2Hz ≈ 45 steps each ≈ 1665 total frames
    # With batch_size=64, ~26 batches/epoch. 10k steps = ~384 epochs.
    max_steps = FieldReference(10000)
    window_size = FieldReference(default=1)

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        batch_size=64,  # smaller batch for small dataset
        shuffle_buffer_size=5000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=2000,
        save_interval=2000,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="octo_finetune_xarm",
            group=placeholder(str),
            entity=placeholder(str),
        ),
        dataset_kwargs=FINETUNING_KWARGS,
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=3e-4,
                warmup_steps=1000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=8,
        ),
        viz_kwargs=dict(
            eval_batch_size=32,
            trajs_for_metrics=37,  # all episodes
            trajs_for_viz=8,
            samples_per_state=8,
        ),
    )

    if task == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 1.0
    elif task == "language_conditioned":
        goal_relabeling_strategy = None
        keep_image_prob = 0.0
    elif task == "multimodal":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.5
    else:
        raise ValueError("Invalid modality")

    traj_transform_kwargs = dict(
        window_size=window_size,
        action_horizon=4,
        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
    )

    workspace_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    wrist_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    frame_transform_kwargs = dict(
        resize_size={
            "primary": (256, 256),
            "wrist": (128, 128),
        },
        image_augment_kwargs=dict(
            primary=workspace_augment_kwargs,
            wrist=wrist_augment_kwargs,
        ),
    )
    config["frame_transform_threads"] = 16

    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs
    return ConfigDict(config)
