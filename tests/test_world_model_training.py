from pathlib import Path

import torch

from data.sequence_dataset import WorldModelSequenceDataset
from models.world_model import EncoderGruDecoderWorldModel, WorldModelConfig
from training.world_model_training import (
    TrainingConfig,
    build_dataloader,
    compute_world_model_loss,
    compute_world_model_loss_and_metrics,
    load_checkpoint,
    save_checkpoint,
    set_deterministic_seed,
)


DATA_DIR = Path("data/generated")


def test_npz_loading_and_sequence_sample_shapes() -> None:
    dataset = WorldModelSequenceDataset(DATA_DIR, "train", sequence_length=8)
    sample = dataset[0]

    assert sample["observations"].shape == (8, 4, 7, 7)
    assert sample["actions"].shape == (8,)
    assert sample["next_observations"].shape == (8, 4, 7, 7)


def test_sequence_construction_count_and_no_crossing_episode_boundaries() -> None:
    dataset = WorldModelSequenceDataset(DATA_DIR, "train", sequence_length=8)

    assert len(dataset) == 210 * (50 - 8 + 1)
    first_info = dataset.sequence_infos[0]
    episode_ids = dataset.episode_ids[first_info.start_index:first_info.end_index]
    time_steps = dataset.time_steps[first_info.start_index:first_info.end_index]

    assert len(set(str(value) for value in episode_ids)) == 1
    assert time_steps.tolist() == list(range(time_steps[0], time_steps[0] + 8))


def test_action_alignment_and_next_observation_alignment() -> None:
    dataset = WorldModelSequenceDataset(DATA_DIR, "train", sequence_length=8)
    sample = dataset[0]

    assert torch.equal(sample["actions"], torch.tensor(dataset.actions[:8], dtype=torch.int64))
    assert torch.equal(sample["observations"][1], sample["next_observations"][0])


def test_model_forward_shapes_and_eval_determinism() -> None:
    set_deterministic_seed(7)
    dataset = WorldModelSequenceDataset(DATA_DIR, "train", sequence_length=8)
    sample = dataset[0]
    model = EncoderGruDecoderWorldModel(WorldModelConfig())
    model.eval()

    observations = sample["observations"].unsqueeze(0)
    actions = sample["actions"].unsqueeze(0)
    logits_a, hidden_a = model(observations, actions)
    logits_b, hidden_b = model(observations, actions)

    assert logits_a.shape == (1, 8, 4, 7, 7)
    assert hidden_a.shape == (1, 1, model.config.gru_hidden_dim)
    assert torch.equal(logits_a, logits_b)
    assert torch.equal(hidden_a, hidden_b)


def test_loss_computation_and_optimizer_step() -> None:
    set_deterministic_seed(11)
    dataset = WorldModelSequenceDataset(DATA_DIR, "train", sequence_length=8)
    loader = build_dataloader(dataset, batch_size=2, shuffle=False, seed=11)
    batch = next(iter(loader))

    model = EncoderGruDecoderWorldModel(WorldModelConfig())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    logits, _ = model(batch["observations"], batch["actions"])
    loss, metrics = compute_world_model_loss_and_metrics(logits, batch["next_observations"])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() > 0.0
    assert "overall_binary_accuracy" in metrics
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_checkpoint_save_and_reload(tmp_path: Path) -> None:
    set_deterministic_seed(13)
    dataset = WorldModelSequenceDataset(DATA_DIR, "train", sequence_length=8)
    sample = dataset[0]

    model = EncoderGruDecoderWorldModel(WorldModelConfig())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    logits, _ = model(sample["observations"].unsqueeze(0), sample["actions"].unsqueeze(0))
    loss, _ = compute_world_model_loss_and_metrics(logits, sample["next_observations"].unsqueeze(0))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    training_config = TrainingConfig(
        data_dir="data/generated",
        epochs=1,
        batch_size=2,
        sequence_length=8,
        learning_rate=1e-3,
        device="cpu",
        seed=13,
        checkpoint_dir=str(tmp_path),
        smoke_test=True,
    )
    checkpoint_path, metadata_path = save_checkpoint(
        checkpoint_dir=tmp_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        validation_metrics={"loss": 0.5, "overall_binary_accuracy": 0.5},
        training_config=training_config,
        imbalance_settings={"loss_mode": "standard_bce", "computed_pos_weights": {}, "pos_weight_cap": 25.0},
    )

    reloaded_model = EncoderGruDecoderWorldModel(WorldModelConfig())
    reloaded_optimizer = torch.optim.Adam(reloaded_model.parameters(), lr=1e-3)
    payload = load_checkpoint(checkpoint_path, reloaded_model, reloaded_optimizer)

    assert checkpoint_path.exists()
    assert metadata_path.exists()
    assert payload["epoch"] == 1
    for parameter_a, parameter_b in zip(model.parameters(), reloaded_model.parameters()):
        assert torch.allclose(parameter_a, parameter_b)


def test_weighted_bce_matches_pytorch_reference() -> None:
    logits = torch.tensor([[[[[0.0]], [[0.5]], [[-0.5]], [[1.0]]]]], dtype=torch.float32)
    targets = torch.tensor([[[[[1.0]], [[0.0]], [[1.0]], [[0.0]]]]], dtype=torch.float32)
    pos_weight = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float32).view(1, 1, 4, 1, 1)

    expected = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, targets)
    actual = compute_world_model_loss(logits, targets, loss_mode="weighted_bce", pos_weight=pos_weight)

    assert torch.allclose(actual, expected)
