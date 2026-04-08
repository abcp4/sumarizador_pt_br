from __future__ import annotations

import argparse
import contextlib
import datetime
import glob
import json
import math
import os
import random
import signal
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass
class TrainConfig:
    model_name: str
    data_glob: str
    output_dir: str
    max_source_length: int
    max_target_length: int
    per_device_train_batch_size: int
    per_device_val_batch_size: int
    ddp_find_unused_parameters: bool
    min_gradient_accumulation_steps: int
    max_gradient_accumulation_steps: int
    grad_accum_growth_factor: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    stable_ratio: float
    decay_ratio: float
    force_decay_on_resume: bool
    resume_load_scheduler_state: bool
    max_grad_norm: float
    save_every_updates: int
    log_every_updates: int
    preview_every_updates: int
    preview_max_new_tokens: int
    preview_log_file: str
    val_ratio: float
    seed: int
    num_workers: int
    resume: str


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Treino de sumarizacao multi-GPU com retomada para vast.ai")
    parser.add_argument("--model-name", type=str, default="/workspace/t5gemma-2-270m-270m")
    parser.add_argument("--data-glob", type=str, default="datasets/*/*/*.json")
    parser.add_argument("--output-dir", type=str, default="checkpoints/t5gemma2-270m-sumarios-ddp")
    parser.add_argument("--max-source-length", type=int, default=3072)
    parser.add_argument("--max-target-length", type=int, default=736)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-val-batch-size", type=int, default=1)
    parser.add_argument(
        "--ddp-find-unused-parameters",
        action="store_true",
        default=True,
        help="Ativa find_unused_parameters no DDP para evitar erro de reducao em modelos com parametros condicionalmente nao usados.",
    )
    parser.add_argument(
        "--no-ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action="store_false",
        help="Desativa find_unused_parameters no DDP para reduzir overhead quando voce tem certeza de que todos os parametros sao sempre usados.",
    )
    parser.add_argument("--min-gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--grad-accum-growth-factor", type=int, default=2)
    parser.add_argument(
        "--gradient-accumulation-steps",
        dest="min_gradient_accumulation_steps",
        type=int,
        help="Alias legado para --min-gradient-accumulation-steps.",
    )
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument(
        "--stable-ratio",
        type=float,
        default=-1,
        help="Proporcao da fase estavel no total de updates. -1 usa o restante apos warmup/decay.",
    )
    parser.add_argument("--decay-ratio", type=float, default=0.1)
    parser.add_argument(
        "--force-decay-on-resume",
        action="store_true",
        help="Ao retomar de checkpoint, reinicia LR scheduler diretamente na fase de decay.",
    )
    parser.add_argument(
        "--resume-load-scheduler-state",
        action="store_true",
        help="Ao retomar, carrega estado do scheduler salvo no checkpoint. Se nao setado, recalcula pelo novo plano.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every-updates", type=int, default=500)
    parser.add_argument("--log-every-updates", type=int, default=50)
    parser.add_argument("--preview-every-updates", type=int, default=500)
    parser.add_argument("--preview-max-new-tokens", type=int, default=256)
    parser.add_argument("--preview-log-file", type=str, default="preview_generations.log")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--resume",
        type=str,
        default="auto",
        help="Caminho de checkpoint .pt, 'auto' para output_dir/latest.pt, ou 'none' para iniciar do zero.",
    )

    args = parser.parse_args()
    return TrainConfig(
        model_name=args.model_name,
        data_glob=args.data_glob,
        output_dir=args.output_dir,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_val_batch_size=args.per_device_val_batch_size,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        min_gradient_accumulation_steps=args.min_gradient_accumulation_steps,
        max_gradient_accumulation_steps=args.max_gradient_accumulation_steps,
        grad_accum_growth_factor=args.grad_accum_growth_factor,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        stable_ratio=args.stable_ratio,
        decay_ratio=args.decay_ratio,
        force_decay_on_resume=args.force_decay_on_resume,
        resume_load_scheduler_state=args.resume_load_scheduler_state,
        max_grad_norm=args.max_grad_norm,
        save_every_updates=args.save_every_updates,
        log_every_updates=args.log_every_updates,
        preview_every_updates=args.preview_every_updates,
        preview_max_new_tokens=args.preview_max_new_tokens,
        preview_log_file=args.preview_log_file,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        resume=args.resume,
    )


def setup_distributed() -> Tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=120))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return distributed, world_size, rank, local_rank, device


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def rank_print(rank: int, *args, **kwargs) -> None:
    if is_main_process(rank):
        print(*args, **kwargs)


def barrier(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: float, device: torch.device, distributed: bool) -> float:
    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor.item()


def seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def infer_summary_type_from_path(path: str) -> str:
    p = path.lower().replace("\\", "/")
    parts = [part for part in p.split("/") if part]

    if "datasets" not in parts:
        raise ValueError(f"Caminho sem pasta 'datasets': {path}")

    datasets_idx = parts.index("datasets")
    if datasets_idx + 1 >= len(parts):
        raise ValueError(f"Nao foi possivel inferir tipo no caminho: {path}")

    type_folder = parts[datasets_idx + 1]
    type_map = {
        "curtos": "curtos",
        "hierarquico": "hierarquico",
        "topicos": "topicos",
        "sem_restricao": "sem_restricao",
        "hierarquicos": "hierarquico",
        "topico": "topicos",
        "curto": "curtos",
        "sem-restricao": "sem_restricao",
    }

    if type_folder not in type_map:
        raise ValueError(f"Tipo desconhecido no path: {type_folder} | path={path}")

    return type_map[type_folder]


def get_target_field(summary_type: str) -> str:
    if summary_type in {"curtos", "hierarquico", "topicos"}:
        return "short_summary"
    return "summary"


PROMPT_BY_TYPE = {
    "curtos": "Resuma em portugues em poucas frases, com foco nas ideias centrais.",
    "hierarquico": "Resuma em portugues em estrutura hierarquica, do geral para o especifico.",
    "topicos": "Resuma em portugues em topicos objetivos e informativos.",
    "sem_restricao": "Resuma em portugues de forma clara, fiel e coesa.",
}


def build_prefixed_input(page_content: str, summary_type: str) -> str:
    instruction = PROMPT_BY_TYPE.get(summary_type, PROMPT_BY_TYPE["sem_restricao"])
    return (
        f"tarefa: sumarizacao_pt\n"
        f"tipo_resumo: {summary_type}\n"
        f"instrucao: {instruction}\n"
        f"texto: {page_content}"
    )


def load_records(data_glob: str, val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    paths = glob.glob(data_glob)
    if not paths:
        raise ValueError(f"Nenhum arquivo encontrado para o padrao: {data_glob}")

    records = []
    for path in tqdm(paths, desc="Carregando JSONs", leave=False):
        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)

        summary_type = infer_summary_type_from_path(path)
        target_field = get_target_field(summary_type)

        source_raw = (item.get("page_content") or "").strip()
        target = (item.get(target_field) or "").strip()
        if not source_raw or not target:
            continue

        source = build_prefixed_input(source_raw, summary_type)
        records.append(
            {
                "path": path,
                "summary_type": summary_type,
                "target_field": target_field,
                "source": source,
                "target": target,
                "source_raw": source_raw,
            }
        )

    if len(records) < 10:
        raise ValueError(f"Poucos exemplos apos filtragem: {len(records)}")

    rng = random.Random(seed)
    rng.shuffle(records)

    val_size = max(1, int(val_ratio * len(records)))
    val_records = records[:val_size]
    train_records = records[val_size:]
    return train_records, val_records


class SummarizationDataset(Dataset):
    def __init__(self, examples: List[Dict], tokenizer, max_source_len: int, max_target_len: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]

        model_inputs = self.tokenizer(
            item["source"],
            max_length=self.max_source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target_tokens = self.tokenizer(
            text_target=item["target"],
            max_length=self.max_target_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        labels_ids = target_tokens["input_ids"].squeeze(0)

        if self.tokenizer.pad_token_id is not None:
            labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_ids,
        }


def get_model_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    if isinstance(model, DDP):
        return model.module.state_dict()
    return model.state_dict()


def load_model_state_dict(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    if isinstance(model, DDP):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def safe_torch_save(state: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def save_checkpoint(
    cfg: TrainConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    step_in_epoch: int,
    global_micro_step: int,
    global_update_step: int,
    rank: int,
    tag: str,
) -> str:
    if not is_main_process(rank):
        return ""

    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{tag}.pt")

    state = {
        "config": asdict(cfg),
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_micro_step": global_micro_step,
        "global_update_step": global_update_step,
        "model_state_dict": get_model_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "saved_at": time.time(),
    }
    safe_torch_save(state, ckpt_path)

    latest_path = os.path.join(cfg.output_dir, "latest.pt")
    safe_torch_save(state, latest_path)
    return ckpt_path


def maybe_resume_path(cfg: TrainConfig) -> Optional[str]:
    if cfg.resume.lower() == "none":
        return None
    if cfg.resume.lower() == "auto":
        path = os.path.join(cfg.output_dir, "latest.pt")
        return path if os.path.exists(path) else None
    return cfg.resume if os.path.exists(cfg.resume) else None


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    load_scheduler_state: bool,
) -> Tuple[int, int, int, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_model_state_dict(model, checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if load_scheduler_state and scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if checkpoint.get("python_random_state") is not None:
        random.setstate(checkpoint["python_random_state"])
    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

    epoch = int(checkpoint.get("epoch", 0))
    step_in_epoch = int(checkpoint.get("step_in_epoch", 0))
    global_micro_step = int(checkpoint.get("global_micro_step", 0))
    global_update_step = int(checkpoint.get("global_update_step", 0))
    return epoch, step_in_epoch, global_micro_step, global_update_step


def run_validation(model: torch.nn.Module, loader: DataLoader, device: torch.device, distributed: bool) -> float:
    model.eval()
    total_val_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()
            total_batches += 1

    local_mean = total_val_loss / max(1, total_batches)
    return all_reduce_mean(local_mean, device, distributed)


def choose_amp_settings(device: torch.device) -> Tuple[bool, Optional[torch.dtype], bool]:
    if device.type != "cuda":
        return False, None, False

    use_bf16 = torch.cuda.is_bf16_supported()
    if use_bf16:
        return True, torch.bfloat16, False
    return True, torch.float16, True


def build_gradient_accum_levels(min_steps: int, max_steps: int, growth_factor: int) -> List[int]:
    if min_steps < 1:
        raise ValueError("min_gradient_accumulation_steps deve ser >= 1")
    if max_steps < min_steps:
        raise ValueError("max_gradient_accumulation_steps deve ser >= min_gradient_accumulation_steps")
    if growth_factor < 2:
        raise ValueError("grad_accum_growth_factor deve ser >= 2")

    levels = [min_steps]
    current = min_steps
    while current < max_steps:
        current = min(max_steps, current * growth_factor)
        if current == levels[-1]:
            break
        levels.append(current)
    return levels


def gradient_accum_for_micro_step(global_micro_step: int, total_micro_steps: int, levels: List[int]) -> int:
    if total_micro_steps <= 0:
        return levels[0]
    progress = global_micro_step / max(1, total_micro_steps)
    phase = min(len(levels) - 1, int(progress * len(levels)))
    return levels[phase]


def estimate_total_training_updates(steps_per_epoch: int, num_epochs: int, levels: List[int]) -> int:
    total_micro_steps = steps_per_epoch * num_epochs
    if total_micro_steps <= 0:
        return 1

    updates = 0
    for epoch in range(num_epochs):
        local_step = 0
        while local_step < steps_per_epoch:
            global_micro_step = epoch * steps_per_epoch + local_step
            acc_steps = gradient_accum_for_micro_step(global_micro_step, total_micro_steps, levels)
            local_step += acc_steps
            updates += 1
    return max(1, updates)


def resolve_wsd_schedule(cfg: TrainConfig, total_training_updates: int) -> Tuple[int, int, int]:
    if cfg.warmup_ratio < 0 or cfg.decay_ratio < 0:
        raise ValueError("warmup_ratio e decay_ratio devem ser >= 0")
    if cfg.stable_ratio < -1:
        raise ValueError("stable_ratio deve ser >= 0 ou -1")
    if cfg.stable_ratio >= 0 and (cfg.warmup_ratio + cfg.stable_ratio + cfg.decay_ratio) > 1.0:
        raise ValueError("A soma warmup_ratio + stable_ratio + decay_ratio deve ser <= 1.0")
    if cfg.stable_ratio < 0 and (cfg.warmup_ratio + cfg.decay_ratio) > 1.0:
        raise ValueError("A soma warmup_ratio + decay_ratio deve ser <= 1.0 quando stable_ratio=-1")

    warmup_updates = int(cfg.warmup_ratio * total_training_updates)
    decay_updates = int(cfg.decay_ratio * total_training_updates)

    if cfg.stable_ratio >= 0:
        stable_updates = int(cfg.stable_ratio * total_training_updates)
    else:
        stable_updates = max(0, total_training_updates - warmup_updates - decay_updates)

    used = warmup_updates + stable_updates + decay_updates
    if used > total_training_updates:
        overflow = used - total_training_updates
        reduce_decay = min(decay_updates, overflow)
        decay_updates -= reduce_decay
        overflow -= reduce_decay
        if overflow > 0:
            stable_updates = max(0, stable_updates - overflow)

    warmup_updates = max(0, warmup_updates)
    stable_updates = max(0, stable_updates)
    decay_updates = max(0, min(decay_updates, total_training_updates - warmup_updates - stable_updates))

    return warmup_updates, stable_updates, decay_updates


def build_wsd_lambda(warmup_updates: int, stable_updates: int, decay_updates: int):
    decay_start = warmup_updates + stable_updates

    def lr_lambda(step_idx: int) -> float:
        if warmup_updates > 0 and step_idx < warmup_updates:
            return float(step_idx + 1) / float(max(1, warmup_updates))
        if step_idx < decay_start:
            return 1.0
        if decay_updates <= 0:
            return 1.0

        decay_step = step_idx - decay_start + 1
        decay_progress = min(1.0, decay_step / float(decay_updates))
        return max(0.0, 1.0 - decay_progress)

    return lr_lambda


def select_preview_examples(val_examples: List[Dict]) -> List[Dict]:
    ordered_types = ["curtos", "hierarquico", "topicos", "sem_restricao"]
    selected_examples: List[Dict] = []
    for summary_type in ordered_types:
        ex = next((item for item in val_examples if item.get("summary_type") == summary_type), None)
        if ex is not None:
            selected_examples.append(ex)
    return selected_examples


def append_preview_log(
    cfg: TrainConfig,
    model: torch.nn.Module,
    tokenizer,
    preview_examples: List[Dict],
    global_update_step: int,
    epoch: int,
    rank: int,
    device: torch.device,
) -> None:
    if not is_main_process(rank):
        return
    if not preview_examples:
        return

    model_for_generation = model.module if isinstance(model, DDP) else model
    model_for_generation.eval()

    log_path = os.path.join(cfg.output_dir, cfg.preview_log_file)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    lines: List[str] = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f"update={global_update_step} | epoch={epoch + 1} | ts={int(time.time())}")

    with torch.no_grad():
        for i, ex in enumerate(preview_examples, start=1):
            inputs = tokenizer(
                ex["source"],
                return_tensors="pt",
                truncation=True,
                max_length=cfg.max_source_length,
            ).to(device)

            generated_ids = model_for_generation.generate(
                **inputs,
                max_new_tokens=cfg.preview_max_new_tokens,
                num_beams=4,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=True,
            )

            pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            lines.append(f"\n[Exemplo {i}] tipo={ex['summary_type']}")
            lines.append(f"fonte: {ex['source_raw'].replace(chr(10), ' ')}")
            lines.append(f"referencia: {ex['target'].replace(chr(10), ' ')}")
            lines.append(f"gerado: {pred.replace(chr(10), ' ')}")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    cfg = parse_args()
    distributed, world_size, rank, local_rank, device = setup_distributed()
    seed_everything(cfg.seed, rank)

    stop_requested = {"value": False}

    def handle_signal(signum, frame):  # noqa: ARG001
        stop_requested["value"] = True
        rank_print(rank, f"Sinal {signum} recebido; checkpoint sera salvo ao fim do step atual.")

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    rank_print(rank, f"Device={device} | world_size={world_size} | rank={rank} | local_rank={local_rank}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    ckpt_to_resume = maybe_resume_path(cfg)

    train_records, val_records = load_records(cfg.data_glob, cfg.val_ratio, cfg.seed)
    rank_print(rank, f"Treino: {len(train_records)} | Validacao: {len(val_records)}")
    preview_examples = select_preview_examples(val_records)
    if is_main_process(rank):
        preview_types = [ex["summary_type"] for ex in preview_examples]
        rank_print(rank, f"Preview de validacao (1 por tipo): {preview_types}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    model.to(device)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=cfg.ddp_find_unused_parameters,
        )

    train_dataset = SummarizationDataset(train_records, tokenizer, cfg.max_source_length, cfg.max_target_length)
    val_dataset = SummarizationDataset(val_records, tokenizer, cfg.max_source_length, cfg.max_target_length)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.per_device_val_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    grad_accum_levels = build_gradient_accum_levels(
        cfg.min_gradient_accumulation_steps,
        cfg.max_gradient_accumulation_steps,
        cfg.grad_accum_growth_factor,
    )
    total_micro_steps = len(train_loader) * cfg.num_epochs
    total_training_updates = estimate_total_training_updates(
        steps_per_epoch=len(train_loader),
        num_epochs=cfg.num_epochs,
        levels=grad_accum_levels,
    )
    warmup_steps, stable_steps, decay_steps = resolve_wsd_schedule(cfg, total_training_updates)

    scheduler_state_should_be_restored = cfg.resume_load_scheduler_state
    if ckpt_to_resume is not None and cfg.force_decay_on_resume:
        warmup_steps = 0
        stable_steps = 0
        decay_steps = max(1, total_training_updates)
        scheduler_state_should_be_restored = False

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_wsd_lambda(warmup_steps, stable_steps, decay_steps),
    )
    decay_start_update = warmup_steps + stable_steps

    amp_enabled, amp_dtype, use_grad_scaler = choose_amp_settings(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    start_epoch = 0
    resume_step_in_epoch = 0
    global_micro_step = 0
    global_update_step = 0

    if ckpt_to_resume is not None:
        barrier(distributed)
        start_epoch, resume_step_in_epoch, global_micro_step, global_update_step = load_checkpoint(
            ckpt_to_resume,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
            load_scheduler_state=scheduler_state_should_be_restored,
        )
        rank_print(
            rank,
            f"Retomando de {ckpt_to_resume} | epoch={start_epoch} | step_in_epoch={resume_step_in_epoch} | micro_steps={global_micro_step} | updates={global_update_step}",
        )
        if cfg.force_decay_on_resume:
            rank_print(rank, "Modo force_decay_on_resume ativo: retomando diretamente em fase de decay de LR.")
    else:
        rank_print(rank, "Treino iniciado do zero (sem checkpoint de retomada).")

    rank_print(rank, f"Scheduler gradient accumulation (niveis): {grad_accum_levels}")
    rank_print(
        rank,
        "WSD LR scheduler "
        f"(warmup_updates={warmup_steps}, stable_updates={stable_steps}, decay_updates={decay_steps}, "
        f"total_updates_estimado={total_training_updates})",
    )

    append_preview_log(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        preview_examples=preview_examples,
        global_update_step=global_update_step,
        epoch=start_epoch,
        rank=rank,
        device=device,
    )

    barrier(distributed)

    for epoch in range(start_epoch, cfg.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        running_updates = 0
        accum_counter = 0
        active_accum_steps = gradient_accum_for_micro_step(global_micro_step, total_micro_steps, grad_accum_levels)

        stable_ckpt_saved = global_update_step >= warmup_steps
        decay_ckpt_saved = global_update_step >= decay_start_update

        if is_main_process(rank):
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{cfg.num_epochs}")
        else:
            pbar = None

        for step, batch in enumerate(train_loader):
            if epoch == start_epoch and step < resume_step_in_epoch:
                if pbar is not None:
                    pbar.update(1)
                continue

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            if accum_counter == 0:
                active_accum_steps = gradient_accum_for_micro_step(
                    global_micro_step,
                    total_micro_steps,
                    grad_accum_levels,
                )

            amp_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
                if amp_enabled
                else contextlib.nullcontext()
            )
            with amp_context:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / active_accum_steps

            if use_grad_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_counter += 1
            global_micro_step += 1

            should_step = (accum_counter >= active_accum_steps) or ((step + 1) == len(train_loader))
            if should_step:
                if cfg.max_grad_norm > 0:
                    if use_grad_scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                if use_grad_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_update_step += 1
                running_updates += 1
                step_loss = loss.item() * active_accum_steps
                running_loss += step_loss
                accum_counter = 0

                if global_update_step % cfg.log_every_updates == 0:
                    avg_loss_local = running_loss / max(1, running_updates)
                    avg_loss = all_reduce_mean(avg_loss_local, device, distributed)
                    if is_main_process(rank):
                        print(
                            f"update={global_update_step} | epoch={epoch + 1} | step={step + 1}/{len(train_loader)} | "
                            f"train_loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | grad_accum={active_accum_steps}"
                        )
                    running_loss = 0.0
                    running_updates = 0

                if global_update_step % cfg.save_every_updates == 0:
                    ckpt_path = save_checkpoint(
                        cfg,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        step + 1,
                        global_micro_step,
                        global_update_step,
                        rank,
                        tag=f"update_{global_update_step}",
                    )
                    if is_main_process(rank):
                        print(f"Checkpoint salvo: {ckpt_path}")

                if cfg.preview_every_updates > 0 and global_update_step % cfg.preview_every_updates == 0:
                    append_preview_log(
                        cfg=cfg,
                        model=model,
                        tokenizer=tokenizer,
                        preview_examples=preview_examples,
                        global_update_step=global_update_step,
                        epoch=epoch,
                        rank=rank,
                        device=device,
                    )
                    if is_main_process(rank):
                        print(
                            f"Preview salvo em {os.path.join(cfg.output_dir, cfg.preview_log_file)} "
                            f"(update={global_update_step})"
                        )

                if (not stable_ckpt_saved) and global_update_step >= warmup_steps:
                    ckpt_path = save_checkpoint(
                        cfg,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        step + 1,
                        global_micro_step,
                        global_update_step,
                        rank,
                        tag=f"stable_start_update_{global_update_step}",
                    )
                    stable_ckpt_saved = True
                    if is_main_process(rank):
                        print(f"Checkpoint inicio fase estavel salvo: {ckpt_path}")

                if (not decay_ckpt_saved) and global_update_step >= decay_start_update:
                    ckpt_path = save_checkpoint(
                        cfg,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        step + 1,
                        global_micro_step,
                        global_update_step,
                        rank,
                        tag=f"decay_start_update_{global_update_step}",
                    )
                    decay_ckpt_saved = True
                    if is_main_process(rank):
                        print(f"Checkpoint inicio decaimento salvo: {ckpt_path}")

            if pbar is not None:
                pbar.update(1)

            if stop_requested["value"]:
                ckpt_path = save_checkpoint(
                    cfg,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    step + 1,
                    global_micro_step,
                    global_update_step,
                    rank,
                    tag=f"signal_update_{global_update_step}",
                )
                if is_main_process(rank):
                    print(f"Parada solicitada. Checkpoint salvo em: {ckpt_path}")
                barrier(distributed)
                cleanup_distributed(distributed)
                return

        if pbar is not None:
            pbar.close()

        val_loss = run_validation(model, val_loader, device, distributed)
        if is_main_process(rank):
            print(f"[Epoch {epoch + 1}] val_loss={val_loss:.4f}")

        ckpt_path = save_checkpoint(
            cfg,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch + 1,
            0,
            global_micro_step,
            global_update_step,
            rank,
            tag=f"epoch_{epoch + 1}",
        )
        if is_main_process(rank):
            print(f"Checkpoint de fim de epoca salvo: {ckpt_path}")

    barrier(distributed)

    if is_main_process(rank):
        final_dir = os.path.join(cfg.output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)

        if isinstance(model, DDP):
            model.module.save_pretrained(final_dir)
        else:
            model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

        with open(os.path.join(cfg.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

        print(f"Treino finalizado. Artefatos em: {cfg.output_dir}")

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
