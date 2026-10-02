#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline local: treina YOLOv7-tiny no dataset dino e exporta para TF.js.

Equivalente ao notebook do Colab, sem google.colab, !pip ou paths /content.

Exemplos:
  python pipeline.py --stage setup --dataset ./dino-dataset.zip
  python pipeline.py --stage all --dataset ./dino-dataset
  python pipeline.py --stage train --epochs 50 --batch-size 16
  python pipeline.py --stage export
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent
YOLOV7_REPO = "https://github.com/Erickinhou/yolov7"
TINY_WEIGHTS_URL = (
    "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt"
)
DEFAULT_NAMES = ["cactus", "bird"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

STAGES = ("setup", "train", "reparam", "eval", "export")


class PipelineError(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    pretty = " ".join(cmd)
    print(f"$ {pretty}", flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(cmd, cwd=cwd, env=merged, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treino local YOLOv7-tiny (dino) e conversão para TF.js",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Etapas:
  setup    clona YOLOv7, baixa pesos, prepara dataset e YAMLs
  train    treina o modelo
  reparam  funde implicit layers para inferência
  eval     detect.py + test.py no split de validação
  export   PyTorch -> ONNX -> SavedModel -> TF.js
  all      executa tudo na ordem

Ambiente recomendado:
  python3 -m venv .venv
  source .venv/bin/activate
  python pipeline.py --install-deps
  python pipeline.py --stage all --dataset ./dino-dataset.zip
""",
    )
    parser.add_argument(
        "--stage",
        choices=(*STAGES, "all"),
        default="all",
        help="Etapa a executar (default: all)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "dino-dataset",
        help="Pasta do dataset ou arquivo .zip",
    )
    parser.add_argument(
        "--yolov7-dir",
        type=Path,
        default=ROOT / "yolov7",
        help="Onde clonar / usar o repositório YOLOv7",
    )
    parser.add_argument("--name", default="yolov7_tiny_dino", help="Nome do run de treino")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="0, 0,1 ou cpu. Se omitido, usa GPU quando disponível",
    )
    parser.add_argument("--nc", type=int, default=2, help="Número de classes")
    parser.add_argument(
        "--names",
        default=",".join(DEFAULT_NAMES),
        help="Nomes das classes, separados por vírgula",
    )
    parser.add_argument("--conf", type=float, default=0.8, help="Confiança no detect.py")
    parser.add_argument("--iou-thres", type=float, default=0.65)
    parser.add_argument("--conf-thres", type=float, default=0.35)
    parser.add_argument("--topk-all", type=int, default=100)
    parser.add_argument("--max-wh", type=int, default=640)
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        default=True,
        help="Reaproveita o mesmo nome de run (default)",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Instala dependências da etapa no Python atual",
    )
    return parser.parse_args()


def class_names(args: argparse.Namespace) -> list[str]:
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    if len(names) != args.nc:
        raise PipelineError(
            f"--names tem {len(names)} itens, mas --nc={args.nc}"
        )
    return names


def resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def weights_dir(args: argparse.Namespace) -> Path:
    return args.yolov7_dir / "runs" / "train" / args.name / "weights"


def best_pt(args: argparse.Namespace) -> Path:
    return weights_dir(args) / "best.pt"


def reparam_pt(args: argparse.Namespace) -> Path:
    return weights_dir(args) / "yolov7-tiny-dino-reparam.pt"


def yolov7_rel(path: Path, yolov7_dir: Path) -> str:
    """YOLOv7 lowercases weight paths; keep them relative to avoid Projetos -> projetos."""
    path = path.resolve()
    root = yolov7_dir.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def data_yaml(args: argparse.Namespace) -> Path:
    dataset_root = dataset_root_from_arg(args.dataset)
    return dataset_root / "data.yaml"


def dataset_root_from_arg(dataset: Path) -> Path:
    path = dataset.expanduser().resolve()
    if path.suffix.lower() == ".zip":
        return ROOT / "dino-dataset"
    return path


def require_exists(path: Path, hint: str) -> Path:
    if not path.exists():
        raise PipelineError(f"{hint}: {path}")
    return path


def find_split_images(root: Path, split: str) -> Path | None:
    candidates = [
        root / "images" / split,
        root / split / "images",
        root / split,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(
            p.suffix.lower() in IMAGE_EXTS for p in candidate.rglob("*")
        ):
            return candidate
    return None


def prepare_dataset(dataset: Path) -> Path:
    dataset = dataset.expanduser().resolve()
    if dataset.suffix.lower() == ".zip":
        require_exists(dataset, "ZIP do dataset não encontrado")
        dest = ROOT / "dino-dataset"
        log(f"Descompactando {dataset} em {dest}")
        extract_zip(dataset, dest)
        return dest

    if not dataset.exists():
        raise PipelineError(
            "Dataset não encontrado.\n"
            f"  Passe uma pasta ou um .zip com --dataset\n"
            f"  Procurado: {dataset}\n"
            "Estrutura esperada:\n"
            "  dino-dataset/images/train\n"
            "  dino-dataset/images/val\n"
            "  dino-dataset/labels/train\n"
            "  dino-dataset/labels/val"
        )
    return dataset


def extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)

    if find_split_images(dest, "train"):
        return

    children = [
        path
        for path in dest.iterdir()
        if path.is_dir() and path.name not in {"__MACOSX"}
    ]
    if len(children) == 1 and find_split_images(children[0], "train"):
        for child in children[0].iterdir():
            target = dest / child.name
            if not target.exists():
                shutil.move(str(child), str(target))


def write_data_yaml(root: Path, nc: int, names: list[str]) -> Path:
    train = find_split_images(root, "train")
    val = find_split_images(root, "val") or find_split_images(root, "valid")
    if train is None or val is None:
        raise PipelineError(
            "Não achei images/train e images/val no dataset.\n"
            f"  Pasta: {root}"
        )

    yaml_path = root / "data.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(names))
    yaml_path.write_text(
        f"train: {train}\n"
        f"val: {val}\n"
        f"\n"
        f"nc: {nc}\n"
        f"\n"
        f"names:\n"
        f"{names_block}\n",
        encoding="utf-8",
    )
    log(f"data.yaml escrito em {yaml_path}")
    print(f"  train: {train}")
    print(f"  val:   {val}")
    return yaml_path


def clone_yolov7(yolov7_dir: Path) -> None:
    if (yolov7_dir / "train.py").exists():
        log(f"YOLOv7 já existe em {yolov7_dir}")
        return
    if yolov7_dir.exists() and any(yolov7_dir.iterdir()):
        raise PipelineError(
            f"{yolov7_dir} existe, mas não parece um clone do YOLOv7"
        )
    log(f"Clonando {YOLOV7_REPO}")
    run(["git", "clone", "--depth", "1", YOLOV7_REPO, str(yolov7_dir)])


def download_tiny_weights(yolov7_dir: Path) -> Path:
    dest = yolov7_dir / "yolov7-tiny.pt"
    if dest.exists():
        log(f"Checkpoint já existe: {dest}")
        return dest
    log(f"Baixando yolov7-tiny.pt")
    urlretrieve(TINY_WEIGHTS_URL, dest)
    return dest


def prepare_model_cfg(yolov7_dir: Path, nc: int) -> Path:
    src = yolov7_dir / "cfg" / "training" / "yolov7-tiny.yaml"
    dest = yolov7_dir / "cfg" / "training" / "yolov7-dino-tiny.yaml"
    require_exists(src, "YAML de treino do tiny não encontrado")
    text = src.read_text(encoding="utf-8")
    text = text.replace("nc: 80", f"nc: {nc}", 1)
    dest.write_text(text, encoding="utf-8")
    log(f"Arquitetura salva em {dest} (nc={nc})")
    return dest


def install_deps(stage: str, yolov7_dir: Path) -> None:
    py = sys.executable
    if stage in {"setup", "train", "reparam", "eval", "all"}:
        req = yolov7_dir / "requirements.txt"
        if req.exists():
            log("Instalando dependências de treino")
            run([py, "-m", "pip", "install", "-r", str(req)])
        else:
            run([py, "-m", "pip", "install", "-r", str(ROOT / "requirements-train.txt")])
        run([py, "-m", "pip", "install", "--no-deps", "thop"])
    if stage in {"export", "all"}:
        log("Instalando dependências de exportação TF.js")
        run([py, "-m", "pip", "install", "-r", str(ROOT / "requirements-export.txt")])


def stage_setup(args: argparse.Namespace) -> None:
    names = class_names(args)
    clone_yolov7(args.yolov7_dir)
    if args.install_deps:
        install_deps("setup", args.yolov7_dir)
    download_tiny_weights(args.yolov7_dir)
    prepare_model_cfg(args.yolov7_dir, args.nc)
    root = prepare_dataset(args.dataset)
    write_data_yaml(root, args.nc, names)


def stage_train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    workers = args.workers if args.workers is not None else (4 if device != "cpu" else 2)
    batch = args.batch_size
    if device == "cpu" and batch > 8:
        log(f"CPU detectada: reduzindo batch-size de {batch} para 4")
        batch = 4

    yaml = require_exists(data_yaml(args), "Rode --stage setup primeiro; falta data.yaml")
    cfg = require_exists(
        args.yolov7_dir / "cfg" / "training" / "yolov7-dino-tiny.yaml",
        "Rode --stage setup primeiro; falta o YAML da arquitetura",
    )
    weights = require_exists(
        args.yolov7_dir / "yolov7-tiny.pt",
        "Rode --stage setup primeiro; falta yolov7-tiny.pt",
    )

    cmd = [
        sys.executable,
        "train.py",
        "--epochs",
        str(args.epochs),
        "--workers",
        str(workers),
        "--device",
        device,
        "--batch-size",
        str(batch),
        "--data",
        str(yaml),
        "--img",
        str(args.img_size),
        str(args.img_size),
        "--cfg",
        str(cfg.relative_to(args.yolov7_dir)),
        "--weights",
        str(weights.name),
        "--name",
        args.name,
        "--hyp",
        "data/hyp.scratch.tiny.yaml",
    ]
    if args.exist_ok:
        cmd.append("--exist-ok")

    log(f"Treinando em device={device}")
    run(cmd, cwd=args.yolov7_dir, env={"WANDB_MODE": "disabled"})
    print(f"Pesos: {best_pt(args)}")


def _detect_layer_index(model) -> int:
    for index, module in enumerate(model.model):
        if module.__class__.__name__ in {"IDetect", "Detect"}:
            return index
    return 77


def stage_reparam(args: argparse.Namespace) -> None:
    yolov7_dir = args.yolov7_dir.resolve()
    sys.path.insert(0, str(yolov7_dir))
    previous_cwd = Path.cwd()
    os.chdir(yolov7_dir)

    import torch
    import yaml
    from models.yolo import Model
    from utils.torch_utils import is_parallel, select_device

    device_id = resolve_device(args.device)
    device = select_device(device_id, batch_size=1)
    ckpt_path = require_exists(best_pt(args), "best.pt do treino não encontrado")
    deploy_cfg = require_exists(
        yolov7_dir / "cfg" / "deploy" / "yolov7-tiny.yaml",
        "cfg/deploy/yolov7-tiny.yaml não encontrado",
    )

    log(f"Reparametrizando {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Model(str(deploy_cfg), ch=3, nc=args.nc).to(device)
    with deploy_cfg.open() as handle:
        yml = yaml.load(handle, Loader=yaml.SafeLoader)
    anchors = len(yml["anchors"][0]) // 2
    state_dict = ckpt["model"].float().state_dict()
    intersect = {
        key: value
        for key, value in state_dict.items()
        if key in model.state_dict() and value.shape == model.state_dict()[key].shape
    }
    model.load_state_dict(intersect, strict=False)
    model.names = ckpt["model"].names
    model.nc = ckpt["model"].nc

    layer = _detect_layer_index(model)
    log(f"Camada de detecção: {layer}")
    for i in range((model.nc + 5) * anchors):
        for head in range(3):
            model.state_dict()[f"model.{layer}.m.{head}.weight"].data[i, :, :, :] *= (
                state_dict[f"model.{layer}.im.{head}.implicit"].data[:, i, :, :].squeeze()
            )
    for head in range(3):
        model.state_dict()[f"model.{layer}.m.{head}.bias"].data += (
            state_dict[f"model.{layer}.m.{head}.weight"]
            .mul(state_dict[f"model.{layer}.ia.{head}.implicit"])
            .sum(1)
            .squeeze()
        )
        model.state_dict()[f"model.{layer}.m.{head}.bias"].data *= state_dict[
            f"model.{layer}.im.{head}.implicit"
        ].data.squeeze()

    fused = deepcopy(model.module if is_parallel(model) else model)
    if device.type != "cpu":
        fused = fused.half()
    out = {
        "model": fused,
        "optimizer": None,
        "training_results": None,
        "epoch": -1,
    }
    dest = reparam_pt(args)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dest)
    os.chdir(previous_cwd)
    log(f"Pesos reparametrizados: {dest}")


def latest_detect_dir(yolov7_dir: Path) -> Path | None:
    detect_root = yolov7_dir / "runs" / "detect"
    if not detect_root.is_dir():
        return None
    runs = [p for p in detect_root.iterdir() if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def stage_eval(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    weights = require_exists(reparam_pt(args), "Rode --stage reparam primeiro")
    yaml = require_exists(data_yaml(args), "data.yaml não encontrado")
    val_images = find_split_images(dataset_root_from_arg(args.dataset), "val")
    if val_images is None:
        val_images = find_split_images(dataset_root_from_arg(args.dataset), "valid")
    if val_images is None:
        raise PipelineError("Pasta de validação não encontrada")

    log("Rodando detect.py")
    run(
        [
            sys.executable,
            "detect.py",
            "--weights",
            yolov7_rel(weights, args.yolov7_dir),
            "--conf-thres",
            str(args.conf),
            "--source",
            str(val_images),
            "--device",
            device,
            "--exist-ok",
            "--name",
            f"{args.name}_val",
        ],
        cwd=args.yolov7_dir,
    )

    log("Rodando test.py")
    run(
        [
            sys.executable,
            "test.py",
            "--weights",
            yolov7_rel(weights, args.yolov7_dir),
            "--data",
            str(yaml),
            "--task",
            "val",
            "--device",
            device,
            "--exist-ok",
            "--name",
            f"{args.name}_val",
        ],
        cwd=args.yolov7_dir,
    )

    vis = args.yolov7_dir / "runs" / "detect" / f"{args.name}_val"
    if not vis.exists():
        vis = latest_detect_dir(args.yolov7_dir)
    if vis:
        images = sorted(
            p for p in vis.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        print(f"Inferências em {vis} ({len(images)} imagens)")


def _which_or_module(binary: str, module_args: list[str]) -> list[str]:
    found = shutil.which(binary)
    if found:
        return [found]
    return [sys.executable, "-m", *module_args]


def stage_export(args: argparse.Namespace) -> None:
    if args.install_deps:
        install_deps("export", args.yolov7_dir)

    weights = require_exists(reparam_pt(args), "Rode --stage reparam primeiro")
    device = resolve_device(args.device)
    log("Exportando PyTorch -> ONNX")
    run(
        [
            sys.executable,
            "export.py",
            "--weights",
            yolov7_rel(weights, args.yolov7_dir),
            "--grid",
            "--topk-all",
            str(args.topk_all),
            "--iou-thres",
            str(args.iou_thres),
            "--conf-thres",
            str(args.conf_thres),
            "--img-size",
            str(args.img_size),
            str(args.img_size),
            "--max-wh",
            str(args.max_wh),
            "--device",
            device,
        ],
        cwd=args.yolov7_dir,
    )

    onnx_path = weights.with_suffix(".onnx")
    require_exists(onnx_path, "ONNX não foi gerado")
    tf_dir = weights_dir(args) / "tf"
    tfjs_dir = weights_dir(args) / "tfjs_model"

    if tf_dir.exists():
        shutil.rmtree(tf_dir)
    if tfjs_dir.exists():
        shutil.rmtree(tfjs_dir)

    log("Convertendo ONNX -> TensorFlow SavedModel")
    onnx2tf = _which_or_module("onnx2tf", ["onnx2tf"])
    run(
        [
            *onnx2tf,
            "-i",
            str(onnx_path),
            "-o",
            str(tf_dir),
            "-osd",
            "-fdosm",
        ]
    )

    log("Convertendo SavedModel -> TF.js")
    converter = _which_or_module(
        "tensorflowjs_converter",
        ["tensorflowjs.converters.converter"],
    )
    stub_dir = ROOT / "_tfjs_stubs"
    converter_env = {}
    if stub_dir.is_dir():
        converter_env["PYTHONPATH"] = (
            f"{stub_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
        )
    run(
        [
            *converter,
            "--input_format=tf_saved_model",
            "--output_format=tfjs_graph_model",
            "--signature_name=serving_default",
            "--saved_model_tags=serve",
            str(tf_dir),
            str(tfjs_dir),
        ],
        env=converter_env or None,
    )

    (tfjs_dir / "labels.json").write_text(
        json.dumps(class_names(args), indent=2) + "\n",
        encoding="utf-8",
    )

    archive = weights_dir(args) / "tfjs_model.tar.gz"
    log(f"Empacotando {archive}")
    run(
        [
            "tar",
            "-czf",
            str(archive),
            "-C",
            str(weights_dir(args)),
            "tfjs_model",
        ]
    )
    print(f"TF.js: {tfjs_dir}")
    print(f"Arquivo: {archive}")


def main() -> int:
    args = parse_args()
    args.dataset = args.dataset.expanduser()
    if args.dataset.suffix.lower() != ".zip" or args.dataset.exists():
        args.dataset = args.dataset.resolve()
    args.yolov7_dir = args.yolov7_dir.expanduser().resolve()
    os.environ["WANDB_MODE"] = "disabled"

    order = STAGES if args.stage == "all" else (args.stage,)
    handlers = {
        "setup": stage_setup,
        "train": stage_train,
        "reparam": stage_reparam,
        "eval": stage_eval,
        "export": stage_export,
    }

    try:
        if args.install_deps and args.stage != "setup":
            if args.stage == "all":
                pass
            else:
                if args.stage == "export":
                    install_deps("export", args.yolov7_dir)
                elif (args.yolov7_dir / "train.py").exists():
                    install_deps(args.stage, args.yolov7_dir)

        for stage in order:
            log(f"Etapa: {stage}")
            handlers[stage](args)
    except PipelineError as exc:
        print(f"\nErro: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"\nComando falhou (exit {exc.returncode}): {exc.cmd}", file=sys.stderr)
        return exc.returncode or 1

    log("Pipeline concluído")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
