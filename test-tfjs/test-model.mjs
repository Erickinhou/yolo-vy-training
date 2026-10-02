#!/usr/bin/env node
/**
 * Testa o YOLOv7-tiny dino exportado em TF.js.
 *
 *   cd test-tfjs && npm install
 *   node test-model.mjs
 *   node test-model.mjs --image ../dino-dataset/images/val/frame_00007.png
 *   node test-model.mjs --dir ../dino-dataset/images/val --limit 10 --conf 0.4
 */
import fs from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import tf from '@tensorflow/tfjs-node'
import sharp from 'sharp'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(HERE, '..')
const INPUT_SIZE = 640
const BOX_STRIDE = 7

const DEFAULT_MODEL = path.join(
  '/home/erickinhou/Projetos/yolov7/runs/train/yolov7_tiny_dino/weights/tfjs_model',
  'model.json',
)
const DEFAULT_IMAGES = path.join(ROOT, 'dino-dataset/images/val')
const DEFAULT_LABELS = path.join(ROOT, 'dino-dataset/labels/val')

function parseArgs(argv) {
  const args = {
    model: DEFAULT_MODEL,
    dir: DEFAULT_IMAGES,
    labelsDir: DEFAULT_LABELS,
    image: null,
    conf: 0.4,
    iou: 0.45,
    limit: 8,
    out: path.join(HERE, 'out'),
    save: true,
  }
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i]
    const next = argv[i + 1]
    if (flag === '--model') args.model = path.resolve(next)
    else if (flag === '--dir') args.dir = path.resolve(next)
    else if (flag === '--image') args.image = path.resolve(next)
    else if (flag === '--labels') args.labelsDir = path.resolve(next)
    else if (flag === '--conf') args.conf = Number(next)
    else if (flag === '--iou') args.iou = Number(next)
    else if (flag === '--limit') args.limit = Number(next)
    else if (flag === '--out') args.out = path.resolve(next)
    else if (flag === '--no-save') args.save = false
    else continue
    if (flag !== '--no-save') i++
  }
  return args
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1)
  const y1 = Math.max(a.y1, b.y1)
  const x2 = Math.min(a.x2, b.x2)
  const y2 = Math.min(a.y2, b.y2)
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1)
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1)
  return inter / (areaA + areaB - inter + 1e-9)
}

function nms(boxes, iouThres) {
  const byClass = new Map()
  for (const box of boxes) {
    if (!byClass.has(box.label)) byClass.set(box.label, [])
    byClass.get(box.label).push(box)
  }
  const kept = []
  for (const group of byClass.values()) {
    group.sort((a, b) => b.score - a.score)
    const picked = []
    for (const box of group) {
      if (picked.every((other) => iou(box, other) < iouThres)) picked.push(box)
    }
    kept.push(...picked)
  }
  return kept.sort((a, b) => b.score - a.score)
}

function decode(candidates, labels, confThres, width, height) {
  const scaleX = width / INPUT_SIZE
  const scaleY = height / INPUT_SIZE
  const boxes = []
  for (let i = 0; i + 6 < candidates.length; i += BOX_STRIDE) {
    const objectness = candidates[i + 4]
    let best = 0
    let bestScore = -1
    for (let c = 0; c < labels.length; c++) {
      const score = objectness * candidates[i + 5 + c]
      if (score > bestScore) {
        bestScore = score
        best = c
      }
    }
    if (bestScore < confThres) continue
    const cx = candidates[i] * scaleX
    const cy = candidates[i + 1] * scaleY
    const w = candidates[i + 2] * scaleX
    const h = candidates[i + 3] * scaleY
    boxes.push({
      label: labels[best] ?? String(best),
      classId: best,
      score: bestScore,
      x: cx,
      y: cy,
      width: w,
      height: h,
      x1: cx - w / 2,
      y1: cy - h / 2,
      x2: cx + w / 2,
      y2: cy + h / 2,
    })
  }
  return boxes
}

async function loadLabelsJson(modelJsonPath) {
  const labelsPath = path.join(path.dirname(modelJsonPath), 'labels.json')
  try {
    return JSON.parse(await fs.readFile(labelsPath, 'utf8'))
  } catch {
    return ['cactus', 'bird']
  }
}

async function listImages(args) {
  if (args.image) return [args.image]
  const names = await fs.readdir(args.dir)
  return names
    .filter((name) => /\.(png|jpe?g|bmp|webp)$/i.test(name))
    .sort()
    .slice(0, args.limit)
    .map((name) => path.join(args.dir, name))
}

async function readYoloLabels(imagePath, labelsDir, width, height) {
  const stem = path.basename(imagePath, path.extname(imagePath))
  const labelPath = path.join(labelsDir, `${stem}.txt`)
  try {
    const text = await fs.readFile(labelPath, 'utf8')
    return text
      .trim()
      .split('\n')
      .filter(Boolean)
      .map((line) => {
        const [cls, cx, cy, w, h] = line.trim().split(/\s+/).map(Number)
        return {
          classId: cls,
          x1: (cx - w / 2) * width,
          y1: (cy - h / 2) * height,
          x2: (cx + w / 2) * width,
          y2: (cy + h / 2) * height,
        }
      })
  } catch {
    return []
  }
}

function colorFor(label) {
  return label === 'bird' ? '#4ea1ff' : '#3ddc84'
}

async function drawBoxes(imagePath, boxes, outPath) {
  const image = sharp(imagePath)
  const { width, height } = await image.metadata()
  const rects = boxes
    .map((box) => {
      const x = Math.max(0, box.x1)
      const y = Math.max(0, box.y1)
      const w = Math.max(1, box.x2 - box.x1)
      const h = Math.max(1, box.y2 - box.y1)
      const color = colorFor(box.label)
      const caption = `${box.label} ${(box.score * 100).toFixed(1)}%`
      return `
        <rect x="${x}" y="${y}" width="${w}" height="${h}" fill="none" stroke="${color}" stroke-width="3"/>
        <rect x="${x}" y="${Math.max(0, y - 20)}" width="${Math.min(w, caption.length * 8 + 8)}" height="20" fill="${color}"/>
        <text x="${x + 4}" y="${Math.max(14, y - 6)}" font-size="14" font-family="sans-serif" fill="#111">${caption}</text>`
    })
    .join('')
  const svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">${rects}</svg>`
  await image
    .composite([{ input: Buffer.from(svg), top: 0, left: 0 }])
    .png()
    .toFile(outPath)
}

async function preprocess(imagePath) {
  const { data, info } = await sharp(imagePath)
    .removeAlpha()
    .resize(INPUT_SIZE, INPUT_SIZE, { fit: 'fill' })
    .raw()
    .toBuffer({ resolveWithObject: true })
  const float = Float32Array.from(data, (value) => value / 255)
  return tf.tensor4d(float, [1, info.height, info.width, 3])
}

async function infer(model, imagePath) {
  const input = await preprocess(imagePath)
  const result = model.execute(input)
  const tensor = Array.isArray(result) ? result[0] : result
  const candidates = await tensor.data()
  input.dispose()
  tensor.dispose()
  if (Array.isArray(result)) {
    for (const item of result) {
      if (item !== tensor) item.dispose()
    }
  }
  return candidates
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  const modelPath = pathToFileURL(args.model).href
  const names = await loadLabelsJson(args.model)

  console.log(`Modelo: ${args.model}`)
  await tf.ready()
  const model = await tf.loadGraphModel(modelPath)
  const inShape = model.inputs[0]?.shape
  const outName = model.outputNodes?.[0] ?? model.outputs?.[0]?.name
  console.log(`Input:  ${JSON.stringify(inShape)}`)
  console.log(`Output: ${outName ?? '(graph)'}`)
  console.log(`Classes: ${names.join(', ')}`)

  const dummy = tf.ones(inShape.map((dim) => (dim == null ? 1 : dim)))
  tf.dispose(model.execute(dummy))
  tf.dispose(dummy)

  const images = await listImages(args)
  if (!images.length) {
    console.error('Nenhuma imagem encontrada.')
    process.exit(1)
  }
  if (args.save) await fs.mkdir(args.out, { recursive: true })

  let totalDets = 0
  for (const imagePath of images) {
    const meta = await sharp(imagePath).metadata()
    const width = meta.width ?? INPUT_SIZE
    const height = meta.height ?? INPUT_SIZE
    const started = performance.now()
    const raw = await infer(model, imagePath)
    const ms = performance.now() - started
    const boxes = nms(decode(raw, names, args.conf, width, height), args.iou)
    totalDets += boxes.length
    const gt = await readYoloLabels(imagePath, args.labelsDir, width, height)

    console.log(`\n${path.basename(imagePath)}  ${width}x${height}  ${ms.toFixed(0)} ms  ${boxes.length} det  ${gt.length} gt`)
    for (const box of boxes) {
      console.log(
        `  ${box.label.padEnd(7)} ${(box.score * 100).toFixed(1).padStart(5)}%  ` +
          `xyxy=${box.x1.toFixed(0)},${box.y1.toFixed(0)},${box.x2.toFixed(0)},${box.y2.toFixed(0)}`,
      )
    }

    if (args.save) {
      const outFile = path.join(
        args.out,
        `${path.basename(imagePath, path.extname(imagePath))}.png`,
      )
      await drawBoxes(imagePath, boxes, outFile)
    }
  }

  console.log(`\n${images.length} imagens, ${totalDets} detecções (conf=${args.conf}, iou=${args.iou})`)
  if (args.save) console.log(`Caixas salvas em ${args.out}`)
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
