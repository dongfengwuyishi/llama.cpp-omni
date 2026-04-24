import { cp, mkdir, rm } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const projectRoot = path.resolve(__dirname, '..')
const sourceDir = path.join(projectRoot, 'dist')
// App layout:
//   apps/frontend-mobile/   ← this project (React source)
//   apps/frontend/mobile/   ← published static bundle served via gateway /mobile/
const targetDir = path.resolve(projectRoot, '../frontend/mobile')

async function main() {
  await rm(targetDir, { recursive: true, force: true })
  await mkdir(targetDir, { recursive: true })
  await cp(sourceDir, targetDir, { recursive: true })
  console.log(`[mobile] published ${sourceDir} -> ${targetDir}`)
}

main().catch((error) => {
  console.error('[mobile] publish failed')
  console.error(error)
  process.exit(1)
})
