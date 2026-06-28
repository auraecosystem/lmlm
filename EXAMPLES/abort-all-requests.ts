/**
 * EXAMPLES/abort-all-requests.ts
 *
 * Demonstrates:
 * - Concurrent streaming generation
 * - Automatic timeout cancellation
 * - Global request abortion
 * - Independent error handling
 * - Clean SDK usage
 */
import { LMLM } from 'lmlm'
const client = new LMLM({
  host: 'http://localhost:11434',
})
const MODEL = 'lmlm-7b'
const TIMEOUT_MS = 5_000
/**
 * Abort all active requests after the timeout.
 */
setTimeout(() => {
  console.log('\n🛑 Aborting all active requests...\n')
  client.abort()
}, TIMEOUT_MS)
/**
 * Stream a generation request.
 */
async function generate(
  id: number,
  prompt: string,
): Promise<void> {
  try {
    console.log(`🚀 Starting request ${id}`)
    const stream = await client.generate({
      model: MODEL,
      prompt,
      stream: true,
    })
    for await (const chunk of stream) {
      process.stdout.write(`[${id}] ${chunk.response}`)
    }
    console.log(`\n✅ Request ${id} completed`)
  } catch (error: any) {
    switch (error.name) {
      case 'AbortError':
        console.log(`🛑 Request ${id} aborted`)
        break
      case 'ModelNotFoundError':
        console.error(`❌ Model "${MODEL}" not found`)
        break
      case 'ConnectionError':
        console.error(`❌ Unable to connect to the LMLM server`)
        break
      default:
        console.error(`❌ Request ${id} failed`, error)
    }
  }
}
/**
 * Application entry point.
 */
async function main() {
  console.log('======================================')
  console.log(' LMLM Concurrent Streaming Demo')
  console.log('======================================\n')
  console.log(`Model   : ${MODEL}`)
  console.log(`Timeout : ${TIMEOUT_MS} ms\n`)
  await Promise.all([
    generate(
      1,
      'Write a long fantasy story about dragons protecting an ancient kingdom.'
    ),
    generate(
      2,
      'Write a long fantasy story about powerful wizards discovering forgotten magic.'
    ),
    generate(
      3,
      'Write a long fantasy story about legendary knights defending a sacred relic.'
    ),
  ])
  console.log('\n🎉 Demo finished.')
}
main().catch((error) => {
  console.error('Fatal Error:', error)
})
