import { Hono } from 'hono'

const app = new Hono()

app.get('/', (c) => {
  c.header('Cache-Control', 'public, max-age=86400')
  return c.json({ message: 'hello' })
})

async function test() {
  const req = new Request('http://localhost/')
  const res = await app.fetch(req)
  console.log('Cache-Control:', res.headers.get('Cache-Control'))
}

test()
