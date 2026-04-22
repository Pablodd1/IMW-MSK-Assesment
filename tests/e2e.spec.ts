import { test, expect, type Page } from '@playwright/test'

// Test credentials
const DEMO_EMAIL = 'demo@physiomotion.com'
const BASE_URL = process.env.BASE_URL || 'http://localhost:3000'

// Helper functions
async function loginAsDemo(page: Page) {
  await page.goto(`${BASE_URL}/static/login.html`)
  await page.click('#demoBtn')
  await page.waitForURL(/dashboard/)
}

test.describe('PhysioMotion E2E Tests', () => {
  test('health endpoint works', async ({ request }) => {
    const response = await request.get(`${BASE_URL}/api/health`)
    expect(response.ok()).toBe(true)
    const data = await response.json()
    expect(data.status).toBe('ok')
  })

  test('login page loads', async ({ page }) => {
    await page.goto(`${BASE_URL}/static/login.html`)
    await expect(page.locator('text=PhysioMotion')).toBeVisible()
    await expect(page.locator('#loginForm')).toBeVisible()
    await expect(page.locator('#demoBtn')).toBeVisible()
  })

  test('demo access works', async ({ page }) => {
    await page.goto(`${BASE_URL}/static/login.html`)
    await page.click('#demoBtn')
    await page.waitForURL(/dashboard/)
    await expect(page.locator('text=Live Movement')).toBeVisible()
  })

  test('dashboard shows camera options', async ({ page }) => {
    await loginAsDemo(page)
    await expect(page.locator('.camera-btn')).toHaveCount(4)
  })

  test('can select camera type', async ({ page }) => {
    await loginAsDemo(page)
    await page.click('button:text("Femto Mega")')
    await expect(page.locator('button:has-text("Femto Mega")')).toHaveClass(/border-cyan-500/)
  })

  test('exercises API endpoint works', async ({ request }) => {
    const response = await request.get(`${BASE_URL}/api/exercises`)
    expect(response.ok()).toBe(true)
    const data = await response.json()
    expect(data.success).toBe(true)
    expect(Array.isArray(data.data)).toBe(true)
  })

  test('patients list loads after login', async ({ page }) => {
    await loginAsDemo(page)
    await page.click('text=Patients')
    // Should show patient list or empty state
    await expect(page.locator('text=Patients') || page.locator('text=No patients')).toBeVisible()
  })
})

test.describe('Joint Tracking Visual Tests', () => {
  test('skeleton overlay renders', async ({ page }) => {
    await loginAsDemo(page)
    
    // Toggle skeleton overlay
    await page.click('button:text("Skeleton")')
    
    // Canvas should exist for drawing
    const canvas = page.locator('#canvas-overlay')
    await expect(canvas).toBeAttached()
  })

  test('FPS counter displays', async ({ page }) => {
    await loginAsDemo(page)
    const fpsCounter = page.locator('#fps-counter')
    await expect(fpsCounter).toBeVisible()
    await expect(fpsCounter).toContainText('FPS')
  })
})

test.describe('Mobile Responsiveness', () => {
  test('login works on mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 })
    await page.goto(`${BASE_URL}/static/login.html`)
    await expect(page.locator('input#email')).toBeVisible()
    await expect(page.locator('input#password')).toBeVisible()
    await expect(page.locator('#demoBtn')).toBeVisible()
  })

  test('dashboard works on mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 })
    await loginAsDemo(page)
    await expect(page.locator('h2:has-text("Live Movement")')).toBeVisible()
  })
})

// Performance tests
test.describe('Performance', () => {
  test('page loads under 3 seconds', async ({ page }) => {
    const startTime = Date.now()
    await page.goto(`${BASE_URL}/static/login.html`)
    const loadTime = Date.now() - startTime
    expect(loadTime).toBeLessThan(3000)
  })

  test('no render-blocking errors in console', async ({ page }) => {
    const errors: string[] = []
    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text())
      }
    })
    
    await page.goto(`${BASE_URL}/static/login.html`)
    await page.waitForTimeout(1000)
    
    // Ignore expected errors (like camera permission)
    const criticalErrors = errors.filter(e => 
      !e.includes('Permission') && 
      !e.includes('Camera') &&
      !e.includes('getUserMedia')
    )
    expect(criticalErrors.length).toBe(0)
  })
})

// Accessibility tests
test.describe('Accessibility', () => {
  test('login form has proper labels', async ({ page }) => {
    await page.goto(`${BASE_URL}/static/login.html`)
    await expect(page.locator('label[for="email"]')).toBeVisible()
    await expect(page.locator('label[for="password"]')).toBeVisible()
  })

  test('buttons have accessible names', async ({ page }) => {
    await page.goto(`${BASE_URL}/static/login.html`)
    await expect(page.locator('#loginButton')).toHaveAttribute('type', 'submit')
    await expect(page.locator('#demoBtn')).toHaveAttribute('type', 'button')
  })
})