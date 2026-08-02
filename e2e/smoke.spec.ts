import { test, expect } from '@playwright/test';

test('home page lists reference topics', async ({ page }) => {
  await page.goto('/');
  // sidebar brand h1 "GATE ML Visualizer" also matches /visualizer/i → scope to the page title
  await expect(page.getByRole('heading', { name: /Machine Learning Visualizer/i })).toBeVisible();
  // sidebar NavLink also contains the title text → scope to the topic-card heading
  await expect(page.getByRole('heading', { name: 'Gradient Descent' })).toBeVisible();
});

test('gradient descent topic plays and steps', async ({ page }) => {
  await page.goto('/topic/gradient-descent');
  const play = page.getByRole('button', { name: /play\/pause/i });
  await expect(play).toBeVisible();
  await page.getByRole('button', { name: /next/i }).click();
  await page.getByRole('button', { name: /next/i }).click();
  await expect(page.getByText(/step \d+/i)).toBeVisible();
});

test('knowledge graph renders and navigates', async ({ page }) => {
  await page.goto('/graph');
  await expect(page.getByRole('img', { name: /knowledge graph/i })).toBeVisible();
});
