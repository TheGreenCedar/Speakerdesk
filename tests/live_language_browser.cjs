/* Real browser + loopback app fixture. Supply an existing Playwright package.
 * node tests/live_language_browser.cjs URL FIXTURE_FOLDER EVIDENCE PLAYWRIGHT_MODULE
 * Start tests/support/live_language_browser.py separately. No downloads/devices.
 */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const [url, folder, evidence, packagePath = '@playwright/test'] = process.argv.slice(2);
const { chromium, expect } = require(packagePath);

async function run() {
  fs.mkdirSync(evidence, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const checks = [], errors = [];
  try {
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, reducedMotion: 'reduce' });
    // All content stays on the local fixture server.
    await context.route('**/*', route => new URL(route.request().url()).origin === url ? route.continue() : route.abort());
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(url);
    await expect(page.locator('#live-language')).toHaveValue('en');
    await expect(page.locator('#live-language')).toBeEnabled();
    await expect(page.locator('.rolling-segment textarea')).toHaveValue('Synthetic en words.');
    await expect(page.locator('#pause-meeting')).toHaveText('Resume');
    assert.equal(await page.evaluate(() => liveLanguageTime(964800)), '01:00.30');
    const token = await page.locator('meta[name="speakerdesk-token"]').getAttribute('content');
    const headers = { 'X-Speakerdesk-Token': token };
    const state = await (await context.request.get(`${url}/api/meeting`)).json();
    const initial = await (await context.request.get(`${url}/api/jobs/${state.id}`)).json();

    // Hold a genuine old status response across PATCH, then deliver it late.
    let releasePoll, acquired;
    const acquiredPoll = new Promise(resolve => { acquired = resolve; });
    const heldPoll = new Promise(resolve => { releasePoll = resolve; });
    let held = false;
    await page.route('**/api/meeting', async route => {
      if (held) return route.continue();
      held = true;
      const response = await route.fetch();
      acquired();
      await heldPoll;
      await route.fulfill({ response });
    });
    const polling = page.evaluate(() => refreshMeeting());
    await acquiredPoll;
    const patch = page.waitForResponse(response => response.request().method() === 'PATCH' && response.url().endsWith('/language'));
    await page.locator('#live-language').selectOption('fr');
    assert.equal((await patch).status(), 200);
    await expect(page.locator('#live-language')).toBeEnabled();
    await expect(page.locator('#live-language-status')).toHaveText('From 00:00.10 · Queued · Changes also set the default');
    releasePoll(); await polling;
    await page.unroute('**/api/meeting');
    await expect(page.locator('#live-language')).toHaveValue('fr');
    checks.push('Late status response cannot restore a superseded language revision.');
    await expect(page.locator('#notice')).toContainText('Earlier words keep their language.');
    assert.equal(await page.locator('#language').inputValue(), 'fr');
    assert.equal((await (await context.request.get(`${url}/api/config`)).json()).default_language, 'fr');
    assert.equal(await page.evaluate(() => localStorage.getItem('speakerdesk.language_mode')), null);
    checks.push('Live selection becomes the server-persisted Settings default without local storage authority.');
    await page.screenshot({ path: path.join(evidence, 'live-language-queued-light.png') });

    fs.writeFileSync(path.join(folder, 'language-release'), '');
    await expect(page.locator('#live-language-status')).toHaveText('From 00:00.10 · Changes also set the default');
    await page.locator('#pause-meeting').click();
    await expect(page.locator('#pause-meeting')).toHaveText('Pause');
    await expect(page.locator('#live-language')).toBeEnabled();
    await page.evaluate(() => speakerdeskTheme.set('dark'));
    await page.screenshot({ path: path.join(evidence, 'live-language-recording-dark.png') });
    await page.locator('#pause-meeting').click();
    await expect(page.locator('#pause-meeting')).toHaveText('Resume');
    await expect(page.locator('.rolling-segment textarea')).toHaveCount(3);
    const after = await (await context.request.get(`${url}/api/jobs/${state.id}`)).json();
    assert.deepEqual(after.document.segments[0], initial.document.segments[0]);
    assert.deepEqual(after.document.segments.slice(1).map(s => [s.language, s.language_generation]), [['fr', 1], ['fr', 1]]);
    checks.push('Resume sends subsequent synthetic PCM to French; the earlier English passage is byte-for-byte unchanged.');

    for (const mode of ['auto', 'fr', 'auto']) {
      await page.locator('#live-language').selectOption(mode);
      await expect(page.locator('#live-language')).toBeEnabled();
      await expect(page.locator('#live-language')).toHaveValue(mode);
    }
    const toggled = await (await context.request.get(`${url}/api/jobs/${state.id}`)).json();
    assert.equal(toggled.language_revision, 4);
    assert.deepEqual(toggled.document.segments, after.document.segments);
    checks.push('Three further changes while paused advance revisions without replacing existing text.');

    await page.setViewportSize({ width: 900, height: 700 });
    await expect(page.locator('#live-language')).toBeInViewport();
    await expect(page.locator('#pause-meeting')).toBeInViewport();
    await expect(page.locator('#stop-meeting')).toBeInViewport();
    for (const id of ['live-language', 'pause-meeting', 'stop-meeting', 'microphone-level', 'system-level']) {
      const box = await page.locator(`#${id}`).boundingBox();
      assert.ok(box && box.x >= 0 && box.x + box.width <= 900 && box.y + box.height <= 700, `${id} fits the compact window`);
    }
    await page.screenshot({ path: path.join(evidence, 'live-language-paused-compact.png') });
    checks.push('Language, source meters, Resume and Stop fit the 900×700 window.');
    await page.locator('#setup-toggle').click();
    await expect(page.locator('#language')).toHaveValue('auto');
    await expect(page.locator('#language option:checked')).toHaveText('Automatic');
    await page.screenshot({ path: path.join(evidence, 'live-language-settings-default.png') });
    await page.locator('#settings-close').click();

    await page.locator('#stop-meeting').click();
    await expect(page.locator('#capture-footer')).toBeHidden();
    const final = await (await context.request.get(`${url}/api/jobs/${state.id}`)).json();
    assert.equal(final.status, 'ready');
    assert.deepEqual(final.document.segments, after.document.segments);
    const exported = await context.request.get(`${url}/api/jobs/${state.id}/export/txt`);
    assert.ok((await exported.text()).includes('Synthetic en words.'));
    assert.ok((await exported.text()).includes('Synthetic fr words.'));
    checks.push('Stop completes normally and text export retains both original-language fixture passages.');
    assert.deepEqual(errors, []);
    const result = { fixture: 'Synthetic PCM; no devices or models', checks, pageErrors: errors };
    fs.writeFileSync(path.join(evidence, 'browser-checks.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result, null, 2));
  } finally {
    await browser.close();
  }
}
run().catch(error => { console.error(error); process.exitCode = 1; });
