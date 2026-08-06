import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test, type APIRequestContext, type Browser, type Page } from "@playwright/test";

const BASE_URL = process.env.CRM_E2E_BASE_URL?.trim() || "https://nwa-agentos.vercel.app";
const OIDC_AUDIENCE = "nwa-agentos-crm-e2e";
const OUTPUT_DIR = join(process.cwd(), "test-results", "nwa-crm-evidence");
const TEST_RECIPIENT_ROOT = "fattohmariemichelle";
const TEST_RECIPIENT_DOMAIN = "gmail.com";

type Account = {
  id: string;
  email: string;
  display_name: string;
  first_name?: string | null;
  last_name?: string | null;
  role: "super_admin" | "admin" | "closer";
  status: string;
};

type PageDiagnostics = {
  consoleErrors: string[];
  failedRequests: string[];
  httpErrors: string[];
};

type AccountEvidence = {
  account: string;
  role: string;
  desktopLeadOptions: Array<{ value: string; text: string }>;
  desktopContactOptions: Array<{ value: string; text: string }>;
  mobileLeadOptions: Array<{ value: string; text: string }>;
  mobileContactOptions: Array<{ value: string; text: string }>;
  desktopPhoneVisible: boolean;
  mobilePhoneVisible: boolean;
  desktopSequenceMatches: number;
  mobileSequenceMatches: number;
  email?: {
    subject: string;
    apiStatus: number;
    providerMessageId: string;
    outboxRows: number;
    historyMessages: number;
    contactHistoryLinked: boolean;
    queueRequests: string[];
  };
  desktopDiagnostics: PageDiagnostics;
  mobileDiagnostics: PageDiagnostics;
};

type EvidenceAttachment = { name: string; content: string; mimeType: string; size: number };

let cachedOidc: { token: string; expiresAt: number } | null = null;

function slug(value: string) {
  return value.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function decodeJwtExpiry(token: string) {
  try {
    const payload = JSON.parse(Buffer.from(token.split(".")[1], "base64url").toString("utf8")) as { exp?: number };
    return Number(payload.exp || 0);
  } catch {
    return 0;
  }
}

async function githubOidcToken() {
  const now = Math.floor(Date.now() / 1000);
  if (cachedOidc && cachedOidc.expiresAt > now + 60) return cachedOidc.token;
  const requestUrl = process.env.ACTIONS_ID_TOKEN_REQUEST_URL;
  const requestToken = process.env.ACTIONS_ID_TOKEN_REQUEST_TOKEN;
  if (!requestUrl || !requestToken) throw new Error("GitHub Actions OIDC is unavailable for this job.");
  const separator = requestUrl.includes("?") ? "&" : "?";
  const response = await fetch(`${requestUrl}${separator}audience=${encodeURIComponent(OIDC_AUDIENCE)}`, {
    headers: { Authorization: `bearer ${requestToken}`, Accept: "application/json" },
  });
  const payload = await response.json().catch(() => ({})) as { value?: string; message?: string };
  if (!response.ok || !payload.value) throw new Error(payload.message || `Unable to obtain GitHub OIDC token (${response.status}).`);
  cachedOidc = { token: payload.value, expiresAt: decodeJwtExpiry(payload.value) };
  return payload.value;
}

async function e2eApi(request: APIRequestContext, action: string, body: Record<string, unknown> = {}, history = false) {
  const endpoint = history ? "/api/crm/e2e/history" : "/api/crm/e2e";
  const response = await request.post(`${BASE_URL}${endpoint}`, {
    headers: { Authorization: `Bearer ${await githubOidcToken()}` },
    data: { action, ...body },
    timeout: 60_000,
  });
  const payload = await response.json().catch(() => ({})) as Record<string, any>;
  if (!response.ok()) throw new Error(`${endpoint} ${action} failed (${response.status()}): ${JSON.stringify(payload)}`);
  return payload;
}

async function waitForProduction(request: APIRequestContext) {
  const deadline = Date.now() + 12 * 60_000;
  let lastError = "";
  while (Date.now() < deadline) {
    try {
      const payload = await e2eApi(request, "accounts");
      if (payload.ok) return payload;
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
    }
    await new Promise((resolve) => setTimeout(resolve, 10_000));
  }
  throw new Error(`Production did not expose the secured verification endpoint. Last error: ${lastError}`);
}

async function magicLink(request: APIRequestContext, email: string) {
  const payload = await e2eApi(request, "magic-link", { email });
  if (!payload.actionLink) throw new Error(`No test session link was generated for ${email}.`);
  return String(payload.actionLink);
}

function diagnosticsFor(page: Page): PageDiagnostics {
  const diagnostics: PageDiagnostics = { consoleErrors: [], failedRequests: [], httpErrors: [] };
  const origin = new URL(BASE_URL).origin;
  page.on("console", (message) => {
    if (message.type() === "error") diagnostics.consoleErrors.push(message.text());
  });
  page.on("requestfailed", (request) => {
    if (request.url().startsWith(origin)) diagnostics.failedRequests.push(`${request.method()} ${request.url()} :: ${request.failure()?.errorText || "unknown"}`);
  });
  page.on("response", (response) => {
    const url = response.url();
    if (!url.startsWith(origin) || response.status() < 400) return;
    const pathname = new URL(url).pathname;
    if (["/favicon.ico", "/apple-touch-icon.png"].includes(pathname)) return;
    diagnostics.httpErrors.push(`${response.status()} ${response.request().method()} ${url}`);
  });
  return diagnostics;
}

async function openAuthenticatedPage(browser: Browser, request: APIRequestContext, account: Account, mobile: boolean) {
  const context = await browser.newContext(mobile ? {
    viewport: { width: 390, height: 844 },
    deviceScaleFactor: 2,
    isMobile: true,
    hasTouch: true,
    locale: "en-GB",
    timezoneId: "Europe/Paris",
  } : {
    viewport: { width: 1440, height: 1000 },
    locale: "en-GB",
    timezoneId: "Europe/Paris",
  });
  const page = await context.newPage();
  const diagnostics = diagnosticsFor(page);
  await page.goto(await magicLink(request, account.email), { waitUntil: "domcontentloaded", timeout: 60_000 });
  await page.waitForURL((url) => url.origin === new URL(BASE_URL).origin && url.pathname.startsWith("/crm"), { timeout: 60_000 });
  await page.goto(`${BASE_URL}/crm?view=leads`, { waitUntil: "domcontentloaded", timeout: 60_000 });
  await expect(page.locator("body")).not.toContainText(/Sign in|Incorrect email or password/i, { timeout: 30_000 });
  await expect(page.getByText(/Leads/i).first()).toBeVisible({ timeout: 30_000 });
  return { context, page, diagnostics };
}

async function closeDrawer(page: Page) {
  const visibleClose = page.locator(".crm-refv2-drawer header button:visible, .crm-modal header button:visible, button[aria-label='Close']:visible, button[aria-label='Fermer']:visible").last();
  if (await visibleClose.count()) await visibleClose.click();
  else await page.keyboard.press("Escape");
}

async function openLeadForm(page: Page) {
  await page.goto(`${BASE_URL}/crm?view=leads`, { waitUntil: "domcontentloaded" });
  const exact = page.getByRole("button", { name: /^\s*\+?\s*(create (a )?lead|créer un lead|lead)\s*$/i }).last();
  if (await exact.count()) await exact.click();
  else await page.locator("button").filter({ hasText: /Create.*lead|Créer.*lead|\+\s*Lead/i }).last().click();
}

async function openContactForm(page: Page) {
  await page.goto(`${BASE_URL}/crm?view=contacts`, { waitUntil: "domcontentloaded" });
  const add = page.getByRole("button", { name: /Add contacts|Ajouter des contacts|New contact/i }).first();
  if (await add.count()) {
    await add.click();
    const create = page.getByRole("button", { name: /^(Create|Créer)$/i }).last();
    if (await create.count()) await create.click();
  } else {
    await page.locator("button").filter({ hasText: /Create.*contact|Créer.*contact|\+\s*Contact/i }).last().click();
  }
}

async function preferredLanguageOptions(page: Page, kind: "lead" | "contact") {
  if (kind === "lead") await openLeadForm(page);
  else await openContactForm(page);
  const label = page.locator("label").filter({ hasText: /Preferred language|Langue préférée/i }).last();
  const select = label.locator("select");
  await expect(select).toBeVisible({ timeout: 20_000 });
  const options = await select.locator("option").evaluateAll((nodes) => nodes.map((node) => ({
    value: (node as HTMLOptionElement).value,
    text: (node.textContent || "").trim(),
  })));
  expect(options).toEqual([
    { value: "fr", text: "French" },
    { value: "en", text: "English" },
  ]);
  await closeDrawer(page);
  return options;
}

function normalizedDigits(value: string) {
  return value.replace(/\D/g, "");
}

async function phoneVisible(page: Page) {
  await page.goto(`${BASE_URL}/crm?view=dialer`, { waitUntil: "domcontentloaded" });
  await expect(page.getByText(/Power Dialer/i).first()).toBeVisible({ timeout: 30_000 });
  const digits = normalizedDigits(await page.locator("body").innerText());
  return digits.includes("0656695823") || digits.includes("33656695823");
}

function isRequestedSequenceText(raw: string) {
  const text = raw.toLowerCase().replace(/\s+/g, " ");
  return text.includes("paused")
    && (text.includes("shared") || text.includes("partagée"))
    && /(steps|étapes)\s*3/.test(text)
    && /(enrollments|inscriptions)\s*0/.test(text)
    && /(daily limit|limite\/jour)\s*40/.test(text)
    && /1\.\s*linkedin/.test(text)
    && /2\.\s*linkedin/.test(text)
    && /(2\s*d|2\s*j)/.test(text)
    && /3\.\s*(manual email|email manual|e-mail manuel)/.test(text)
    && /(3\s*d|3\s*j)/.test(text)
    && /(by|par)\s+michael/.test(text)
    && /(activate|activer)/.test(text);
}

async function requestedSequenceCards(page: Page) {
  await page.goto(`${BASE_URL}/crm?view=sequences`, { waitUntil: "domcontentloaded" });
  await expect(page.getByText(/Sequences|Séquences|Follow-up sequences|Suivi Sales Navigator/i).first()).toBeVisible({ timeout: 30_000 });
  const cards = await page.locator(".crm-feature-card").allTextContents();
  return cards.filter(isRequestedSequenceText);
}

async function openCompose(page: Page) {
  await page.goto(`${BASE_URL}/crm?view=email`, { waitUntil: "domcontentloaded" });
  let heading = page.getByRole("heading", { name: /Compose an email|Rédiger un e-mail/i }).last();
  if (!(await heading.count())) {
    await page.goto(`${BASE_URL}/crm?view=inbox`, { waitUntil: "domcontentloaded" });
    await page.getByRole("button", { name: /Compose|Rédiger/i }).last().click();
    heading = page.getByRole("heading", { name: /Compose an email|Rédiger un e-mail/i }).last();
  }
  await expect(heading).toBeVisible({ timeout: 30_000 });
  return heading;
}

async function sendDirectEmail(page: Page, request: APIRequestContext, account: Account, recipient: string, subject: string) {
  const queueRequests: string[] = [];
  const listener = (browserRequest: any) => {
    const url = browserRequest.url();
    if ((url.includes("crm_outbox") && browserRequest.method() !== "GET") || url.includes("/api/crm/email/process")) {
      queueRequests.push(`${browserRequest.method()} ${url}`);
    }
  };
  page.on("request", listener);
  const heading = await openCompose(page);
  await page.getByLabel(/Recipient|Destinataire/i).fill(recipient);
  await page.getByLabel(/Subject|Objet/i).fill(subject);
  await page.getByLabel(/Message/i).fill(`Controlled Playwright delivery verification for ${account.display_name}.`);
  const responsePromise = page.waitForResponse((response) => response.url().includes("/api/crm/email/send") && response.request().method() === "POST", { timeout: 90_000 });
  await page.getByRole("button", { name: /Send email|Envoyer l'e-mail|Envoyer l’e-mail/i }).click();
  const response = await responsePromise;
  const payload = await response.json().catch(() => ({})) as Record<string, any>;
  expect(response.status()).toBe(200);
  expect(payload.result?.sent).toBe(true);
  expect(payload.result?.recorded).toBe(true);
  expect(payload.result?.status).toBe("sent");
  await expect(heading).toHaveCount(0, { timeout: 30_000 });
  page.off("request", listener);

  const evidence = await e2eApi(request, "evidence", { subject }, true);
  expect(queueRequests).toEqual([]);
  expect(evidence.outbox).toEqual([]);
  expect(evidence.messages.some((message: any) => message.delivery_status === "sent")).toBe(true);
  const contactHistoryLinked = evidence.threads.some((thread: any) => thread.owner_id === account.id && thread.contact?.email === recipient);
  expect(contactHistoryLinked).toBe(true);
  return {
    subject,
    apiStatus: response.status(),
    providerMessageId: String(payload.result.providerMessageId || ""),
    outboxRows: evidence.outbox.length,
    historyMessages: evidence.messages.length,
    contactHistoryLinked,
    queueRequests,
  };
}

function isMichael(account: Account) {
  const identity = `${account.email} ${account.display_name} ${account.first_name || ""} ${account.last_name || ""}`.toLowerCase();
  return identity.includes("crm-michael") || identity.includes("michael@nowordsagency.com") || identity.includes("michael inthasane");
}

function assertNoDiagnostics(account: string, viewport: string, diagnostics: PageDiagnostics) {
  expect(diagnostics.consoleErrors, `${account} ${viewport}: console errors`).toEqual([]);
  expect(diagnostics.failedRequests, `${account} ${viewport}: failed requests`).toEqual([]);
  expect(diagnostics.httpErrors, `${account} ${viewport}: HTTP errors`).toEqual([]);
}

function attachment(path: string, name: string): EvidenceAttachment {
  const data = readFileSync(path);
  return { name, content: data.toString("base64"), mimeType: name.endsWith(".json") ? "application/json" : "image/jpeg", size: data.length };
}

async function secureReportDelivery(request: APIRequestContext, report: Record<string, unknown>, paths: Array<{ path: string; name: string }>) {
  const reportPath = join(OUTPUT_DIR, "report.json");
  writeFileSync(reportPath, JSON.stringify(report, null, 2));
  const attachments = [attachment(reportPath, "nwa-crm-playwright-report.json"), ...paths.filter((item) => {
    try { readFileSync(item.path); return true; } catch { return false; }
  }).map((item) => attachment(item.path, item.name))];
  await e2eApi(request, "deliver-report", {
    report: JSON.stringify(report, null, 2),
    attachments,
  }, true);
}

test.describe.configure({ mode: "serial" });
test.setTimeout(35 * 60_000);

test("all requested CRM corrections are verified on real accounts", async ({ browser, request }) => {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  const report: Record<string, any> = {
    startedAt: new Date().toISOString(),
    baseUrl: BASE_URL,
    deployment: "production",
    status: "running",
    accounts: [],
    cleanup: null,
    preCleanupSequenceMatches: null,
    postCleanupSequenceMatches: null,
    accountEvidence: [] as AccountEvidence[],
    failure: null,
  };
  const reportScreenshots: Array<{ path: string; name: string }> = [];
  const testContacts: string[] = [];
  let failure: unknown = null;

  try {
    const initial = await waitForProduction(request);
    const activeAccounts = (initial.accounts as Account[]).filter((account) => account.status === "active");
    const admin = activeAccounts.find((account) => account.role === "super_admin") || activeAccounts.find((account) => account.role === "admin");
    expect(admin, "An active main/admin CRM account is required.").toBeTruthy();
    const closers = activeAccounts.filter((account) => account.role === "closer");
    expect(closers.length, "Exactly seven active closer spaces must be tested.").toBe(7);
    const michael = closers.find(isMichael);
    expect(michael, "Michael's CRM profile was not identified uniquely.").toBeTruthy();
    const accounts = [admin!, ...closers];
    report.accounts = accounts.map((account) => ({ id: account.id, email: account.email, display_name: account.display_name, role: account.role }));

    const beforeSession = await openAuthenticatedPage(browser, request, michael!, false);
    const preMatches = await requestedSequenceCards(beforeSession.page);
    expect(preMatches.length).toBeLessThanOrEqual(1);
    report.preCleanupSequenceMatches = preMatches.length;
    const beforeSequencePath = join(OUTPUT_DIR, "before-michael-sequence.jpg");
    await beforeSession.page.screenshot({ path: beforeSequencePath, type: "jpeg", quality: 65, fullPage: true });
    reportScreenshots.push({ path: beforeSequencePath, name: "before-michael-sequence.jpg" });
    await beforeSession.context.close();

    const cleanup = await e2eApi(request, "cleanup");
    report.cleanup = cleanup.result;
    expect(cleanup.result.phoneOwners).toHaveLength(1);
    expect(cleanup.result.phoneOwners[0].profile_id).toBe(michael!.id);
    expect(cleanup.result.phoneOwners[0].phone_number).toBe("+33656695823");
    expect(cleanup.result.remainingRequestedSequenceMatches).toBe(0);

    const stamp = Date.now();
    for (const account of accounts) {
      const recipient = `${TEST_RECIPIENT_ROOT}+crm-e2e-${slug(account.display_name)}-${stamp}@${TEST_RECIPIENT_DOMAIN}`;
      testContacts.push(recipient);
      await e2eApi(request, "create-contact", { ownerEmail: account.email, email: recipient }, true);

      const desktop = await openAuthenticatedPage(browser, request, account, false);
      const desktopLeadOptions = await preferredLanguageOptions(desktop.page, "lead");
      const desktopContactOptions = await preferredLanguageOptions(desktop.page, "contact");
      const desktopPhoneVisible = await phoneVisible(desktop.page);
      expect(desktopPhoneVisible).toBe(account.id === michael!.id);
      const desktopSequenceMatches = (await requestedSequenceCards(desktop.page)).length;
      expect(desktopSequenceMatches).toBe(0);
      await desktop.page.reload({ waitUntil: "domcontentloaded" });
      expect((await requestedSequenceCards(desktop.page)).length).toBe(0);

      const subject = `NWA CRM direct send ${slug(account.display_name)} ${stamp}`;
      const email = await sendDirectEmail(desktop.page, request, account, recipient, subject);
      if (account.id === michael!.id) {
        await desktop.page.goto(`${BASE_URL}/crm?view=dialer`, { waitUntil: "domcontentloaded" });
        const michaelDialerPath = join(OUTPUT_DIR, "after-michael-power-dialer.jpg");
        await desktop.page.screenshot({ path: michaelDialerPath, type: "jpeg", quality: 65, fullPage: true });
        reportScreenshots.push({ path: michaelDialerPath, name: "after-michael-power-dialer.jpg" });
        await desktop.page.goto(`${BASE_URL}/crm?view=sequences`, { waitUntil: "domcontentloaded" });
        const afterSequencePath = join(OUTPUT_DIR, "after-michael-sequence.jpg");
        await desktop.page.screenshot({ path: afterSequencePath, type: "jpeg", quality: 65, fullPage: true });
        reportScreenshots.push({ path: afterSequencePath, name: "after-michael-sequence.jpg" });
      }
      assertNoDiagnostics(account.display_name, "desktop", desktop.diagnostics);

      const mobile = await openAuthenticatedPage(browser, request, account, true);
      const mobileLeadOptions = await preferredLanguageOptions(mobile.page, "lead");
      const mobileContactOptions = await preferredLanguageOptions(mobile.page, "contact");
      const mobilePhoneVisible = await phoneVisible(mobile.page);
      expect(mobilePhoneVisible).toBe(account.id === michael!.id);
      const mobileSequenceMatches = (await requestedSequenceCards(mobile.page)).length;
      expect(mobileSequenceMatches).toBe(0);
      assertNoDiagnostics(account.display_name, "mobile", mobile.diagnostics);

      if (account.id === michael!.id) {
        await openLeadForm(mobile.page);
        const mobileLanguagePath = join(OUTPUT_DIR, "after-michael-mobile-language.jpg");
        await mobile.page.screenshot({ path: mobileLanguagePath, type: "jpeg", quality: 65, fullPage: true });
        reportScreenshots.push({ path: mobileLanguagePath, name: "after-michael-mobile-language.jpg" });
        await closeDrawer(mobile.page);
      }

      (report.accountEvidence as AccountEvidence[]).push({
        account: account.display_name,
        role: account.role,
        desktopLeadOptions,
        desktopContactOptions,
        mobileLeadOptions,
        mobileContactOptions,
        desktopPhoneVisible,
        mobilePhoneVisible,
        desktopSequenceMatches,
        mobileSequenceMatches,
        email,
        desktopDiagnostics: desktop.diagnostics,
        mobileDiagnostics: mobile.diagnostics,
      });
      await desktop.context.close();
      await mobile.context.close();
    }

    const relogin = await openAuthenticatedPage(browser, request, michael!, false);
    expect((await requestedSequenceCards(relogin.page)).length).toBe(0);
    await relogin.context.close();

    const finalEvidence = await e2eApi(request, "evidence");
    expect(finalEvidence.phoneRows).toHaveLength(1);
    expect(finalEvidence.phoneRows[0].profile_id).toBe(michael!.id);
    expect(finalEvidence.requestedSequenceMatches).toBe(0);
    report.postCleanupSequenceMatches = finalEvidence.requestedSequenceMatches;
    report.finalEvidence = finalEvidence;
    report.status = "passed";
  } catch (error) {
    failure = error;
    report.status = "failed";
    report.failure = error instanceof Error ? { name: error.name, message: error.message, stack: error.stack } : String(error);
  } finally {
    for (const email of testContacts) {
      try { await e2eApi(request, "archive-contact", { email }, true); } catch { /* report delivery is more important */ }
    }
    report.finishedAt = new Date().toISOString();
    try {
      await secureReportDelivery(request, report, reportScreenshots);
    } catch (deliveryError) {
      report.reportDeliveryError = deliveryError instanceof Error ? deliveryError.message : String(deliveryError);
      writeFileSync(join(OUTPUT_DIR, "report.json"), JSON.stringify(report, null, 2));
    }
    console.log(`NWA_CRM_PLAYWRIGHT_REPORT=${JSON.stringify({ status: report.status, accounts: report.accounts, cleanup: report.cleanup, failure: report.failure })}`);
  }

  if (failure) throw failure;
});
