import { readFileSync, writeFileSync } from "node:fs";

const path = "tests/nwa-crm.spec.ts";
let source = readFileSync(path, "utf8");

const currentMagicLink = `async function magicLink(request: APIRequestContext, email: string) {
  const payload = await e2eApi(request, "magic-link", { email });
  if (!payload.actionLink) throw new Error(\`No test session link was generated for \${email}.\`);
  return String(payload.actionLink);
}`;

const replacementMagicLink = `async function magicLink(request: APIRequestContext, email: string) {
  const response = await request.post(\`\${BASE_URL}/api/crm/e2e/session\`, {
    headers: { Authorization: \`Bearer \${await githubOidcToken()}\` },
    data: { email },
    timeout: 60_000,
  });
  const payload = await response.json().catch(() => ({})) as Record<string, any>;
  if (!response.ok() || !payload.callbackUrl) {
    throw new Error(payload.error || \`No test session callback was generated for \${email}.\`);
  }
  return String(payload.callbackUrl);
}`;

if (!source.includes(currentMagicLink)) {
  throw new Error("The expected magic-link helper was not found in the Playwright test.");
}
source = source.replace(currentMagicLink, replacementMagicLink);

const currentCallbackWait = `  await page.waitForURL((url) => url.origin === new URL(BASE_URL).origin && url.pathname.startsWith("/crm"), { timeout: 60_000 });`;
const replacementCallbackWait = `  await page.waitForURL((url) => url.origin === new URL(BASE_URL).origin && url.pathname === "/crm", { timeout: 60_000 });`;
if (!source.includes(currentCallbackWait)) {
  throw new Error("The expected CRM callback wait was not found.");
}
source = source.replace(currentCallbackWait, replacementCallbackWait);

const currentReadiness = `  await expect(page.locator("body")).not.toContainText(/Sign in|Incorrect email or password/i, { timeout: 30_000 });
  await expect(page.getByText(/Leads/i).first()).toBeVisible({ timeout: 30_000 });
  return { context, page, diagnostics };`;

const replacementReadiness = `  await expect(page.locator("body")).not.toContainText(/Sign in|Incorrect email or password/i, { timeout: 30_000 });
  try {
    await expect(page.getByText(/Leads/i).first()).toBeVisible({ timeout: 30_000 });
  } catch {
    const bodyText = (await page.locator("body").innerText()).replace(/\\s+/g, " ").slice(0, 1200);
    throw new Error(\`Authenticated CRM shell did not render Leads for \${account.email}. URL: \${page.url()}. Body: \${bodyText}\`);
  }
  return { context, page, diagnostics };`;

if (!source.includes(currentReadiness)) {
  throw new Error("The expected authenticated CRM readiness block was not found.");
}
source = source.replace(currentReadiness, replacementReadiness);

const currentAccounts = `    const admin = activeAccounts.find((account) => account.role === "super_admin") || activeAccounts.find((account) => account.role === "admin");
    expect(admin, "An active main/admin CRM account is required.").toBeTruthy();
    const closers = activeAccounts.filter((account) => account.role === "closer");
    expect(closers.length, "Exactly seven active closer spaces must be tested.").toBe(7);
    const michael = closers.find(isMichael);
    expect(michael, "Michael's CRM profile was not identified uniquely.").toBeTruthy();
    const accounts = [admin!, ...closers];`;

const replacementAccounts = `    const adminCandidates = activeAccounts.filter((account) => account.role === "super_admin" || account.role === "admin");
    const admin = adminCandidates.find((account) => account.email.trim().toLowerCase() === "fattohmariemichelle@gmail.com")
      || adminCandidates.find((account) => account.role === "super_admin")
      || adminCandidates[0];
    expect(admin, "An active main/admin CRM account is required.").toBeTruthy();

    const intendedCloserIdentities = [
      { name: "Michael Inthasane", professional: "michael@nowordsagency.com", legacy: "fattohmariemichelle+crm-michael@gmail.com" },
      { name: "Bethel Igbelina", professional: "bethel@nowordsagency.com", legacy: "fattohmariemichelle+crm-bethel@gmail.com" },
      { name: "Cyril Chukwu Ikaechukwu", professional: "cyril@nowordsagency.com", legacy: "fattohmariemichelle+crm-cyril@gmail.com" },
      { name: "Ike-Nwemem Destiny Oguchialu", professional: "destiny@nowordsagency.com", legacy: "fattohmariemichelle+crm-destiny@gmail.com" },
      { name: "Kenneth Chibuzor Okanu", professional: "kenneth@nowordsagency.com", legacy: "fattohmariemichelle+crm-kenneth@gmail.com" },
      { name: "Lilian Michael Chimdike", professional: "lilian@nowordsagency.com", legacy: "fattohmariemichelle+crm-lilian@gmail.com" },
      { name: "Nokwanda Ntuli", professional: "nokwanda@nowordsagency.com", legacy: "fattohmariemichelle+crm-nokwanda@gmail.com" },
    ] as const;
    const activeCloserProfiles = activeAccounts.filter((account) => account.role === "closer");
    const closers = intendedCloserIdentities.map((identity) => {
      const professional = activeCloserProfiles.filter((account) => account.email.trim().toLowerCase() === identity.professional);
      const legacy = activeCloserProfiles.filter((account) => account.email.trim().toLowerCase() === identity.legacy);
      const byName = activeCloserProfiles.filter((account) => account.display_name.trim().toLowerCase() === identity.name.toLowerCase());
      const candidates = professional.length ? professional : legacy.length ? legacy : byName;
      expect(candidates.length, identity.name + " must resolve to exactly one active closer CRM profile.").toBe(1);
      return candidates[0];
    });
    expect(new Set(closers.map((account) => account.id)).size, "The seven selected closer spaces must be distinct.").toBe(7);
    const michael = closers.find(isMichael);
    expect(michael, "Michael's CRM profile was not identified uniquely.").toBeTruthy();
    const accounts = [admin!, ...closers];`;

if (!source.includes(currentAccounts)) {
  throw new Error("The expected closer selection block was not found in the Playwright test.");
}
source = source.replace(currentAccounts, replacementAccounts);
writeFileSync(path, source, "utf8");
console.log("Playwright waits for the completed OTP callback, then tests the exact seven-account matrix.");
