import { readFileSync, writeFileSync } from "node:fs";

const path = "tests/nwa-crm.spec.ts";
let source = readFileSync(path, "utf8");

const current = `    const admin = activeAccounts.find((account) => account.role === "super_admin") || activeAccounts.find((account) => account.role === "admin");
    expect(admin, "An active main/admin CRM account is required.").toBeTruthy();
    const closers = activeAccounts.filter((account) => account.role === "closer");
    expect(closers.length, "Exactly seven active closer spaces must be tested.").toBe(7);
    const michael = closers.find(isMichael);
    expect(michael, "Michael's CRM profile was not identified uniquely.").toBeTruthy();
    const accounts = [admin!, ...closers];`;

const replacement = `    const adminCandidates = activeAccounts.filter((account) => account.role === "super_admin" || account.role === "admin");
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

if (!source.includes(current)) {
  throw new Error("The expected closer selection block was not found in the Playwright test.");
}
source = source.replace(current, replacement);
writeFileSync(path, source, "utf8");
console.log("Playwright matrix restricted to the seven explicitly requested closer identities.");
