// Site identity shared by the About / Contact pages and the footers. One place, so a repo move or
// a new contact address is a one-line change.

export const SITE_URL = "https://brawldraft.com";

// The project is open source; the public issue tracker is the primary contact channel.
export const GITHUB_REPO_URL = "https://github.com/mctatge/Brawlstars-Draft-Tool";
export const GITHUB_ISSUES_URL = `${GITHUB_REPO_URL}/issues`;

export const AUTHOR_NAME = "Mitchell Tatge";
export const AUTHOR_GITHUB_URL = "https://github.com/mctatge";

// PLACEHOLDER — no contact address is published anywhere in this repo, so none is hardcoded here.
// Set this to the address that should appear on /contact (a dedicated mailbox rather than a
// personal one is the usual choice). While it is null, the Contact page shows GitHub only.
export const CONTACT_EMAIL: string | null = null;
