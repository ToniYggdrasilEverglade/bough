# Integrate the Bough UI refresh with Copilot

These assets are supplied for integration; index.html is intentionally unchanged. Preserve local modifications when applying this patch.

## Copilot prompt

Integrate frontend/professional.css and frontend/theme.js into the existing Bough frontend. In frontend/index.html, load theme.js synchronously in head before the stylesheet links: <script src="theme.js?v=1"></script>. Keep style.css and load <link rel="stylesheet" href="professional.css?v=1"> immediately after it. Add exactly one button inside .controls: <button id="theme-toggle" class="ghost" type="button" aria-label="Switch theme">Switch theme</button>. Set the page title and visible brand heading to Bough. Set the brand subtitle to: A clearer path from prompt to response. Preserve all existing element IDs, event handlers, authentication, memory controls, mock-model warning, WebSockets and backend behavior. Remove duplicate theme handlers or duplicate theme buttons if already added locally. Do not add dependencies or external fonts. Review backend static-file serving to ensure professional.css and theme.js can be fetched and add narrow routes if the existing server explicitly lists allowed files. Do not broadly expose the project directory.

## Validation

- Verify both new asset URLs return 200 with appropriate content types.
- Verify system light and dark defaults with no bough-theme storage entry.
- Toggle both ways and reload; the selected theme should persist.
- Verify the page still works when localStorage is unavailable.
- Check desktop and narrow mobile layouts, keyboard focus and hidden auth/mock panels.
- Confirm sending, stopping, memory controls and sign-in still work.
- Run the existing tests; report actual results.

The assets have not been browser-tested. This is a presentation change, not a backend or authentication fix.
