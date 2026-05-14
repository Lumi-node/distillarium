// @ts-check
import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://distillarium.app",
  trailingSlash: "ignore",
  build: {
    format: "directory",
  },
  vite: {
    server: {
      // Allow serving the dev preview from the bind address used by Cloudflare Tunnels / LAN
      host: true,
    },
  },
});
