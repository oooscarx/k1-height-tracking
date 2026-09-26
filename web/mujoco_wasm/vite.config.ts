import {defineConfig} from "vite";

const repository = process.env.GITHUB_REPOSITORY?.split("/")[1];

export default defineConfig({
  base: repository ? `/${repository}/` : "/",
  build: {
    target: "es2022",
  },
});
