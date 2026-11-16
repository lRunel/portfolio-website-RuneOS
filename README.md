# Rune-os — Shubham Chowta's portfolio

A portfolio site that runs on a small operating-system kernel written in C and compiled to WebAssembly.
The desktop, windows and terminal are plain JavaScript; every terminal command (`ls`, `cd`, `cat`, `ps`, …)
is sent to the kernel as a syscall.

## What's in the kernel

`kernel/` (~1,000 lines of C, built with Emscripten):

- **Processes** — process control blocks, spawn/exit/reap, and a cooperative round-robin scheduler.
- **Virtual filesystem** — an in-memory tree with directories and files, plus pseudo-files:
  `/proc` (process list) and `/kernel/state` (live kernel stats, shown in the desktop widget).
- **Syscalls** — `PING, SPAWN, EXIT, REAP, READ, WRITE, MKDIR, LS, TOUCH, CHDIR, GETCWD, SETSTATE`,
  dispatched in `core/kernel.c` and exposed to JavaScript through `kernel_wasm.c`.
- **Content** — `~/resume.txt` and `~/projects/*.txt` are short summaries compiled into the kernel.

"Cache" (`cache <question>` in the terminal) is a scripted Q&A helper: it matches questions against
hand-written answers in `js/qa_database.js`. It is not a language model.

## Layout

```
index.html            Desktop + mobile layouts
css/index.css         All styles for the main page
js/                   Shell (syscall bridge), window manager, apps, desktop background
kernel/               C kernel source + Makefile
wasm/                 Compiled kernel (kernel.js + kernel.wasm) — commit after rebuilding
projects/<name>/      One standalone page per project, with its docs and figures
assets/               Resume PDF, favicon, wallpapers
build-site.sh         Copies just the site files into dist/ for deployment
```

## Run locally

```bash
python3 -m http.server 8000     # then open http://localhost:8000
```

## Rebuild the kernel

Requires the [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html) installed at `./emsdk`
(or edit `EMCC` in `kernel/Makefile`).

```bash
make -C kernel
```

## Deploy (Cloudflare Pages)

```bash
./build-site.sh
npx wrangler pages deploy dist
```

## Credits

Interactive line figures: [hairline](https://hairline.lucasmarkes.com) by Lucas Marques (MIT), vendored at `js/vendor/hairline.js`.
