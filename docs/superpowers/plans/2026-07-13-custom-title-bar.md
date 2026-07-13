# Custom Window Title Bar Implementation Plan

> **For agentic workers:** Use subagent-driven-development or executing-plans to implement this plan task-by-task.

**Goal:** Replace the native OS window title bar with a custom Slint title bar featuring window controls and a theme toggle, inspired by wsl-dashboard's design.

**Architecture:** Set `no-frame: true` on the Window element to remove native decorations, then place a custom `TitleBar` component at the top of the layout. Window drag and control callbacks are wired in both Slint (button interactions) and Rust (window positioning/manipulation).

**Tech Stack:** Slint 1.13.1, Rust

---

### Task 1: Create the TitleBar Slint component

**Files:**
- Create: `ui/title_bar.slint`

- [ ] **Step 1: Create `ui/title_bar.slint`**

### Task 2: Modify app.slint — wire custom title bar

**Files:**
- Modify: `ui/app.slint`

### Task 3: Update sidebar.slint — remove "Zeedle" title

**Files:**
- Modify: `ui/sidebar.slint`

### Task 4: Wire Rust callbacks for window operations

**Files:**
- Modify: `src/main.rs`

### Task 5: Build and verify

- [ ] **Step 1: Run `cargo build` and fix any compilation errors**
- [ ] **Step 2: Run `cargo run` to visually verify the title bar**
