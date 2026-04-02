# AGENTS.md - VirtualBhoomi Studio UI Design & Implementation Guide

## Project Overview
VirtualBhoomi is a web-based SaaS platform for Indian real estate promoters and builders. It converts video walkthroughs or blueprint images into **walkable, customizable 3D virtual showflats** using World Labs Marble (Gaussian Splats).

The core experience is the **VirtualBhoomi Studio** — a clean, professional, Marble-like 3D editor where users can:
- View the generated 3D world
- Customize wall colors, add furniture, and place stickers
- Walk in first-person or use VR
- Switch between rendering engines (Gauzilla / WebSplatter)
- Publish and share branded links

## UI Design Philosophy (Very Important)
- **Style**: Modern, dark theme (like Marble World Labs), clean, minimal, professional, and trustworthy for real estate professionals.
- **Color Palette**: Dark background (#0a0a0a or #111111), accents in vibrant blue (#3b82f6) or teal (#14b8a6), white/gray text.
- **Feel**: Premium, fast, spatial. Avoid clutter. Make the 3D canvas as large as possible.
- **Target Users**: Builders/promoters (not gamers) → prioritize clarity, ease of use, and professional look.
- **Mobile Friendly**: Responsive. On mobile, sidebar collapses into bottom sheet or hamburger menu.

## Main Screen: VirtualBhoomi Studio Layout

***User stroies
    - User can search for property that already exist - Like explore
    - Upload images to create the 3d model
    - User can wait for the 3d model to generate
    - User see the list of their uploaded proeprties
    - By clicking on the project, user can view the 3d model in the web. It can have walk , vr style view, 
    - User can edit their property 3d model using Studio window page
    - It can share or invite others to collobrate like (team)
    - Have chat bot option to alter the furniture , wall color on the 3d rendered page and Property edit studio


**Overall Layout (Desktop First)**:

- **Top Navigation Bar** (fixed, dark):
  - Left: Logo "VirtualBhoomi Studio" + Project Name (e.g., "Demo - North Facing House")
  - Center: Status indicators (e.g., "Ready", "Generating...")
  - Right side (from left to right):
    - "Attach PLY" button (for manual .ply upload)
    - Engine Selector: Segmented control → **Gauzilla (Recommended)** | **WebSplatter**
    - "Publish" primary button (blue)
    - User avatar / profile

- **Left Sidebar** (narrow, ~280px, collapsible):
  - Tools / Categories (with icons):
    - 🎨 Colors (Wall painting with Asian Paints palette)
    - 🪑 Furniture (Catalog with drag & drop)
    - 🖼️ Stickers / Photos (Upload personal images)
  - Optional: Room list (Hall, Kitchen, Bedroom 1, etc.) for targeted editing

- **Main Area**: Full 3D Canvas (the WebSplatter / Gauzilla renderer)
  - Should take maximum available space
  - Show FPS counter in corner when enabled
  - Overlay controls:
    - First-person / Orbit toggle
    - VR button (WebXR)
    - Reset Camera / Auto Center button
    - Help tooltip

- **Right Sidebar** (optional, collapsible):
  - Properties panel (when object selected)
  - Current customizations list
  - Version history

- **Bottom Bar** (optional):
  - Quick actions: Undo, Redo, Save Version, Return to Original

## Key UI Screens to Generate

1. **Studio Main View** (most important)
   - Dark header with project name
   - Left tool sidebar (Colors, Furniture, Stickers)
   - Large central 3D viewer
   - Top-right engine selector + Publish button
   - Bottom-right mini controls (VR, Reset Camera, etc.)

2. **Generation / Upload Screen**
   - Clean upload area for Video or Blueprint
   - Progress indicator during World Labs generation
   - Rich prompt editor

3. **Customization Panels**
   - Colors panel: Grid of wall colors + eyedropper
   - Furniture panel: Scrollable grid of 3D models with search
   - Stickers panel: Upload + recent stickers

4. **Mobile View**
   - Collapsed left sidebar becomes bottom tab bar
   - 3D viewer takes full height

## Design Requirements for All Screens
- Use **Tailwind CSS** classes
- Dark modern UI with subtle glassmorphism / borders where needed
- Icons: Use Lucide React or Heroicons (consistent style)
- Buttons: Clear hierarchy — primary (blue), secondary (gray), ghost
- Typography: Inter or system sans-serif, clean headings
- Loading states: Smooth skeletons + progress bars
- Responsiveness: Mobile-first where possible, but desktop is primary for studio

## Current Implementation Notes
- 3D renderer is already integrated using WebSplatter (Rust + Wasm + WebGPU)
- Engine switching (Gauzilla ↔ WebSplatter) must be clearly visible in UI
- Camera auto-centering is implemented or in progress
- Sidebar currently has "Colors", "Furniture", "Stickers"

## Instructions for AI When Generating UI
When I ask you to generate or stitch a UI screen:
- Always use Tailwind CSS + shadcn/ui style components where possible
- Make the 3D canvas take maximum space
- Keep the layout very close to Marble.worldlabs.ai studio feel
- Include the renderer engine selector prominently
- Make it look premium and suitable for real estate promoters
- Provide both desktop and mobile variants if relevant
- Add helpful comments in the code

**Priority Right Now**: Generate clean, production-ready React + Tailwind components for the main Studio layout based on the screenshot and description above.

Let's create a beautiful, intuitive, and professional VirtualBhoomi Studio UI.