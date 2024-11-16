import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

/**
 * Quartz 4.0 Configuration
 *
 * See https://quartz.jzhao.xyz/configuration for more information.
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "Eric W. Tramel",
    pageTitleSuffix: " ", 
    enableSPA: true,
    enablePopovers: true,
    analytics: {
      provider: "plausible",
    },
    locale: "en-US",
    baseUrl: "eric-tramel.github.io/eric-quartz-demo",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "created",
    generateSocialImages: false,
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: "Lora",
        body: "EB Garamond",
        code: "Fira Mono",
      },
      colors: {
        lightMode: {
          light: "#f4f1e0",             // Background with a parchment-like feel
          lightgray: "#d9d2c4",         // Soft borders reminiscent of aged paper
          gray: "#a09a8e",              // Graph links and heavier borders in a muted gray
          darkgray: "#4f4b45",          // Body text in a slightly warmer, ink-like dark gray
          dark: "#2f2c28",              // Header text and icons in a deeper brownish-gray
          secondary: "#5b4d3a",         // Link color in a subtle brown
          tertiary: "#927b61",          // Hover states in a warm, aged-paper accent color
          highlight: "rgba(144, 134, 121, 0.2)", // Soft highlight with a translucent parchment shade
          textHighlight: "#fff5a488",   // Highlighted text with a warm yellowish overlay
        },
        darkMode: {
          light: "#1e1e1c",               // Background with a deep, muted charcoal tone
          lightgray: "#3c3a36",           // Soft borders in a warm, dark taupe
          gray: "#646056",                // Graph links and heavier borders in a medium warm gray
          darkgray: "#bcb8ae",            // Body text in a softer, off-white for comfortable reading
          dark: "#e8e5de",                // Header text and icons in a light parchment-inspired tone
          secondary: "#9a8c7e",           // Link color in a warm, desaturated brown
          tertiary: "#7f7263",            // Hover states in a dark, sepia-toned accent color
          highlight: "rgba(102, 93, 82, 0.2)", // Soft highlight with a translucent warm gray overlay
          textHighlight: "#e3c17688",     // Highlighted text with a muted, warm gold overlay
        }
      },
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "filesystem"],
      }),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "catppuccin-latte",
          dark: "catppuccin-mocha",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents(),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({ renderEngine: "katex" }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.NotFoundPage(),
    ],
  },
}

export default config
