// NovaSight 营销入口页 (简化版)
// 8 个 FPS 场景入口 + 首访/回访按钮状态 + 简化的热启动动效 → 跳转 /
// 不触碰控制台 React 工程；通过 localStorage 记忆访问状态

const VISITED_KEY = "novasight_visited";
const CONTROL_ROUTE = "/"; // 控制台 (React) 入口

type GameSlide = {
  type: "Hero 图" | "AI Vision 图";
  src: string;
  caption: string;
};

type Game = {
  title: string;
  subtitle: string;
  desc: string;
  slides: GameSlide[];
};

const GAMES: Record<string, Game> = {
  "delta-force": {
    title: "Delta Force",
    subtitle: "三角洲行动",
    desc: "NovaSight × Delta Force 的 Hero 宣传图与 AI Vision 实时分析图。",
    slides: [
      { type: "Hero 图", src: "/landing/assets/games/delta-force-hero.png", caption: "Delta Force Hero：以战术基地为背景，融合 NovaSight 品牌与 HUD 宣传语言。" },
      { type: "AI Vision 图", src: "/landing/assets/games/delta-force-ai-vision.png", caption: "Delta Force AI Vision：展示实时场景理解、多目标识别、环境分析与信息增强。" }
    ]
  },
  "crossfire": {
    title: "CrossFire",
    subtitle: "穿越火线",
    desc: "NovaSight × CrossFire 的 Hero 宣传图与 AI Vision 实时分析图。",
    slides: [
      { type: "Hero 图", src: "/landing/assets/games/crossfire-hero.png", caption: "CrossFire Hero：以高强度都市交战氛围塑造品牌主视觉。" },
      { type: "AI Vision 图", src: "/landing/assets/games/crossfire-ai-vision.png", caption: "CrossFire AI Vision：展示场景洞察、路线标注、热区分析与实时数据叠加。" }
    ]
  },
  "valorant": {
    title: "VALORANT",
    subtitle: "无畏契约",
    desc: "NovaSight × VALORANT 的 Hero 宣传图与 AI Vision 实时分析图。",
    slides: [
      { type: "Hero 图", src: "/landing/assets/games/valorant-hero.png", caption: "VALORANT Hero：在近未来竞技地图中融合 NovaSight 的品牌宣传表达。" },
      { type: "AI Vision 图", src: "/landing/assets/games/valorant-ai-vision.png", caption: "VALORANT AI Vision：展示多目标识别、战术区域理解与信息增强。" }
    ]
  },
  "cs2": {
    title: "Counter-Strike 2",
    subtitle: "CS2 / 反恐精英 2",
    desc: "NovaSight × Counter-Strike 2 的 Hero 宣传图与 AI Vision 实时分析图。",
    slides: [
      { type: "Hero 图", src: "/landing/assets/games/cs2-hero.png", caption: "CS2 Hero：突出高压战术环境中的清晰感、品牌感与视觉冲击。" },
      { type: "AI Vision 图", src: "/landing/assets/games/cs2-ai-vision.png", caption: "CS2 AI Vision：展示烟雾分析、区域控制、目标识别与信息增强。" }
    ]
  },
  "pubg": {
    title: "PUBG: Battlegrounds",
    subtitle: "绝地求生",
    desc: "NovaSight × PUBG 的 Hero 宣传图与 AI Vision 实时分析图。",
    slides: [
      { type: "Hero 图", src: "/landing/assets/games/pubg-hero.png", caption: "PUBG Hero：以开阔战场、空投与载具场景作为品牌主视觉。" },
      { type: "AI Vision 图", src: "/landing/assets/games/pubg-ai-vision.png", caption: "PUBG AI Vision：展示广域场景理解、威胁方向、掩体分析与实时环境信息。" }
    ]
  },
  "arena-breakout-infinite": {
    title: "Arena Breakout: Infinite",
    subtitle: "暗区突围：无限",
    desc: "NovaSight × Arena Breakout: Infinite 的 Hero 宣传图与 AI Vision 实时分析图。",
    slides: [
      { type: "Hero 图", src: "/landing/assets/games/arena-breakout-infinite-hero.png", caption: "Arena Breakout: Infinite Hero：强调工业废墟、撤离氛围与高风险高回报。" },
      { type: "AI Vision 图", src: "/landing/assets/games/arena-breakout-infinite-ai-vision.png", caption: "Arena Breakout: Infinite AI Vision：展示兴趣点、敌方小队、撤离点与空间路线信息。" }
    ]
  },
  "overwatch-2": {
    title: "Overwatch 2",
    subtitle: "守望先锋 2",
    desc: "NovaSight × Overwatch 2 的 Hero 宣传图与 AI Vision 实时分析图。",
    slides: [
      { type: "Hero 图", src: "/landing/assets/games/overwatch-2-hero.png", caption: "Overwatch 2 Hero：强调明亮未来风、协同氛围与品牌主视觉。" },
      { type: "AI Vision 图", src: "/landing/assets/games/overwatch-2-ai-vision.png", caption: "Overwatch 2 AI Vision：展示目标推进、队伍结构、路线流与实时事件信息。" }
    ]
  },
  "csol": {
    title: "Counter-Strike Online",
    subtitle: "反恐精英 Online",
    desc: "NovaSight × Counter-Strike Online 的 Hero 宣传图与 AI Vision 实时分析图。",
    slides: [
      { type: "Hero 图", src: "/landing/assets/games/csol-hero.png", caption: "CSOL Hero：保留经典线上 FPS 氛围，同时以 NovaSight 重塑品牌主视觉。" },
      { type: "AI Vision 图", src: "/landing/assets/games/csol-ai-vision.png", caption: "CSOL AI Vision：展示路线分析、威胁识别、环境理解与实时信息增强。" }
    ]
  }
};

const hasVisited = (): boolean => localStorage.getItem(VISITED_KEY) === "1";

const escapeAttr = (s: string): string =>
  s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");

const tpl = (strings: TemplateStringsArray, ...values: Array<string | number>): string =>
  strings.reduce((acc, str, i) => acc + str + (i < values.length ? String(values[i]) : ""), "");

function renderTopbar(visited: boolean): string {
  return tpl`
    <header class="topbar" data-reveal>
      <a class="brand" href="#top" aria-label="NovaSight 首页">
        <img src="/landing/assets/logo.svg" alt="NovaSight" />
      </a>
      <div class="topbar-right">
        <button class="reset-link" id="resetVisitBtn" type="button" title="清除首次访问标记">重置</button>
        <button class="primary-cta" id="topLaunchBtn" type="button">
          <span id="topLaunchText">${visited ? "进入控制台" : "启动 NovaSight"}</span>
        </button>
      </div>
    </header>
  `;
}

function renderHero(visited: boolean): string {
  return tpl`
    <section class="hero" id="top">
      <h1 data-reveal>面向游戏视觉体验的<br><span>实时 AI 视觉工作台</span></h1>
      <p class="hero-lead" data-reveal>
        8 个 FPS 场景的统一启动入口。点击下方任意游戏可查看 Hero 图与 AI Vision 分析图，点击启动按钮进入控制台。
      </p>
      <div class="hero-actions" data-reveal>
        <button class="primary-cta lg" id="launchBtn" type="button">
          <span id="launchBtnText">${visited ? "直接进入控制台" : "一键启动"}</span>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h12M13 6l6 6-6 6"/></svg>
        </button>
      </div>
    </section>
  `;
}

function renderGameGrid(): string {
  const tiles = Object.entries(GAMES).map(
    ([key, game]) => tpl`
      <button class="game-tile" type="button" data-game="${escapeAttr(key)}">
        <span class="tile-title">${escapeAttr(game.title)}</span>
        <span class="tile-sub">${escapeAttr(game.subtitle)}</span>
      </button>
    `
  ).join("");
  return tpl`
    <section class="game-section" id="games">
      <div class="section-heading" data-reveal>
        <h2>选择 FPS 场景</h2>
      </div>
      <div class="game-grid">${tiles}</div>
    </section>
  `;
}

function renderGalleryModal(): string {
  return tpl`
    <div class="gallery-modal" id="galleryModal" hidden aria-hidden="true">
      <div class="modal-backdrop" data-close-modal></div>
      <div class="modal-panel" role="dialog" aria-modal="true" aria-labelledby="modalTitle">
        <button class="modal-close" type="button" aria-label="关闭" data-close-modal>×</button>
        <div class="modal-head">
          <div>
            <h2 id="modalTitle">Game Title</h2>
            <p id="modalDesc">游戏宣传图展示</p>
          </div>
          <div class="modal-tabs">
            <button type="button" class="modal-tab is-active" data-slide="0">Hero</button>
            <button type="button" class="modal-tab" data-slide="1">AI Vision</button>
          </div>
        </div>
        <div class="modal-stage">
          <img id="modalImage" src="" alt="" />
        </div>
        <p class="modal-caption" id="modalCaption">当前为 Hero 图</p>
      </div>
    </div>
  `;
}

function renderLaunchOverlay(): string {
  return tpl`
    <div class="launch-overlay" id="launchOverlay" hidden aria-hidden="true">
      <div class="launch-overlay-backdrop"></div>
      <div class="launch-dialog" role="dialog" aria-modal="true" aria-labelledby="launchDialogTitle">
        <h2 id="launchDialogTitle">NovaSight 启动中</h2>
        <div class="power-gauge" id="powerStage" style="--progress:0;">
          <svg viewBox="0 0 120 120" class="power-ring" aria-hidden="true">
            <circle class="ring-bg" cx="60" cy="60" r="54" />
            <circle class="ring-fg" cx="60" cy="60" r="54" />
          </svg>
          <div class="power-value"><strong id="powerValue">0</strong><span>%</span></div>
        </div>
        <p class="launch-status" id="launchStatusText">正在启动核心模块…</p>
        <button type="button" class="primary-cta" id="closeLaunchOverlay" disabled>启动中</button>
      </div>
    </div>
  `;
}

function renderToast(): string {
  return tpl`
    <div class="toast" id="toast" aria-live="polite" aria-hidden="true">
      <span id="toastMessage">已重置首次访问状态</span>
    </div>
  `;
}

function setupReveal(root: HTMLElement): void {
  const items = Array.from(root.querySelectorAll<HTMLElement>("[data-reveal]"));
  if (items.length === 0) return;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    items.forEach((item) => item.classList.add("is-visible"));
    return;
  }
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.08, rootMargin: "0px 0px -8% 0px" }
  );
  items.forEach((item) => observer.observe(item));
}

function setupGallery(root: HTMLElement): void {
  const modal = root.querySelector<HTMLElement>("#galleryModal");
  const modalImage = root.querySelector<HTMLImageElement>("#modalImage");
  const modalTitle = root.querySelector<HTMLElement>("#modalTitle");
  const modalDesc = root.querySelector<HTMLElement>("#modalDesc");
  const modalCaption = root.querySelector<HTMLElement>("#modalCaption");
  const modalTabs = Array.from(root.querySelectorAll<HTMLElement>(".modal-tab"));
  const gameButtons = Array.from(root.querySelectorAll<HTMLElement>(".game-tile"));

  if (!modal || !modalImage || !modalTitle || !modalDesc || !modalCaption) return;

  let currentGame = "delta-force";
  let currentSlide = 0;

  const updateSlide = (): void => {
    const game = GAMES[currentGame];
    const slide = game.slides[currentSlide];
    modalImage.src = slide.src;
    modalImage.alt = `${game.title} ${slide.type}`;
    modalCaption.textContent = `${slide.type} · ${slide.caption}`;
    modalTabs.forEach((tab, idx) => tab.classList.toggle("is-active", idx === currentSlide));
  };

  const openModal = (gameKey: string, slideIndex = 0): void => {
    const game = GAMES[gameKey];
    if (!game) return;
    currentGame = gameKey;
    currentSlide = slideIndex;
    modalTitle.textContent = game.title;
    modalDesc.textContent = game.subtitle;
    updateSlide();
    modal.hidden = false;
    modal.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
  };

  const closeModal = (): void => {
    modal.hidden = true;
    modal.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
  };

  gameButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.game;
      if (key) openModal(key);
    });
  });

  root.querySelectorAll<HTMLElement>("[data-close-modal]").forEach((el) =>
    el.addEventListener("click", closeModal)
  );
  modalTabs.forEach((tab) =>
    tab.addEventListener("click", () => {
      currentSlide = Number(tab.dataset.slide ?? 0);
      updateSlide();
    })
  );

  document.addEventListener("keydown", (event) => {
    if (modal.hidden) return;
    if (event.key === "Escape") closeModal();
  });

  // Preload all slides
  Object.values(GAMES).forEach((game) => {
    game.slides.forEach((slide) => {
      const img = new Image();
      img.src = slide.src;
    });
  });
}

function setupLaunch(root: HTMLElement): void {
  const overlay = root.querySelector<HTMLElement>("#launchOverlay");
  const topLaunchBtn = root.querySelector<HTMLElement>("#topLaunchBtn");
  const launchBtn = root.querySelector<HTMLElement>("#launchBtn");
  const powerStage = root.querySelector<HTMLElement>("#powerStage");
  const powerValue = root.querySelector<HTMLElement>("#powerValue");
  const launchStatusText = root.querySelector<HTMLElement>("#launchStatusText");
  const closeLaunchOverlay = root.querySelector<HTMLButtonElement>("#closeLaunchOverlay");

  if (!overlay || !powerStage || !powerValue || !launchStatusText || !closeLaunchOverlay) return;

  let running = false;

  const showOverlay = (): void => {
    overlay.hidden = false;
    overlay.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
  };

  const hideOverlay = (): void => {
    overlay.hidden = true;
    overlay.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
  };

  const renderProgress = (progress: number): void => {
    const value = Math.round(progress);
    powerValue.textContent = String(value);
    powerStage.style.setProperty("--progress", String(value));
    if (value < 35) {
      launchStatusText.textContent = "正在加载核心模块…";
    } else if (value < 75) {
      launchStatusText.textContent = "正在初始化视觉分析管线…";
    } else if (value < 100) {
      launchStatusText.textContent = "即将进入控制台…";
    } else {
      launchStatusText.textContent = "启动完成";
    }
  };

  const runSequence = (): void => {
    if (running) return;
    running = true;
    showOverlay();
    closeLaunchOverlay.disabled = true;
    closeLaunchOverlay.textContent = "启动中";
    renderProgress(0);

    const duration = 2400;
    const start = performance.now();

    const tick = (now: number): void => {
      const elapsed = now - start;
      const eased = Math.min(1, elapsed / duration);
      const progress = 100 * (1 - Math.pow(1 - eased, 1.6));
      renderProgress(progress);

      if (eased < 1) {
        requestAnimationFrame(tick);
      } else {
        renderProgress(100);
        closeLaunchOverlay.disabled = false;
        closeLaunchOverlay.textContent = "进入控制台";
        running = false;
      }
    };

    requestAnimationFrame(tick);
  };

  const enterConsole = (): void => {
    localStorage.setItem(VISITED_KEY, "1");
    window.location.href = CONTROL_ROUTE;
  };

  topLaunchBtn?.addEventListener("click", runSequence);
  launchBtn?.addEventListener("click", runSequence);
  closeLaunchOverlay.addEventListener("click", () => {
    if (running) return;
    hideOverlay();
    enterConsole();
  });

  overlay.querySelector<HTMLElement>(".launch-overlay-backdrop")?.addEventListener("click", () => {
    if (!running) hideOverlay();
  });
}

function setupReset(root: HTMLElement): void {
  const btn = root.querySelector<HTMLButtonElement>("#resetVisitBtn");
  const toast = root.querySelector<HTMLElement>("#toast");
  const toastMessage = root.querySelector<HTMLElement>("#toastMessage");
  if (!btn || !toast || !toastMessage) return;
  let timer: number | undefined;
  const showToast = (msg: string): void => {
    toastMessage.textContent = msg;
    toast.setAttribute("aria-hidden", "false");
    if (timer) window.clearTimeout(timer);
    timer = window.setTimeout(() => toast.setAttribute("aria-hidden", "true"), 1800);
  };
  btn.addEventListener("click", () => {
    localStorage.removeItem(VISITED_KEY);
    showToast("已重置首次访问状态，刷新页面以查看效果");
  });
}

export function mountLanding(root: HTMLElement): void {
  const visited = hasVisited();
  root.innerHTML = tpl`
    <div class="site-shell">
      <div class="bg-layer" aria-hidden="true">
        <img src="/landing/assets/hero-bg.png" alt="" class="bg-image" />
        <div class="bg-vignette"></div>
      </div>

      ${renderTopbar(visited)}

      <main>
        ${renderHero(visited)}
        ${renderGameGrid()}
      </main>
    </div>

    ${renderGalleryModal()}
    ${renderLaunchOverlay()}
    ${renderToast()}
  `;

  setupReveal(root);
  setupGallery(root);
  setupLaunch(root);
  setupReset(root);
}
