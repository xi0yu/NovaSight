const CONTROL_ROUTE = "/";

type LauncherEntry = {
  title: string;
  detail: string;
  href: string;
};

const ENTRIES: LauncherEntry[] = [
  {
    title: "采集工作台",
    detail: "设备、格式、ROI 与预览",
    href: "/?page=capture"
  },
  {
    title: "模型推理",
    detail: "当前模型、推理参数与 ROI 输入",
    href: "/?page=infer"
  },
  {
    title: "控制链路",
    detail: "目标、预测、控制和输出状态",
    href: "/?page=control"
  },
  {
    title: "参数配置",
    detail: "输出门、预测开关和控制参数",
    href: "/?page=params"
  },
  {
    title: "设备诊断",
    detail: "kmNet 连接、按键和单步移动",
    href: "/?page=control-test"
  },
  {
    title: "延迟分析",
    detail: "推理耗时、结果帧龄和统计窗口",
    href: "/?page=latency"
  }
];

const tpl = (strings: TemplateStringsArray, ...values: Array<string | number>): string =>
  strings.reduce((acc, str, i) => acc + str + (i < values.length ? String(values[i]) : ""), "");

function renderTopbar(): string {
  return tpl`
    <header class="topbar">
      <a class="brand" href="${CONTROL_ROUTE}" aria-label="进入 NovaSight Studio">
        <img src="/landing/assets/logo.svg" alt="NovaSight" />
      </a>
      <a class="primary-cta" href="${CONTROL_ROUTE}">进入 Studio</a>
    </header>
  `;
}

function renderHero(): string {
  return tpl`
    <section class="launcher-hero" aria-labelledby="launcher-title">
      <div class="launcher-copy">
        <span class="eyebrow">NOVASIGHT STUDIO</span>
        <h1 id="launcher-title">实时视觉控制台</h1>
        <p>
          打开 Studio 后按当前后端状态进入采集、推理、控制和参数配置。后端未连接时，Studio 会显示连接和授权恢复入口。
        </p>
      </div>
      <div class="launcher-actions">
        <a class="primary-cta lg" href="${CONTROL_ROUTE}">
          进入 Studio
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h12M13 6l6 6-6 6"/></svg>
        </a>
        <a class="secondary-cta" href="/?visual-system=1">查看视觉系统</a>
      </div>
    </section>
  `;
}

function renderEntries(): string {
  return tpl`
    <section class="entry-section" aria-labelledby="entry-title">
      <div class="section-heading">
        <h2 id="entry-title">常用入口</h2>
        <p>这些入口都进入同一个 Studio，只是默认打开不同工作页。</p>
      </div>
      <div class="entry-grid">
        ${ENTRIES.map((entry) => tpl`
          <a class="entry-tile" href="${entry.href}">
            <span>${entry.title}</span>
            <small>${entry.detail}</small>
          </a>
        `).join("")}
      </div>
    </section>
  `;
}

function renderRuntimeNote(): string {
  return tpl`
    <section class="runtime-note" aria-label="运行入口说明">
      <strong>用户启动方式</strong>
      <span>使用 NovaSight 启动器打开 Web UI；开发者和用户都通过同一套入口访问 Studio。</span>
    </section>
  `;
}

export function mountLanding(root: HTMLElement): void {
  root.innerHTML = tpl`
    <div class="site-shell">
      <div class="bg-layer" aria-hidden="true">
        <img src="/landing/assets/hero-bg.png" alt="" class="bg-image" />
        <div class="bg-vignette"></div>
      </div>

      ${renderTopbar()}

      <main>
        ${renderHero()}
        ${renderEntries()}
        ${renderRuntimeNote()}
      </main>
    </div>
  `;
}
