const THEME_STORIES = [
  {
    id: "momo",
    index: "01",
    work: "NOVA GIFT",
    character: "Momo",
    title: "桃粉礼物",
    line: "个人授权被包装成一份可追溯的生日礼物；每一步启动都有下一步。",
    tags: ["成年向导", "礼物贴纸", "粉色工作台"]
  },
  {
    id: "elysia",
    index: "02",
    work: "NOVA SAKURA",
    character: "Sakura",
    title: "樱晶庭院",
    line: "把视觉链路收进温柔的粉白层次，保留工作台的清晰边界。",
    tags: ["樱色档案", "水晶花庭", "柔和校准"]
  },
  {
    id: "rem",
    index: "03",
    work: "NOVA AZURE",
    character: "Azure",
    title: "苍雪校准",
    line: "蓝色只服务稳定和状态，不把画面变成另一个系统。",
    tags: ["低温校准", "蓝白档案", "稳定优先"]
  },
  {
    id: "lusha",
    index: "04",
    work: "NOVA GILDED",
    character: "Gilded",
    title: "鎏金王庭",
    line: "金色只出现在收藏和档案层，运行控件保持克制。",
    tags: ["典藏档案", "暖金边界", "礼装感"]
  },
  {
    id: "tayama",
    index: "05",
    work: "NOVA CRIMSON",
    character: "Crimson",
    title: "绯夜模式",
    line: "夜间主题降低亮面干扰，把红色留给目标和危险动作。",
    tags: ["夜间工作台", "深红信号", "成熟漫画"]
  }
] as const;

export function ThemeGallery() {
  return (
    <aside className="theme-gallery" aria-label="当前角色主题画廊">
      <div className="theme-gallery-scene" aria-hidden="true">
        <span>NOVA THEME</span>
      </div>
      <div className="theme-gallery-stage" aria-hidden="true">
        <span className="theme-gallery-halo" />
        <span className="theme-gallery-spark theme-gallery-spark-a">✦</span>
        <span className="theme-gallery-spark theme-gallery-spark-b">✧</span>
      </div>
      <div className="theme-gallery-portrait" aria-hidden="true" />
      <div className="theme-gallery-stories">
        {THEME_STORIES.map((story) => (
          <section className="theme-gallery-story" data-theme-story={story.id} key={story.id}>
            <div className="theme-gallery-kicker">
              <span>{story.index}</span>
              <small>{story.work}</small>
            </div>
            <strong>{story.title}</strong>
            <h2>{story.character}</h2>
            <p>{story.line}</p>
            <div className="theme-gallery-tags">
              {story.tags.map((tag) => <span key={tag}>{tag}</span>)}
            </div>
          </section>
        ))}
      </div>
      <div className="theme-gallery-footer">
        <span>THEME COLLECTION</span>
        <b>NovaSight 自有角色主题档案</b>
      </div>
    </aside>
  );
}
