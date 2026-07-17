const THEME_STORIES = [
  {
    id: "elysia",
    index: "01",
    work: "崩坏三",
    character: "爱莉希雅",
    title: "樱晶花园",
    line: "愿每一次命中，都像飞花般绚烂。",
    tags: ["粉色妖精", "水晶花庭", "浪漫主义"]
  },
  {
    id: "rem",
    index: "02",
    work: "Re:Zero",
    character: "雷姆",
    title: "苍雪女仆",
    line: "蓝色月光之下，守望稳定而精确。",
    tags: ["鬼族女仆", "月下绣球", "蓝白幻想"]
  },
  {
    id: "lusha",
    index: "03",
    work: "死馆 2",
    character: "露莎公主",
    title: "鎏金王庭",
    line: "黄金照耀王庭，也照亮每一条轨迹。",
    tags: ["褐金王女", "白金宫殿", "华丽幻想"]
  },
  {
    id: "tayama",
    index: "04",
    work: "超市后烟二人",
    character: "田山小姐",
    title: "绯夜烟巷",
    line: "霓虹熄灭之前，保持一点从容。",
    tags: ["夜班休憩", "黑红霓虹", "成熟漫画"]
  }
] as const;

export function ThemeGallery() {
  return (
    <aside className="theme-gallery" aria-label="当前角色主题画廊">
      <div className="theme-gallery-scene" aria-hidden="true">
        <span>CHARACTER THEME</span>
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
        <b>NovaSight × 二次元主题档案</b>
      </div>
    </aside>
  );
}
