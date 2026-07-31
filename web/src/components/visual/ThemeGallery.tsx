import { THEME_STORIES } from "./themeRegistry";

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
