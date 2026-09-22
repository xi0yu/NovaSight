use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use novasight_store::model_catalog::{ModelCatalogError, ModelCatalogNode, SqliteModelCatalog};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "novasight-model-folders-{}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn empty_folders_are_visible_and_unsafe_paths_leave_the_filesystem_unchanged() {
    let scratch = Scratch::new();
    let model_root = scratch.0.join("data/models");
    let catalog =
        SqliteModelCatalog::open_with_model_root(scratch.0.join("registry.db"), &model_root)
            .unwrap();

    assert_eq!(catalog.create_catalog_directory("Arena").unwrap(), "Arena");
    assert_eq!(
        catalog.create_catalog_directory("Arena/第二组").unwrap(),
        "Arena/第二组"
    );
    let snapshot = catalog.catalog(false).unwrap();
    let arena = snapshot
        .root
        .children
        .iter()
        .find_map(|node| match node {
            ModelCatalogNode::Directory(dir) if dir.relative_path == "Arena" => Some(dir),
            _ => None,
        })
        .expect("new empty folder visible without a model file");
    assert!(arena.children.iter().any(|node| matches!(node,
        ModelCatalogNode::Directory(dir) if dir.relative_path == "Arena/第二组")));

    for invalid in [
        "../escape",
        "/tmp/escape",
        "Arena//x",
        "Arena/./x",
        "Arena/../x",
        "Arena\\x",
        "Arena/",
    ] {
        assert!(
            matches!(
                catalog.create_catalog_directory(invalid),
                Err(ModelCatalogError::InvalidCatalogDirectoryPath(_))
            ),
            "{invalid}"
        );
    }
    assert!(matches!(
        catalog.create_catalog_directory("Arena"),
        Err(ModelCatalogError::CatalogDirectoryExists(_))
    ));
    assert!(matches!(
        catalog.create_catalog_directory("missing/child"),
        Err(ModelCatalogError::CatalogDirectoryParentInvalid(_))
    ));
    assert!(!scratch.0.join("escape").exists());
    assert!(!model_root.join("missing").exists());

    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&scratch.0, model_root.join("alias")).unwrap();
        assert!(matches!(
            catalog.create_catalog_directory("alias/escape"),
            Err(ModelCatalogError::CatalogDirectoryParentInvalid(_))
        ));
        assert!(!scratch.0.join("escape").exists());
    }
}
