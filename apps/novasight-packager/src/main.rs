use std::ffi::OsStr;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, ExitCode};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};

const DEFAULT_OUTPUT: &str = "out/package/NovaSight";
const RUST_PACKAGES: [&str; 3] = ["novasight", "novasightctl", "novasightd"];

#[derive(Parser, Debug)]
#[command(
    name = "novasight-packager",
    about = "Build a NovaSight portable package"
)]
struct Args {
    /// Package build profile.
    #[arg(long, value_enum, default_value_t = PackageProfile::Debug)]
    profile: PackageProfile,

    /// Output package directory. It must stay below the workspace out/ tree.
    #[arg(long, default_value = DEFAULT_OUTPUT)]
    output: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum PackageProfile {
    /// Developer package with debug Rust binaries.
    Debug,
    /// Optimized package with release Rust binaries.
    Release,
}

#[derive(Clone, Debug)]
struct PackageLayout {
    root: PathBuf,
    bin: PathBuf,
    web: PathBuf,
    data: PathBuf,
    logs: PathBuf,
    run: PathBuf,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("NOVASIGHT_PACKAGE_FAILED: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    let workspace = find_workspace_root(&std::env::current_dir().context("read current dir")?)?;
    let output = resolve_output_path(&workspace, &args.output)?;
    build_artifacts(&workspace, args.profile)?;
    assemble_package(&workspace, args.profile, &output)?;
    validate_package(&output)?;
    println!("{}", output.display());
    Ok(())
}

fn find_workspace_root(start: &Path) -> Result<PathBuf> {
    let mut cursor = start.to_owned();
    loop {
        if cursor.join("Cargo.toml").is_file()
            && cursor.join("apps").is_dir()
            && cursor.join("web").is_dir()
        {
            return Ok(cursor);
        }
        if !cursor.pop() {
            bail!(
                "could not find NovaSight workspace root from {}",
                start.display()
            );
        }
    }
}

fn resolve_output_path(workspace: &Path, output: &Path) -> Result<PathBuf> {
    let output = if output.is_absolute() {
        output.to_owned()
    } else {
        workspace.join(output)
    };
    if output
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        bail!("package output cannot contain '..': {}", output.display());
    }
    let out_root = workspace.join("out");
    if !output.starts_with(&out_root) {
        bail!(
            "package output must stay below {}; got {}",
            out_root.display(),
            output.display()
        );
    }
    Ok(output)
}

fn build_artifacts(workspace: &Path, profile: PackageProfile) -> Result<()> {
    build_rust_artifacts(workspace, profile)?;
    run_command(
        workspace,
        "pnpm",
        ["--dir", "web", "build"],
        &[("CI", "true")],
    )
}

fn build_rust_artifacts(workspace: &Path, profile: PackageProfile) -> Result<()> {
    run_command(workspace, "cargo", rust_build_args(profile), &[])
}

fn rust_build_args(profile: PackageProfile) -> Vec<&'static str> {
    let mut args = vec!["build", "--locked"];
    for package in RUST_PACKAGES {
        args.push("-p");
        args.push(package);
    }
    if profile == PackageProfile::Release {
        args.push("--release");
    }
    args
}

fn run_command<I, S>(cwd: &Path, program: &str, args: I, envs: &[(&str, &str)]) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut command = Command::new(program);
    command.args(args).current_dir(cwd);
    for (key, value) in envs {
        command.env(key, value);
    }
    let status = command.status().with_context(|| format!("run {program}"))?;
    if !status.success() {
        bail!("{program} exited with {status}");
    }
    Ok(())
}

fn assemble_package(workspace: &Path, profile: PackageProfile, output: &Path) -> Result<()> {
    if output.join("run/ready.json").exists() {
        bail!(
            "{} appears to be running; stop NovaSight before rebuilding the package",
            output.display()
        );
    }
    if output.exists() {
        fs::remove_dir_all(output)
            .with_context(|| format!("remove previous package {}", output.display()))?;
    }

    let layout = PackageLayout::new(output);
    fs::create_dir_all(&layout.bin).with_context(|| format!("create {}", layout.bin.display()))?;
    fs::create_dir_all(&layout.web).with_context(|| format!("create {}", layout.web.display()))?;
    fs::create_dir_all(layout.data.join("models"))
        .with_context(|| format!("create {}", layout.data.join("models").display()))?;
    fs::create_dir_all(layout.data.join("runtime/deepstream")).with_context(|| {
        format!(
            "create {}",
            layout.data.join("runtime/deepstream").display()
        )
    })?;
    fs::create_dir_all(&layout.logs)
        .with_context(|| format!("create {}", layout.logs.display()))?;
    fs::create_dir_all(&layout.run).with_context(|| format!("create {}", layout.run.display()))?;

    let artifact_dir = workspace.join("out/cargo").join(profile.artifact_dir());
    copy_executable(
        &artifact_dir.join(executable_name("novasight")),
        &layout.root.join("NovaSight"),
    )?;
    copy_executable(
        &artifact_dir.join(executable_name("novasightd")),
        &layout.bin.join(executable_name("novasightd")),
    )?;
    copy_executable(
        &artifact_dir.join(executable_name("novasightctl")),
        &layout.bin.join(executable_name("novasightctl")),
    )?;
    copy_runtime_web_tree(&workspace.join("out/web"), &layout.web)?;
    copy_file(
        &workspace.join("deploy/novasight.production.yaml"),
        &layout.data.join("novasight.yaml"),
    )?;
    copy_file(
        &workspace.join("deploy/deepstream-tracker-iou.yml"),
        &layout
            .data
            .join("runtime/deepstream/deepstream-tracker-iou.yml"),
    )?;
    fs::write(
        layout.root.join("README-USER.txt"),
        "Run NovaSight from this directory. NovaSight-owned data, logs, and runtime files stay inside this folder.\n",
    )
    .with_context(|| format!("write {}", layout.root.join("README-USER.txt").display()))?;
    Ok(())
}

impl PackageProfile {
    const fn artifact_dir(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Release => "release",
        }
    }
}

impl PackageLayout {
    fn new(root: &Path) -> Self {
        Self {
            root: root.to_owned(),
            bin: root.join("bin"),
            web: root.join("web"),
            data: root.join("data"),
            logs: root.join("logs"),
            run: root.join("run"),
        }
    }
}

fn executable_name(name: &'static str) -> &'static str {
    if cfg!(windows) {
        match name {
            "novasight" => "novasight.exe",
            "novasightd" => "novasightd.exe",
            "novasightctl" => "novasightctl.exe",
            _ => name,
        }
    } else {
        name
    }
}

fn copy_executable(source: &Path, destination: &Path) -> Result<()> {
    copy_file(source, destination)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(destination, fs::Permissions::from_mode(0o755))
            .with_context(|| format!("set executable permissions on {}", destination.display()))?;
    }
    Ok(())
}

fn copy_file(source: &Path, destination: &Path) -> Result<()> {
    let parent = destination
        .parent()
        .ok_or_else(|| anyhow!("{} has no parent directory", destination.display()))?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    fs::copy(source, destination)
        .with_context(|| format!("copy {} to {}", source.display(), destination.display()))?;
    Ok(())
}

fn copy_runtime_web_tree(source: &Path, destination: &Path) -> Result<()> {
    copy_runtime_web_tree_from_root(source, source, destination)
}

fn copy_runtime_web_tree_from_root(
    web_root: &Path,
    source: &Path,
    destination: &Path,
) -> Result<()> {
    if !source.is_dir() {
        bail!("{} is not a directory", source.display());
    }
    fs::create_dir_all(destination).with_context(|| format!("create {}", destination.display()))?;
    for entry in fs::read_dir(source).with_context(|| format!("read {}", source.display()))? {
        let entry = entry.with_context(|| format!("read entry below {}", source.display()))?;
        let source_path = entry.path();
        if !should_package_runtime_web_path(web_root, &source_path) {
            continue;
        }
        let destination_path = destination.join(entry.file_name());
        let metadata = entry
            .metadata()
            .with_context(|| format!("inspect {}", source_path.display()))?;
        if metadata.is_dir() {
            copy_runtime_web_tree_from_root(web_root, &source_path, &destination_path)?;
        } else if metadata.is_file() {
            copy_file(&source_path, &destination_path)?;
        }
    }
    Ok(())
}

fn should_package_runtime_web_path(web_root: &Path, path: &Path) -> bool {
    let Ok(relative) = path.strip_prefix(web_root) else {
        return true;
    };
    let mut components = relative.components();
    let Some(first) = components.next() else {
        return true;
    };
    if first.as_os_str() == OsStr::new("landing") || first.as_os_str() == OsStr::new("landing.html")
    {
        return false;
    }
    if first.as_os_str() == OsStr::new("assets")
        && let Some(second) = components.next()
        && components.next().is_none()
    {
        let name = second.as_os_str().to_string_lossy();
        return !name.starts_with("landing-");
    }
    true
}

fn validate_package(output: &Path) -> Result<()> {
    for path in [
        output.join("NovaSight"),
        output.join("bin").join(executable_name("novasightd")),
        output.join("bin").join(executable_name("novasightctl")),
        output.join("web/index.html"),
        output.join("data/novasight.yaml"),
        output.join("logs"),
        output.join("run"),
    ] {
        if !path.exists() {
            bail!("packaged artifact is missing {}", path.display());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_selects_the_expected_artifact_directory() {
        assert_eq!(PackageProfile::Debug.artifact_dir(), "debug");
        assert_eq!(PackageProfile::Release.artifact_dir(), "release");
    }

    #[test]
    fn profiles_use_the_daemon_default_production_features() {
        assert_eq!(
            rust_build_args(PackageProfile::Debug),
            vec![
                "build",
                "--locked",
                "-p",
                "novasight",
                "-p",
                "novasightctl",
                "-p",
                "novasightd",
            ]
        );
        assert_eq!(
            rust_build_args(PackageProfile::Release),
            vec![
                "build",
                "--locked",
                "-p",
                "novasight",
                "-p",
                "novasightctl",
                "-p",
                "novasightd",
                "--release",
            ]
        );
    }

    #[test]
    fn output_must_stay_under_workspace_out() {
        let workspace = Path::new("/tmp/novasight");

        assert_eq!(
            resolve_output_path(workspace, Path::new("out/package/NovaSight")).unwrap(),
            PathBuf::from("/tmp/novasight/out/package/NovaSight")
        );
        assert!(resolve_output_path(workspace, Path::new("../NovaSight")).is_err());
        assert!(resolve_output_path(workspace, Path::new("/tmp/NovaSight")).is_err());
    }

    #[test]
    fn package_layout_is_stable() {
        let layout = PackageLayout::new(Path::new("/tmp/NovaSight"));

        assert_eq!(layout.bin, Path::new("/tmp/NovaSight/bin"));
        assert_eq!(layout.web, Path::new("/tmp/NovaSight/web"));
        assert_eq!(layout.data, Path::new("/tmp/NovaSight/data"));
        assert_eq!(layout.logs, Path::new("/tmp/NovaSight/logs"));
        assert_eq!(layout.run, Path::new("/tmp/NovaSight/run"));
    }

    #[test]
    fn runtime_web_package_excludes_landing_entrypoint() {
        let web_root = Path::new("/tmp/novasight/out/web");

        assert!(should_package_runtime_web_path(
            web_root,
            &web_root.join("index.html")
        ));
        assert!(should_package_runtime_web_path(
            web_root,
            &web_root.join("assets/main-abc123.js")
        ));
        assert!(!should_package_runtime_web_path(
            web_root,
            &web_root.join("landing.html")
        ));
        assert!(!should_package_runtime_web_path(
            web_root,
            &web_root.join("landing/assets/hero-bg.png")
        ));
        assert!(!should_package_runtime_web_path(
            web_root,
            &web_root.join("assets/landing-abc123.js")
        ));
    }
}
