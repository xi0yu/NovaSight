use std::ffi::OsStr;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, ExitCode};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};

const DEFAULT_OUTPUT: &str = "out/package/NovaSight";

#[derive(Parser, Debug)]
#[command(
    name = "novasight-packager",
    about = "Build a NovaSight portable package"
)]
struct Args {
    /// Package build profile.
    #[arg(long, value_enum, default_value_t = PackageProfile::Dev)]
    profile: PackageProfile,

    /// Output package directory. It must stay below the workspace out/ tree.
    #[arg(long, default_value = DEFAULT_OUTPUT)]
    output: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum PackageProfile {
    /// Developer package: debug Rust binaries, no DeepStream feature, same launcher path.
    Dev,
    /// Jetson package: release Rust binaries with the DeepStream feature enabled.
    JetsonRelease,
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
    build_rust_package(workspace, "novasight", profile, false)?;
    build_rust_package(workspace, "novasightctl", profile, false)?;
    build_rust_package(
        workspace,
        "novasightd",
        profile,
        profile == PackageProfile::JetsonRelease,
    )?;
    run_command(
        workspace,
        "pnpm",
        ["--dir", "web", "build"],
        &[("CI", "true")],
    )
}

fn build_rust_package(
    workspace: &Path,
    package: &str,
    profile: PackageProfile,
    deepstream: bool,
) -> Result<()> {
    let mut args = vec!["build", "--locked", "-p", package];
    if profile == PackageProfile::JetsonRelease {
        args.push("--release");
    }
    if deepstream {
        args.push("--features");
        args.push("deepstream");
    }
    run_command(workspace, "cargo", args, &[])
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
    copy_tree(&workspace.join("out/web"), &layout.web)?;
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
            Self::Dev => "debug",
            Self::JetsonRelease => "release",
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

fn copy_tree(source: &Path, destination: &Path) -> Result<()> {
    if !source.is_dir() {
        bail!("{} is not a directory", source.display());
    }
    fs::create_dir_all(destination).with_context(|| format!("create {}", destination.display()))?;
    for entry in fs::read_dir(source).with_context(|| format!("read {}", source.display()))? {
        let entry = entry.with_context(|| format!("read entry below {}", source.display()))?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let metadata = entry
            .metadata()
            .with_context(|| format!("inspect {}", source_path.display()))?;
        if metadata.is_dir() {
            copy_tree(&source_path, &destination_path)?;
        } else if metadata.is_file() {
            copy_file(&source_path, &destination_path)?;
        }
    }
    Ok(())
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
        assert_eq!(PackageProfile::Dev.artifact_dir(), "debug");
        assert_eq!(PackageProfile::JetsonRelease.artifact_dir(), "release");
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
}
