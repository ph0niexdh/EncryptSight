use tauri_plugin_shell::ShellExt;

fn main() {
  tauri::Builder::default()
    .plugin(tauri_plugin_shell::init())
    .setup(|app| {
      // The sidecar is a PyInstaller build of backend/sidecar.py. The backend
      // itself falls back to SQLite when packaged Postgres is unavailable.
      let _child = app.shell().sidecar("encryptsight-backend")?.args(["--port", "8000"]).spawn()?;
      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("failed to launch EncryptSight desktop");
}
