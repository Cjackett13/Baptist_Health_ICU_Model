/// Toggle live API vs bundled seed JSON.
abstract final class ApiConfig {
  /// Set true when FastAPI is running (see docs/MODEL_INTEGRATION.md).
  static const useLiveApi = false;

  /// Chrome / Windows desktop: localhost
  /// Android emulator: 10.0.2.2
  /// Physical device on same Wi‑Fi: your PC's LAN IP, e.g. 192.168.1.10
  static const baseUrl = 'http://localhost:8000';
}
