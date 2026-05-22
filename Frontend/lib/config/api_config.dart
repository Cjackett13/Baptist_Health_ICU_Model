/// API endpoints for live predictions.
abstract final class ApiConfig {
  /// Cardiogenic shock escalation API (MCS + VA-ECMO models).
  static const shockApiBaseUrl = 'http://127.0.0.1:8000';

  /// When true, loads seed JSON then refreshes MCS/ECMO from [shockApiBaseUrl]
  /// using full parquet-backed feature rows (/predict/cohort).
  static const useShockEscalationApi = true;

  /// Full patient list from aggregator backend (optional, separate port).
  static const useLiveApi = false;
  static const baseUrl = 'http://localhost:8001';
}
