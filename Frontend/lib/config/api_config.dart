/// API endpoints for live predictions.
abstract final class ApiConfig {
  /// Cardiogenic shock escalation API (MCS + VA-ECMO models).
  static const shockApiBaseUrl = 'http://127.0.0.1:8000';

  /// When true, loads seed JSON then refreshes MCS/ECMO from [shockApiBaseUrl]
  /// using full parquet-backed feature rows (/predict/cohort).
  static const useShockEscalationApi = true;

  /// Full patient list from aggregator backend (mortality + LOS on port 8001).
  static const useLiveApi = true;
  static const baseUrl = 'http://127.0.0.1:8001';
}
