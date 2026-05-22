import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import '../config/api_config.dart';
import '../models/patient_prediction.dart';
import 'prediction_api_client.dart';
import 'shock_escalation_api_client.dart';

/// Loads patients from seed JSON and optionally enriches MCS/ECMO via shock API.
class PatientRepository extends ChangeNotifier {
  PatientRepository._();
  static final PatientRepository instance = PatientRepository._();

  List<PatientRecord>? _cache;
  final PredictionApiClient _aggregatorApi = PredictionApiClient();
  final ShockEscalationApiClient _shockApi = ShockEscalationApiClient();

  /// Set when seed loads but live MCS/ECMO refresh failed (API down).
  String? lastShockApiWarning;

  Future<List<PatientRecord>> loadPatients({bool forceRefresh = false}) async {
    if (!forceRefresh && _cache != null) return _cache!;

    List<PatientRecord> patients;

    if (ApiConfig.useLiveApi) {
      patients = await _aggregatorApi.fetchPatients();
    } else {
      final raw = await rootBundle.loadString('assets/patients_seed.json');
      final decoded = jsonDecode(raw) as Map<String, dynamic>;
      final list = decoded['patients'] as List<dynamic>;
      patients = list
          .map((e) => PatientRecord.fromSeedJson(e as Map<String, dynamic>))
          .toList();
    }

    lastShockApiWarning = null;
    if (ApiConfig.useShockEscalationApi) {
      try {
        patients = await _shockApi.enrichPatientsBatch(patients);
      } catch (e) {
        lastShockApiWarning =
            'Using saved MCS/ECMO scores — live API unavailable at '
            '${ApiConfig.shockApiBaseUrl}. Start api_server.py to refresh.';
        debugPrint('Shock API enrich failed, using seed: $e');
      }
    }

    _cache = patients;
    notifyListeners();
    return _cache!;
  }

  /// Latest chart for one patient (includes clinician edits this session).
  PatientRecord? patientById(String id) {
    final list = _cache;
    if (list == null) return null;
    for (final p in list) {
      if (p.id == id) return p;
    }
    return null;
  }

  /// Saves recommendations for patient and family views (in-memory for demo).
  void updateRecommendations(
    String patientId,
    List<HomeCareSuggestion> recommendations,
  ) {
    final list = _cache;
    if (list == null) return;
    final index = list.indexWhere((p) => p.id == patientId);
    if (index < 0) return;
    list[index] = list[index].copyWithRecommendations(
      List<HomeCareSuggestion>.from(recommendations),
    );
    notifyListeners();
  }

  void clearCache() {
    _cache = null;
    notifyListeners();
  }
}
