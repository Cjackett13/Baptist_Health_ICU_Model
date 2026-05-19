import 'dart:convert';

import 'package:flutter/services.dart';

import '../config/api_config.dart';
import '../models/patient_prediction.dart';
import 'prediction_api_client.dart';

/// Loads patients from FastAPI (live models) or bundled seed JSON (offline demo).
class PatientRepository {
  PatientRepository._();
  static final PatientRepository instance = PatientRepository._();

  List<PatientRecord>? _cache;
  final PredictionApiClient _api = PredictionApiClient();

  Future<List<PatientRecord>> loadPatients({bool forceRefresh = false}) async {
    if (!forceRefresh && _cache != null) return _cache!;

    if (ApiConfig.useLiveApi) {
      _cache = await _api.fetchPatients();
      return _cache!;
    }

    final raw = await rootBundle.loadString('assets/patients_seed.json');
    final decoded = jsonDecode(raw) as Map<String, dynamic>;
    final list = decoded['patients'] as List<dynamic>;
    _cache = list
        .map((e) => PatientRecord.fromSeedJson(e as Map<String, dynamic>))
        .toList();
    return _cache!;
  }

  void clearCache() => _cache = null;
}
