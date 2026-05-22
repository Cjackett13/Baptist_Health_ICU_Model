import 'dart:convert';

import 'package:http/http.dart' as http;

import '../config/api_config.dart';
import '../models/patient_prediction.dart';

/// Base URL for mortality endpoints (same aggregator as [ApiConfig.baseUrl]).
///
/// Override at build time:
/// `flutter run --dart-define=MORTALITY_API_BASE_URL=https://...`
String mortalityApiBaseUrl() => const String.fromEnvironment(
      'MORTALITY_API_BASE_URL',
      defaultValue: ApiConfig.baseUrl,
    );

class MortalityApiResponse {
  MortalityApiResponse({
    required this.hospitalMortality,
    required this.icuMortality,
    required this.inHospitalExpiry,
    required this.modelVersion,
    required this.source,
  });

  final double hospitalMortality;
  final double icuMortality;
  final double inHospitalExpiry;
  final String modelVersion;
  final String source;
}

/// POST `/v1/predict/mortality` with [PatientFeatures]. Returns null on failure.
Future<MortalityApiResponse?> fetchMortalityRemote(PatientFeatures features) async {
  final base = mortalityApiBaseUrl().replaceAll(RegExp(r'/$'), '');
  final uri = Uri.parse('$base/v1/predict/mortality');
  try {
    final res = await http
        .post(
          uri,
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode(features.toJson()),
        )
        .timeout(const Duration(seconds: 15));
    if (res.statusCode < 200 || res.statusCode >= 300) {
      return null;
    }
    final map = jsonDecode(res.body) as Map<String, dynamic>;
    return MortalityApiResponse(
      hospitalMortality: (map['hospital_mortality'] as num).toDouble(),
      icuMortality: (map['icu_mortality'] as num).toDouble(),
      inHospitalExpiry: (map['in_hospital_expiry'] as num).toDouble(),
      modelVersion: map['model_version'] as String? ?? 'unknown',
      source: map['source'] as String? ?? 'unknown',
    );
  } on Object {
    return null;
  }
}

/// GET `/v1/mortality/shap-bundle` — same JSON as `mortality_shap_bundle.json` from export.
Future<Map<String, dynamic>?> fetchMortalityShapBundleJson() async {
  final base = mortalityApiBaseUrl().replaceAll(RegExp(r'/$'), '');
  final uri = Uri.parse('$base/v1/mortality/shap-bundle');
  try {
    final res =
        await http.get(uri).timeout(const Duration(seconds: 20));
    if (res.statusCode < 200 || res.statusCode >= 300) {
      return null;
    }
    return jsonDecode(res.body) as Map<String, dynamic>;
  } on Object {
    return null;
  }
}

/// Parses `shap_mortality` from the bundle for [ShapValue.fromJson].
List<ShapValue> shapMortalityFromBundle(Map<String, dynamic> bundle) {
  final raw = bundle['shap_mortality'];
  if (raw is! List<dynamic>) {
    return [];
  }
  return raw
      .whereType<Map<String, dynamic>>()
      .map(ShapValue.fromJson)
      .toList();
}

/// GET `/v1/mortality/predictions` — rows from the export PKL as JSON.
Future<Map<String, dynamic>?> fetchMortalityPredictionsPage({
  int page = 0,
  int pageSize = 50,
}) async {
  final base = mortalityApiBaseUrl().replaceAll(RegExp(r'/$'), '');
  final uri = Uri.parse(
    '$base/v1/mortality/predictions?page=$page&page_size=$pageSize',
  );
  try {
    final res =
        await http.get(uri).timeout(const Duration(seconds: 30));
    if (res.statusCode < 200 || res.statusCode >= 300) {
      return null;
    }
    return jsonDecode(res.body) as Map<String, dynamic>;
  } on Object {
    return null;
  }
}
