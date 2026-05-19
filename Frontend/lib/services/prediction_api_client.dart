import 'dart:convert';

import 'package:http/http.dart' as http;

import '../config/api_config.dart';
import '../models/patient_prediction.dart';

/// Fetches patients with live model predictions from FastAPI.
class PredictionApiClient {
  PredictionApiClient({http.Client? client}) : _client = client ?? http.Client();

  final http.Client _client;

  Future<List<PatientRecord>> fetchPatients() async {
    final uri = Uri.parse('${ApiConfig.baseUrl}/patients?live=true');
    final response = await _client.get(uri).timeout(const Duration(seconds: 30));

    if (response.statusCode != 200) {
      throw Exception(
        'API ${response.statusCode}: ${response.body.substring(0, response.body.length.clamp(0, 200))}',
      );
    }

    final decoded = jsonDecode(response.body) as Map<String, dynamic>;
    final list = decoded['patients'] as List<dynamic>;
    return list
        .map((e) => PatientRecord.fromSeedJson(e as Map<String, dynamic>))
        .toList();
  }

  Future<PatientRecord> fetchPatient(String encounterId) async {
    final uri = Uri.parse('${ApiConfig.baseUrl}/patients/$encounterId');
    final response = await _client.get(uri).timeout(const Duration(seconds: 15));

    if (response.statusCode != 200) {
      throw Exception('API ${response.statusCode}');
    }

    final decoded = jsonDecode(response.body) as Map<String, dynamic>;
    return PatientRecord.fromSeedJson(decoded);
  }

  void dispose() => _client.close();
}
