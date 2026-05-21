import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import '../config/api_config.dart';
import '../models/patient_prediction.dart';

/// Live MCS / VA-ECMO predictions from the cardiogenic shock escalation API.
class ShockEscalationApiClient {
  ShockEscalationApiClient({http.Client? client})
      : _client = client ?? http.Client();

  final http.Client _client;

  /// Replaces MCS/ECMO seed placeholders with POST /predict/cohort results.
  Future<List<PatientRecord>> enrichPatientsBatch(
    List<PatientRecord> patients,
  ) async {
    final cohortItems = <Map<String, dynamic>>[];
    final cohortIndices = <int>[];

    for (var i = 0; i < patients.length; i++) {
      final patient = patients[i];
      final encounterId = int.tryParse(patient.encounterId ?? '');
      if (encounterId == null) continue;
      cohortItems.add({
        'encounter_id': encounterId,
        'hour_from_admit': _hourFromAdmit(patient),
      });
      cohortIndices.add(i);
    }

    if (cohortItems.isEmpty) {
      throw ShockApiException('No valid encounter_id on patients for cohort predict');
    }

    final uri = Uri.parse('${ApiConfig.shockApiBaseUrl}/predict/cohort/batch');
    final response = await _client
        .post(
          uri,
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'patients': cohortItems}),
        )
        .timeout(const Duration(seconds: 120));

    if (response.statusCode != 200) {
      throw ShockApiException(
        'batch ${response.statusCode}: ${response.body}',
      );
    }

    final decoded = jsonDecode(response.body) as Map<String, dynamic>;
    final predictions = decoded['predictions'] as List<dynamic>;
    if (predictions.length != cohortIndices.length) {
      throw ShockApiException(
        'batch size mismatch: sent ${cohortIndices.length}, got ${predictions.length}',
      );
    }

    final updated = List<PatientRecord>.from(patients);
    for (var j = 0; j < predictions.length; j++) {
      final item = predictions[j] as Map<String, dynamic>;
      final idx = cohortIndices[j];
      final preds = _mergeResults(
        updated[idx].predictions,
        item['results'] as List<dynamic>,
      );
      updated[idx] = updated[idx].copyWithPredictions(preds);
    }

    debugPrint(
      'Shock API: updated MCS/ECMO for ${predictions.length} patients',
    );
    return updated;
  }

  int _hourFromAdmit(PatientRecord patient) {
    if (patient.hourFromAdmit != null && patient.hourFromAdmit! >= 4) {
      return patient.hourFromAdmit!.clamp(4, 240);
    }
    final days = patient.daysAdmitted ?? 1;
    return (days * 24).clamp(4, 240);
  }

  PatientPredictions _mergeResults(
    PatientPredictions base,
    List<dynamic> results,
  ) {
    var mcsProb = base.mcs12hProbability;
    var mcsNeeded = base.mcs12hNeeded;
    var ecmoProb = base.vaEcmo12hProbability;
    var ecmoNeeded = base.vaEcmo12hNeeded;
    var mcsReasons = base.shapMcs12h;
    var ecmoReasons = base.shapVaEcmo12h;

    for (final raw in results) {
      final r = raw as Map<String, dynamic>;
      final model = (r['model'] as String).toLowerCase();
      final prob = (r['probability'] as num).toDouble();
      final alert = r['alert'] as bool;
      final reasons = ShapValue.topByShapMagnitude(
        (r['reasons'] as List<dynamic>? ?? [])
            .map((e) => ShapValue.fromJson(e as Map<String, dynamic>))
            .toList(),
        k: 5,
      );

      if (model.contains('mcs')) {
        mcsProb = prob;
        mcsNeeded = alert;
        if (reasons.isNotEmpty) mcsReasons = reasons;
      } else if (model.contains('ecmo')) {
        ecmoProb = prob;
        ecmoNeeded = alert;
        if (reasons.isNotEmpty) ecmoReasons = reasons;
      }
    }

    return base.copyWithMcsEcmo(
      mcs12hProbability: mcsProb,
      mcs12hNeeded: mcsNeeded,
      vaEcmo12hProbability: ecmoProb,
      vaEcmo12hNeeded: ecmoNeeded,
      shapMcs12h: mcsReasons,
      shapVaEcmo12h: ecmoReasons,
      lastUpdated: DateTime.now(),
    );
  }

  void dispose() => _client.close();
}

class ShockApiException implements Exception {
  ShockApiException(this.message);
  final String message;
  @override
  String toString() => message;
}
