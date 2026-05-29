import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import '../config/api_config.dart';
import '../features/home/demo_hospitals.dart';
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
    } else if (ApiConfig.useDashboardJson) {
      patients = await _loadFromDashboardJson();
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

  Future<List<PatientRecord>> _loadFromDashboardJson() async {
    final raw = await rootBundle.loadString('assets/dashboard_results.json');
    final decoded = jsonDecode(raw) as Map<String, dynamic>;

    final patientList = decoded['patients'] as List<dynamic>;
    final models = decoded['models'] as Map<String, dynamic>;

    Map<int, Map<String, dynamic>> indexByEncounter(String key) {
      final preds = (models[key]?['predictions'] as List<dynamic>?) ?? [];
      return {
        for (final p in preds)
          (p as Map<String, dynamic>)['encounter_id'] as int: p,
      };
    }

    final mortByEnc = indexByEncounter('mortality');
    final losByEnc  = indexByEncounter('los');
    final scaiByEnc = indexByEncounter('scai');
    final vasoByEnc = indexByEncounter('vasopressor');
    final mcsByEnc  = indexByEncounter('mcs');
    final ecmoByEnc = indexByEncounter('ecmo');

    List<ShapValue> parseShap(dynamic raw) {
      if (raw is! List) return const [];
      return ShapValue.topByShapMagnitude(
        raw.map((e) => ShapValue.fromJson(e as Map<String, dynamic>)).toList(),
        k: 5,
      );
    }

    final List<PatientRecord> allPatients = [];

    for (final item in patientList) {
      final p = item as Map<String, dynamic>;
      final encId = p['encounter_id'] as int;

      final mort = mortByEnc[encId];
      if (mort == null) continue;

      final los  = losByEnc[encId];
      final scai = scaiByEnc[encId];
      final vaso = vasoByEnc[encId];
      final mcs  = mcsByEnc[encId];
      final ecmo = ecmoByEnc[encId];

      final mortalityProb = (mort['prediction'] as num).toDouble();
      final losDays       = los  != null ? (los['prediction']  as num).toDouble() : 3.0;
      final scaiProb      = scai != null ? (scai['prediction'] as num).toDouble() : 0.0;
      final scaiStage     = scai?['current_stage'] as String? ?? 'B';
      final vasoProb      = vaso != null ? (vaso['alert_proba'] as num).toDouble() : 0.0;
      final vasoCount     = vaso != null ? (vaso['ordinal_pred'] as int? ?? 0) : 0;
      final mcsProb       = mcs  != null ? (mcs['prediction']  as num).toDouble() : 0.0;
      final mcsNeeded     = mcs  != null && (mcs['alert_flag']  as int? ?? 0) == 1;
      final ecmoProb      = ecmo != null ? (ecmo['prediction'] as num).toDouble() : 0.0;
      final ecmoNeeded    = ecmo != null && (ecmo['alert_flag'] as int? ?? 0) == 1;

      final readmissionRisk = (mortalityProb * 0.5 +
              (losDays / 14.0).clamp(0.0, 1.0) * 0.5)
          .clamp(0.0, 1.0);
      final icuLosDays = (losDays * 0.4).clamp(0.5, losDays);
      final scaiAlert  = scai != null && (scai['alert_flag'] as int? ?? 0) == 1;

      final predictions = PatientPredictions(
        readmissionRisk:         readmissionRisk,
        hospitalLosDays:         losDays,
        icuLosDays:              icuLosDays,
        homeCareSuggestions:     const [],
        hospitalMortality:       mortalityProb,
        icuMortality:            (mortalityProb * 1.1).clamp(0.0, 1.0),
        inHospitalExpiry:        mortalityProb,
        icuTransferRisk:         scaiProb,
        shapTransfer:            parseShap(scai?['shap_values']),
        shapReadmission:         const [],
        shapMortality:           parseShap(mort['shap_values']),
        lastUpdated:             DateTime.now(),
        scaiDeterioration6hProb: scaiProb,
        scaiDeterioration6hLabel: scaiAlert ? 'Likely to worsen' : 'Unlikely to worsen',
        currentScaiStage:        scaiStage,
        vasopressorProbability:  vasoProb,
        predictedVasopressorCount: vasoCount,
        mortalityRisk:           mortalityProb,
        mcs12hProbability:       mcsProb,
        mcs12hNeeded:            mcsNeeded,
        vaEcmo12hProbability:    ecmoProb,
        vaEcmo12hNeeded:         ecmoNeeded,
        shapMcs12h:              parseShap(mcs?['shap_values']),
        shapVaEcmo12h:           parseShap(ecmo?['shap_values']),
        shapScai:                parseShap(scai?['shap_values']),
        shapVasopressor:         parseShap(vaso?['shap_values']),
      );

      final age          = p['age'] as int?;
      final losDaysInt   = losDays.round().clamp(1, 60);
      final hospitalIdx  = encId % demoHospitals.length;
      final hospitalId   = demoHospitals[hospitalIdx].id;
      final roomLetter   = ['A', 'B', 'C', 'D'][encId % 4];
      final roomNum      = (encId % 20) + 1;

      final rawVitals = p['vitals'] as List<dynamic>? ?? [];
      final clinicalVitals = rawVitals
          .map((v) => PatientVital.fromJson(v as Map<String, dynamic>))
          .toList();

      final rawMeds = p['medications'] as List<dynamic>? ?? [];
      final medications = rawMeds
          .map((m) => PatientMedication.fromJson(m as Map<String, dynamic>))
          .toList();

      allPatients.add(PatientRecord(
        rank:         0,
        name:         p['name'] as String,
        id:           encId.toString(),
        encounterId:  encId.toString(),
        roomNumber:   'ICU $roomNum$roomLetter',
        predictions:  predictions,
        clinicalVitals: clinicalVitals,
        medications: medications,
        features: PatientFeatures(
          numMedications:    0,
          numberInpatient:   1,
          numLabProcedures:  0,
          timeInHospital:    losDaysInt,
          numberDiagnoses:   1,
          numberEmergency:   1,
          numberOutpatient:  0,
          ageMid:            (age ?? 65).toDouble(),
        ),
        age:           age,
        gender:        p['sex'] as String?,
        daysAdmitted:  losDaysInt,
        demoHospitalId: hospitalId,
      ));
    }

    // Per hospital: top N by mortality risk; assign sequential rank
    final List<PatientRecord> result = [];
    final maxPerHospital = ApiConfig.dashboardPatientsPerHospital;

    for (final hospital in demoHospitals) {
      final bucket = allPatients
          .where((r) => r.demoHospitalId == hospital.id)
          .toList()
        ..sort((a, b) => b.predictions.mortalityRisk
            .compareTo(a.predictions.mortalityRisk));

      var rank = 1;
      for (final r in bucket.take(maxPerHospital)) {
        result.add(r.copyWithRank(rank++));
      }
    }

    return result;
  }
}
