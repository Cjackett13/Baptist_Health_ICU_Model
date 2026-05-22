// lib/models/patient_prediction.dart
//
// Single source of truth for all prediction data structures.
// When FastAPI is connected, these models map 1:1 to the API response.
// When Firestore is connected, these models map 1:1 to the document schema.

import 'dart:math';

import 'package:flutter/material.dart';

import 'shap_factor_descriptions.dart';

// ─────────────────────────────────────────────────────────────────────────────
// SHAP VALUE
// One feature's contribution to a prediction.
// positive → pushes risk UP, negative → pushes risk DOWN.
// ─────────────────────────────────────────────────────────────────────────────
class ShapValue {
  const ShapValue({
    required this.feature,
    required this.value,
    required this.direction,
    required this.description,
  });

  final String feature;
  final double value;
  final String direction; // 'positive' | 'negative'
  final String description;

  bool get isPositive => direction == 'positive';

  factory ShapValue.fromJson(Map<String, dynamic> json) {
    final rawDir = json['direction'] as String?;
    final isPositive = json['is_positive'] as bool?;
    final value = (json['value'] as num).toDouble();
    final direction = rawDir ??
        (isPositive != null
            ? (isPositive ? 'positive' : 'negative')
            : (value >= 0 ? 'positive' : 'negative'));
    final feature = json['feature'] as String? ?? 'Unknown factor';
    final increasesRisk = direction == 'positive';
    return ShapValue(
      feature: feature,
      value: value,
      direction: direction,
      description: json['description'] as String? ??
          describeShapFactor(feature, increasesRisk: increasesRisk),
    );
  }

  Map<String, dynamic> toJson() => {
        'feature': feature,
        'value': value,
        'is_positive': isPositive,
        'direction': direction,
        'description': description,
      };

  /// Top [k] unique features ranked by |SHAP| (for MCS/ECMO escalation).
  static List<ShapValue> topByShapMagnitude(
    List<ShapValue> raw, {
    int k = 5,
  }) {
    final sorted = List<ShapValue>.from(raw)
      ..sort((a, b) => b.value.abs().compareTo(a.value.abs()));
    final seen = <String>{};
    final out = <ShapValue>[];
    for (final s in sorted) {
      if (!seen.add(s.feature)) continue;
      out.add(s);
      if (out.length >= k) break;
    }
    return out;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// PATIENT FEATURES
// The editable clinical features used by the what-if simulator.
// These are the exact column names your XGBoost model was trained on.
// ─────────────────────────────────────────────────────────────────────────────
class PatientFeatures {
  const PatientFeatures({
    required this.numMedications,
    required this.numberInpatient,
    required this.numLabProcedures,
    required this.timeInHospital,
    required this.numberDiagnoses,
    required this.numberEmergency,
    required this.numberOutpatient,
    required this.ageMid,
  });

  final int numMedications;
  final int numberInpatient;
  final int numLabProcedures;
  final int timeInHospital;
  final int numberDiagnoses;
  final int numberEmergency;
  final int numberOutpatient;
  final double ageMid;

  PatientFeatures copyWith({
    int? numMedications,
    int? numberInpatient,
    int? numLabProcedures,
    int? timeInHospital,
    int? numberDiagnoses,
    int? numberEmergency,
    int? numberOutpatient,
    double? ageMid,
  }) =>
      PatientFeatures(
        numMedications: numMedications ?? this.numMedications,
        numberInpatient: numberInpatient ?? this.numberInpatient,
        numLabProcedures: numLabProcedures ?? this.numLabProcedures,
        timeInHospital: timeInHospital ?? this.timeInHospital,
        numberDiagnoses: numberDiagnoses ?? this.numberDiagnoses,
        numberEmergency: numberEmergency ?? this.numberEmergency,
        numberOutpatient: numberOutpatient ?? this.numberOutpatient,
        ageMid: ageMid ?? this.ageMid,
      );

  factory PatientFeatures.fromJson(Map<String, dynamic> json) =>
      PatientFeatures(
        numMedications: json['num_medications'] as int,
        numberInpatient: json['number_inpatient'] as int,
        numLabProcedures: json['num_lab_procedures'] as int,
        timeInHospital: json['time_in_hospital'] as int,
        numberDiagnoses: json['number_diagnoses'] as int,
        numberEmergency: json['number_emergency'] as int,
        numberOutpatient: json['number_outpatient'] as int,
        ageMid: (json['age_mid'] as num).toDouble(),
      );

  Map<String, dynamic> toJson() => {
        'num_medications': numMedications,
        'number_inpatient': numberInpatient,
        'num_lab_procedures': numLabProcedures,
        'time_in_hospital': timeInHospital,
        'number_diagnoses': numberDiagnoses,
        'number_emergency': numberEmergency,
        'number_outpatient': numberOutpatient,
        'age_mid': ageMid,
      };
}

// ─────────────────────────────────────────────────────────────────────────────
// HOME CARE SUGGESTION
// One personalized recommendation generated by the AI for the patient's family.
// Generated by Claude API in FastAPI — displayed on HomeCareScreen.
// ─────────────────────────────────────────────────────────────────────────────
class HomeCareSuggestion {
  const HomeCareSuggestion({
    required this.category,
    required this.title,
    required this.description,
    required this.impact,
  });

  // category: 'diet' | 'exercise' | 'medication' | 'monitoring' | 'lifestyle'
  final String category;
  final String title;
  final String description;

  // e.g. "Reduces readmission risk by ~12%"
  final String impact;

  factory HomeCareSuggestion.fromJson(Map<String, dynamic> json) =>
      HomeCareSuggestion(
        category: json['category'] as String,
        title: json['title'] as String,
        description: json['description'] as String,
        impact: json['impact'] as String,
      );

  Map<String, dynamic> toJson() => {
        'category': category,
        'title': title,
        'description': description,
        'impact': impact,
      };

  HomeCareSuggestion copyWith({
    String? category,
    String? title,
    String? description,
    String? impact,
  }) =>
      HomeCareSuggestion(
        category: category ?? this.category,
        title: title ?? this.title,
        description: description ?? this.description,
        impact: impact ?? this.impact,
      );
}

// ─────────────────────────────────────────────────────────────────────────────
// CLINICAL VITAL / MEDICATION (from parquet clinical_event & medication_admin)
// ─────────────────────────────────────────────────────────────────────────────
class PatientVital {
  const PatientVital({
    required this.code,
    required this.title,
    required this.value,
    required this.units,
    required this.normalcy,
    this.recordedAt,
  });

  final String code;
  final String title;
  final double value;
  final String units;
  final String normalcy;
  final DateTime? recordedAt;

  factory PatientVital.fromJson(Map<String, dynamic> json) {
    final raw = json['value'];
    if (raw == null) {
      throw FormatException('Vital ${json['code']} has null value');
    }
    return PatientVital(
      code: json['code'] as String,
      title: json['title'] as String,
      value: (raw as num).toDouble(),
      units: json['units'] as String? ?? '',
      normalcy: json['normalcy'] as String? ?? 'NORMAL',
      recordedAt: json['recorded_at'] != null
          ? DateTime.tryParse(json['recorded_at'] as String)
          : null,
    );
  }
}

class PatientMedication {
  const PatientMedication({
    required this.name,
    required this.code,
    required this.dosage,
    required this.route,
    required this.isVasopressor,
    this.start,
  });

  final String name;
  final String code;
  final String dosage;
  final String route;
  final bool isVasopressor;
  final DateTime? start;

  factory PatientMedication.fromJson(Map<String, dynamic> json) =>
      PatientMedication(
        name: json['name'] as String,
        code: json['code'] as String,
        dosage: json['dosage'] as String? ?? '',
        route: json['route'] as String? ?? 'IV',
        isVasopressor: json['is_vasopressor'] as bool? ?? false,
        start: json['start'] != null
            ? DateTime.tryParse(json['start'] as String)
            : null,
      );
}

class PatientDiagnosis {
  const PatientDiagnosis({
    required this.code,
    required this.text,
    required this.priority,
    required this.classification,
  });

  final String code;
  final String text;
  final int priority;
  final String classification;

  factory PatientDiagnosis.fromJson(Map<String, dynamic> json) =>
      PatientDiagnosis(
        code: json['code'] as String,
        text: json['text'] as String,
        priority: json['priority'] as int,
        classification: json['classification'] as String,
      );
}

// ─────────────────────────────────────────────────────────────────────────────
// PATIENT PREDICTIONS
// ML outputs + legacy fields for list cards / home care.
// ─────────────────────────────────────────────────────────────────────────────
class PatientPredictions {
  const PatientPredictions({
    required this.readmissionRisk,
    required this.hospitalLosDays,
    required this.icuLosDays,
    required this.homeCareSuggestions,
    required this.hospitalMortality,
    required this.icuMortality,
    required this.inHospitalExpiry,
    required this.icuTransferRisk,
    required this.shapTransfer,
    required this.shapReadmission,
    required this.shapMortality,
    required this.lastUpdated,
    required this.scaiDeterioration6hProb,
    required this.scaiDeterioration6hLabel,
    required this.currentScaiStage,
    required this.vasopressorProbability,
    required this.predictedVasopressorCount,
    required this.mortalityRisk,
    required this.mcs12hProbability,
    required this.mcs12hNeeded,
    required this.vaEcmo12hProbability,
    required this.vaEcmo12hNeeded,
    this.shapMcs12h = const [],
    this.shapVaEcmo12h = const [],
  });

  final double readmissionRisk;
  final double hospitalLosDays;
  final double icuLosDays;
  final List<HomeCareSuggestion> homeCareSuggestions;
  final double hospitalMortality;
  final double icuMortality;
  final double inHospitalExpiry;
  final double icuTransferRisk;
  final List<ShapValue> shapTransfer;
  final List<ShapValue> shapReadmission;
  final List<ShapValue> shapMortality;
  final DateTime lastUpdated;

  /// SCAI shock stage worsening within 6 hours (probability 0–1).
  final double scaiDeterioration6hProb;
  final String scaiDeterioration6hLabel;
  final String currentScaiStage;
  final double vasopressorProbability;
  final int predictedVasopressorCount;
  final double mortalityRisk;
  final double mcs12hProbability;
  final bool mcs12hNeeded;
  final double vaEcmo12hProbability;
  final bool vaEcmo12hNeeded;
  final List<ShapValue> shapMcs12h;
  final List<ShapValue> shapVaEcmo12h;

  // Convenience — highest mortality across all three scopes
  double get peakMortality =>
      [hospitalMortality, icuMortality, inHospitalExpiry].reduce(
        (a, b) => a > b ? a : b,
      );

  /// List acuity tier — SCAI stage first; not MCS/ECMO screening scores.
  String get overallTier {
    final stage = currentScaiStage.toUpperCase();
    if (stage == 'D' || stage == 'E') return 'Critical';
    if (stage == 'C') return 'Moderate';
    if (stage == 'A' || stage == 'B') {
      if (mortalityRisk >= 0.75 || scaiDeterioration6hProb >= 0.65) {
        return 'Critical';
      }
      if (mortalityRisk >= 0.45 || scaiDeterioration6hProb >= 0.45) {
        return 'Moderate';
      }
      return 'Stable';
    }
    if (mortalityRisk >= 0.70) return 'Critical';
    if (mortalityRisk >= 0.40) return 'Moderate';
    return 'Stable';
  }

  factory PatientPredictions.fromSeedMap(Map<String, dynamic> p) {
    final lastUpdated = DateTime.tryParse(p['last_updated'] as String? ?? '') ??
        DateTime.now();
    final mortality = (p['mortality_risk'] as num?)?.toDouble() ??
        (p['hospital_mortality'] as num?)?.toDouble() ??
        0.0;
    final icuRisk = (p['icu_transfer_risk'] as num?)?.toDouble() ?? mortality;
    final readmit = (p['readmission_risk'] as num?)?.toDouble() ?? mortality * 0.7;

    return PatientPredictions(
      readmissionRisk: readmit,
      hospitalLosDays: (p['hospital_los_days'] as num?)?.toDouble() ?? 0.0,
      icuLosDays: (p['icu_los_days'] as num?)?.toDouble() ?? 0.0,
      homeCareSuggestions: const [],
      hospitalMortality:
          (p['hospital_mortality'] as num?)?.toDouble() ?? mortality,
      icuMortality:
          (p['icu_mortality'] as num?)?.toDouble() ??
          (mortality * 1.1).clamp(0.0, 1.0),
      inHospitalExpiry:
          (p['in_hospital_expiry'] as num?)?.toDouble() ?? mortality,
      icuTransferRisk: icuRisk,
      shapTransfer: const [],
      shapReadmission: const [],
      shapMortality: const [],
      lastUpdated: lastUpdated,
      scaiDeterioration6hProb:
          (p['scai_deterioration_6h_prob'] as num?)?.toDouble() ?? 0.2,
      scaiDeterioration6hLabel:
          p['scai_deterioration_6h_label'] as String? ?? 'Unknown',
      currentScaiStage: p['current_scai_stage'] as String? ?? 'B',
      vasopressorProbability:
          (p['vasopressor_probability'] as num?)?.toDouble() ?? 0.2,
      predictedVasopressorCount:
          p['predicted_vasopressor_count'] as int? ?? 0,
      mortalityRisk: mortality,
      mcs12hProbability:
          (p['mcs_12h_probability'] as num?)?.toDouble() ?? 0.15,
      mcs12hNeeded: p['mcs_12h_needed'] as bool? ?? false,
      vaEcmo12hProbability:
          (p['va_ecmo_12h_probability'] as num?)?.toDouble() ?? 0.1,
      vaEcmo12hNeeded: p['va_ecmo_12h_needed'] as bool? ?? false,
      shapMcs12h: _parseShapList(p['shap_mcs_12h']),
      shapVaEcmo12h: _parseShapList(p['shap_va_ecmo_12h']),
    );
  }

  static List<ShapValue> _parseShapList(dynamic raw) {
    if (raw is! List<dynamic>) return const [];
    final parsed = raw
        .map((e) => ShapValue.fromJson(e as Map<String, dynamic>))
        .toList();
    return ShapValue.topByShapMagnitude(parsed, k: 5);
  }

  factory PatientPredictions.fromJson(Map<String, dynamic> json) =>
      PatientPredictions(
        readmissionRisk:
            (json['readmission_risk'] as num).toDouble(),
        hospitalLosDays:
            (json['hospital_los_days'] as num).toDouble(),
        icuLosDays: (json['icu_los_days'] as num).toDouble(),
        homeCareSuggestions:
            (json['home_care_suggestions'] as List<dynamic>)
                .map((e) => HomeCareSuggestion.fromJson(
                    e as Map<String, dynamic>))
                .toList(),
        hospitalMortality:
            (json['hospital_mortality'] as num).toDouble(),
        icuMortality: (json['icu_mortality'] as num).toDouble(),
        inHospitalExpiry:
            (json['in_hospital_expiry'] as num).toDouble(),
        icuTransferRisk:
            (json['icu_transfer_risk'] as num).toDouble(),
        shapTransfer:
            (json['shap_transfer'] as List<dynamic>)
                .map((e) =>
                    ShapValue.fromJson(e as Map<String, dynamic>))
                .toList(),
        shapReadmission:
            (json['shap_readmission'] as List<dynamic>)
                .map((e) =>
                    ShapValue.fromJson(e as Map<String, dynamic>))
                .toList(),
        shapMortality:
            (json['shap_mortality'] as List<dynamic>)
                .map((e) =>
                    ShapValue.fromJson(e as Map<String, dynamic>))
                .toList(),
        lastUpdated: DateTime.parse(json['last_updated'] as String),
        scaiDeterioration6hProb:
            (json['scai_deterioration_6h_prob'] as num?)?.toDouble() ?? 0.2,
        scaiDeterioration6hLabel:
            json['scai_deterioration_6h_label'] as String? ?? 'Unknown',
        currentScaiStage: json['current_scai_stage'] as String? ?? 'B',
        vasopressorProbability:
            (json['vasopressor_probability'] as num?)?.toDouble() ?? 0.2,
        predictedVasopressorCount:
            json['predicted_vasopressor_count'] as int? ?? 0,
        mortalityRisk:
            (json['mortality_risk'] as num?)?.toDouble() ??
                (json['hospital_mortality'] as num).toDouble(),
        mcs12hProbability:
            (json['mcs_12h_probability'] as num?)?.toDouble() ?? 0.15,
        mcs12hNeeded: json['mcs_12h_needed'] as bool? ?? false,
        vaEcmo12hProbability:
            (json['va_ecmo_12h_probability'] as num?)?.toDouble() ?? 0.1,
        vaEcmo12hNeeded: json['va_ecmo_12h_needed'] as bool? ?? false,
        shapMcs12h: _parseShapList(json['shap_mcs_12h']),
        shapVaEcmo12h: _parseShapList(json['shap_va_ecmo_12h']),
      );

  PatientPredictions copyWithMcsEcmo({
    required double mcs12hProbability,
    required bool mcs12hNeeded,
    required double vaEcmo12hProbability,
    required bool vaEcmo12hNeeded,
    List<ShapValue>? shapMcs12h,
    List<ShapValue>? shapVaEcmo12h,
    DateTime? lastUpdated,
  }) =>
      PatientPredictions(
        readmissionRisk: readmissionRisk,
        hospitalLosDays: hospitalLosDays,
        icuLosDays: icuLosDays,
        homeCareSuggestions: homeCareSuggestions,
        hospitalMortality: hospitalMortality,
        icuMortality: icuMortality,
        inHospitalExpiry: inHospitalExpiry,
        icuTransferRisk: icuTransferRisk,
        shapTransfer: shapTransfer,
        shapReadmission: shapReadmission,
        shapMortality: shapMortality,
        lastUpdated: lastUpdated ?? this.lastUpdated,
        scaiDeterioration6hProb: scaiDeterioration6hProb,
        scaiDeterioration6hLabel: scaiDeterioration6hLabel,
        currentScaiStage: currentScaiStage,
        vasopressorProbability: vasopressorProbability,
        predictedVasopressorCount: predictedVasopressorCount,
        mortalityRisk: mortalityRisk,
        mcs12hProbability: mcs12hProbability,
        mcs12hNeeded: mcs12hNeeded,
        vaEcmo12hProbability: vaEcmo12hProbability,
        vaEcmo12hNeeded: vaEcmo12hNeeded,
        shapMcs12h: shapMcs12h ?? this.shapMcs12h,
        shapVaEcmo12h: shapVaEcmo12h ?? this.shapVaEcmo12h,
      );

  PatientPredictions copyWith({
    double? hospitalMortality,
    double? icuMortality,
    double? inHospitalExpiry,
    DateTime? lastUpdated,
  }) {
    final h = hospitalMortality ?? this.hospitalMortality;
    final i = icuMortality ?? this.icuMortality;
    final e = inHospitalExpiry ?? this.inHospitalExpiry;
    final peak = [h, i, e].reduce((a, b) => a > b ? a : b);
    return PatientPredictions(
      readmissionRisk: readmissionRisk,
      hospitalLosDays: hospitalLosDays,
      icuLosDays: icuLosDays,
      homeCareSuggestions: homeCareSuggestions,
      hospitalMortality: h,
      icuMortality: i,
      inHospitalExpiry: e,
      icuTransferRisk: icuTransferRisk,
      shapTransfer: shapTransfer,
      shapReadmission: shapReadmission,
      shapMortality: shapMortality,
      lastUpdated: lastUpdated ?? this.lastUpdated,
      scaiDeterioration6hProb: scaiDeterioration6hProb,
      scaiDeterioration6hLabel: scaiDeterioration6hLabel,
      currentScaiStage: currentScaiStage,
      vasopressorProbability: vasopressorProbability,
      predictedVasopressorCount: predictedVasopressorCount,
      mortalityRisk: peak,
      mcs12hProbability: mcs12hProbability,
      mcs12hNeeded: mcs12hNeeded,
      vaEcmo12hProbability: vaEcmo12hProbability,
      vaEcmo12hNeeded: vaEcmo12hNeeded,
      shapMcs12h: shapMcs12h,
      shapVaEcmo12h: shapVaEcmo12h,
    );
  }

  PatientPredictions copyWithHomeCare(List<HomeCareSuggestion> recs) =>
      PatientPredictions(
        readmissionRisk: readmissionRisk,
        hospitalLosDays: hospitalLosDays,
        icuLosDays: icuLosDays,
        homeCareSuggestions: recs,
        hospitalMortality: hospitalMortality,
        icuMortality: icuMortality,
        inHospitalExpiry: inHospitalExpiry,
        icuTransferRisk: icuTransferRisk,
        shapTransfer: shapTransfer,
        shapReadmission: shapReadmission,
        shapMortality: shapMortality,
        lastUpdated: lastUpdated,
        scaiDeterioration6hProb: scaiDeterioration6hProb,
        scaiDeterioration6hLabel: scaiDeterioration6hLabel,
        currentScaiStage: currentScaiStage,
        vasopressorProbability: vasopressorProbability,
        predictedVasopressorCount: predictedVasopressorCount,
        mortalityRisk: mortalityRisk,
        mcs12hProbability: mcs12hProbability,
        mcs12hNeeded: mcs12hNeeded,
        vaEcmo12hProbability: vaEcmo12hProbability,
        vaEcmo12hNeeded: vaEcmo12hNeeded,
        shapMcs12h: shapMcs12h,
        shapVaEcmo12h: shapVaEcmo12h,
      );

  Map<String, dynamic> toJson() => {
        'readmission_risk': readmissionRisk,
        'hospital_los_days': hospitalLosDays,
        'icu_los_days': icuLosDays,
        'home_care_suggestions':
            homeCareSuggestions.map((s) => s.toJson()).toList(),
        'hospital_mortality': hospitalMortality,
        'icu_mortality': icuMortality,
        'in_hospital_expiry': inHospitalExpiry,
        'icu_transfer_risk': icuTransferRisk,
        'shap_transfer':
            shapTransfer.map((s) => s.toJson()).toList(),
        'shap_readmission':
            shapReadmission.map((s) => s.toJson()).toList(),
        'shap_mortality':
            shapMortality.map((s) => s.toJson()).toList(),
        'last_updated': lastUpdated.toIso8601String(),
      };
}

// ─────────────────────────────────────────────────────────────────────────────
// MECHANICAL SUPPORT (from procedure_event at scoring hour)
// ─────────────────────────────────────────────────────────────────────────────
class PatientMechanicalSupport {
  const PatientMechanicalSupport({
    required this.onMcs,
    required this.onVaEcmo,
    required this.activeDevices,
  });

  final bool onMcs;
  final bool onVaEcmo;
  final List<ActiveSupportDevice> activeDevices;

  factory PatientMechanicalSupport.fromJson(Map<String, dynamic>? json) {
    if (json == null) {
      return const PatientMechanicalSupport(
        onMcs: false,
        onVaEcmo: false,
        activeDevices: [],
      );
    }
    final devices = (json['active_devices'] as List<dynamic>? ?? [])
        .map((e) => ActiveSupportDevice.fromJson(e as Map<String, dynamic>))
        .toList();
    return PatientMechanicalSupport(
      onMcs: json['on_mcs'] as bool? ?? devices.isNotEmpty,
      onVaEcmo: json['on_va_ecmo'] as bool? ?? false,
      activeDevices: devices,
    );
  }

  String get summaryLabel {
    if (!onMcs) return 'No MCS/ECMO at this hour';
    if (onVaEcmo) return 'On VA-ECMO';
    if (activeDevices.length == 1) return 'On ${activeDevices.first.label}';
    return 'On ${activeDevices.length} MCS devices';
  }
}

class ActiveSupportDevice {
  const ActiveSupportDevice({
    required this.code,
    required this.label,
    this.start,
    this.end,
  });

  final String code;
  final String label;
  final DateTime? start;
  final DateTime? end;

  factory ActiveSupportDevice.fromJson(Map<String, dynamic> json) =>
      ActiveSupportDevice(
        code: json['code'] as String,
        label: json['label'] as String,
        start: json['start'] != null
            ? DateTime.tryParse(json['start'] as String)
            : null,
        end:
            json['end'] != null ? DateTime.tryParse(json['end'] as String) : null,
      );
}

// ─────────────────────────────────────────────────────────────────────────────
// PATIENT RECORD
// Central model used everywhere in the app.
// condition is derived from predictions.overallTier — never set manually.
// ─────────────────────────────────────────────────────────────────────────────
class PatientRecord {
  const PatientRecord({
    required this.rank,
    required this.name,
    required this.id,
    required this.roomNumber,
    required this.predictions,
    required this.features,
    this.encounterId,
    this.primaryDoctor,
    this.issue,
    this.age,
    this.gender,
    this.diagnosis,
    this.daysAdmitted,
    this.unitCd,
    this.facilityCd,
    this.demoHospitalId,
    this.scaiStageCurrent,
    this.hourFromAdmit,
    this.clinicalVitals = const [],
    this.diagnoses = const [],
    this.medications = const [],
    this.recommendations = const [],
    this.conditionOverride,
    this.mechanicalSupport = const PatientMechanicalSupport(
      onMcs: false,
      onVaEcmo: false,
      activeDevices: [],
    ),
  });

  final int rank;
  final String name;
  final String id;
  final String roomNumber;
  final PatientPredictions predictions;
  final PatientFeatures features;
  final String? encounterId;
  final String? primaryDoctor;
  final String? issue;
  final int? age;
  final String? gender;
  final String? diagnosis;
  final int? daysAdmitted;
  final String? unitCd;
  final String? facilityCd;
  /// Demo hospital picker id (`bh-jax`, `bh-mia`, …) from seed export.
  final String? demoHospitalId;
  final String? scaiStageCurrent;
  /// ICU hour aligned to the MCS/ECMO model feature row in parquet cache.
  final int? hourFromAdmit;
  final List<PatientVital> clinicalVitals;
  final List<PatientDiagnosis> diagnoses;
  final List<PatientMedication> medications;
  final List<HomeCareSuggestion> recommendations;
  final String? conditionOverride;
  final PatientMechanicalSupport mechanicalSupport;

  double get icuRisk => predictions.icuTransferRisk;
  double get readmissionRisk => predictions.readmissionRisk;
  String get condition =>
      conditionOverride ?? predictions.overallTier;
  List<ShapValue> get shapValues => predictions.shapTransfer;

  factory PatientRecord.fromSeedJson(Map<String, dynamic> json) {
    final predMap = json['predictions'] as Map<String, dynamic>;
    final predictions = PatientPredictions.fromSeedMap(predMap);
    final recs = (json['recommendations'] as List<dynamic>? ?? [])
        .map((e) => HomeCareSuggestion.fromJson(e as Map<String, dynamic>))
        .toList();
    final vitals = <PatientVital>[];
    for (final raw in json['vitals'] as List<dynamic>? ?? []) {
      final map = raw as Map<String, dynamic>;
      if (map['value'] == null) continue;
      vitals.add(PatientVital.fromJson(map));
    }
    final dx = (json['diagnoses'] as List<dynamic>? ?? [])
        .map((e) => PatientDiagnosis.fromJson(e as Map<String, dynamic>))
        .toList();
    final meds = (json['medications'] as List<dynamic>? ?? [])
        .map((e) => PatientMedication.fromJson(e as Map<String, dynamic>))
        .toList();

    final age = json['age'] as int?;
    final days = json['days_admitted'] as int?;
    final hourFromAdmit = json['hour_from_admit'] as int?;

    final modelPersonId = json['model_person_id'] as String? ?? json['id'] as String;
    final modelEncounterId =
        json['model_encounter_id'] as String? ?? json['encounter_id'] as String?;

    return PatientRecord(
      rank: json['rank'] as int,
      name: json['name'] as String,
      id: modelPersonId,
      encounterId: modelEncounterId,
      roomNumber: json['room_number'] as String,
      predictions: predictions.copyWithHomeCare(recs),
      features: PatientFeatures(
        numMedications: meds.length,
        numberInpatient: 1,
        numLabProcedures: vitals.where((v) => v.code.length > 3).length,
        timeInHospital: days ?? 1,
        numberDiagnoses: dx.length.clamp(1, 16),
        numberEmergency: 1,
        numberOutpatient: 0,
        ageMid: (age ?? 65).toDouble(),
      ),
      primaryDoctor: json['primary_doctor'] as String?,
      issue: json['issue'] as String?,
      age: age,
      gender: json['gender'] as String?,
      diagnosis: json['diagnosis'] as String?,
      daysAdmitted: days,
      unitCd: json['unit_cd'] as String?,
      facilityCd: json['facility_cd'] as String?,
      demoHospitalId: json['demo_hospital_id'] as String?,
      scaiStageCurrent: json['scai_stage_current'] as String?,
      hourFromAdmit: hourFromAdmit,
      clinicalVitals: vitals,
      diagnoses: dx,
      medications: meds,
      recommendations: recs,
      conditionOverride: json['condition'] as String?,
      mechanicalSupport: PatientMechanicalSupport.fromJson(
        json['mechanical_support'] as Map<String, dynamic>?,
      ),
    );
  }

  PatientRecord copyWithRecommendations(List<HomeCareSuggestion> recs) =>
      PatientRecord(
        rank: rank,
        name: name,
        id: id,
        roomNumber: roomNumber,
        predictions: predictions.copyWithHomeCare(recs),
        features: features,
        encounterId: encounterId,
        primaryDoctor: primaryDoctor,
        issue: issue,
        age: age,
        gender: gender,
        diagnosis: diagnosis,
        daysAdmitted: daysAdmitted,
        unitCd: unitCd,
        facilityCd: facilityCd,
        demoHospitalId: demoHospitalId,
        scaiStageCurrent: scaiStageCurrent,
        hourFromAdmit: hourFromAdmit,
        clinicalVitals: clinicalVitals,
        diagnoses: diagnoses,
        medications: medications,
        recommendations: recs,
        mechanicalSupport: mechanicalSupport,
        conditionOverride: conditionOverride,
      );

  PatientRecord copyWithPredictions(PatientPredictions newPredictions) =>
      PatientRecord(
        rank: rank,
        name: name,
        id: id,
        roomNumber: roomNumber,
        predictions: newPredictions,
        features: features,
        encounterId: encounterId,
        primaryDoctor: primaryDoctor,
        issue: issue,
        age: age,
        gender: gender,
        diagnosis: diagnosis,
        daysAdmitted: daysAdmitted,
        unitCd: unitCd,
        facilityCd: facilityCd,
        demoHospitalId: demoHospitalId,
        scaiStageCurrent: scaiStageCurrent,
        hourFromAdmit: hourFromAdmit,
        clinicalVitals: clinicalVitals,
        diagnoses: diagnoses,
        medications: medications,
        recommendations: recommendations,
        mechanicalSupport: mechanicalSupport,
        conditionOverride: conditionOverride,
      );

  PatientRecord copyWithRank(int newRank) => PatientRecord(
        rank: newRank,
        name: name,
        id: id,
        roomNumber: roomNumber,
        predictions: predictions,
        features: features,
        encounterId: encounterId,
        primaryDoctor: primaryDoctor,
        issue: issue,
        age: age,
        gender: gender,
        diagnosis: diagnosis,
        daysAdmitted: daysAdmitted,
        unitCd: unitCd,
        facilityCd: facilityCd,
        demoHospitalId: demoHospitalId,
        scaiStageCurrent: scaiStageCurrent,
        hourFromAdmit: hourFromAdmit,
        clinicalVitals: clinicalVitals,
        diagnoses: diagnoses,
        medications: medications,
        recommendations: recommendations,
        mechanicalSupport: mechanicalSupport,
        conditionOverride: conditionOverride,
      );
}

// ─────────────────────────────────────────────────────────────────────────────
// COLOR + LABEL HELPERS
// Shared across all prediction widgets.
// ─────────────────────────────────────────────────────────────────────────────

Color riskColor(double risk) {
  if (risk >= 0.65) return const Color(0xFFE05A5A);
  if (risk >= 0.40) return const Color(0xFFD4A030);
  return const Color(0xFF4A9E6A);
}

Color conditionColor(String condition) {
  switch (condition) {
    case 'Critical':
      return const Color(0xFFE05A5A);
    case 'Moderate':
      return const Color(0xFFD4A030);
    default:
      return const Color(0xFF4A9E6A);
  }
}

String riskLabel(double risk) {
  if (risk >= 0.65) return 'High risk';
  if (risk >= 0.40) return 'Elevated';
  return 'Stable';
}

Color losColor(double days) {
  if (days >= 7) return const Color(0xFFE05A5A);
  if (days >= 3) return const Color(0xFFD4A030);
  return const Color(0xFF4A9E6A);
}

/// Vitals/labs urgency from NORMALCY_CD (e.g. CRITICAL_HIGH, LOW, NORMAL).
Color normalcyColor(String normalcy) {
  final n = normalcy.toUpperCase().replaceAll(' ', '_');
  if (n.contains('CRITICAL')) return const Color(0xFFE05A5A);
  if (n == 'HIGH' || n == 'LOW' || n == 'ABNORMAL') {
    return const Color(0xFFD4A030);
  }
  if (n == 'NORMAL') return const Color(0xFF4A9E6A);
  if (n == 'UNKNOWN') return const Color(0xFF9E9E9E);
  return const Color(0xFF9E9E9E);
}

String normalcyLabel(String normalcy) {
  final n = normalcy.toUpperCase().replaceAll(' ', '_');
  switch (n) {
    case 'CRITICAL_HIGH':
      return 'Critical high';
    case 'CRITICAL_LOW':
      return 'Critical low';
    case 'CRITICAL':
      return 'Critical';
    case 'HIGH':
      return 'High';
    case 'LOW':
      return 'Low';
    case 'ABNORMAL':
      return 'Abnormal';
    case 'NORMAL':
      return 'Normal';
    case 'UNKNOWN':
      return 'No reference range';
    default:
      return normalcy.replaceAll('_', ' ').toLowerCase();
  }
}

String categoryIcon(String category) {
  switch (category) {
    case 'diet':
      return '🥗';
    case 'exercise':
      return '🚶';
    case 'medication':
      return '💊';
    case 'monitoring':
      return '📊';
    case 'lifestyle':
      return '🌿';
    default:
      return '💡';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// MOCK DATA GENERATORS
// Replace these with real FastAPI calls later.
// Every function signature stays the same — just swap the body.
// ─────────────────────────────────────────────────────────────────────────────
List<ShapValue> generateMockShapValues(
    double risk, Random rng, String predictionType) {
  final featureSets = {
    'transfer': [
      ('Prior inpatient visits', true),
      ('No. of medications', true),
      ('Age', true),
      ('Lab procedures', true),
      ('Emergency visits', true),
      ('Days in hospital', true),
      ('Insulin adjusted', false),
      ('Outpatient visits', false),
    ],
    'readmission': [
      ('Discharge disposition', true),
      ('Number inpatient', true),
      ('Diagnosis group', true),
      ('No. of medications', true),
      ('Time in hospital', true),
      ('Metformin usage', false),
      ('Admission source', false),
    ],
    'mortality': [
      ('Age', true),
      ('No. of diagnoses', true),
      ('ICU procedures', true),
      ('Emergency admissions', true),
      ('Lab abnormalities', true),
      ('Medication count', true),
      ('Prior outpatient visits', false),
    ],
  };

  final features =
      featureSets[predictionType] ?? featureSets['transfer']!;
  final shapValues = <ShapValue>[];

  for (var i = 0; i < features.length.clamp(0, 5); i++) {
    final feat = features[i];
    final magnitude =
        (risk * (0.5 - i * 0.07) * (0.85 + rng.nextDouble() * 0.3))
            .clamp(0.04, 0.55);
    final increases = feat.$2;
    shapValues.add(ShapValue(
      feature: feat.$1,
      value: double.parse(magnitude.toStringAsFixed(2)),
      direction: increases ? 'positive' : 'negative',
      description: describeShapFactor(
        feat.$1,
        increasesRisk: increases,
      ),
    ));
  }

  // Add one protective factor
  final protective = features.lastWhere(
    (f) => !f.$2,
    orElse: () => ('Stable vitals', false),
  );
  shapValues.add(ShapValue(
    feature: protective.$1,
    value: double.parse(
        (0.04 + rng.nextDouble() * 0.14).toStringAsFixed(2)),
    direction: 'negative',
    description: describeShapFactor(
      protective.$1,
      increasesRisk: false,
    ),
  ));

  return shapValues;
}

/// Top 5 SHAP-style factors for MCS / VA-ECMO when the live API is unavailable.
List<ShapValue> generateMockEscalationShapValues(
  double risk,
  Random rng, {
  required bool forEcmo,
}) {
  final features = forEcmo
      ? [
          ('SCAI stage (current)', true),
          ('Shock burden (now)', true),
          ('Mean arterial pressure', true),
          ('Lactate', true),
          ('Vasoactive-inotrope score', true),
          ('Oxygen saturation', false),
          ('Hours on current SCAI stage', true),
        ]
      : [
          ('SCAI stage (current)', true),
          ('Shock burden (now)', true),
          ('Vasoactive-inotrope score', true),
          ('Mean arterial pressure', true),
          ('Lactate', true),
          ('Vasopressor use', true),
          ('Oxygen saturation', false),
        ];

  final shapValues = <ShapValue>[];
  for (var i = 0; i < 5; i++) {
    final feat = features[i];
    final base = risk * (0.55 - i * 0.08) * (0.9 + rng.nextDouble() * 0.25);
    final signed = feat.$2 ? base : -base * 0.65;
    final magnitude = signed.abs().clamp(0.05, 0.85);
    shapValues.add(
      ShapValue(
        feature: feat.$1,
        value: double.parse(
          (feat.$2 ? magnitude : -magnitude).toStringAsFixed(4),
        ),
        direction: feat.$2 ? 'positive' : 'negative',
        description: describeShapFactor(
          feat.$1,
          increasesRisk: feat.$2,
        ),
      ),
    );
  }
  return ShapValue.topByShapMagnitude(shapValues, k: 5);
}

List<HomeCareSuggestion> generateMockHomeCareSuggestions(
    PatientFeatures features, double readmissionRisk) {
  // Mock suggestions — replaced by Claude API call in FastAPI later
  return [
    HomeCareSuggestion(
      category: 'diet',
      title: 'Reduce sodium intake',
      description:
          'Aim for less than 2,300mg of sodium per day. Avoid processed foods, canned soups, and fast food. '
          'Cook at home using herbs and spices instead of salt.',
      impact: 'Reduces readmission risk by ~8%',
    ),
    HomeCareSuggestion(
      category: 'medication',
      title: 'Take medications as prescribed',
      description:
          'Set a daily alarm to take all ${features.numMedications} prescribed medications. '
          'Never skip doses even if feeling better. '
          'Use a pill organiser to track daily intake.',
      impact: 'Reduces ICU transfer risk by ~15%',
    ),
    HomeCareSuggestion(
      category: 'monitoring',
      title: 'Monitor blood pressure daily',
      description:
          'Check blood pressure every morning before eating or taking medications. '
          'Keep a log and share it at your next appointment. '
          'Seek immediate care if readings exceed 180/120.',
      impact: 'Enables early intervention, reducing mortality risk by ~11%',
    ),
    HomeCareSuggestion(
      category: 'exercise',
      title: 'Light daily activity',
      description:
          'Start with 10-minute walks twice a day and gradually increase. '
          'Avoid strenuous exercise for the first two weeks post-discharge. '
          'Stop and rest if you feel short of breath or chest pain.',
      impact: 'Improves long-term outcomes by ~9%',
    ),
    HomeCareSuggestion(
      category: 'lifestyle',
      title: 'Schedule follow-up appointments',
      description:
          'Book a follow-up with your primary care physician within 7 days of discharge. '
          'Bring your complete medication list to every appointment. '
          'Do not wait for symptoms to worsen before calling your doctor.',
      impact: 'Reduces 30-day readmission risk by ~18%',
    ),
  ];
}

PatientPredictions generateMockPredictions(
    double icuRisk, PatientFeatures features, Random rng) {
  // Readmission loosely correlated with ICU risk
  final readmission =
      (icuRisk * 0.65 + rng.nextDouble() * 0.35).clamp(0.05, 0.95);

  // LOS: higher risk = longer stay
  final hospitalLos = icuRisk >= 0.65
      ? 6.0 + rng.nextDouble() * 8.0
      : icuRisk >= 0.40
          ? 3.0 + rng.nextDouble() * 4.0
          : 1.0 + rng.nextDouble() * 3.0;
  final icuLos = (hospitalLos * 0.45 + rng.nextDouble() * 1.5)
      .clamp(0.5, hospitalLos);

  // Mortality — correlated but not identical to ICU risk
  final hospMort =
      (icuRisk * 0.40 + rng.nextDouble() * 0.20).clamp(0.01, 0.85);
  final icuMort =
      (hospMort * 1.3 + rng.nextDouble() * 0.10).clamp(0.01, 0.95);
  final inhospExpiry =
      (hospMort * 0.85 + rng.nextDouble() * 0.10).clamp(0.01, 0.90);

  final scaiProb = (icuRisk * 0.5 + rng.nextDouble() * 0.3).clamp(0.05, 0.95);
  final vasoProb = (icuRisk * 0.6 + rng.nextDouble() * 0.25).clamp(0.05, 0.95);

  return PatientPredictions(
    readmissionRisk: double.parse(readmission.toStringAsFixed(2)),
    hospitalLosDays: double.parse(hospitalLos.toStringAsFixed(1)),
    icuLosDays: double.parse(icuLos.toStringAsFixed(1)),
    homeCareSuggestions:
        generateMockHomeCareSuggestions(features, readmission),
    hospitalMortality: double.parse(hospMort.toStringAsFixed(2)),
    icuMortality: double.parse(icuMort.toStringAsFixed(2)),
    inHospitalExpiry: double.parse(inhospExpiry.toStringAsFixed(2)),
    icuTransferRisk: double.parse(icuRisk.toStringAsFixed(2)),
    shapTransfer: generateMockShapValues(icuRisk, rng, 'transfer'),
    shapReadmission: generateMockShapValues(readmission, rng, 'readmission'),
    shapMortality: generateMockShapValues(hospMort, rng, 'mortality'),
    lastUpdated: DateTime.now(),
    scaiDeterioration6hProb: double.parse(scaiProb.toStringAsFixed(2)),
    scaiDeterioration6hLabel:
        scaiProb >= 0.5 ? 'Likely to worsen' : 'Unlikely to worsen',
    currentScaiStage: ['A', 'B', 'C', 'D', 'E'][rng.nextInt(5)],
    vasopressorProbability: double.parse(vasoProb.toStringAsFixed(2)),
    predictedVasopressorCount: (vasoProb * 3).round().clamp(0, 4),
    mortalityRisk: double.parse(hospMort.toStringAsFixed(2)),
    mcs12hProbability:
        double.parse((icuRisk * 0.55).clamp(0.02, 0.9).toStringAsFixed(2)),
    mcs12hNeeded: icuRisk >= 0.55,
    vaEcmo12hProbability:
        double.parse((icuRisk * 0.45).clamp(0.02, 0.85).toStringAsFixed(2)),
    vaEcmo12hNeeded: icuRisk >= 0.65,
    shapMcs12h: generateMockEscalationShapValues(icuRisk, rng, forEcmo: false),
    shapVaEcmo12h: generateMockEscalationShapValues(icuRisk, rng, forEcmo: true),
  );
}