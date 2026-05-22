// Local fallback mortality estimates (no ML server required).
// Kept in sync with `back_end/app/fallback_mortality.py`.

import 'dart:math' as math;

class MortalityScores {
  const MortalityScores({
    required this.hospitalMortality,
    required this.icuMortality,
    required this.inHospitalExpiry,
  });

  final double hospitalMortality;
  final double icuMortality;
  final double inHospitalExpiry;

  double get peak => math.max(
        hospitalMortality,
        math.max(icuMortality, inHospitalExpiry),
      );
}

class MortalityModel {
  MortalityModel._();

  static double _sigmoid(double z) => 1.0 / (1.0 + math.exp(-z));

  static double _clamp(double v, double lo, double hi) =>
      v < lo ? lo : (v > hi ? hi : v);

  /// Deterministic heuristic when API / trained bundle is unavailable.
  static MortalityScores predict({
    required int numMedications,
    required int numberInpatient,
    required int numLabProcedures,
    required int timeInHospital,
    required int numberDiagnoses,
    required int numberEmergency,
    required int numberOutpatient,
    required double ageMid,
  }) {
    final z = -5.15 +
        0.042 * (ageMid - 55.0) +
        0.20 * _clamp(timeInHospital - 3.0, -2, 11) +
        0.017 * _clamp(numMedications - 8.0, -7, 32) +
        0.11 * _clamp(numberInpatient.toDouble(), 0, 8) +
        0.048 * _clamp(numberDiagnoses - 4.0, -3, 12) +
        0.075 * _clamp(numberEmergency.toDouble(), 0, 6) +
        0.0055 * _clamp(numLabProcedures - 35.0, -34, 85) -
        0.014 * _clamp(numberOutpatient.toDouble(), 0, 15);

    final hospital = _clamp(_sigmoid(z), 0.01, 0.85);
    final icu = _clamp(hospital * 1.12 + 0.018, 0.01, 0.92);
    final expiry = _clamp(hospital * 0.90 + 0.01, 0.01, 0.88);

    return MortalityScores(
      hospitalMortality: double.parse(hospital.toStringAsFixed(4)),
      icuMortality: double.parse(icu.toStringAsFixed(4)),
      inHospitalExpiry: double.parse(expiry.toStringAsFixed(4)),
    );
  }
}
