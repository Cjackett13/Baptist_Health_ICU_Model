// lib/widgets/prediction_cards.dart
//
// All prediction display components used in the patient detail screen.
// Each section is self-contained — plug real data in without touching layout.

import 'package:flutter/material.dart';
import '../models/patient_prediction.dart';

// ─────────────────────────────────────────────────────────────────────────────
// SECTION HEADER
// Consistent header used above each prediction group.
// ─────────────────────────────────────────────────────────────────────────────
class PredictionSectionHeader extends StatelessWidget {
  const PredictionSectionHeader({
    required this.title,
    required this.subtitle,
    required this.icon,
    super.key,
  });

  final String title;
  final String subtitle;
  final IconData icon;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: const Color(0xFF222831),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(icon, color: Colors.white, size: 16),
          ),
          const SizedBox(width: 10),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                  color: Colors.black87,
                ),
              ),
              Text(
                subtitle,
                style: const TextStyle(
                  fontSize: 11,
                  color: Colors.black45,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// PREDICTION CARD SHELL
// White rounded container used by all prediction sections.
// ─────────────────────────────────────────────────────────────────────────────
class _PredictionCard extends StatelessWidget {
  const _PredictionCard({required this.child, this.padding});

  final Widget child;
  final EdgeInsetsGeometry? padding;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: padding ?? const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.black.withOpacity(0.07)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.03),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: child,
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// RISK GAUGE TILE
// Used inside prediction cards for a single % risk value.
// ─────────────────────────────────────────────────────────────────────────────
class RiskGaugeTile extends StatelessWidget {
  const RiskGaugeTile({
    required this.label,
    required this.value,
    this.description,
    super.key,
  });

  final String label;
  final double value;
  final String? description;

  @override
  Widget build(BuildContext context) {
    final color = riskColor(value);
    final pct = (value * 100).round();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(fontSize: 11, color: Colors.black45),
        ),
        const SizedBox(height: 6),
        Row(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text(
              '$pct%',
              style: TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.w700,
                color: color,
                height: 1,
              ),
            ),
            const SizedBox(width: 8),
            Padding(
              padding: const EdgeInsets.only(bottom: 3),
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  riskLabel(value),
                  style: TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                    color: color,
                  ),
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: value,
            backgroundColor: const Color(0xFFEEEEEE),
            valueColor: AlwaysStoppedAnimation<Color>(color),
            minHeight: 6,
          ),
        ),
        if (description != null) ...[
          const SizedBox(height: 5),
          Text(
            description!,
            style: const TextStyle(fontSize: 10, color: Colors.black38),
          ),
        ],
      ],
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 1 — TRANSFER & ADMISSION
// Prediction 1: Readmission risk
// Prediction 7: ICU transfer risk
// ─────────────────────────────────────────────────────────────────────────────
class TransferAdmissionSection extends StatelessWidget {
  const TransferAdmissionSection({
    required this.predictions,
    super.key,
  });

  final PatientPredictions predictions;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const PredictionSectionHeader(
          title: 'Transfer & Admission',
          subtitle: 'ICU transfer probability and readmission risk',
          icon: Icons.swap_horiz_rounded,
        ),
        _PredictionCard(
          child: Column(
            children: [
              RiskGaugeTile(
                label: 'ICU transfer risk (24-hour window)',
                value: predictions.icuTransferRisk,
                description:
                    'Probability patient requires ICU escalation within 24 hrs',
              ),
              const SizedBox(height: 18),
              const Divider(height: 1, color: Color(0xFFF0F0F0)),
              const SizedBox(height: 18),
              RiskGaugeTile(
                label: '30-day readmission risk',
                value: predictions.readmissionRisk,
                description:
                    'Probability of ER or hospital readmission within 30 days post-discharge',
              ),
            ],
          ),
        ),
      ],
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 2 — LENGTH OF STAY
// Prediction 2: Hospital LOS + ICU LOS (regression outputs — days not %)
// ─────────────────────────────────────────────────────────────────────────────
class LengthOfStaySection extends StatelessWidget {
  const LengthOfStaySection({
    required this.predictions,
    super.key,
  });

  final PatientPredictions predictions;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const PredictionSectionHeader(
          title: 'Length of Stay',
          subtitle: 'Predicted hospital and ICU stay duration',
          icon: Icons.calendar_today_outlined,
        ),
        _PredictionCard(
          child: Row(
            children: [
              Expanded(
                child: _LosTile(
                  label: 'Hospital stay',
                  days: predictions.hospitalLosDays,
                  description: 'Total predicted\nhospital duration',
                ),
              ),
              Container(
                width: 1,
                height: 80,
                color: const Color(0xFFF0F0F0),
                margin: const EdgeInsets.symmetric(horizontal: 16),
              ),
              Expanded(
                child: _LosTile(
                  label: 'ICU stay',
                  days: predictions.icuLosDays,
                  description: 'Predicted time\nin ICU specifically',
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

/// LOS tiles only — for collapsible profile sections.
class LengthOfStayBody extends StatelessWidget {
  const LengthOfStayBody({required this.predictions, super.key});
  final PatientPredictions predictions;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: _LosTile(
            label: 'Hospital stay',
            days: predictions.hospitalLosDays,
            description: 'Total predicted\nhospital duration',
          ),
        ),
        Container(
          width: 1,
          height: 80,
          color: const Color(0xFFF0F0F0),
          margin: const EdgeInsets.symmetric(horizontal: 16),
        ),
        Expanded(
          child: _LosTile(
            label: 'ICU stay',
            days: predictions.icuLosDays,
            description: 'Predicted time\nin ICU specifically',
          ),
        ),
      ],
    );
  }
}

class _LosTile extends StatelessWidget {
  const _LosTile({
    required this.label,
    required this.days,
    required this.description,
  });

  final String label;
  final double days;
  final String description;

  @override
  Widget build(BuildContext context) {
    final color = losColor(days);
    final whole = days.floor();
    final fraction = ((days - whole) * 10).round();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(fontSize: 11, color: Colors.black45),
        ),
        const SizedBox(height: 8),
        Row(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text(
              '$whole',
              style: TextStyle(
                fontSize: 34,
                fontWeight: FontWeight.w700,
                color: color,
                height: 1,
              ),
            ),
            Text(
              '.$fraction',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w500,
                color: color.withOpacity(0.7),
                height: 1.4,
              ),
            ),
            const SizedBox(width: 4),
            const Padding(
              padding: EdgeInsets.only(bottom: 3),
              child: Text(
                'days',
                style: TextStyle(
                  fontSize: 13,
                  color: Colors.black45,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),
        Text(
          description,
          style: const TextStyle(fontSize: 10, color: Colors.black38),
        ),
      ],
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 3 — MORTALITY RISK (grouped)
// Prediction 4: Hospital mortality
// Prediction 5: ICU mortality
// Prediction 6: In-hospital expiry
// ─────────────────────────────────────────────────────────────────────────────
class MortalityRiskSection extends StatelessWidget {
  const MortalityRiskSection({
    required this.predictions,
    super.key,
  });

  final PatientPredictions predictions;

  @override
  Widget build(BuildContext context) {
    final peak = predictions.peakMortality;
    final peakColor = riskColor(peak);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const PredictionSectionHeader(
          title: 'Mortality Risk',
          subtitle:
              'Hospital, ICU, and in-hospital expiry predictions',
          icon: Icons.monitor_heart_outlined,
        ),
        _PredictionCard(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Peak indicator
              Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 12, vertical: 8),
                decoration: BoxDecoration(
                  color: peakColor.withOpacity(0.08),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(
                      color: peakColor.withOpacity(0.25)),
                ),
                child: Row(
                  children: [
                    Icon(Icons.warning_amber_rounded,
                        color: peakColor, size: 16),
                    const SizedBox(width: 6),
                    Text(
                      'Peak mortality risk: ${(peak * 100).round()}%',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                        color: peakColor,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),

              // Three mortality predictions
              _MortalityRow(
                label: 'Hospital mortality',
                value: predictions.hospitalMortality,
                description: 'Overall in-hospital death risk',
              ),
              const SizedBox(height: 14),
              _MortalityRow(
                label: 'ICU mortality',
                value: predictions.icuMortality,
                description: 'Death risk specific to ICU stay',
              ),
              const SizedBox(height: 14),
              _MortalityRow(
                label: 'In-hospital expiry',
                value: predictions.inHospitalExpiry,
                description: 'Broader expiry estimate during admission',
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _MortalityRow extends StatelessWidget {
  const _MortalityRow({
    required this.label,
    required this.value,
    required this.description,
  });

  final String label;
  final double value;
  final String description;

  @override
  Widget build(BuildContext context) {
    final color = riskColor(value);
    final pct = (value * 100).round();

    return Row(
      children: [
        SizedBox(
          width: 48,
          child: Text(
            '$pct%',
            style: TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.w700,
              color: color,
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: Colors.black87,
                ),
              ),
              const SizedBox(height: 2),
              ClipRRect(
                borderRadius: BorderRadius.circular(3),
                child: LinearProgressIndicator(
                  value: value,
                  backgroundColor: const Color(0xFFEEEEEE),
                  valueColor:
                      AlwaysStoppedAnimation<Color>(color),
                  minHeight: 5,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                description,
                style: const TextStyle(
                    fontSize: 10, color: Colors.black38),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SHAP PANEL
// Expandable — shows drivers for a selected prediction group.
// ─────────────────────────────────────────────────────────────────────────────
class ShapPanel extends StatefulWidget {
  const ShapPanel({
    required this.predictions,
    super.key,
  });

  final PatientPredictions predictions;

  @override
  State<ShapPanel> createState() => _ShapPanelState();
}

class _ShapPanelState extends State<ShapPanel> {
  int _selected = 0; // 0=transfer, 1=readmission, 2=mortality

  static const _tabs = ['ICU Transfer', 'Readmission', 'Mortality'];

  List<ShapValue> get _activeShap {
    switch (_selected) {
      case 1:
        return widget.predictions.shapReadmission;
      case 2:
        return widget.predictions.shapMortality;
      default:
        return widget.predictions.shapTransfer;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const PredictionSectionHeader(
          title: 'Why these predictions?',
          subtitle:
              'Top factors driving each risk score (SHAP values)',
          icon: Icons.insights_outlined,
        ),
        _PredictionCard(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Tab selector
              Container(
                decoration: BoxDecoration(
                  color: const Color(0xFFF2F2F2),
                  borderRadius: BorderRadius.circular(10),
                ),
                padding: const EdgeInsets.all(3),
                child: Row(
                  children: List.generate(
                    _tabs.length,
                    (i) => Expanded(
                      child: GestureDetector(
                        onTap: () =>
                            setState(() => _selected = i),
                        child: AnimatedContainer(
                          duration:
                              const Duration(milliseconds: 180),
                          padding: const EdgeInsets.symmetric(
                              vertical: 7),
                          decoration: BoxDecoration(
                            color: _selected == i
                                ? Colors.white
                                : Colors.transparent,
                            borderRadius:
                                BorderRadius.circular(8),
                            boxShadow: _selected == i
                                ? [
                                    BoxShadow(
                                      color: Colors.black
                                          .withOpacity(0.08),
                                      blurRadius: 4,
                                      offset:
                                          const Offset(0, 1),
                                    )
                                  ]
                                : null,
                          ),
                          child: Text(
                            _tabs[i],
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              fontSize: 11,
                              fontWeight: _selected == i
                                  ? FontWeight.w700
                                  : FontWeight.w400,
                              color: _selected == i
                                  ? Colors.black87
                                  : Colors.black45,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 14),

              // SHAP rows
              ..._activeShap.map((s) => _ShapRow(shap: s)),

              const SizedBox(height: 8),
              Row(
                children: [
                  _LegendDot(
                    color: const Color(0xFFE05A5A),
                    label: 'Increases risk',
                  ),
                  const SizedBox(width: 16),
                  _LegendDot(
                    color: const Color(0xFF4A9E6A),
                    label: 'Decreases risk',
                  ),
                ],
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _ShapRow extends StatelessWidget {
  const _ShapRow({required this.shap});
  final ShapValue shap;

  @override
  Widget build(BuildContext context) {
    final color = shap.isPositive
        ? const Color(0xFFE05A5A)
        : const Color(0xFF4A9E6A);
    final barWidth =
        (shap.value.abs() / 0.5).clamp(0.05, 1.0);
    final sign = shap.isPositive ? '+' : '−';

    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          SizedBox(
            width: 140,
            child: Text(
              shap.feature,
              style: const TextStyle(
                  fontSize: 12, color: Colors.black87),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(3),
              child: LinearProgressIndicator(
                value: barWidth,
                backgroundColor: const Color(0xFFEEEEEE),
                valueColor:
                    AlwaysStoppedAnimation<Color>(color),
                minHeight: 8,
              ),
            ),
          ),
          const SizedBox(width: 8),
          SizedBox(
            width: 44,
            child: Text(
              '$sign${shap.value.toStringAsFixed(2)}',
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w600,
                color: color,
              ),
              textAlign: TextAlign.right,
            ),
          ),
        ],
      ),
    );
  }
}

class _LegendDot extends StatelessWidget {
  const _LegendDot({required this.color, required this.label});
  final Color color;
  final String label;

  @override
  Widget build(BuildContext context) => Row(
        children: [
          Container(
            width: 8,
            height: 8,
            decoration:
                BoxDecoration(shape: BoxShape.circle, color: color),
          ),
          const SizedBox(width: 5),
          Text(label,
              style: const TextStyle(
                  fontSize: 10, color: Colors.black45)),
        ],
      );
}

// ─────────────────────────────────────────────────────────────────────────────
// WHAT-IF SIMULATOR
// Sliders update all 6 risk predictions simultaneously.
// Replace _recalcAll() body with real FastAPI call later.
// ─────────────────────────────────────────────────────────────────────────────
class WhatIfSimulator extends StatefulWidget {
  const WhatIfSimulator({
    required this.patient,
    super.key,
  });

  final PatientRecord patient;

  @override
  State<WhatIfSimulator> createState() => _WhatIfSimulatorState();
}

class _WhatIfSimulatorState extends State<WhatIfSimulator> {
  late PatientFeatures _features;
  late double _simTransfer;
  late double _simReadmission;
  late double _simMortality;
  late double _simHospLos;
  late double _simIcuLos;

  @override
  void initState() {
    super.initState();
    _features = widget.patient.features;
    _syncFromPredictions(widget.patient.predictions);
  }

  void _syncFromPredictions(PatientPredictions p) {
    _simTransfer = p.icuTransferRisk;
    _simReadmission = p.readmissionRisk;
    _simMortality = p.peakMortality;
    _simHospLos = p.hospitalLosDays;
    _simIcuLos = p.icuLosDays;
  }

  // Heuristic simulation — replaced by real FastAPI /predict call later.
  void _recalcAll() {
    final base = widget.patient.predictions;
    final orig = widget.patient.features;

    final medDelta =
        (_features.numMedications - orig.numMedications) * 0.008;
    final inpDelta =
        (_features.numberInpatient - orig.numberInpatient) * 0.035;
    final labDelta =
        (_features.numLabProcedures - orig.numLabProcedures) * 0.002;
    final hosDelta =
        (_features.timeInHospital - orig.timeInHospital) * 0.015;
    final dxDelta =
        (_features.numberDiagnoses - orig.numberDiagnoses) * 0.012;
    final emDelta =
        (_features.numberEmergency - orig.numberEmergency) * 0.025;
    final totalDelta =
        medDelta + inpDelta + labDelta + hosDelta + dxDelta + emDelta;

    _simTransfer =
        (base.icuTransferRisk + totalDelta).clamp(0.02, 0.99);
    _simReadmission =
        (base.readmissionRisk + totalDelta * 0.7).clamp(0.02, 0.99);
    _simMortality =
        (base.peakMortality + totalDelta * 0.5).clamp(0.02, 0.99);
    _simHospLos =
        (base.hospitalLosDays + hosDelta * 8 + medDelta * 3)
            .clamp(0.5, 21.0);
    _simIcuLos =
        (_simHospLos * 0.45 + inpDelta * 0.5).clamp(0.2, _simHospLos);
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const PredictionSectionHeader(
          title: 'What-if Simulator',
          subtitle:
              'Adjust values to see how all predictions change',
          icon: Icons.tune_outlined,
        ),
        _PredictionCard(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Sliders
              _SimSlider(
                label: 'Medications',
                value: _features.numMedications.toDouble(),
                min: 1,
                max: 40,
                onChanged: (v) => setState(() {
                  _features = _features.copyWith(
                      numMedications: v.round());
                  _recalcAll();
                }),
              ),
              _SimSlider(
                label: 'Prior inpatient',
                value: _features.numberInpatient.toDouble(),
                min: 0,
                max: 10,
                onChanged: (v) => setState(() {
                  _features = _features.copyWith(
                      numberInpatient: v.round());
                  _recalcAll();
                }),
              ),
              _SimSlider(
                label: 'Lab procedures',
                value: _features.numLabProcedures.toDouble(),
                min: 1,
                max: 120,
                onChanged: (v) => setState(() {
                  _features = _features.copyWith(
                      numLabProcedures: v.round());
                  _recalcAll();
                }),
              ),
              _SimSlider(
                label: 'Days in hospital',
                value: _features.timeInHospital.toDouble(),
                min: 1,
                max: 14,
                onChanged: (v) => setState(() {
                  _features = _features.copyWith(
                      timeInHospital: v.round());
                  _recalcAll();
                }),
              ),
              _SimSlider(
                label: 'No. of diagnoses',
                value: _features.numberDiagnoses.toDouble(),
                min: 1,
                max: 16,
                onChanged: (v) => setState(() {
                  _features = _features.copyWith(
                      numberDiagnoses: v.round());
                  _recalcAll();
                }),
              ),

              const SizedBox(height: 12),
              const Divider(height: 1, color: Color(0xFFF0F0F0)),
              const SizedBox(height: 12),

              // Simulated results grid
              const Text(
                'Simulated predictions',
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  color: Colors.black45,
                ),
              ),
              const SizedBox(height: 10),
              _SimResultGrid(
                transfer: _simTransfer,
                readmission: _simReadmission,
                mortality: _simMortality,
                hospLos: _simHospLos,
                icuLos: _simIcuLos,
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _SimResultGrid extends StatelessWidget {
  const _SimResultGrid({
    required this.transfer,
    required this.readmission,
    required this.mortality,
    required this.hospLos,
    required this.icuLos,
  });

  final double transfer;
  final double readmission;
  final double mortality;
  final double hospLos;
  final double icuLos;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          children: [
            Expanded(
                child: _SimResultTile(
                    label: 'ICU transfer',
                    value: '${(transfer * 100).round()}%',
                    color: riskColor(transfer))),
            const SizedBox(width: 8),
            Expanded(
                child: _SimResultTile(
                    label: 'Readmission',
                    value: '${(readmission * 100).round()}%',
                    color: riskColor(readmission))),
            const SizedBox(width: 8),
            Expanded(
                child: _SimResultTile(
                    label: 'Mortality',
                    value: '${(mortality * 100).round()}%',
                    color: riskColor(mortality))),
          ],
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
                child: _SimResultTile(
                    label: 'Hospital LOS',
                    value: '${hospLos.toStringAsFixed(1)}d',
                    color: losColor(hospLos))),
            const SizedBox(width: 8),
            Expanded(
                child: _SimResultTile(
                    label: 'ICU LOS',
                    value: '${icuLos.toStringAsFixed(1)}d',
                    color: losColor(icuLos))),
            const SizedBox(width: 8),
            const Expanded(child: SizedBox()),
          ],
        ),
      ],
    );
  }
}

class _SimResultTile extends StatelessWidget {
  const _SimResultTile({
    required this.label,
    required this.value,
    required this.color,
  });

  final String label;
  final String value;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding:
          const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F5F5),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: const TextStyle(
                fontSize: 10, color: Colors.black45),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w700,
              color: color,
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// VITALS (from clinical_event parquet columns)
// ─────────────────────────────────────────────────────────────────────────────
class VitalsSection extends StatelessWidget {
  const VitalsSection({required this.patient, super.key});
  final PatientRecord patient;

  static const _vitalOrder = [
    'HR',
    'SBP',
    'DBP',
    'MAP',
    'RR',
    'SPO2',
    'TEMP',
    'URINE_OUT_HR',
    'CVP',
    'CO',
    'CI',
    'LACTATE',
    'CREATININE',
    'BUN',
    'NT_PROBNP',
    'TROPONIN_I',
  ];

  @override
  Widget build(BuildContext context) {
    final sorted = [...patient.clinicalVitals];
    sorted.sort((a, b) {
      final ai = _vitalOrder.indexOf(a.code);
      final bi = _vitalOrder.indexOf(b.code);
      return (ai == -1 ? 999 : ai).compareTo(bi == -1 ? 999 : bi);
    });

    if (sorted.isEmpty) {
      return const Text(
        'No vitals on file for this encounter.',
        style: TextStyle(fontSize: 13, color: Colors.black45),
      );
    }
    return Wrap(
      spacing: 10,
      runSpacing: 10,
      children: sorted.map((v) => _VitalChip(vital: v)).toList(),
    );
  }
}

class _VitalChip extends StatelessWidget {
  const _VitalChip({required this.vital});
  final PatientVital vital;

  Color get _normalcyColor {
    switch (vital.normalcy.toUpperCase()) {
      case 'HIGH':
      case 'LOW':
      case 'CRITICAL':
        return const Color(0xFFE05A5A);
      default:
        return const Color(0xFF4A9E6A);
    }
  }

  @override
  Widget build(BuildContext context) {
    final units = vital.units.isNotEmpty ? ' ${vital.units}' : '';
    return Container(
      width: 160,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFF8F8F8),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: _normalcyColor.withOpacity(0.25)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            vital.title,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(fontSize: 10, color: Colors.black45),
          ),
          const SizedBox(height: 4),
          Text(
            '${vital.value}$units',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w700,
              color: _normalcyColor,
            ),
          ),
          Text(
            vital.normalcy,
            style: TextStyle(fontSize: 9, color: _normalcyColor),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CLINICIAN — ML PREDICTIONS (cardiogenic shock model outputs)
// ─────────────────────────────────────────────────────────────────────────────
class ClinicalPredictionsSection extends StatelessWidget {
  const ClinicalPredictionsSection({
    required this.predictions,
    super.key,
  });
  final PatientPredictions predictions;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        _PredictionRow(
                label: 'SCAI stage deterioration (6h)',
                subtitle:
                    'Current stage ${predictions.currentScaiStage} · ${predictions.scaiDeterioration6hLabel}',
                value: predictions.scaiDeterioration6hProb,
                trailing: predictions.scaiDeterioration6hLabel,
              ),
              const SizedBox(height: 14),
              _PredictionRow(
                label: 'Vasopressor need',
                subtitle:
                    'Predicted count: ${predictions.predictedVasopressorCount}',
                value: predictions.vasopressorProbability,
                trailing: '${predictions.predictedVasopressorCount} agents',
              ),
              const SizedBox(height: 14),
              _PredictionRow(
                label: 'Mortality risk',
                subtitle: 'In-hospital mortality probability',
                value: predictions.mortalityRisk,
              ),
              const SizedBox(height: 14),
              _LosPredictionRow(
                label: 'Length of stay (hospital)',
                days: predictions.hospitalLosDays,
              ),
              const SizedBox(height: 10),
              _LosPredictionRow(
                label: 'Length of stay (ICU)',
                days: predictions.icuLosDays,
              ),
              const SizedBox(height: 14),
              _BinaryPredictionRow(
                label: 'MCS within 12 hours',
                probability: predictions.mcs12hProbability,
                needed: predictions.mcs12hNeeded,
              ),
              const SizedBox(height: 14),
        _BinaryPredictionRow(
          label: 'VA-ECMO within 12 hours',
          probability: predictions.vaEcmo12hProbability,
          needed: predictions.vaEcmo12hNeeded,
        ),
      ],
    );
  }
}

class _PredictionRow extends StatelessWidget {
  const _PredictionRow({
    required this.label,
    required this.subtitle,
    required this.value,
    this.trailing,
  });

  final String label;
  final String subtitle;
  final double value;
  final String? trailing;

  @override
  Widget build(BuildContext context) {
    final color = riskColor(value);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(
                label,
                style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: Colors.black87,
                ),
              ),
            ),
            Text(
              '${(value * 100).round()}%',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w700,
                color: color,
              ),
            ),
          ],
        ),
        const SizedBox(height: 2),
        Text(subtitle,
            style: const TextStyle(fontSize: 10, color: Colors.black38)),
        if (trailing != null) ...[
          const SizedBox(height: 2),
          Text(trailing!,
              style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  color: color)),
        ],
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(3),
          child: LinearProgressIndicator(
            value: value,
            backgroundColor: const Color(0xFFEEEEEE),
            valueColor: AlwaysStoppedAnimation<Color>(color),
            minHeight: 5,
          ),
        ),
      ],
    );
  }
}

class _LosPredictionRow extends StatelessWidget {
  const _LosPredictionRow({required this.label, required this.days});
  final String label;
  final double days;

  @override
  Widget build(BuildContext context) {
    final color = losColor(days);
    return Row(
      children: [
        Expanded(
          child: Text(label,
              style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: Colors.black87)),
        ),
        Text(
          '${days.toStringAsFixed(1)} days',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w700,
            color: color,
          ),
        ),
      ],
    );
  }
}

class _BinaryPredictionRow extends StatelessWidget {
  const _BinaryPredictionRow({
    required this.label,
    required this.probability,
    required this.needed,
  });

  final String label;
  final double probability;
  final bool needed;

  @override
  Widget build(BuildContext context) {
    final color = riskColor(probability);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(
                label,
                style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: Colors.black87,
                ),
              ),
            ),
            Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: (needed ? color : const Color(0xFF4A9E6A))
                    .withOpacity(0.12),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                needed ? 'Likely needed' : 'Unlikely',
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  color: needed ? color : const Color(0xFF4A9E6A),
                ),
              ),
            ),
            const SizedBox(width: 8),
            Text(
              '${(probability * 100).round()}%',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w700,
                color: color,
              ),
            ),
          ],
        ),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(3),
          child: LinearProgressIndicator(
            value: probability,
            backgroundColor: const Color(0xFFEEEEEE),
            valueColor: AlwaysStoppedAnimation<Color>(color),
            minHeight: 5,
          ),
        ),
      ],
    );
  }
}

class DiagnosesSection extends StatelessWidget {
  const DiagnosesSection({required this.diagnoses, super.key});
  final List<PatientDiagnosis> diagnoses;

  @override
  Widget build(BuildContext context) {
    if (diagnoses.isEmpty) {
      return const Text(
        'No diagnoses on file.',
        style: TextStyle(fontSize: 13, color: Colors.black45),
      );
    }
    return Column(
      children: diagnoses.map((d) {
        return Padding(
          padding: const EdgeInsets.only(bottom: 10),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: const Color(0xFFF2F2F2),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Text(
                  d.code,
                  style: const TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  d.text,
                  style: const TextStyle(
                    fontSize: 13,
                    color: Colors.black87,
                  ),
                ),
              ),
            ],
          ),
        );
      }).toList(),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// PATIENT PORTAL SECTIONS
// ─────────────────────────────────────────────────────────────────────────────
class PatientLengthOfStaySection extends StatelessWidget {
  const PatientLengthOfStaySection({required this.predictions, super.key});
  final PatientPredictions predictions;

  @override
  Widget build(BuildContext context) =>
      LengthOfStayBody(predictions: predictions);
}

class PatientRecommendationsSection extends StatelessWidget {
  const PatientRecommendationsSection({
    required this.recommendations,
    super.key,
  });
  final List<HomeCareSuggestion> recommendations;

  @override
  Widget build(BuildContext context) {
    if (recommendations.isEmpty) {
      return const Text(
        'No recommendations at this time.',
        style: TextStyle(fontSize: 13, color: Colors.black45),
      );
    }
    return Column(
      children: recommendations
          .map(
            (s) => Padding(
              padding: const EdgeInsets.only(bottom: 10),
              child: _PredictionCard(
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(categoryIcon(s.category),
                        style: const TextStyle(fontSize: 24)),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            s.title,
                            style: const TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            s.description,
                            style: const TextStyle(
                              fontSize: 12,
                              color: Colors.black54,
                              height: 1.4,
                            ),
                          ),
                          const SizedBox(height: 6),
                          Text(
                            s.impact,
                            style: const TextStyle(
                              fontSize: 11,
                              fontWeight: FontWeight.w600,
                              color: Color(0xFF4A9E6A),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          )
          .toList(),
    );
  }
}

class PatientMedicationsSection extends StatelessWidget {
  const PatientMedicationsSection({required this.medications, super.key});
  final List<PatientMedication> medications;

  @override
  Widget build(BuildContext context) {
    if (medications.isEmpty) {
      return const Text(
        'No medications on file.',
        style: TextStyle(fontSize: 13, color: Colors.black45),
      );
    }
    return Column(
      children: medications.map((m) {
        return Padding(
          padding: const EdgeInsets.only(bottom: 12),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(
                m.isVasopressor
                    ? Icons.bolt_outlined
                    : Icons.medication_liquid_outlined,
                size: 20,
                color: m.isVasopressor
                    ? const Color(0xFFE05A5A)
                    : Colors.black38,
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      m.name,
                      style: const TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    if (m.dosage.isNotEmpty)
                      Text(
                        'Dosage: ${m.dosage}',
                        style: const TextStyle(
                          fontSize: 12,
                          color: Colors.black54,
                        ),
                      ),
                    Text(
                      'Route: ${m.route}',
                      style: const TextStyle(
                        fontSize: 11,
                        color: Colors.black38,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        );
      }).toList(),
    );
  }
}

class _SimSlider extends StatelessWidget {
  const _SimSlider({
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    required this.onChanged,
  });

  final String label;
  final double value;
  final double min;
  final double max;
  final ValueChanged<double> onChanged;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          SizedBox(
            width: 110,
            child: Text(
              label,
              style: const TextStyle(
                  fontSize: 12, color: Colors.black87),
            ),
          ),
          Expanded(
            child: SliderTheme(
              data: SliderTheme.of(context).copyWith(
                activeTrackColor: const Color(0xFF222831),
                inactiveTrackColor: const Color(0xFFDDDDDD),
                thumbColor: const Color(0xFF222831),
                overlayColor: const Color(0x22222831),
                trackHeight: 3,
                thumbShape: const RoundSliderThumbShape(
                    enabledThumbRadius: 7),
              ),
              child: Slider(
                value: value.clamp(min, max),
                min: min,
                max: max,
                divisions: (max - min).round(),
                onChanged: onChanged,
              ),
            ),
          ),
          SizedBox(
            width: 28,
            child: Text(
              value.round().toString(),
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: Colors.black87,
              ),
              textAlign: TextAlign.right,
            ),
          ),
        ],
      ),
    );
  }
}