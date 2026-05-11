// lib/alert_system.dart
//
// Patient list card + full detail sheet.
// All prediction data structures live in models/patient_prediction.dart.
// All prediction display widgets live in widgests/prediction_cards.dart.

import 'package:flutter/material.dart';
import 'models/patient_prediction.dart';
import 'widgests/prediction_cards.dart';
import 'screens/home_care_screen.dart';

export 'models/patient_prediction.dart';

// ─────────────────────────────────────────────────────────────────────────────
// PATIENT LIST CARD
// ─────────────────────────────────────────────────────────────────────────────
class PatientListCard extends StatelessWidget {
  const PatientListCard({
    required this.patient,
    required this.onTap,
    super.key,
  });

  final PatientRecord patient;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final condColor = conditionColor(patient.condition);

    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(18),
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(18),
          border: Border(
            left: BorderSide(color: condColor, width: 4),
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.05),
              blurRadius: 12,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 38,
                  height: 38,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: const Color(0xFFF2F2F2),
                    border: Border.all(color: Colors.black12),
                  ),
                  child: Center(
                    child: Text(
                      '${patient.rank}',
                      style: const TextStyle(
                          fontWeight: FontWeight.w700, fontSize: 13),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        patient.name,
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w700,
                          color: Colors.black,
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        _subtitle(patient),
                        style: const TextStyle(
                            fontSize: 12, color: Colors.black54),
                      ),
                    ],
                  ),
                ),
                _RiskBadge(label: patient.condition, color: condColor),
                const SizedBox(width: 4),
                const Icon(Icons.chevron_right_rounded,
                    color: Colors.black26),
              ],
            ),
            const SizedBox(height: 14),
            Row(
              children: [
                Expanded(
                  child: _RiskBar(
                    label: 'ICU transfer',
                    value: patient.predictions.icuTransferRisk,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _RiskBar(
                    label: 'Readmission',
                    value: patient.predictions.readmissionRisk,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                const Icon(Icons.calendar_today_outlined,
                    size: 11, color: Colors.black38),
                const SizedBox(width: 4),
                Text(
                  'Est. stay: ${patient.predictions.hospitalLosDays.toStringAsFixed(1)}d hospital  ·  '
                  '${patient.predictions.icuLosDays.toStringAsFixed(1)}d ICU',
                  style: const TextStyle(
                      fontSize: 11, color: Colors.black45),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  String _subtitle(PatientRecord p) {
    final parts = <String>[];
    if (p.gender != null && p.age != null) {
      parts.add('${p.gender}, ${p.age}');
    }
    if (p.diagnosis != null) parts.add(p.diagnosis!);
    if (p.daysAdmitted != null) parts.add('${p.daysAdmitted}d admitted');
    return parts.isNotEmpty ? parts.join(' · ') : 'ID ${p.id}';
  }
}

class _RiskBar extends StatelessWidget {
  const _RiskBar({required this.label, required this.value});
  final String label;
  final double value;

  @override
  Widget build(BuildContext context) {
    final color = riskColor(value);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label,
            style: const TextStyle(fontSize: 10, color: Colors.black45)),
        const SizedBox(height: 3),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: value,
            backgroundColor: const Color(0xFFEEEEEE),
            valueColor: AlwaysStoppedAnimation<Color>(color),
            minHeight: 6,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          '${(value * 100).round()}%',
          style: TextStyle(
              fontSize: 12, fontWeight: FontWeight.w600, color: color),
        ),
      ],
    );
  }
}

class _RiskBadge extends StatelessWidget {
  const _RiskBadge({required this.label, required this.color});
  final String label;
  final Color color;

  @override
  Widget build(BuildContext context) => Container(
        padding:
            const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(
          color: color.withOpacity(0.12),
          borderRadius: BorderRadius.circular(20),
        ),
        child: Text(
          label,
          style: TextStyle(
              fontSize: 11, fontWeight: FontWeight.w700, color: color),
        ),
      );
}

// ─────────────────────────────────────────────────────────────────────────────
// SHOW PATIENT DETAILS
// ─────────────────────────────────────────────────────────────────────────────
void showPatientDetails(BuildContext context, PatientRecord patient) {
  showModalBottomSheet<void>(
    context: context,
    backgroundColor: const Color(0xFFF5F5F5),
    isScrollControlled: true,
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
    ),
    builder: (context) => _PatientDetailSheet(patient: patient),
  );
}

class _PatientDetailSheet extends StatelessWidget {
  const _PatientDetailSheet({required this.patient});
  final PatientRecord patient;

  @override
  Widget build(BuildContext context) {
    final screenHeight = MediaQuery.sizeOf(context).height;
    return SizedBox(
      height: screenHeight * 0.94,
      child: Column(
        children: [
          _DetailHeader(patient: patient),
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 32),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  TransferAdmissionSection(
                      predictions: patient.predictions),
                  const SizedBox(height: 16),
                  LengthOfStaySection(predictions: patient.predictions),
                  const SizedBox(height: 16),
                  MortalityRiskSection(predictions: patient.predictions),
                  const SizedBox(height: 16),
                  ShapPanel(predictions: patient.predictions),
                  const SizedBox(height: 16),
                  WhatIfSimulator(patient: patient),
                  const SizedBox(height: 16),
                  _HomeCareCta(patient: patient),
                  const SizedBox(height: 16),
                  _ClinicalDetailsCard(patient: patient),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _DetailHeader extends StatelessWidget {
  const _DetailHeader({required this.patient});
  final PatientRecord patient;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      padding: const EdgeInsets.fromLTRB(20, 14, 20, 16),
      child: Column(
        children: [
          Container(
            width: 40,
            height: 4,
            decoration: BoxDecoration(
              color: Colors.black12,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              Container(
                width: 44,
                height: 44,
                decoration: const BoxDecoration(
                  shape: BoxShape.circle,
                  color: Color(0xFFF2F2F2),
                ),
                child: const Icon(Icons.person_outline,
                    color: Colors.black45, size: 24),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      patient.name,
                      style: const TextStyle(
                          fontSize: 18, fontWeight: FontWeight.w700),
                    ),
                    Text(
                      'Room ${patient.roomNumber}'
                      '${patient.gender != null && patient.age != null ? '  ·  ${patient.gender}, ${patient.age}' : ''}'
                      '${patient.diagnosis != null ? '  ·  ${patient.diagnosis}' : ''}',
                      style: const TextStyle(
                          fontSize: 12, color: Colors.black45),
                    ),
                  ],
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 10, vertical: 5),
                decoration: BoxDecoration(
                  color: conditionColor(patient.condition)
                      .withOpacity(0.12),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  patient.condition,
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    color: conditionColor(patient.condition),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              IconButton(
                onPressed: () => Navigator.pop(context),
                icon: const Icon(Icons.close),
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _HomeCareCta extends StatelessWidget {
  const _HomeCareCta({required this.patient});
  final PatientRecord patient;

  @override
  Widget build(BuildContext context) => GestureDetector(
        onTap: () => Navigator.push(
          context,
          MaterialPageRoute<void>(
            builder: (_) => HomeCareScreen(patient: patient),
          ),
        ),
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: const Color(0xFF222831),
            borderRadius: BorderRadius.circular(16),
          ),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: const Color(0xFF4A9E6A).withOpacity(0.2),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.home_outlined,
                    color: Color(0xFF4A9E6A), size: 22),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'View home care plan',
                      style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                          color: Colors.white),
                    ),
                    Text(
                      'AI-generated recovery instructions for family',
                      style: TextStyle(
                          fontSize: 12,
                          color: Colors.white.withOpacity(0.6)),
                    ),
                  ],
                ),
              ),
              const Icon(Icons.arrow_forward_ios_rounded,
                  color: Color(0xFF4A9E6A), size: 16),
            ],
          ),
        ),
      );
}

class _ClinicalDetailsCard extends StatelessWidget {
  const _ClinicalDetailsCard({required this.patient});
  final PatientRecord patient;

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.black.withOpacity(0.07)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Clinical details',
              style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  color: Colors.black87),
            ),
            const SizedBox(height: 12),
            _row('Patient ID', patient.id),
            _row('Room', patient.roomNumber),
            if (patient.primaryDoctor != null)
              _row('Doctor', patient.primaryDoctor!),
            if (patient.daysAdmitted != null)
              _row('Days admitted', '${patient.daysAdmitted}'),
            if (patient.issue != null && patient.issue!.isNotEmpty)
              _row('Issue', patient.issue!),
            _row(
              'Last AI update',
              '${patient.predictions.lastUpdated.month}/'
              '${patient.predictions.lastUpdated.day}/'
              '${patient.predictions.lastUpdated.year}',
            ),
          ],
        ),
      );

  Widget _row(String label, String value) => Padding(
        padding: const EdgeInsets.only(bottom: 8),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              width: 120,
              child: Text(label,
                  style: const TextStyle(
                      fontSize: 13, color: Colors.black45)),
            ),
            Expanded(
              child: Text(value,
                  style: const TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                    color: Colors.black87,
                  )),
            ),
          ],
        ),
      );
}