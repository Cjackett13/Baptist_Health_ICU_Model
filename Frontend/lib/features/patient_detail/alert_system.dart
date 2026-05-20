// Patient list card + full detail sheet.

import 'package:flutter/material.dart';

import '../../models/patient_prediction.dart';
import '../../screens/role_selection_screen.dart';
import '../../services/patient_summary_pdf.dart';
import '../../widgets/collapsible_section.dart';
import '../../widgets/prediction_cards.dart';

export '../../models/patient_prediction.dart';

// ─────────────────────────────────────────────────────────────────────────────
// PATIENT LIST CARD
// ─────────────────────────────────────────────────────────────────────────────
class PatientListCard extends StatelessWidget {
  const PatientListCard({
    required this.patient,
    required this.onTap,
    this.isClinician = true,
    super.key,
  });

  final PatientRecord patient;
  final VoidCallback onTap;
  final bool isClinician;

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
            if (isClinician) ...[
              Row(
                children: [
                  Expanded(
                    child: _RiskBar(
                      label: 'Mortality',
                      value: patient.predictions.mortalityRisk,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _RiskBar(
                      label: 'SCAI worsen (6h)',
                      value: patient.predictions.scaiDeterioration6hProb,
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
                    'SCAI ${patient.predictions.currentScaiStage}',
                    style: const TextStyle(
                        fontSize: 11, color: Colors.black45),
                  ),
                ],
              ),
            ] else ...[
              Row(
                children: [
                  const Icon(Icons.calendar_today_outlined,
                      size: 11, color: Colors.black38),
                  const SizedBox(width: 4),
                  Text(
                    'Predicted stay: ${patient.predictions.hospitalLosDays.toStringAsFixed(1)} days',
                    style: const TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: Colors.black54),
                  ),
                ],
              ),
            ],
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
void showPatientDetails(
  BuildContext context,
  PatientRecord patient, {
  required AppRole role,
  String? sessionPatientId,
}) {
  if (role == AppRole.patient &&
      sessionPatientId != null &&
      patient.id != sessionPatientId) {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('You can only view your own care record.'),
      ),
    );
    return;
  }

  showModalBottomSheet<void>(
    context: context,
    backgroundColor: const Color(0xFFF5F5F5),
    isScrollControlled: true,
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
    ),
    builder: (context) => _PatientDetailSheet(patient: patient, role: role),
  );
}

class _PatientDetailSheet extends StatelessWidget {
  const _PatientDetailSheet({required this.patient, required this.role});
  final PatientRecord patient;
  final AppRole role;

  bool get _isClinician => role == AppRole.clinician;

  @override
  Widget build(BuildContext context) {
    final screenHeight = MediaQuery.sizeOf(context).height;
    return SizedBox(
      height: screenHeight * 0.94,
      child: Column(
        children: [
          _DetailHeader(patient: patient, isClinician: _isClinician),
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 32),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: _isClinician
                    ? _clinicianSections()
                    : _patientSections(),
              ),
            ),
          ),
        ],
      ),
    );
  }

  List<Widget> _clinicianSections() => [
        CollapsibleProfileSection(
          title: 'Vitals & labs',
          subtitle: 'Latest values from clinical_event (monitor / lab)',
          icon: Icons.monitor_heart_outlined,
          child: VitalsSection(patient: patient),
        ),
        const SizedBox(height: 12),
        CollapsibleProfileSection(
          title: 'Predictions',
          subtitle:
              'SCAI, vasopressors, mortality, hospital LOS, MCS, VA-ECMO',
          icon: Icons.analytics_outlined,
          initiallyExpanded: true,
          child: ClinicalPredictionsSection(
            predictions: patient.predictions,
            mechanicalSupport: patient.mechanicalSupport,
          ),
        ),
        const SizedBox(height: 12),
        if (patient.diagnoses.isNotEmpty) ...[
          CollapsibleProfileSection(
            title: 'Diagnoses',
            subtitle: 'Principal and secondary (diagnosis table)',
            icon: Icons.medical_information_outlined,
            initiallyExpanded: false,
            child: DiagnosesSection(diagnoses: patient.diagnoses),
          ),
          const SizedBox(height: 12),
        ],
        CollapsibleProfileSection(
          title: 'Clinical details',
          subtitle: 'Encounter, unit, attending, record metadata',
          icon: Icons.description_outlined,
          initiallyExpanded: false,
          child: _ClinicalDetailsBody(patient: patient),
        ),
      ];

  List<Widget> _patientSections() => [
        CollapsibleProfileSection(
          title: 'Length of stay',
          subtitle: 'Predicted hospital stay duration',
          icon: Icons.calendar_today_outlined,
          child: PatientLengthOfStaySection(predictions: patient.predictions),
        ),
        const SizedBox(height: 12),
        CollapsibleProfileSection(
          title: 'Recommendations',
          subtitle: 'Guidance based on your vitals and diagnoses',
          icon: Icons.lightbulb_outline,
          child: PatientRecommendationsSection(
            recommendations: patient.recommendations,
          ),
        ),
        const SizedBox(height: 12),
        CollapsibleProfileSection(
          title: 'Prescribed medications',
          subtitle: 'Name, dosage, and route from your care team',
          icon: Icons.medication_outlined,
          child: PatientMedicationsSection(medications: patient.medications),
        ),
        const SizedBox(height: 12),
        CollapsibleProfileSection(
          title: 'Your record',
          subtitle: 'Room, doctor, and last model update',
          icon: Icons.badge_outlined,
          initiallyExpanded: false,
          child: _ClinicalDetailsBody(patient: patient, minimal: true),
        ),
      ];
}

class _DetailHeader extends StatelessWidget {
  const _DetailHeader({required this.patient, required this.isClinician});
  final PatientRecord patient;
  final bool isClinician;

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
                      '${patient.diagnosis != null ? '  ·  ${patient.diagnosis}' : ''}'
                      '${isClinician && patient.scaiStageCurrent != null ? '  ·  SCAI ${patient.scaiStageCurrent}' : ''}'
                      '${patient.mechanicalSupport.onMcs ? '  ·  ${patient.mechanicalSupport.summaryLabel}' : ''}',
                      style: const TextStyle(
                          fontSize: 12, color: Colors.black45),
                    ),
                  ],
                ),
              ),
              if (isClinician)
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
              if (isClinician) const SizedBox(width: 8),
              IconButton(
                tooltip: 'Export PDF summary for family',
                onPressed: () => _exportPdf(context),
                icon: const Icon(Icons.picture_as_pdf_outlined, size: 22),
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(),
              ),
              const SizedBox(width: 4),
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

  Future<void> _exportPdf(BuildContext context) async {
    try {
      await PatientSummaryPdf.share(
        patient,
        forFamily: !isClinician,
      );
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not export PDF: $e')),
        );
      }
    }
  }
}

class _ClinicalDetailsBody extends StatelessWidget {
  const _ClinicalDetailsBody({required this.patient, this.minimal = false});
  final PatientRecord patient;
  final bool minimal;

  @override
  Widget build(BuildContext context) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _row('Patient ID', patient.id),
          if (!minimal) ...[
            _row(
              'Mechanical support',
              patient.mechanicalSupport.summaryLabel,
            ),
            if (patient.mechanicalSupport.onMcs)
              for (final d in patient.mechanicalSupport.activeDevices)
                _row(
                  d.label,
                  d.start != null
                      ? '${d.start!.month}/${d.start!.day}'
                          '${d.end != null ? ' – ${d.end!.month}/${d.end!.day}' : ' – ongoing'}'
                      : 'Active at scoring hour',
                ),
          ],
          if (!minimal && patient.encounterId != null)
            _row('Encounter', patient.encounterId!),
          _row('Room', patient.roomNumber),
          if (!minimal && patient.unitCd != null) _row('Unit', patient.unitCd!),
          if (patient.primaryDoctor != null)
            _row('Doctor', patient.primaryDoctor!),
          if (!minimal && patient.daysAdmitted != null)
            _row('Days admitted', '${patient.daysAdmitted}'),
          if (!minimal && patient.issue != null && patient.issue!.isNotEmpty)
            _row('Principal diagnosis', patient.issue!),
          _row(
            'Last model update',
            '${patient.predictions.lastUpdated.month}/'
            '${patient.predictions.lastUpdated.day}/'
            '${patient.predictions.lastUpdated.year}',
          ),
        ],
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