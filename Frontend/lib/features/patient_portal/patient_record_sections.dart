import 'package:flutter/material.dart';

import '../../models/patient_prediction.dart';
import '../../widgets/collapsible_section.dart';
import '../../widgets/prediction_cards.dart';

/// Family-facing sections shown on the patient home screen.
class PatientRecordSections extends StatelessWidget {
  const PatientRecordSections({required this.patient, super.key});

  final PatientRecord patient;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        CollapsibleProfileSection(
          title: 'Length of stay',
          subtitle: 'Predicted hospital stay duration',
          icon: Icons.calendar_today_outlined,
          initiallyExpanded: true,
          child: PatientLengthOfStaySection(predictions: patient.predictions),
        ),
        const SizedBox(height: 12),
        CollapsibleProfileSection(
          title: 'Recommendations',
          subtitle: 'Guidance from your care team',
          icon: Icons.lightbulb_outline,
          initiallyExpanded: true,
          child: PatientRecommendationsSection(
            recommendations: patient.recommendations,
          ),
        ),
        const SizedBox(height: 12),
        CollapsibleProfileSection(
          title: 'Prescribed medications',
          subtitle: 'Name, dosage, and route from your care team',
          icon: Icons.medication_outlined,
          initiallyExpanded: true,
          child: PatientMedicationsSection(medications: patient.medications),
        ),
        const SizedBox(height: 12),
        CollapsibleProfileSection(
          title: 'Your care team & room',
          subtitle: 'Who is caring for you at this hospital',
          icon: Icons.badge_outlined,
          initiallyExpanded: false,
          child: _PatientCareTeamBody(patient: patient),
        ),
      ],
    );
  }
}

class _PatientCareTeamBody extends StatelessWidget {
  const _PatientCareTeamBody({required this.patient});
  final PatientRecord patient;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _row('Room', patient.roomNumber),
        if (patient.primaryDoctor != null) _row('Doctor', patient.primaryDoctor!),
        if (patient.diagnosis != null) _row('Diagnosis', patient.diagnosis!),
        if (patient.daysAdmitted != null)
          _row('Days in hospital', '${patient.daysAdmitted}'),
        _row('Patient ID', patient.id),
      ],
    );
  }

  Widget _row(String label, String value) => Padding(
        padding: const EdgeInsets.only(bottom: 8),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              width: 110,
              child: Text(
                label,
                style: const TextStyle(fontSize: 13, color: Colors.black45),
              ),
            ),
            Expanded(
              child: Text(
                value,
                style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w500,
                  color: Colors.black87,
                ),
              ),
            ),
          ],
        ),
      );
}
