import 'package:flutter/material.dart';

import '../../models/patient_prediction.dart';
import '../../services/patient_repository.dart';
import '../../services/patient_summary_pdf.dart';
import '../home/demo_hospitals.dart';
import 'patient_care_chat_sheet.dart';
import 'patient_record_sections.dart';
import '../../screens/patient_hospital_screen.dart';
import '../../screens/role_selection_screen.dart';
import '../../theme/app_colors.dart';

/// Patient dashboard — shows one patient's care details (no multi-patient list).
class PatientHomeScreen extends StatelessWidget {
  const PatientHomeScreen({
    required this.patientId,
    required this.hospital,
    super.key,
  });

  final String patientId;
  final DemoHospital hospital;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: PatientRepository.instance,
      builder: (context, _) {
        final patient = PatientRepository.instance.patientById(patientId);
        if (patient == null) {
          return Scaffold(
            appBar: AppBar(title: const Text('My care')),
            body: const Center(
              child: Text('Could not load your care record.'),
            ),
          );
        }
        return _PatientHomeBody(patient: patient, hospital: hospital);
      },
    );
  }
}

class _PatientHomeBody extends StatelessWidget {
  const _PatientHomeBody({
    required this.patient,
    required this.hospital,
  });

  final PatientRecord patient;
  final DemoHospital hospital;

  @override
  Widget build(BuildContext context) {
    final firstName = patient.name.split(' ').first;

    return Scaffold(
      backgroundColor: BhColors.background,
      appBar: AppBar(
        backgroundColor: BhColors.ink,
        foregroundColor: Colors.white,
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              firstName,
              style: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w700,
              ),
            ),
            Text(
              hospital.city,
              style: TextStyle(
                fontSize: 11,
                color: Colors.white.withValues(alpha: 0.75),
              ),
            ),
          ],
        ),
        actions: [
          IconButton(
            tooltip: 'Export PDF for family or work',
            onPressed: () => _sharePdf(context, patient, hospital),
            icon: const Icon(Icons.picture_as_pdf_outlined),
          ),
          TextButton(
            onPressed: () => Navigator.of(context).pushAndRemoveUntil(
              MaterialPageRoute<void>(
                builder: (_) => const RoleSelectionScreen(),
              ),
              (_) => false,
            ),
            child: const Text(
              'Switch View',
              style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _WelcomeCard(
                    patient: patient,
                    hospital: hospital,
                    onExportPdf: () => _sharePdf(context, patient, hospital),
                  ),
                  const SizedBox(height: 16),
                  PatientRecordSections(patient: patient),
                ],
              ),
            ),
          ),
          _ChatBar(
            onTap: () => PatientCareChatSheet.show(context, patient),
          ),
        ],
      ),
    );
  }
}

Future<void> _sharePdf(
  BuildContext context,
  PatientRecord patient,
  DemoHospital hospital,
) async {
  try {
    await PatientSummaryPdf.share(
      patient,
      forFamily: true,
      hospitalName: hospital.name,
      hospitalCity: hospital.city,
    );
  } catch (e) {
    if (context.mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not export PDF: $e')),
      );
    }
  }
}

class _WelcomeCard extends StatelessWidget {
  const _WelcomeCard({
    required this.patient,
    required this.hospital,
    required this.onExportPdf,
  });
  final PatientRecord patient;
  final DemoHospital hospital;
  final VoidCallback onExportPdf;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.black.withValues(alpha: 0.06)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.04),
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
              CircleAvatar(
                radius: 26,
                backgroundColor: BhColors.primary.withValues(alpha: 0.15),
                child: Text(
                  patient.name.isNotEmpty ? patient.name[0].toUpperCase() : '?',
                  style: const TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.w700,
                    color: BhColors.ink,
                  ),
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      patient.name,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w700,
                        color: BhColors.ink,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Room ${patient.roomNumber} · ${hospital.name}',
                      style: const TextStyle(
                        fontSize: 12,
                        color: Colors.black54,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          if (patient.diagnosis != null) ...[
            const SizedBox(height: 12),
            Text(
              patient.diagnosis!,
              style: const TextStyle(
                fontSize: 13,
                color: Colors.black87,
                height: 1.35,
              ),
            ),
          ],
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed: onExportPdf,
            style: FilledButton.styleFrom(
              backgroundColor: BhColors.primary,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 14),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
            icon: const Icon(Icons.picture_as_pdf_outlined, size: 20),
            label: const Text(
              'Export PDF for family or work',
              style: TextStyle(fontWeight: FontWeight.w700),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Share length of stay, diagnoses, medications, and recovery tips. '
            'Clinical risk scores are not included.',
            style: TextStyle(
              fontSize: 11,
              color: Colors.black.withValues(alpha: 0.45),
              height: 1.35,
            ),
          ),
          const SizedBox(height: 12),
          OutlinedButton.icon(
            onPressed: () => Navigator.of(context).pushAndRemoveUntil(
              MaterialPageRoute<void>(
                builder: (_) => const PatientHospitalScreen(),
              ),
              (_) => false,
            ),
            icon: const Icon(Icons.swap_horiz, size: 18),
            label: const Text('Not you? Change hospital or profile'),
          ),
        ],
      ),
    );
  }
}

class _ChatBar extends StatelessWidget {
  const _ChatBar({required this.onTap});
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white,
      elevation: 8,
      child: SafeArea(
        top: false,
        child: InkWell(
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            child: Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: BhColors.primary.withValues(alpha: 0.12),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(
                    Icons.chat_outlined,
                    color: BhColors.primary,
                  ),
                ),
                const SizedBox(width: 12),
                const Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Ask about your care',
                        style: TextStyle(
                          fontSize: 15,
                          fontWeight: FontWeight.w700,
                          color: BhColors.ink,
                        ),
                      ),
                      Text(
                        'Medications, costs, and recovery tips',
                        style: TextStyle(fontSize: 12, color: Colors.black54),
                      ),
                    ],
                  ),
                ),
                const Icon(Icons.chevron_right, color: Colors.black26),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
