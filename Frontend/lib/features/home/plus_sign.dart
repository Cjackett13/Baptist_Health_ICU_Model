// lib/plus_sign.dart
import 'dart:math';
import 'package:flutter/material.dart';
import '../patient_detail/alert_system.dart';

class PlusSignButton extends StatelessWidget {
  const PlusSignButton({required this.onPatientCreated, super.key});

  final ValueChanged<PatientRecord> onPatientCreated;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 52,
      height: 52,
      child: FilledButton(
        onPressed: () => _showCreatePatientDialog(context),
        style: FilledButton.styleFrom(
          backgroundColor: const Color(0xFF393E46),
          foregroundColor: Colors.white,
          padding: EdgeInsets.zero,
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
        child: const Icon(Icons.add, size: 24),
      ),
    );
  }

  Future<void> _showCreatePatientDialog(BuildContext context) async {
    final nameController = TextEditingController();
    final idController = TextEditingController();
    final roomController = TextEditingController();
    final issueController = TextEditingController();
    final doctorController = TextEditingController();

    final createdPatient = await showDialog<PatientRecord>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: const Text('Create new patient'),
        content: SizedBox(
          width: 420,
          child: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                _field(nameController, 'Name', required: true),
                const SizedBox(height: 10),
                _field(idController, 'ID', required: true),
                const SizedBox(height: 10),
                _field(roomController, 'Room number', required: true),
                const SizedBox(height: 10),
                _field(issueController, 'Issue'),
                const SizedBox(height: 10),
                _field(doctorController, 'Doctor'),
              ],
            ),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () {
              final name = nameController.text.trim();
              final id = idController.text.trim();
              final room = roomController.text.trim();

              if (name.isEmpty || id.isEmpty || room.isEmpty) {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text(
                        'Please fill in name, ID, and room number.'),
                  ),
                );
                return;
              }

              // New patients start with low default risk.
              // Overwritten by real model predictions once FastAPI connected.
              final rng = Random();
              final icuRisk = 0.08 + rng.nextDouble() * 0.18;
              const features = PatientFeatures(
                numMedications: 5,
                numberInpatient: 0,
                numLabProcedures: 20,
                timeInHospital: 1,
                numberDiagnoses: 3,
                numberEmergency: 0,
                numberOutpatient: 0,
                ageMid: 55,
              );

              Navigator.of(dialogContext).pop(
                PatientRecord(
                  rank: 0,
                  name: name,
                  id: id,
                  roomNumber: room,
                  predictions:
                      generateMockPredictions(icuRisk, features, rng),
                  features: features,
                  issue: issueController.text.trim().isEmpty
                      ? null
                      : issueController.text.trim(),
                  primaryDoctor: doctorController.text.trim().isEmpty
                      ? null
                      : doctorController.text.trim(),
                ),
              );
            },
            child: const Text('Create'),
          ),
        ],
      ),
    );

    if (createdPatient != null) {
      onPatientCreated(createdPatient);
    }

    nameController.dispose();
    idController.dispose();
    roomController.dispose();
    issueController.dispose();
    doctorController.dispose();
  }

  static Widget _field(
    TextEditingController c,
    String label, {
    bool required = false,
  }) =>
      TextField(
        controller: c,
        decoration: InputDecoration(
          labelText: required ? '$label *' : label,
          border: const OutlineInputBorder(),
        ),
      );
}