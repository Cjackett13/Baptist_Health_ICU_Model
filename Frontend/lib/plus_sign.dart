import 'package:flutter/material.dart';

import 'alert_system.dart';

class PlusSignButton extends StatelessWidget {
  const PlusSignButton({
    required this.onPatientCreated,
    super.key,
  });

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
          padding: const EdgeInsets.all(12),
          elevation: 0,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        ),
        child: const Icon(Icons.add, size: 28),
      ),
    );
  }

  Future<void> _showCreatePatientDialog(BuildContext context) async {
    final nameController = TextEditingController();
    final idController = TextEditingController();
    final roomController = TextEditingController();
    final issueController = TextEditingController();
    final doctorController = TextEditingController();
    final vitalsController = TextEditingController();

    final createdPatient = await showDialog<PatientRecord>(
      context: context,
      builder: (dialogContext) {
        return AlertDialog(
          title: const Text('Create new patient'),
          content: SizedBox(
            width: 420,
            child: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  _buildTextField(
                    controller: nameController,
                    label: 'Name',
                    requiredField: true,
                  ),
                  const SizedBox(height: 10),
                  _buildTextField(
                    controller: idController,
                    label: 'ID',
                    requiredField: true,
                  ),
                  const SizedBox(height: 10),
                  _buildTextField(
                    controller: roomController,
                    label: 'Room number',
                    requiredField: true,
                  ),
                  const SizedBox(height: 10),
                  _buildTextField(controller: issueController, label: 'Issue'),
                  const SizedBox(height: 10),
                  _buildTextField(
                    controller: doctorController,
                    label: 'Doctor',
                  ),
                  const SizedBox(height: 10),
                  _buildTextField(
                    controller: vitalsController,
                    label: 'Vitals',
                    maxLines: 2,
                  ),
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
                      content: Text('Please fill in name, ID, and room number.'),
                    ),
                  );
                  return;
                }

                Navigator.of(dialogContext).pop(
                  PatientRecord(
                    rank: 0,
                    name: name,
                    id: id,
                    condition: 'Stable',
                    roomNumber: room,
                    issue: _optionalValue(issueController.text),
                    primaryDoctor: _optionalValue(doctorController.text),
                    vitals: _optionalValue(vitalsController.text),
                  ),
                );
              },
              child: const Text('Create'),
            ),
          ],
        );
      },
    );

    if (createdPatient != null) {
      onPatientCreated(createdPatient);
    }

    nameController.dispose();
    idController.dispose();
    roomController.dispose();
    issueController.dispose();
    doctorController.dispose();
    vitalsController.dispose();
  }

  static Widget _buildTextField({
    required TextEditingController controller,
    required String label,
    bool requiredField = false,
    int maxLines = 1,
  }) {
    return TextField(
      controller: controller,
      maxLines: maxLines,
      decoration: InputDecoration(
        labelText: requiredField ? '$label *' : label,
        border: const OutlineInputBorder(),
      ),
    );
  }

  static String? _optionalValue(String value) {
    final trimmed = value.trim();
    return trimmed.isEmpty ? null : trimmed;
  }
}