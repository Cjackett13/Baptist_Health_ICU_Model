import 'package:flutter/material.dart';

class PatientRecord {
  const PatientRecord({
    required this.rank,
    required this.name,
    required this.id,
    required this.condition,
    required this.roomNumber,
    this.primaryDoctor,
    this.issue,
    this.vitals,
  });

  final int rank;
  final String name;
  final String id;
  final String condition;
  final String roomNumber;
  final String? primaryDoctor;
  final String? issue;
  final String? vitals;
}

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
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(18),
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: Colors.black12),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.06),
              blurRadius: 16,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: Row(
          children: [
            Container(
              width: 42,
              height: 42,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(color: Colors.black12),
                color: const Color(0xFFF7F7F7),
              ),
              child: Center(
                child: Text(
                  '${patient.rank}',
                  style: const TextStyle(
                    color: Colors.black,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    patient.name,
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                      color: Colors.black,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Patient ID ${patient.id}',
                    style: const TextStyle(fontSize: 14, color: Colors.black54),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    _conditionLabel(patient.condition),
                    style: TextStyle(
                      fontSize: 14,
                      color: _conditionColor(patient.condition),
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
            const Icon(Icons.chevron_right_rounded, color: Colors.black26),
          ],
        ),
      ),
    );
  }
}

void showPatientDetails(BuildContext context, PatientRecord patient) {
  showModalBottomSheet<void>(
    context: context,
    backgroundColor: Colors.white,
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
    ),
    builder: (context) {
      return Padding(
        padding: const EdgeInsets.fromLTRB(20, 24, 20, 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    color: const Color(0xFFF7F7F7),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(
                    Icons.person_outline,
                    color: Colors.black54,
                    size: 26,
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Text(
                    'Patient details',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
                IconButton(
                  onPressed: () => Navigator.pop(context),
                  icon: const Icon(Icons.close),
                ),
              ],
            ),
            const SizedBox(height: 22),
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        patient.name,
                        style: const TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Patient ID: ${patient.id}',
                        style: const TextStyle(
                          color: Colors.black54,
                          fontSize: 14,
                        ),
                      ),
                    ],
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 8,
                  ),
                  decoration: BoxDecoration(
                    color: _conditionColor(patient.condition).withOpacity(0.12),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    _conditionLabel(patient.condition),
                    style: TextStyle(
                      color: _conditionColor(patient.condition),
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 22),
            _detailRow('Room number', patient.roomNumber),
            const SizedBox(height: 12),
            _detailRow('Primary care doctor', patient.primaryDoctor),
            const SizedBox(height: 12),
            _detailRow('Issue', patient.issue),
            const SizedBox(height: 12),
            _detailRow('Vitals', patient.vitals),
            const SizedBox(height: 24),
            Center(
              child: FilledButton(
                style: FilledButton.styleFrom(
                  backgroundColor: Colors.black,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(
                    horizontal: 32,
                    vertical: 14,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                onPressed: () => Navigator.pop(context),
                child: const Text('Close'),
              ),
            ),
          ],
        ),
      );
    },
  );
}

String _conditionLabel(String condition) {
  final normalized = condition.toLowerCase();
  if (normalized.contains('critical')) {
    return 'Critical';
  }
  if (normalized.contains('severe') ||
      normalized.contains('serious') ||
      normalized.contains('moderate')) {
    return 'Moderate';
  }
  return condition;
}

Color _conditionColor(String condition) {
  final normalized = condition.toLowerCase();
  if (normalized.contains('critical')) {
    return Colors.red;
  }
  if (normalized.contains('severe') ||
      normalized.contains('serious') ||
      normalized.contains('moderate')) {
    return Colors.orange;
  }
  return Colors.green;
}

Widget _detailRow(String title, String? value) {
  final displayValue = (value == null || value.trim().isEmpty) ? 'N/A' : value;
  return Row(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      SizedBox(
        width: 130,
        child: Text(
          title,
          style: const TextStyle(
            color: Colors.black54,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      Expanded(
        child: Text(
          displayValue,
          style: const TextStyle(
            color: Colors.black,
            fontWeight: FontWeight.w500,
          ),
        ),
      ),
    ],
  );
}