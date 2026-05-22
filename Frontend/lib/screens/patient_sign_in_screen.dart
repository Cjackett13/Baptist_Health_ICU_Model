import 'package:flutter/material.dart';

import '../features/home/demo_hospitals.dart';
import '../features/patient_portal/patient_home_screen.dart';
import '../models/patient_prediction.dart';
import '../services/patient_repository.dart';
import '../theme/app_colors.dart';

/// Patient flow step 2 — confirm identity, then open personal dashboard.
class PatientSignInScreen extends StatefulWidget {
  const PatientSignInScreen({required this.hospital, super.key});

  final DemoHospital hospital;

  @override
  State<PatientSignInScreen> createState() => _PatientSignInScreenState();
}

class _PatientSignInScreenState extends State<PatientSignInScreen> {
  List<PatientRecord> _patients = [];
  PatientRecord? _selected;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final patients = await PatientRepository.instance.loadPatients();
      if (!mounted) return;
      final atSite = patients
          .where((p) => p.demoHospitalId == widget.hospital.id)
          .toList();
      setState(() {
        _patients = atSite;
        _loading = false;
        if (atSite.length == 1) {
          _selected = atSite.first;
        }
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  void _continue() {
    final patient = _selected;
    if (patient == null) return;
    Navigator.of(context).pushReplacement(
      MaterialPageRoute<void>(
        builder: (_) => PatientHomeScreen(
          patientId: patient.id,
          hospital: widget.hospital,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: BhColors.background,
      appBar: AppBar(
        backgroundColor: BhColors.ink,
        foregroundColor: Colors.white,
        title: Text(widget.hospital.city),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? Center(child: Text('Error: $_error'))
              : SingleChildScrollView(
                  padding: const EdgeInsets.all(24),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      const Text(
                        'Confirm your identity',
                        style: TextStyle(
                          fontSize: 22,
                          fontWeight: FontWeight.w700,
                          color: BhColors.ink,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Select your name at ${widget.hospital.name}. '
                        'You will go straight to your personal care page.',
                        style: TextStyle(
                          fontSize: 14,
                          color: Colors.black.withValues(alpha: 0.55),
                          height: 1.4,
                        ),
                      ),
                      const SizedBox(height: 24),
                      Container(
                        padding: const EdgeInsets.all(20),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(
                            color: Colors.black.withValues(alpha: 0.08),
                          ),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            const Text(
                              'I am',
                              style: TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.w600,
                                color: Colors.black54,
                              ),
                            ),
                            const SizedBox(height: 10),
                            DropdownButtonFormField<PatientRecord>(
                              initialValue: _selected,
                              isExpanded: true,
                              decoration: InputDecoration(
                                filled: true,
                                fillColor: const Color(0xFFF8F8F8),
                                border: OutlineInputBorder(
                                  borderRadius: BorderRadius.circular(12),
                                ),
                              ),
                              hint: const Text('Choose your name'),
                              items: _patients
                                  .map(
                                    (p) => DropdownMenuItem(
                                      value: p,
                                      child: Text(
                                        '${p.name} · Room ${p.roomNumber}',
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                    ),
                                  )
                                  .toList(),
                              onChanged: (p) =>
                                  setState(() => _selected = p),
                            ),
                            if (_selected != null) ...[
                              const SizedBox(height: 16),
                              _IdentityPreview(patient: _selected!),
                            ],
                          ],
                        ),
                      ),
                      const SizedBox(height: 24),
                      FilledButton(
                        onPressed: _selected == null ? null : _continue,
                        style: FilledButton.styleFrom(
                          backgroundColor: BhColors.primary,
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                        child: const Text(
                          'View my care',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
    );
  }
}

class _IdentityPreview extends StatelessWidget {
  const _IdentityPreview({required this.patient});
  final PatientRecord patient;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: BhColors.primary.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          CircleAvatar(
            backgroundColor: BhColors.primary.withValues(alpha: 0.2),
            child: Text(
              patient.name.isNotEmpty ? patient.name[0].toUpperCase() : '?',
              style: const TextStyle(
                fontWeight: FontWeight.w700,
                color: BhColors.ink,
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
                    fontWeight: FontWeight.w700,
                    fontSize: 15,
                  ),
                ),
                if (patient.diagnosis != null)
                  Text(
                    patient.diagnosis!,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
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
    );
  }
}
