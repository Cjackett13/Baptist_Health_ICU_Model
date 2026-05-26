import 'package:flutter/material.dart';

import '../features/home/demo_hospitals.dart';
import '../theme/app_colors.dart';
import 'patient_sign_in_screen.dart';
import 'role_selection_screen.dart';

/// Patient flow step 1 — choose which hospital site you are at.
class PatientHospitalScreen extends StatefulWidget {
  const PatientHospitalScreen({super.key});

  @override
  State<PatientHospitalScreen> createState() => _PatientHospitalScreenState();
}

class _PatientHospitalScreenState extends State<PatientHospitalScreen> {
  final TextEditingController _searchController = TextEditingController();
  String _query = '';

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  List<DemoHospital> get _filtered {
    final q = _query.trim().toLowerCase();
    if (q.isEmpty) return demoHospitals;
    return demoHospitals.where((h) {
      final haystack =
          '${h.name} ${h.city} ${h.state} ${h.zip} ${h.addressLine1}'
              .toLowerCase();
      return haystack.contains(q);
    }).toList();
  }

  void _openSignIn(DemoHospital hospital) {
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => PatientSignInScreen(hospital: hospital),
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
        title: const Text('Select your hospital'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.of(context).pushReplacement(
            MaterialPageRoute<void>(
              builder: (_) => const RoleSelectionScreen(),
            ),
          ),
        ),
      ),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            color: Colors.white,
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Which location are you at?',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.w700,
                    color: BhColors.ink,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  'You will then choose your profile from the '
                  '$patientsPerHospital patients at that site.',
                  style: TextStyle(
                    fontSize: 13,
                    color: Colors.black.withValues(alpha: 0.55),
                    height: 1.4,
                  ),
                ),
                const SizedBox(height: 14),
                TextField(
                  controller: _searchController,
                  onChanged: (v) => setState(() => _query = v),
                  decoration: InputDecoration(
                    hintText: 'Search by city, zip, or address',
                    prefixIcon: const Icon(Icons.search, size: 20),
                    filled: true,
                    fillColor: const Color(0xFFF2F2F2),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                      borderSide: BorderSide.none,
                    ),
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            child: ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: _filtered.length,
              separatorBuilder: (_, __) => const SizedBox(height: 10),
              itemBuilder: (context, index) {
                final hospital = _filtered[index];
                return Material(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(14),
                  child: InkWell(
                    onTap: () => _openSignIn(hospital),
                    borderRadius: BorderRadius.circular(14),
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.all(10),
                            decoration: BoxDecoration(
                              color: BhColors.primary.withValues(alpha: 0.12),
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: const Icon(
                              Icons.local_hospital_outlined,
                              color: BhColors.primary,
                              size: 28,
                            ),
                          ),
                          const SizedBox(width: 14),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  hospital.name,
                                  style: const TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.w700,
                                    color: BhColors.ink,
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  hospital.addressShort,
                                  style: const TextStyle(
                                    fontSize: 12,
                                    color: Colors.black54,
                                  ),
                                ),
                              ],
                            ),
                          ),
                          const Icon(
                            Icons.arrow_forward_ios_rounded,
                            size: 16,
                            color: Colors.black26,
                          ),
                        ],
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
