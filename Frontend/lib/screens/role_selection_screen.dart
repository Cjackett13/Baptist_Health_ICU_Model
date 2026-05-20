import 'package:flutter/material.dart';

import '../theme/app_colors.dart';
import 'home_page.dart';
import 'patient_sign_in_screen.dart';

enum AppRole { clinician, patient }

class RoleSelectionScreen extends StatelessWidget {
  const RoleSelectionScreen({super.key});

  static const double _boxGap = 32;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: BhColors.background,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(height: 32),
              Image.asset(
                'assets/baptist_logo.png',
                height: 56,
                fit: BoxFit.contain,
              ),
              const SizedBox(height: 24),
              Text(
                'Baptist Health Cardiogenic Shock Tracker',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                      color: BhColors.ink,
                      fontWeight: FontWeight.w700,
                      fontFamily: 'Georgia',
                    ),
              ),
              const SizedBox(height: 8),
              Text(
                'Choose how you want to use this workspace',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: BhColors.slate,
                    ),
              ),
              Expanded(
                child: LayoutBuilder(
                  builder: (context, constraints) {
                    final maxW = constraints.maxWidth;
                    final maxH = constraints.maxHeight;
                    final boxSize = _squareSize(maxW, maxH);

                    return Center(
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          SizedBox(
                            width: boxSize,
                            height: boxSize,
                            child: _RoleCard(
                              title: 'Nurse / Physician',
                              subtitle:
                                  'Patient list, vitals, and ML predictions.',
                              icon: Icons.local_hospital_outlined,
                              accent: BhColors.ink,
                              onTap: () => _open(context, AppRole.clinician),
                            ),
                          ),
                          const SizedBox(width: _boxGap),
                          SizedBox(
                            width: boxSize,
                            height: boxSize,
                            child: _RoleCard(
                              title: 'Patient',
                              subtitle:
                                  'Length of stay, recommendations, and meds.',
                              icon: Icons.person_outline,
                              accent: BhColors.primary,
                              onTap: () => _openPatientSignIn(context),
                            ),
                          ),
                        ],
                      ),
                    );
                  },
                ),
              ),
              Text(
                'Demo data sourced from project parquet tables (person, encounter, '
                'clinical_event, diagnosis, medication_admin, scai_stage_hourly).',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: Colors.black38,
                      fontSize: 11,
                    ),
              ),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }

  /// Equal square tiles, centered, with fixed gap between them.
  double _squareSize(double maxWidth, double maxHeight) {
    const maxBox = 286.0; // 220 × 1.3
    const minBox = 182.0; // 140 × 1.3
    final fromWidth = (maxWidth - _boxGap) / 2;
    final fromHeight = maxHeight;
    return (fromWidth < fromHeight ? fromWidth : fromHeight)
        .clamp(minBox, maxBox);
  }

  void _open(BuildContext context, AppRole role) {
    Navigator.of(context).pushReplacement(
      MaterialPageRoute<void>(
        builder: (_) => HomePage(role: role),
      ),
    );
  }

  void _openPatientSignIn(BuildContext context) {
    Navigator.of(context).pushReplacement(
      MaterialPageRoute<void>(
        builder: (_) => const PatientSignInScreen(),
      ),
    );
  }
}

class _RoleCard extends StatelessWidget {
  const _RoleCard({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.accent,
    required this.onTap,
  });

  final String title;
  final String subtitle;
  final IconData icon;
  final Color accent;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white,
      borderRadius: BorderRadius.circular(20),
      elevation: 0,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(20),
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: Colors.black.withValues(alpha: 0.08)),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.05),
                blurRadius: 16,
                offset: const Offset(0, 6),
              ),
            ],
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: accent.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(icon, color: accent, size: 36),
              ),
              const SizedBox(height: 14),
              Text(
                title,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w700,
                  color: BhColors.ink,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                subtitle,
                textAlign: TextAlign.center,
                maxLines: 3,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                  fontSize: 11,
                  color: Colors.black54,
                  height: 1.35,
                ),
              ),
              const SizedBox(height: 12),
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    'Continue',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                      color: accent,
                    ),
                  ),
                  const SizedBox(width: 4),
                  Icon(Icons.arrow_forward_rounded, size: 14, color: accent),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
